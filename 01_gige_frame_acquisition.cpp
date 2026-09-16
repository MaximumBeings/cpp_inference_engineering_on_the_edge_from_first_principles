// Chapter 18.1 -- Every earlier chapter's real file was already sitting on
// disk before this book's own code ever touched it: a GGUF checkpoint, a
// conversation's own text. A manufacturing line's camera produces its
// frames over a network, one unreliable UDP-sized packet at a time, and
// the first job any edge inference engine has is turning that packet
// stream back into a single, complete image before Section 18.2's vision
// encoder can see a single pixel.
//
// A note on scope, stated up front because this book states its scope
// honestly everywhere else: this section's own wire format is INSPIRED
// BY, not a byte-exact implementation of, the real GigE Vision Streaming
// Protocol (GVSP) the machine-vision industry actually uses. The real
// standard's packet-format byte packs several flag bits (extended IDs,
// resend markers) this book has no way to verify against real hardware
// in this environment, and claiming byte-exact spec compliance without
// a real camera to check it against would be exactly the kind of
// unearned promise Section 16.1 already refused to make about a CLI
// flag. What this section DOES teach faithfully is the real, general
// problem every frame-grabber protocol like GVSP exists to solve: a
// LEADER packet announces a new frame's own dimensions and format, a
// run of PAYLOAD packets carry the actual pixel bytes at explicit,
// independent offsets (so out-of-order delivery is not corruption), and
// a TRAILER packet closes the frame -- at which point the only question
// that matters is whether every payload chunk this frame promised
// actually arrived.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_gige_frame_acquisition.cpp -o 01_gige_frame_acquisition
// Run:     ./01_gige_frame_acquisition

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: this section's own GVSP-inspired packet format. Every packet
// shares one small header; what follows it is interpreted differently
// depending on the header's own `format` field, exactly as the real
// protocol's leader/payload/trailer split works.
// =======================================================================
enum class PacketFormat : uint8_t { LEADER = 1, PAYLOAD = 2, TRAILER = 3 };
enum class PixelFormat : uint32_t { MONO8 = 0, RGB8 = 1 };

int bytes_per_pixel(PixelFormat fmt) {
    switch (fmt) {
        case PixelFormat::MONO8: return 1;
        case PixelFormat::RGB8: return 3;
    }
    return 0;
}

struct GVSPHeader {
    uint16_t status = 0;         // 0 == success, matching the real spec's own convention
    uint16_t block_id = 0;       // which frame this packet belongs to
    PacketFormat format = PacketFormat::PAYLOAD;
    uint32_t packet_id = 0;      // leader=0, trailer=UINT32_MAX, payload=chunk index (NOT arrival order)
};

struct LeaderPayload {
    PixelFormat pixel_format = PixelFormat::MONO8;
    uint32_t size_x = 0, size_y = 0;
    uint64_t timestamp_ns = 0;
};

struct TrailerPayload {
    uint32_t total_payload_packets = 0;   // how many PAYLOAD packets this frame promises
};

struct GVSPPacket {
    GVSPHeader header;
    LeaderPayload leader;               // valid only when header.format == LEADER
    TrailerPayload trailer;             // valid only when header.format == TRAILER
    std::vector<uint8_t> chunk;         // valid only when header.format == PAYLOAD
};

constexpr uint32_t TRAILER_PACKET_ID = 0xFFFFFFFFu;
constexpr size_t CHUNK_BYTES = 256;    // this section's own fixed payload-packet size

// =======================================================================
// PART 2: the frame reassembler. This is the section's real teaching
// point: reconstructing one contiguous frame buffer from packets that
// may arrive out of order, and refusing to report a frame as complete
// when a payload packet never arrived at all.
// =======================================================================
struct FrameResult {
    bool ok = false;
    uint16_t block_id = 0;
    uint32_t size_x = 0, size_y = 0;
    PixelFormat pixel_format = PixelFormat::MONO8;
    std::vector<uint8_t> pixels;
    uint32_t packets_expected = 0, packets_received = 0;
};

class FrameReassembler {
public:
    // Feeds one packet in. Returns a completed (or incomplete-but-closed)
    // frame the instant a TRAILER for the currently active block arrives;
    // returns std::nullopt while a frame is still being assembled.
    std::optional<FrameResult> on_packet(const GVSPPacket& pkt) {
        const auto& h = pkt.header;
        if (h.format == PacketFormat::LEADER) {
            // A new leader always starts a fresh block, even if the
            // previous one was never closed -- a real camera's next
            // frame does not wait for this book's own reassembler to
            // finish complaining about the last one.
            active_ = true;
            block_id_ = h.block_id;
            size_x_ = pkt.leader.size_x;
            size_y_ = pkt.leader.size_y;
            pixel_format_ = pkt.leader.pixel_format;
            const size_t total_bytes = static_cast<size_t>(size_x_) * size_y_ * bytes_per_pixel(pixel_format_);
            buffer_.assign(total_bytes, 0);
            received_chunks_.clear();
            return std::nullopt;
        }
        if (!active_ || h.block_id != block_id_) {
            // A payload or trailer for a block we never saw a leader
            // for (or the wrong block): this book's own reassembler
            // drops it rather than guessing where it belongs.
            return std::nullopt;
        }
        if (h.format == PacketFormat::PAYLOAD) {
            const size_t offset = static_cast<size_t>(h.packet_id) * CHUNK_BYTES;
            if (offset < buffer_.size() && received_chunks_.insert(h.packet_id).second) {
                const size_t n = std::min(pkt.chunk.size(), buffer_.size() - offset);
                std::memcpy(buffer_.data() + offset, pkt.chunk.data(), n);
            }
            return std::nullopt;
        }
        // TRAILER: close the block regardless of completeness, and
        // report the truth about how much of it actually arrived.
        FrameResult res;
        res.block_id = block_id_;
        res.size_x = size_x_;
        res.size_y = size_y_;
        res.pixel_format = pixel_format_;
        res.packets_expected = pkt.trailer.total_payload_packets;
        res.packets_received = static_cast<uint32_t>(received_chunks_.size());
        res.ok = (res.packets_received == res.packets_expected);
        res.pixels = std::move(buffer_);
        active_ = false;
        return res;
    }

private:
    bool active_ = false;
    uint16_t block_id_ = 0;
    uint32_t size_x_ = 0, size_y_ = 0;
    PixelFormat pixel_format_ = PixelFormat::MONO8;
    std::vector<uint8_t> buffer_;
    std::unordered_set<uint32_t> received_chunks_;
};

// =======================================================================
// PART 3: hardware triggering. A real inspection camera on a
// manufacturing line does not free-run -- it exposes exactly once per
// part, on a signal from the line itself (a photoelectric sensor, a PLC
// output), and the controller's own job is refusing a second exposure
// request while the first one's frame is still in flight, and reporting
// -- never silently hanging on -- a trigger whose frame never arrives at
// all, or arrives too late to matter to a line that has already moved
// the part along.
// =======================================================================
enum class TriggerMode { OFF, SOFTWARE, HARDWARE_RISING_EDGE };

struct TriggerEvent {
    bool accepted = false;
    std::string reason;     // set when accepted == false
    uint64_t trigger_tick = 0;
};

struct InspectionResult {
    bool ok = false;             // true: a complete frame arrived within the deadline
    bool timed_out = false;      // true: the deadline passed with no frame at all
    uint64_t trigger_tick = 0;
    uint64_t frame_ready_tick = 0;
    uint64_t latency_ticks = 0;
};

// Time here is a caller-supplied, monotonically increasing tick count,
// never a real wall clock -- exactly this book's own standing discipline
// against locking real timing into cross-architecture-compared output,
// applied to a state machine instead of a benchmark this time. A real
// deployment feeds real timestamps in; this section's own self-tests
// feed in deterministic, hand-chosen ones.
class TriggerController {
public:
    TriggerController(TriggerMode mode, uint64_t deadline_ticks)
        : mode_(mode), deadline_ticks_(deadline_ticks) {}

    TriggerMode mode() const { return mode_; }
    bool busy() const { return busy_; }

    TriggerEvent fire(uint64_t now_tick) {
        TriggerEvent ev;
        if (mode_ == TriggerMode::OFF) { ev.reason = "trigger mode is OFF"; return ev; }
        if (busy_) { ev.reason = "camera busy: previous trigger's frame has not completed"; return ev; }
        busy_ = true;
        pending_trigger_tick_ = now_tick;
        ev.accepted = true;
        ev.trigger_tick = now_tick;
        return ev;
    }

    // Called once a FrameReassembler reports a completed (or
    // incomplete-but-closed) frame for this trigger. Ends the "busy"
    // state either way -- a dropped frame still frees the camera for the
    // NEXT trigger, exactly as a real inspection cell must keep moving
    // rather than stall the whole line over one bad frame.
    InspectionResult on_frame(bool frame_ok, uint64_t frame_ready_tick) {
        InspectionResult res;
        res.trigger_tick = pending_trigger_tick_;
        res.frame_ready_tick = frame_ready_tick;
        res.latency_ticks = frame_ready_tick - pending_trigger_tick_;
        res.ok = frame_ok && res.latency_ticks <= deadline_ticks_;
        busy_ = false;
        return res;
    }

    // Called when the deadline passes with no frame at all -- the
    // camera link itself is down, not just one dropped packet.
    InspectionResult on_timeout(uint64_t now_tick) {
        InspectionResult res;
        res.trigger_tick = pending_trigger_tick_;
        res.timed_out = true;
        res.latency_ticks = now_tick - pending_trigger_tick_;
        busy_ = false;
        return res;
    }

private:
    TriggerMode mode_;
    uint64_t deadline_ticks_;
    bool busy_ = false;
    uint64_t pending_trigger_tick_ = 0;
};

// =======================================================================
// PART 4: self-tests, against this section's own synthetic packet
// streams -- built, shuffled, and truncated by this file itself, so
// every check is fully deterministic and needs no real camera.
// =======================================================================
std::vector<GVSPPacket> build_frame_packets(uint16_t block_id, uint32_t w, uint32_t h,
                                             PixelFormat fmt, unsigned seed, uint64_t timestamp_ns) {
    std::vector<GVSPPacket> packets;
    GVSPHeader lh; lh.status = 0; lh.block_id = block_id; lh.format = PacketFormat::LEADER; lh.packet_id = 0;
    GVSPPacket leader; leader.header = lh;
    leader.leader.pixel_format = fmt; leader.leader.size_x = w; leader.leader.size_y = h;
    leader.leader.timestamp_ns = timestamp_ns;
    packets.push_back(leader);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    const size_t total_bytes = static_cast<size_t>(w) * h * bytes_per_pixel(fmt);
    std::vector<uint8_t> image(total_bytes);
    for (auto& b : image) b = static_cast<uint8_t>(byte_dist(rng));

    const uint32_t n_chunks = static_cast<uint32_t>((total_bytes + CHUNK_BYTES - 1) / CHUNK_BYTES);
    for (uint32_t i = 0; i < n_chunks; ++i) {
        GVSPHeader ph; ph.status = 0; ph.block_id = block_id; ph.format = PacketFormat::PAYLOAD; ph.packet_id = i;
        GVSPPacket pkt; pkt.header = ph;
        const size_t off = static_cast<size_t>(i) * CHUNK_BYTES;
        const size_t n = std::min(CHUNK_BYTES, total_bytes - off);
        pkt.chunk.assign(image.begin() + static_cast<long>(off), image.begin() + static_cast<long>(off + n));
        packets.push_back(pkt);
    }

    GVSPHeader th; th.status = 0; th.block_id = block_id; th.format = PacketFormat::TRAILER; th.packet_id = TRAILER_PACKET_ID;
    GVSPPacket trailer; trailer.header = th; trailer.trailer.total_payload_packets = n_chunks;
    packets.push_back(trailer);
    return packets;
}

std::vector<uint8_t> reference_image(uint32_t w, uint32_t h, PixelFormat fmt, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::vector<uint8_t> image(static_cast<size_t>(w) * h * bytes_per_pixel(fmt));
    for (auto& b : image) b = static_cast<uint8_t>(byte_dist(rng));
    return image;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 18.1: GigE Vision Frame Acquisition and Hardware Triggering\n";
    std::cout << "========================================================\n";

    constexpr uint32_t W = 32, H = 32;   // 1024 bytes MONO8 -> 4 payload chunks at CHUNK_BYTES=256

    std::cout << "\n-- Test 1: in-order packet stream reassembles exactly --\n";
    {
        auto packets = build_frame_packets(1, W, H, PixelFormat::MONO8, /*seed=*/7, /*ts=*/1000);
        auto ref = reference_image(W, H, PixelFormat::MONO8, 7);
        FrameReassembler asm1;
        std::optional<FrameResult> result;
        for (const auto& p : packets) { auto r = asm1.on_packet(p); if (r) result = r; }
        CHECK(result.has_value());
        CHECK(result->ok);
        CHECK(result->size_x == W && result->size_y == H);
        CHECK(result->packets_expected == result->packets_received);
        CHECK(result->pixels == ref);
        std::cout << "  " << result->packets_received << "/" << result->packets_expected
                   << " payload packets received, frame ok: " << (result->ok ? "yes" : "no")
                   << ", pixels match reference: " << (result->pixels == ref ? "yes" : "no") << "\n";
    }

    std::cout << "\n-- Test 2: shuffled (out-of-order) delivery reassembles identically --\n";
    {
        auto packets = build_frame_packets(2, W, H, PixelFormat::MONO8, /*seed=*/11, /*ts=*/2000);
        auto ref = reference_image(W, H, PixelFormat::MONO8, 11);
        // Shuffle only the PAYLOAD packets (indices 1..n-2); leader stays
        // first and trailer stays last, exactly as a real link guarantees
        // block boundaries even when individual packets within a block
        // race each other over the network.
        std::mt19937 shuffle_rng(99);
        std::shuffle(packets.begin() + 1, packets.end() - 1, shuffle_rng);
        FrameReassembler asm2;
        std::optional<FrameResult> result;
        for (const auto& p : packets) { auto r = asm2.on_packet(p); if (r) result = r; }
        CHECK(result.has_value());
        CHECK(result->ok);
        CHECK(result->pixels == ref);
        std::cout << "  reassembled from shuffled packet order, pixels still match reference: "
                   << (result->pixels == ref ? "yes" : "no") << "\n";
    }

    std::cout << "\n-- Test 3: a dropped payload packet is reported, not silently accepted --\n";
    {
        auto packets = build_frame_packets(3, W, H, PixelFormat::MONO8, /*seed=*/23, /*ts=*/3000);
        // Drop the third payload packet (index 2 among payloads, i.e.
        // packets[3]) -- a real link losing exactly one UDP-sized chunk.
        packets.erase(packets.begin() + 3);
        FrameReassembler asm3;
        std::optional<FrameResult> result;
        for (const auto& p : packets) { auto r = asm3.on_packet(p); if (r) result = r; }
        CHECK(result.has_value());
        CHECK(!result->ok);
        CHECK(result->packets_received == result->packets_expected - 1);
        std::cout << "  received " << result->packets_received << "/" << result->packets_expected
                   << " payload packets, frame correctly flagged incomplete (ok=" << (result->ok ? "true" : "false") << ")\n";
    }

    std::cout << "\n-- Test 4: trigger controller mode and busy-state enforcement --\n";
    {
        TriggerController off_ctrl(TriggerMode::OFF, /*deadline_ticks=*/100);
        auto ev0 = off_ctrl.fire(10);
        CHECK(!ev0.accepted);

        TriggerController ctrl(TriggerMode::SOFTWARE, /*deadline_ticks=*/100);
        auto ev1 = ctrl.fire(10);
        CHECK(ev1.accepted);
        CHECK(ctrl.busy());
        auto ev2 = ctrl.fire(20);   // second trigger while the first is still outstanding
        CHECK(!ev2.accepted);

        auto insp1 = ctrl.on_frame(/*frame_ok=*/true, /*frame_ready_tick=*/45);
        CHECK(!ctrl.busy());
        CHECK(insp1.ok);
        CHECK(insp1.latency_ticks == 35);

        auto ev3 = ctrl.fire(50);   // camera is free again now
        CHECK(ev3.accepted);
        std::cout << "  OFF mode rejects a fire(): yes; a second fire() while busy is rejected: yes; "
                     "camera frees up after on_frame(): yes; latency computed as trigger-to-frame-ready: "
                  << insp1.latency_ticks << " ticks\n";
    }

    std::cout << "\n-- Test 5: a missing frame times out, and a late frame is flagged, not accepted --\n";
    {
        TriggerController ctrl(TriggerMode::HARDWARE_RISING_EDGE, /*deadline_ticks=*/30);
        ctrl.fire(100);
        auto timeout_res = ctrl.on_timeout(/*now_tick=*/131);
        CHECK(timeout_res.timed_out);
        CHECK(!timeout_res.ok);
        CHECK(!ctrl.busy());

        ctrl.fire(200);
        auto late_res = ctrl.on_frame(/*frame_ok=*/true, /*frame_ready_tick=*/240);   // 40 ticks, deadline was 30
        CHECK(!late_res.timed_out);
        CHECK(!late_res.ok);   // frame arrived complete, but too late to matter
        std::cout << "  a trigger with no frame at all times out (timed_out=" << (timeout_res.timed_out ? "true" : "false")
                   << "); a COMPLETE frame arriving after the deadline is still flagged not ok "
                     "(ok=" << (late_res.ok ? "true" : "false") << ", latency=" << late_res.latency_ticks << " ticks > 30 tick deadline)\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
