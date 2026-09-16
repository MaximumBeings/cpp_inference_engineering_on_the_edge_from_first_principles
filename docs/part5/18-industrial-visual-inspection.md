# Chapter 18: Industrial Visual Inspection: Edge-Deployed Defect Detection on Manufacturing Lines

**What you will understand by the end of this chapter:**

- How a real machine-vision camera's frame actually arrives at an inference engine -- not as a single, ready-made image file, but as an unreliable stream of UDP-sized packets that a frame-grabber protocol like GigE Vision Streaming Protocol (GVSP) must reassemble, in the right order or not, before a single pixel is usable.
- How to build a from-scratch vision encoder in the style of Qwen2.5-VL: patchifying a raw pixel buffer, embedding each patch, applying two-dimensional rotary position encoding so a patch's row and column are both recoverable from its own vector, running a small stack of bidirectional transformer blocks reusing this book's own already-verified RMSNorm/matmul/SwiGLU machinery, and merging four spatially adjacent patches into one visual token sized to match a text decoder's embedding space.
- How to splice real visual tokens into a real text token sequence and run the SAME, unchanged Qwen2 decoder this book has verified since Chapter 15 over the combined sequence -- extending a locked function to accept a precomputed starting embedding instead of rewriting anything about how it processes a sequence once that embedding exists.
- How to turn a vision-language model's own confidence and verdict into one of three dispositions a conveyor's downstream gate actually understands, with an ASYMMETRIC threshold between "reject" and "accept" that reflects a real industrial tradeoff, and how to log every decision this line ever made into a real, queryable SQLite database using nothing but that library's own long-stable C ABI, declared directly rather than assumed to be available as a development package.
- How to model OPC UA's real data and service pattern -- typed, named nodes; Read, Write, and Subscribe services; real published status codes -- as a from-scratch, interface-level abstraction, stated honestly as exactly that rather than dressed up as a byte-exact wire-protocol implementation this environment has no real PLC or SCADA endpoint to verify against.

**What you need to know first:**

- Chapter 15's real Qwen2 transformer block and Chapter 16's complete `decode_step`/`prefill` pipeline -- this chapter's own vision-language fusion in Section 18.3 extends those exact, already-verified functions rather than building a second decoder that would need its own independent trust.
- Chapter 9's 1D rotary position encoding -- Section 18.2's two-dimensional version reuses the identical rotate-half mechanism, applied twice over disjoint halves of each attention head's dimension.
- This chapter's own honest-exception shape is broader than any earlier chapter's, and stated plainly rather than smoothed over: this is the first chapter in the book with no real hardware available to verify against at all -- no real GigE Vision camera, no multi-gigabyte Qwen2.5-VL checkpoint download, and no real PLC or SCADA endpoint. Every section states exactly which of its own pieces are real, fully verifiable, from-scratch implementations; which are stated pedagogical simplifications of a real spec this book cannot check byte-for-byte without hardware it does not have; and which are interface-level simulations of a real protocol, built to teach that protocol's real data model and service pattern honestly without claiming wire-level compliance nobody here can confirm.

---

Every earlier chapter's real file was already sitting on disk before this book's own code ever touched it -- a GGUF checkpoint, a conversation's own text. A manufacturing line is a different kind of deployment target entirely: its camera produces frames over a network in real time, its inspection model has to fuse what it sees with what it is told, its verdict has to become an action a conveyor's gate can act on within milliseconds, and its every decision has to be durably logged and reported upward to the SCADA system that runs the rest of the plant. This chapter builds all five of those pieces from scratch, in order -- frame acquisition, vision encoding, multimodal fusion, disposition and logging, and SCADA integration -- stating exactly where each one is a real, verifiable implementation and where it is an honest, clearly labeled simplification of a real industrial protocol this environment has no hardware to check byte-for-byte.

## 18.1 GigE Vision Frame Acquisition and Hardware Triggering

### Intuition

A manufacturing line's camera does not hand an inference engine a finished image. It hands it a stream of network packets, arriving over UDP, which offers no guarantee of order and no guarantee that every packet even arrives. The very first job any edge inference engine on a real line has -- before a single pixel reaches a vision encoder -- is turning that unreliable packet stream back into one complete, correctly ordered frame, and knowing honestly when it has failed to do so.

### The Concept, In Detail

This section's own wire format is stated up front, in the same voice this book has used everywhere else it cannot verify a real external spec byte-for-byte: it is INSPIRED BY, not a byte-exact implementation of, the real GigE Vision Streaming Protocol (GVSP) the machine-vision industry actually uses. The real standard packs several flag bits (extended IDs, resend markers) this book has no real camera to verify against, and claiming byte-exact compliance without one would be exactly the kind of unearned promise Section 16.1 already refused to make about a CLI flag. What this section teaches faithfully is the real, general problem every frame-grabber protocol like GVSP exists to solve: a LEADER packet announces a new frame's own dimensions and pixel format, a run of PAYLOAD packets carry the actual pixel bytes at explicit, OFFSET-based positions rather than positions implied by arrival order, and a TRAILER packet closes the frame -- at which point the only question that matters is whether every payload chunk this frame promised actually arrived.

`FrameReassembler::on_packet` places each payload chunk at `packet_id * CHUNK_BYTES`, an explicit byte offset computed from the packet's own header field, never from the order packets happen to arrive in. This is the one design choice this section's own Test 2 exists to prove matters: reassembling a frame from a deliberately SHUFFLED packet order produces byte-identical pixels to the in-order case, because placement never depended on order in the first place. A protocol that instead appended each arriving chunk to a growing buffer would have silently corrupted any frame whose packets arrived out of sequence -- a real, common occurrence on any real UDP-based network -- without any way to detect that it had happened.

Detecting a genuinely incomplete frame requires tracking which specific chunks were actually received, not just counting how many arrived: `received_chunks_`, an `unordered_set<uint32_t>` of packet IDs, lets the reassembler both ignore an accidental duplicate delivery (a real UDP behavior) and report an honest `packets_received` count that a naive running counter could not distinguish from a set of unique arrivals. When the TRAILER packet closes the frame, `FrameResult::ok` is computed by comparing `packets_received` against `packets_expected` -- never assumed true, never silently defaulted to true and corrected only on a caught exception. A dropped payload packet is exactly the situation Test 3 verifies: the reassembler reports `ok=false` with the correct, honest received-versus-expected counts, rather than handing Section 18.2's vision encoder a frame quietly padded with zero bytes it never received.

Hardware triggering is a real, separate concern from frame reassembly, and this section's `TriggerController` keeps it that way: a photoelectric sensor on a real line fires a hardware trigger the instant a part passes a fixed point, and the camera must refuse a second trigger while still busy capturing the first (Test 4), then correctly time out if the triggered frame never arrives at all, and correctly flag as failed a frame that DOES arrive, complete, but past its own deadline (Test 5) -- late is not the same as wrong, but on a line moving at a fixed speed, a correct-but-late frame is still a frame the disposition logic downstream cannot safely act on in time. Every one of these timing checks runs on caller-supplied TICK counts, never real wall-clock time, consistent with this book's own standing discipline -- established since Chapter 17's own timing methodology -- against locking real, machine-specific timing into output this book's four-way cross-architecture check compares byte-for-byte.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_gige_frame_acquisition.cpp -o 01_gige_frame_acquisition
./01_gige_frame_acquisition
```

**Sample input:** a synthetic reference image split into GVSP-inspired leader/payload/trailer packets, reassembled and checked byte-for-byte against the reference in both in-order and deliberately shuffled delivery order; a dropped payload packet checked to produce `ok=false` with correct received-versus-expected counts rather than a silently accepted, incomplete frame; and `TriggerController` checked for busy-state rejection, timeout on a missing frame, and correct failure of a complete-but-late frame -- all using caller-supplied tick counts, never real wall-clock time.

```text
========================================================
Chapter 18.1: GigE Vision Frame Acquisition and Hardware Triggering
========================================================

-- Test 1: in-order packet stream reassembles exactly --
  4/4 payload packets received, frame ok: yes, pixels match reference: yes

-- Test 2: shuffled (out-of-order) delivery reassembles identically --
  reassembled from shuffled packet order, pixels still match reference: yes

-- Test 3: a dropped payload packet is reported, not silently accepted --
  received 3/4 payload packets, frame correctly flagged incomplete (ok=false)

-- Test 4: trigger controller mode and busy-state enforcement --
  OFF mode rejects a fire(): yes; a second fire() while busy is rejected: yes; camera frees up after on_frame(): yes; latency computed as trigger-to-frame-ready: 35 ticks

-- Test 5: a missing frame times out, and a late frame is flagged, not accepted --
  a trigger with no frame at all times out (timed_out=true); a COMPLETE frame arriving after the deadline is still flagged not ok (ok=false, latency=40 ticks > 30 tick deadline)

24/24 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] placing payload bytes by arrival order instead of by the packet's own offset field"
    It is tempting to reassemble a frame by simply appending each arriving payload packet's bytes to a growing buffer, in the order they happen to arrive -- and on a quiet local test network, this often even appears to work, because packets frequently do arrive in order. Real UDP delivery offers no such guarantee, and a single out-of-order packet under that scheme corrupts the entire frame silently, with no error and no way downstream to detect it happened. The fix `FrameReassembler` applies is placing every payload chunk at the explicit byte offset its own header field states (`packet_id * CHUNK_BYTES`), independent of arrival order entirely -- proven correct here not by hoping a test network stays quiet, but by a self-test that deliberately shuffles delivery order and confirms the reassembled pixels are still byte-identical to the reference.

## 18.2 Image Preprocessing and the Qwen2.5-VL Vision Encoder

### Intuition

Section 18.1's `FrameReassembler` already hands this section exactly the input a real deployment has: a raw MONO8/RGB8 pixel buffer, straight off the sensor. That is one genuine advantage a machine-vision pipeline has over a general-purpose photo pipeline -- there is no JPEG or PNG to decode at all, because a GigE Vision camera never encodes one in the first place. This section starts from that raw buffer and builds everything Qwen2.5-VL's own vision encoder needs from it.

### The Concept, In Detail

Patchification splits a resized image into a regular grid of fixed-size square patches, each one flattened into a single vector of raw pixel floats -- the vision-transformer equivalent of tokenizing text, except the "vocabulary" here is the space of possible small image patches rather than a fixed lookup table. `patchify` and `extract_patch` produce exactly this grid, verified in Test 1 by confirming the expected grid dimensions, the expected per-patch vector length, and that every value in every patch is finite.

Two-dimensional rotary position encoding solves a real problem 1D RoPE cannot: a text sequence has one coordinate (its position), but an image patch has two (its row and its column), and a position encoding that only captured a flattened row-major index would conflate a patch's true 2D neighborhood with an arbitrary linear ordering. `RoPE2DTables` and `apply_rope2d` solve this by splitting each attention head's dimension into two disjoint halves and independently rotating each half by the patch's row coordinate and its column coordinate respectively, reusing the exact 1D rotate-half mechanism Chapter 9 already verified, applied twice. Test 2 checks three real invariants of this construction directly: position (0, 0) is the identity rotation (nothing to rotate away from at the origin), identical patch content at two different (row, col) positions produces genuinely different output vectors (position information survives into the vector), and changing ONLY the row or ONLY the column independently changes the output (the two halves are actually functioning as two independent coordinates, not collapsing into one).

The vision transformer blocks themselves reuse this book's own already-verified RMSNorm, matmul, and SwiGLU building blocks from Section 15.3, wired into FULL bidirectional attention -- every patch attends to every other patch, with no causal mask, because an image has no "future" a patch needs to be prevented from seeing. This section states its own scope honestly here too: the real Qwen2.5-VL uses windowed attention (full attention in only a few of its many layers, attention restricted to nearby patches everywhere else) to keep a high-resolution image computationally affordable, and teaching that real windowing scheme correctly would need its own section's worth of index arithmetic without changing a single idea this section actually exists to teach -- patch embedding, 2D RoPE, and the merger below. The simplification is stated plainly rather than quietly shipped as if it were the genuine article.

The 2x2 spatial patch merger is where a real correctness pitfall lives, and this section's Test 3 is built specifically to catch it: merging four patches into one visual token must group four SPATIALLY ADJACENT patches -- a 2x2 neighborhood in the actual image grid -- not four ROW-MAJOR-CONSECUTIVE patches, which on anything but a 2-wide grid are a completely different, spatially meaningless set. `block_patch_indices(br, bc, grid_w)` computes `{r0*gw+c0, r0*gw+c1, r1*gw+c0, r1*gw+c1}` -- the real four indices of a 2x2 neighborhood -- and Test 3 confirms directly, on a 4x4 grid, that block (0,0) gathers patches `{0, 1, 4, 5}` (a genuine 2x2 square) rather than `{0, 1, 2, 3}` (one full row), which is exactly the bug a naive "just take four consecutive patches" implementation would produce silently, with no crash and no obviously wrong-looking output, only a vision encoder that has scrambled every image's own spatial structure.

### Code and Verification

```cpp
// Chapter 18.2 -- Section 18.1's FrameReassembler already hands this
// section exactly the input a real deployment has: a raw MONO8/RGB8
// pixel buffer, straight off the sensor. That is the one genuine
// advantage a machine-vision pipeline has over a general-purpose photo
// pipeline: there is no JPEG or PNG to decode at all, because a GigE
// Vision camera never encodes one in the first place. This section
// starts from exactly that raw buffer and builds everything Qwen2.5-VL's
// own vision encoder needs from it: patchification, a linear patch
// embedding, the SAME two-dimensional rotary position encoding the real
// model uses to tell a patch's row and column apart, a small stack of
// bidirectional (non-causal) transformer blocks reusing this book's own
// already-verified RMSNorm/matmul/SwiGLU building blocks, and the
// 2x2 spatial patch merger that turns four neighboring patches into one
// visual token sized to match the text decoder's own embedding space.
//
// A note on scope, in this book's own recurring voice: this section's
// transformer blocks use FULL bidirectional attention over every patch,
// not the real Qwen2.5-VL's own window-attention scheme (full attention
// in only a few of its many layers, windowed attention restricted to
// nearby patches everywhere else, to keep a high-resolution image
// affordable). Teaching the real windowing scheme correctly would need
// its own section's worth of index arithmetic without changing a single
// idea this section is actually here to teach -- patch embedding, 2D
// RoPE, and the merger -- so this section states the simplification
// plainly rather than quietly shipping windowed attention as if it were
// the genuine article.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 02_vision_encoder.cpp -o 02_vision_encoder
// Run:     ./02_vision_encoder

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: raw-buffer preprocessing. Section 18.1's FrameResult::pixels
// is already exactly this input -- an HxWxC row-major byte buffer, no
// codec involved anywhere in this pipeline.
// =======================================================================
struct RawImage {
    uint32_t width = 0, height = 0, channels = 0;
    std::vector<uint8_t> pixels;   // row-major, HxWxC, matching FrameResult::pixels exactly
};

// Nearest-neighbor resize to a target size that is an exact multiple of
// the ViT's own patch size. Real deployments use a higher-quality
// filter; nearest-neighbor is this section's own stated simplification,
// chosen because it is exactly reproducible in integer arithmetic across
// every architecture this book locks against, with no floating-point
// rounding difference to chase.
RawImage resize_nearest(const RawImage& src, uint32_t dst_w, uint32_t dst_h) {
    RawImage dst;
    dst.width = dst_w; dst.height = dst_h; dst.channels = src.channels;
    dst.pixels.resize(static_cast<size_t>(dst_w) * dst_h * src.channels);
    for (uint32_t y = 0; y < dst_h; ++y) {
        uint32_t sy = std::min(src.height - 1, (y * src.height) / dst_h);
        for (uint32_t x = 0; x < dst_w; ++x) {
            uint32_t sx = std::min(src.width - 1, (x * src.width) / dst_w);
            const uint8_t* sp = &src.pixels[(static_cast<size_t>(sy) * src.width + sx) * src.channels];
            uint8_t* dp = &dst.pixels[(static_cast<size_t>(y) * dst_w + x) * src.channels];
            std::memcpy(dp, sp, src.channels);
        }
    }
    return dst;
}

// CLIP/SigLIP-style per-channel normalization constants -- stated, not
// derived: a real deployment would use whichever mean/std the specific
// checkpoint's own preprocessor_config.json declares.
struct NormStats { float mean[3] = {0.481f, 0.458f, 0.408f}; float std[3] = {0.269f, 0.261f, 0.276f}; };

// One patch's flattened, normalized pixel data: patch_size * patch_size
// * channels floats, in row-major (row, then col, then channel) order.
std::vector<float> extract_patch(const RawImage& img, uint32_t patch_row, uint32_t patch_col,
                                  uint32_t patch_size, const NormStats& norm) {
    std::vector<float> out(static_cast<size_t>(patch_size) * patch_size * img.channels);
    size_t idx = 0;
    for (uint32_t py = 0; py < patch_size; ++py) {
        uint32_t y = patch_row * patch_size + py;
        for (uint32_t px = 0; px < patch_size; ++px) {
            uint32_t x = patch_col * patch_size + px;
            const uint8_t* sp = &img.pixels[(static_cast<size_t>(y) * img.width + x) * img.channels];
            for (uint32_t c = 0; c < img.channels; ++c) {
                float v = static_cast<float>(sp[c]) / 255.0f;
                out[idx++] = (v - norm.mean[c % 3]) / norm.std[c % 3];
            }
        }
    }
    return out;
}

// Patchifies the WHOLE image into a row-major grid of flattened patches:
// patches[row * grid_w + col] is that (row, col) patch's own flattened,
// normalized pixel vector.
std::vector<std::vector<float>> patchify(const RawImage& img, uint32_t patch_size, const NormStats& norm,
                                          uint32_t& grid_h, uint32_t& grid_w) {
    grid_h = img.height / patch_size;
    grid_w = img.width / patch_size;
    std::vector<std::vector<float>> patches(static_cast<size_t>(grid_h) * grid_w);
    for (uint32_t r = 0; r < grid_h; ++r)
        for (uint32_t c = 0; c < grid_w; ++c)
            patches[static_cast<size_t>(r) * grid_w + c] = extract_patch(img, r, c, patch_size, norm);
    return patches;
}

// =======================================================================
// PART 2: this book's own RMSNorm/matmul/SwiGLU, repeated verbatim from
// Section 15.3 -- the vision encoder's own transformer blocks are built
// from the identical primitives the text decoder already uses.
// =======================================================================
void rms_norm(std::span<float> out, std::span<const float> x, std::span<const float> weights, float epsilon = 1e-6f) {
    const size_t d = x.size();
    double sum_sq = 0.0;
    for (size_t i = 0; i < d; ++i) sum_sq += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    float rms_inv = 1.0f / std::sqrt(static_cast<float>(sum_sq / static_cast<double>(d)) + epsilon);
    for (size_t i = 0; i < d; ++i) out[i] = (x[i] * rms_inv) * weights[i];
}
void matmul(std::span<float> out, std::span<const float> x, std::span<const float> W, size_t in_dim, size_t out_dim) {
    for (size_t j = 0; j < out_dim; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < in_dim; ++i) sum += static_cast<double>(x[i]) * static_cast<double>(W[j * in_dim + i]);
        out[j] = static_cast<float>(sum);
    }
}
inline float silu(float x) { return x * (1.0f / (1.0f + std::exp(-x))); }
void swiglu_ffn(std::span<float> out, std::span<const float> x, std::span<const float> W_gate,
                 std::span<const float> W_up, std::span<const float> W_down, size_t dim, size_t d_ff) {
    std::vector<float> gate_proj(d_ff), up_proj(d_ff), hidden(d_ff);
    matmul(gate_proj, x, W_gate, dim, d_ff);
    matmul(up_proj, x, W_up, dim, d_ff);
    for (size_t i = 0; i < d_ff; ++i) hidden[i] = silu(gate_proj[i]) * up_proj[i];
    matmul(out, hidden, W_down, d_ff, dim);
}
void softmax_inplace(std::span<float> scores) {
    float max_val = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;
    for (float& s : scores) { s = std::exp(s - max_val); sum += s; }
    float inv_sum = 1.0f / sum;
    for (float& s : scores) s *= inv_sum;
}

// =======================================================================
// PART 3: two-dimensional rotary position encoding. A text token has one
// position; a patch has TWO (its row and its column), and Qwen2-VL's own
// real fix is splitting each attention head's dimension in half, rotating
// the FIRST half by the patch's row and the SECOND half by its column --
// the same 1D rotate-half mechanism this book has used since Chapter 9,
// applied twice, to two different coordinates, over two disjoint slices
// of the same vector.
// =======================================================================
struct RoPE2DTables {
    std::vector<float> cos_row, sin_row, cos_col, sin_col;
    int quarter_dim;   // half_dim (per axis) is head_dim/2; each axis's own rotate-half pairs cover head_dim/4
    RoPE2DTables(int max_coord, int head_dim, float base) : quarter_dim(head_dim / 4) {
        cos_row.resize(static_cast<size_t>(max_coord) * quarter_dim);
        sin_row.resize(static_cast<size_t>(max_coord) * quarter_dim);
        cos_col.resize(static_cast<size_t>(max_coord) * quarter_dim);
        sin_col.resize(static_cast<size_t>(max_coord) * quarter_dim);
        for (int coord = 0; coord < max_coord; ++coord) {
            for (int k = 0; k < quarter_dim; ++k) {
                float theta = 1.0f / std::pow(base, (2.0f * static_cast<float>(k)) / static_cast<float>(head_dim / 2));
                float angle = static_cast<float>(coord) * theta;
                size_t idx = static_cast<size_t>(coord) * quarter_dim + k;
                cos_row[idx] = std::cos(angle); sin_row[idx] = std::sin(angle);
                cos_col[idx] = std::cos(angle); sin_col[idx] = std::sin(angle);
            }
        }
    }
};
// Rotates a length-`2*half`-element slice using the standard rotate-half
// pairing (index k paired with index k+half), exactly Section 9's own
// `apply_rope` -- factored out so both the row-half and the col-half of
// a patch's query/key vector can call the identical primitive.
void rotate_half_inplace(std::span<float> vec, std::span<const float> cos_tab, std::span<const float> sin_tab,
                          int coord, int quarter_dim) {
    for (int k = 0; k < quarter_dim; ++k) {
        float x1 = vec[static_cast<size_t>(k)], x2 = vec[static_cast<size_t>(k + quarter_dim)];
        float c = cos_tab[static_cast<size_t>(coord) * quarter_dim + k];
        float s = sin_tab[static_cast<size_t>(coord) * quarter_dim + k];
        vec[static_cast<size_t>(k)] = x1 * c - x2 * s;
        vec[static_cast<size_t>(k + quarter_dim)] = x1 * s + x2 * c;
    }
}
// Applies 2D RoPE to one attention head's full head_dim vector: the
// FIRST half rotated by `row` (using its own rotate-half pairing over
// that half's two quarters), the SECOND half rotated by `col`.
void apply_rope2d(std::span<float> head_vec, int row, int col, const RoPE2DTables& t) {
    const int half = static_cast<int>(head_vec.size()) / 2;
    rotate_half_inplace(head_vec.subspan(0, static_cast<size_t>(half)), t.cos_row, t.sin_row, row, t.quarter_dim);
    rotate_half_inplace(head_vec.subspan(static_cast<size_t>(half), static_cast<size_t>(half)), t.cos_col, t.sin_col, col, t.quarter_dim);
}

// =======================================================================
// PART 4: the vision transformer block itself -- RMSNorm, QKV
// projection, 2D RoPE, FULL (non-causal, no KV cache) bidirectional
// attention over every patch, output projection, a residual, a second
// RMSNorm, the SwiGLU FFN, and a second residual.
// =======================================================================
struct ViTShape { int dim, n_heads, head_dim, d_ff; int qkv_dim() const { return n_heads * head_dim; } };
struct ViTBlockWeights {
    std::vector<float> attn_norm, Wq, Wk, Wv, Wo;
    std::vector<float> ffn_norm, Wgate, Wup, Wdown;
};

// One full self-attention pass over ALL patches -- unlike the text
// decoder's causal, one-position-at-a-time attention, every patch here
// attends to every other patch in a single call, because an image has
// no "future" to mask and no cache to build incrementally.
void vit_full_attention(const std::vector<std::vector<float>>& q, const std::vector<std::vector<float>>& k,
                         const std::vector<std::vector<float>>& v, std::vector<std::vector<float>>& out,
                         int n_heads, int head_dim) {
    const int n = static_cast<int>(q.size());
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int h = 0; h < n_heads; ++h) {
        for (int i = 0; i < n; ++i) {
            std::vector<float> scores(static_cast<size_t>(n));
            std::span<const float> qi(q[static_cast<size_t>(i)].data() + h * head_dim, static_cast<size_t>(head_dim));
            for (int j = 0; j < n; ++j) {
                std::span<const float> kj(k[static_cast<size_t>(j)].data() + h * head_dim, static_cast<size_t>(head_dim));
                double d = 0.0;
                for (int c = 0; c < head_dim; ++c) d += static_cast<double>(qi[static_cast<size_t>(c)]) * kj[static_cast<size_t>(c)];
                scores[static_cast<size_t>(j)] = static_cast<float>(d) * scale;
            }
            softmax_inplace(scores);
            float* o = out[static_cast<size_t>(i)].data() + h * head_dim;
            for (int c = 0; c < head_dim; ++c) o[c] = 0.0f;
            for (int j = 0; j < n; ++j) {
                std::span<const float> vj(v[static_cast<size_t>(j)].data() + h * head_dim, static_cast<size_t>(head_dim));
                float w = scores[static_cast<size_t>(j)];
                for (int c = 0; c < head_dim; ++c) o[c] += w * vj[static_cast<size_t>(c)];
            }
        }
    }
}

void vit_block_forward(std::vector<std::vector<float>>& x, const ViTShape& shape, const ViTBlockWeights& w,
                        const std::vector<int>& rows, const std::vector<int>& cols, const RoPE2DTables& rope) {
    const int n = static_cast<int>(x.size());
    std::vector<std::vector<float>> q(static_cast<size_t>(n)), k(static_cast<size_t>(n)), v(static_cast<size_t>(n)), attn_out(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        std::vector<float> normed(static_cast<size_t>(shape.dim));
        rms_norm(normed, x[static_cast<size_t>(i)], w.attn_norm);
        q[static_cast<size_t>(i)].resize(static_cast<size_t>(shape.qkv_dim()));
        k[static_cast<size_t>(i)].resize(static_cast<size_t>(shape.qkv_dim()));
        v[static_cast<size_t>(i)].resize(static_cast<size_t>(shape.qkv_dim()));
        attn_out[static_cast<size_t>(i)].resize(static_cast<size_t>(shape.qkv_dim()));
        matmul(q[static_cast<size_t>(i)], normed, w.Wq, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.qkv_dim()));
        matmul(k[static_cast<size_t>(i)], normed, w.Wk, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.qkv_dim()));
        matmul(v[static_cast<size_t>(i)], normed, w.Wv, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.qkv_dim()));
        for (int h = 0; h < shape.n_heads; ++h) {
            apply_rope2d(std::span<float>(q[static_cast<size_t>(i)].data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)),
                         rows[static_cast<size_t>(i)], cols[static_cast<size_t>(i)], rope);
            apply_rope2d(std::span<float>(k[static_cast<size_t>(i)].data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)),
                         rows[static_cast<size_t>(i)], cols[static_cast<size_t>(i)], rope);
        }
    }
    vit_full_attention(q, k, v, attn_out, shape.n_heads, shape.head_dim);
    for (int i = 0; i < n; ++i) {
        std::vector<float> proj(static_cast<size_t>(shape.dim));
        matmul(proj, attn_out[static_cast<size_t>(i)], w.Wo, static_cast<size_t>(shape.qkv_dim()), static_cast<size_t>(shape.dim));
        for (int d = 0; d < shape.dim; ++d) x[static_cast<size_t>(i)][static_cast<size_t>(d)] += proj[static_cast<size_t>(d)];
        std::vector<float> normed2(static_cast<size_t>(shape.dim)), ffn_out(static_cast<size_t>(shape.dim));
        rms_norm(normed2, x[static_cast<size_t>(i)], w.ffn_norm);
        swiglu_ffn(ffn_out, normed2, w.Wgate, w.Wup, w.Wdown, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.d_ff));
        for (int d = 0; d < shape.dim; ++d) x[static_cast<size_t>(i)][static_cast<size_t>(d)] += ffn_out[static_cast<size_t>(d)];
    }
}

// =======================================================================
// PART 5: the 2x2 spatial patch merger. Four SPATIALLY ADJACENT patches
// -- (2r,2c), (2r,2c+1), (2r+1,2c), (2r+1,2c+1) -- are concatenated and
// projected down to the LLM's own embedding dimension, cutting the
// visual token count by 4x. Grouping by spatial adjacency, not by
// row-major list order, matters: for any grid wider than two patches,
// four row-major-consecutive patches are the START OF ONE ROW, not a 2x2
// neighborhood, and merging them would blend unrelated regions of the
// image into one token.
// =======================================================================
struct MergerWeights { std::vector<float> W1, b1, W2, b2; int hidden_dim; };

// Returns the flat patch-list indices of the 4 patches spatially
// adjacent to merged block (br, bc), in a fixed, testable order:
// top-left, top-right, bottom-left, bottom-right. Factored out on its
// own specifically so this section's own self-tests can check the
// SPATIAL grouping directly, independent of the matmul math around it.
std::array<int, 4> block_patch_indices(int br, int bc, int grid_w) {
    const int r0 = 2 * br, r1 = 2 * br + 1, c0 = 2 * bc, c1 = 2 * bc + 1;
    return {r0 * grid_w + c0, r0 * grid_w + c1, r1 * grid_w + c0, r1 * grid_w + c1};
}

std::vector<std::vector<float>> merge_all_2x2(const std::vector<std::vector<float>>& patches, int grid_h, int grid_w,
                                               int vit_dim, const MergerWeights& mw, int llm_dim) {
    const int out_h = grid_h / 2, out_w = grid_w / 2;
    std::vector<std::vector<float>> merged(static_cast<size_t>(out_h) * out_w);
    for (int br = 0; br < out_h; ++br) {
        for (int bc = 0; bc < out_w; ++bc) {
            std::vector<float> concat(static_cast<size_t>(4 * vit_dim));
            auto idx = block_patch_indices(br, bc, grid_w);
            for (int slot = 0; slot < 4; ++slot) {
                const auto& p = patches[static_cast<size_t>(idx[static_cast<size_t>(slot)])];
                std::copy(p.begin(), p.end(), concat.begin() + slot * vit_dim);
            }
            std::vector<float> h1(static_cast<size_t>(mw.hidden_dim));
            matmul(h1, concat, mw.W1, static_cast<size_t>(4 * vit_dim), static_cast<size_t>(mw.hidden_dim));
            for (int i = 0; i < mw.hidden_dim; ++i) h1[static_cast<size_t>(i)] = silu(h1[static_cast<size_t>(i)] + mw.b1[static_cast<size_t>(i)]);
            std::vector<float> out(static_cast<size_t>(llm_dim));
            matmul(out, h1, mw.W2, static_cast<size_t>(mw.hidden_dim), static_cast<size_t>(llm_dim));
            for (int i = 0; i < llm_dim; ++i) out[static_cast<size_t>(i)] += mw.b2[static_cast<size_t>(i)];
            merged[static_cast<size_t>(br) * out_w + bc] = std::move(out);
        }
    }
    return merged;
}

// =======================================================================
// PART 6: self-tests, against a small synthetic image and small
// synthetic weights -- fast, deterministic, and needing no real
// checkpoint to prove this section's own machinery is correct.
// =======================================================================
std::vector<float> rand_vec(std::mt19937& rng, size_t n) {
    std::normal_distribution<float> dist(0.0f, 0.3f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 18.2: Image Preprocessing and the Qwen2.5-VL Vision Encoder\n";
    std::cout << "========================================================\n";

    constexpr int PATCH = 14, CH = 3;
    constexpr int VIT_DIM = 24, N_HEADS = 2, HEAD_DIM = 8, D_FF = 32, N_LAYERS = 2;
    constexpr int LLM_DIM = 32, MERGER_HIDDEN = 40;

    std::cout << "\n-- Test 1: resize and patchify produce the expected grid and patch shape --\n";
    {
        RawImage src; src.width = 100; src.height = 80; src.channels = CH;
        std::mt19937 rng(1);
        std::uniform_int_distribution<int> byte_dist(0, 255);
        src.pixels.resize(static_cast<size_t>(src.width) * src.height * CH);
        for (auto& b : src.pixels) b = static_cast<uint8_t>(byte_dist(rng));

        RawImage resized = resize_nearest(src, 56, 56);   // 4x4 grid of 14x14 patches
        CHECK(resized.width == 56 && resized.height == 56);

        NormStats norm;
        uint32_t grid_h = 0, grid_w = 0;
        auto patches = patchify(resized, PATCH, norm, grid_h, grid_w);
        CHECK(grid_h == 4 && grid_w == 4);
        CHECK(patches.size() == 16);
        CHECK(patches[0].size() == static_cast<size_t>(PATCH) * PATCH * CH);
        bool all_finite = true;
        for (const auto& p : patches) for (float v : p) if (!std::isfinite(v)) all_finite = false;
        CHECK(all_finite);
        std::cout << "  resized to 56x56, patchified into " << grid_h << "x" << grid_w
                   << " grid, each patch " << patches[0].size() << " floats, all finite: "
                   << (all_finite ? "yes" : "no") << "\n";
    }

    std::cout << "\n-- Test 2: 2D RoPE rotates by row in the first half-dim and by column in the second --\n";
    {
        RoPE2DTables rope(8, HEAD_DIM, 10000.0f);
        std::vector<float> base(HEAD_DIM);
        std::mt19937 rng(2);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto& v : base) v = dist(rng);

        // Position (0,0): every rotation angle is coord * theta = 0, so
        // the identity rotation should leave the vector byte-for-byte
        // unchanged, exactly like Section 9's own 1D RoPE at position 0.
        std::vector<float> at_origin = base;
        apply_rope2d(at_origin, 0, 0, rope);
        CHECK(at_origin == base);

        // Identical content at two DIFFERENT (row, col) positions must
        // produce DIFFERENT vectors -- this is the entire reason a
        // position encoding exists at all.
        std::vector<float> at_a = base, at_b = base;
        apply_rope2d(at_a, 1, 3, rope);
        apply_rope2d(at_b, 5, 2, rope);
        CHECK(at_a != at_b);

        // Changing ONLY the row (column held fixed) must still change
        // the result, and changing ONLY the column must too -- proving
        // both halves are actually wired to their own coordinate rather
        // than one half silently controlling both axes' output.
        std::vector<float> at_c = base, at_d = base;
        apply_rope2d(at_c, 1, 3, rope);
        apply_rope2d(at_d, 7, 3, rope);   // same col, different row
        CHECK(at_c != at_d);
        std::vector<float> at_e = base, at_f = base;
        apply_rope2d(at_e, 1, 3, rope);
        apply_rope2d(at_f, 1, 6, rope);   // same row, different col
        CHECK(at_e != at_f);

        std::cout << "  position (0,0) is the identity rotation: yes; identical content at different "
                     "(row,col) positions produces different vectors: yes; row-only and col-only "
                     "changes each independently change the output: yes\n";
    }

    std::cout << "\n-- Test 3: the 2x2 merger groups spatially adjacent patches, not row-major-consecutive ones --\n";
    {
        // A 4x4 grid: block (0,0) must gather patches (0,0),(0,1),(1,0),
        // (1,1) -- flat indices 0,1,4,5 -- NOT the row-major-consecutive
        // 0,1,2,3, which for a grid wider than 2 patches would silently
        // merge the first FOUR PATCHES OF ONE ROW into a single token
        // instead of a genuine 2x2 spatial neighborhood.
        auto idx00 = block_patch_indices(0, 0, /*grid_w=*/4);
        CHECK((idx00 == std::array<int, 4>{0, 1, 4, 5}));
        auto idx11 = block_patch_indices(1, 1, /*grid_w=*/4);
        CHECK((idx11 == std::array<int, 4>{10, 11, 14, 15}));
        auto idx01 = block_patch_indices(0, 1, /*grid_w=*/4);
        CHECK((idx01 == std::array<int, 4>{2, 3, 6, 7}));
        std::cout << "  block(0,0) on a 4x4 grid gathers patches {0,1,4,5} (a real 2x2 neighborhood), "
                     "not {0,1,2,3} (one row): correct\n";
    }

    std::cout << "\n-- Test 4: full encoder forward pass (embed -> " << N_LAYERS << " ViT blocks -> merger) --\n";
    {
        RawImage src; src.width = 56; src.height = 56; src.channels = CH;
        std::mt19937 img_rng(3);
        std::uniform_int_distribution<int> byte_dist(0, 255);
        src.pixels.resize(static_cast<size_t>(src.width) * src.height * CH);
        for (auto& b : src.pixels) b = static_cast<uint8_t>(byte_dist(img_rng));

        NormStats norm;
        uint32_t grid_h = 0, grid_w = 0;
        auto raw_patches = patchify(src, PATCH, norm, grid_h, grid_w);
        const int n_patches = static_cast<int>(raw_patches.size());
        const size_t patch_vec_len = raw_patches[0].size();

        std::mt19937 w_rng(42);
        std::vector<float> W_embed = rand_vec(w_rng, static_cast<size_t>(VIT_DIM) * patch_vec_len);

        std::vector<std::vector<float>> x(static_cast<size_t>(n_patches));
        std::vector<int> rows(static_cast<size_t>(n_patches)), cols(static_cast<size_t>(n_patches));
        for (int i = 0; i < n_patches; ++i) {
            x[static_cast<size_t>(i)].resize(VIT_DIM);
            matmul(x[static_cast<size_t>(i)], raw_patches[static_cast<size_t>(i)], W_embed, patch_vec_len, VIT_DIM);
            rows[static_cast<size_t>(i)] = i / static_cast<int>(grid_w);
            cols[static_cast<size_t>(i)] = i % static_cast<int>(grid_w);
        }

        std::vector<ViTBlockWeights> layers(N_LAYERS);
        for (auto& l : layers) {
            l.attn_norm.assign(VIT_DIM, 1.0f);
            l.Wq = rand_vec(w_rng, static_cast<size_t>(N_HEADS * HEAD_DIM) * VIT_DIM);
            l.Wk = rand_vec(w_rng, static_cast<size_t>(N_HEADS * HEAD_DIM) * VIT_DIM);
            l.Wv = rand_vec(w_rng, static_cast<size_t>(N_HEADS * HEAD_DIM) * VIT_DIM);
            l.Wo = rand_vec(w_rng, static_cast<size_t>(VIT_DIM) * (N_HEADS * HEAD_DIM));
            l.ffn_norm.assign(VIT_DIM, 1.0f);
            l.Wgate = rand_vec(w_rng, static_cast<size_t>(D_FF) * VIT_DIM);
            l.Wup = rand_vec(w_rng, static_cast<size_t>(D_FF) * VIT_DIM);
            l.Wdown = rand_vec(w_rng, static_cast<size_t>(VIT_DIM) * D_FF);
        }
        ViTShape shape{VIT_DIM, N_HEADS, HEAD_DIM, D_FF};
        RoPE2DTables rope(static_cast<int>(std::max(grid_h, grid_w)) + 1, HEAD_DIM, 10000.0f);

        auto run_encoder = [&]() {
            auto xx = x;
            for (const auto& l : layers) vit_block_forward(xx, shape, l, rows, cols, rope);
            MergerWeights mw;
            mw.hidden_dim = MERGER_HIDDEN;
            std::mt19937 m_rng(99);
            mw.W1 = rand_vec(m_rng, static_cast<size_t>(MERGER_HIDDEN) * (4 * VIT_DIM));
            mw.b1 = rand_vec(m_rng, static_cast<size_t>(MERGER_HIDDEN));
            mw.W2 = rand_vec(m_rng, static_cast<size_t>(LLM_DIM) * MERGER_HIDDEN);
            mw.b2 = rand_vec(m_rng, static_cast<size_t>(LLM_DIM));
            return merge_all_2x2(xx, static_cast<int>(grid_h), static_cast<int>(grid_w), VIT_DIM, mw, LLM_DIM);
        };

        auto merged1 = run_encoder();
        auto merged2 = run_encoder();

        CHECK(merged1.size() == (grid_h / 2) * (grid_w / 2));
        CHECK(merged1[0].size() == static_cast<size_t>(LLM_DIM));
        bool all_finite = true;
        for (const auto& tok : merged1) for (float v : tok) if (!std::isfinite(v)) all_finite = false;
        CHECK(all_finite);
        bool deterministic = (merged1.size() == merged2.size());
        for (size_t i = 0; deterministic && i < merged1.size(); ++i) deterministic = (merged1[i] == merged2[i]);
        CHECK(deterministic);

        std::cout << "  " << n_patches << " patches (" << grid_h << "x" << grid_w << ") -> " << N_LAYERS
                   << " ViT blocks -> " << merged1.size() << " merged visual tokens, each " << LLM_DIM
                   << "-dim; all finite: " << (all_finite ? "yes" : "no")
                   << "; deterministic across two runs: " << (deterministic ? "yes" : "no") << "\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_vision_encoder.cpp -o 02_vision_encoder
./02_vision_encoder
```

**Sample input:** a synthetic raw image resized and patchified into a known grid, checked for the expected patch count, patch size, and finiteness; 2D RoPE checked against three real geometric invariants (identity at the origin, distinguishable positions, and row/column independence) rather than any single hand-computed vector; the 2x2 merger checked against a naive row-major bug it does NOT commit, on a grid wide enough (4x4) to make the two groupings genuinely different sets of patches; and a full encoder forward pass (embed, two ViT blocks, merger) checked for finiteness and determinism across two runs.

```text
========================================================
Chapter 18.2: Image Preprocessing and the Qwen2.5-VL Vision Encoder
========================================================

-- Test 1: resize and patchify produce the expected grid and patch shape --
  resized to 56x56, patchified into 4x4 grid, each patch 588 floats, all finite: yes

-- Test 2: 2D RoPE rotates by row in the first half-dim and by column in the second --
  position (0,0) is the identity rotation: yes; identical content at different (row,col) positions produces different vectors: yes; row-only and col-only changes each independently change the output: yes

-- Test 3: the 2x2 merger groups spatially adjacent patches, not row-major-consecutive ones --
  block(0,0) on a 4x4 grid gathers patches {0,1,4,5} (a real 2x2 neighborhood), not {0,1,2,3} (one row): correct

-- Test 4: full encoder forward pass (embed -> 2 ViT blocks -> merger) --
  16 patches (4x4) -> 2 ViT blocks -> 4 merged visual tokens, each 32-dim; all finite: yes; deterministic across two runs: yes

16/16 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] a 2x2 merge on a 4-wide grid can look correct even when it silently is not"
    On a grid exactly 2 patches wide, "four row-major-consecutive patches" and "one real 2x2 spatial block" happen to describe the SAME four indices, so a bug that merges by row-major order instead of by real spatial adjacency would pass a test built on a 2-wide grid without ever being exercised. This section's own Test 3 deliberately uses a 4x4 grid specifically because a 4-wide grid is where the two groupings first diverge -- block (0,0)'s real spatial neighborhood is `{0, 1, 4, 5}`, while four row-major-consecutive patches starting at index 0 would wrongly be `{0, 1, 2, 3}`, one entire row rather than a square. The general lesson: a geometric correctness property should be tested on the smallest input where the bug it is guarding against and the correct behavior would actually produce two DIFFERENT answers, not the smallest input where the code merely runs without crashing.

## 18.3 Wiring the Vision Encoder into the Qwen2 Decoder

### Intuition

Section 18.2 turned a raw camera frame into a small handful of visual tokens, each one already projected into the text decoder's own embedding space. This section does the one thing that makes "vision-LANGUAGE model" more than two separate models bolted together: splicing those visual tokens directly into a text token sequence, in place of a reserved placeholder ID, and running the IDENTICAL, unchanged Qwen2 decoder this book has verified since Chapter 15 over the combined sequence.

### The Concept, In Detail

A visual token and a text token differ only in how their own embedding vector was produced -- one came from looking up a token ID in `token_embd.weight`, the other came from Section 18.2's own vision encoder and merger. Once both are sitting in the same sequence of vectors, one per position, `qwen2_block_forward` cannot tell the difference and was never asked to: there is no separate "vision decoder" and no special-cased attention path for image positions anywhere in this section's own code, because a real vision-language model's whole architectural point is that there does not need to be one.

The one real seam this requires lives in exactly one place. Every earlier chapter's `decode_step` always computed a position's own starting vector by looking up `token_id` in the embedding table -- a visual token has no such row to look up, because its vector was already computed upstream. `decode_step_with_embedding` factors the ORIGINAL `decode_step` into its two real halves -- "get a starting embedding," then "run every layer" -- and the original `decode_step` becomes a thin, behavior-preserving wrapper around it that supplies the embedding-table lookup as its one specific way of getting a starting vector. This is exactly Chapter 16.4's own pattern of extending an already-locked function with an optional new capability rather than rewriting its internals, applied here to embeddings instead of profiling.

`build_fused_embeddings` walks a token ID sequence and replaces each reserved `IMAGE_PLACEHOLDER_ID` with the next unused visual token vector, in order -- and this is where this section's own fail-loud discipline matters most, because a silent mismatch here would be a real, hard-to-diagnose deployment bug: too few placeholders for the visual tokens actually produced (some of the image would simply be discarded, silently) or too many placeholders for the tokens available (the fusion would need to invent vectors that do not exist). `build_fused_embeddings` refuses outright in both directions, returning a specific, human-readable error rather than silently truncating or padding -- exactly Test 1's own check, confirming both a matching count succeeds and each kind of mismatch is rejected with a message that names which direction the mismatch went.

`prefill_multimodal` calls `build_fused_embeddings` once, then loops `decode_step_with_embedding` over the resulting sequence exactly as `prefill` already loops `decode_step` over a text-only one -- reusing, not reimplementing, the KV-capacity guard and NaN detection Chapter 16 and Chapter 17 already verified. Test 4 confirms that guard survives the fusion unchanged: a 9-position fused sequence against a deliberately undersized 5-position cache still correctly reports `exceeded_capacity=true`, rather than the capacity check having been quietly lost somewhere in the new fusion logic. Test 3 checks the property that actually matters for calling this a vision-LANGUAGE model rather than a language model that happens to accept an extra argument: the SAME text tokens, fused with two DIFFERENT images, produce two DIFFERENT final hidden states -- proof the decoder's attention is genuinely conditioning on the image content, not silently ignoring vectors it was never trained to expect.

### Code and Verification

```cpp
// Chapter 18.3 -- Section 18.2 turned a raw camera frame into a small
// handful of visual tokens, each one already projected into the text
// decoder's own embedding space. This section does the one thing that
// makes "vision-LANGUAGE model" more than two separate models bolted
// together: splicing those visual tokens directly into a text token
// sequence, in place of a reserved placeholder ID, and running the
// IDENTICAL, unchanged Qwen2 decoder this book has verified since
// Chapter 15 over the combined sequence -- no separate "vision decoder,"
// no special-cased attention path for image positions. A visual token
// and a text token differ only in how their own embedding vector was
// produced; once both are sitting in the same sequence of vectors, one
// per position, `qwen2_block_forward` cannot tell the difference and
// was never asked to.
//
// The one real seam this requires is in exactly one place: today,
// `decode_step` always computes a position's own starting vector by
// looking up `token_id` in `token_embd.weight`. A visual token has no
// such row to look up -- its vector was already computed by Section
// 18.2's own merger. `decode_step_with_embedding` factors the ORIGINAL
// `decode_step` into "get a starting embedding, then run every layer,"
// and `decode_step` becomes the thin, unchanged-behavior special case
// of it that looks the embedding up by ID -- exactly Chapter 16.4's own
// pattern of extending a locked function with an optional capability
// rather than rewriting it, applied to embeddings instead of profiling.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 03_multimodal_fusion.cpp -o 03_multimodal_fusion
// Run:     ./03_multimodal_fusion

#include <mdspan/mdspan.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: Section 15.1's GGUF reader/writer (Section 15.4's own copy).
// =======================================================================
enum GGUFValueType : uint32_t {
    V_UINT8 = 0, V_INT8 = 1, V_UINT16 = 2, V_INT16 = 3, V_UINT32 = 4,
    V_INT32 = 5, V_FLOAT32 = 6, V_BOOL = 7, V_STRING = 8, V_ARRAY = 9,
    V_UINT64 = 10, V_INT64 = 11, V_FLOAT64 = 12,
};
enum GGMLType : uint32_t { GGML_F32 = 0, GGML_Q8_0 = 8 };
size_t scalar_byte_size(uint32_t t) {
    switch (t) {
        case V_UINT8: case V_INT8: case V_BOOL: return 1;
        case V_UINT16: case V_INT16: return 2;
        case V_UINT32: case V_INT32: case V_FLOAT32: return 4;
        case V_UINT64: case V_INT64: case V_FLOAT64: return 8;
        default: return 0;
    }
}
class GGUFWriter {
    std::ofstream out;
    size_t pos = 0;
    void write_raw(const void* data, size_t size) { out.write(reinterpret_cast<const char*>(data), size); pos += size; }
public:
    explicit GGUFWriter(const std::string& path) : out(path, std::ios::binary) {}
    void write_magic() { write_raw("GGUF", 4); }
    void write_u32(uint32_t v) { write_raw(&v, 4); }
    void write_u64(uint64_t v) { write_raw(&v, 8); }
    void write_f32(float v) { write_raw(&v, 4); }
    void write_string(const std::string& s) { uint64_t len = s.size(); write_raw(&len, 8); write_raw(s.data(), s.size()); }
    void write_kv_string(const std::string& k, const std::string& v) { write_string(k); write_u32(V_STRING); write_string(v); }
    void write_kv_u32(const std::string& k, uint32_t v) { write_string(k); write_u32(V_UINT32); write_u32(v); }
    void write_kv_f32(const std::string& k, float v) { write_string(k); write_u32(V_FLOAT32); write_f32(v); }
    void write_tensor_info(const std::string& name, const std::vector<uint64_t>& dims, GGMLType type, uint64_t offset) {
        write_string(name);
        write_u32(static_cast<uint32_t>(dims.size()));
        for (uint64_t d : dims) write_u64(d);
        write_u32(static_cast<uint32_t>(type));
        write_u64(offset);
    }
    void align(size_t alignment) {
        size_t rem = pos % alignment;
        if (rem != 0) { std::vector<char> zeros(alignment - rem, 0); write_raw(zeros.data(), zeros.size()); }
    }
    void write_bytes(const void* data, size_t size) { write_raw(data, size); }
    size_t tell() const { return pos; }
    bool good() const { return out.good(); }
};
struct TensorInfo {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t type = 0;
    uint64_t offset = 0;
    uint64_t n_elements = 0;
};
using MetaValue = std::variant<std::string, uint32_t, float, bool, std::vector<std::string>>;
class GGUFReader {
    std::ifstream in;
    void read_raw(void* data, size_t size) { in.read(reinterpret_cast<char*>(data), size); }
    void skip_value(uint32_t type) {
        if (type == V_STRING) { read_string(); return; }
        if (type == V_ARRAY) {
            uint32_t elem_type; read_raw(&elem_type, 4);
            uint64_t count; read_raw(&count, 8);
            for (uint64_t i = 0; i < count; ++i) skip_value(elem_type);
            return;
        }
        in.seekg(static_cast<std::streamoff>(scalar_byte_size(type)), std::ios::cur);
    }
public:
    uint32_t magic = 0, version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    std::unordered_map<std::string, MetaValue> metadata;
    std::vector<TensorInfo> tensors;
    uint64_t data_section_offset = 0;
    bool open(const std::string& path) {
        in.open(path, std::ios::binary);
        if (!in.is_open()) return false;
        char magic_bytes[4];
        read_raw(magic_bytes, 4);
        if (std::memcmp(magic_bytes, "GGUF", 4) != 0) return false;
        std::memcpy(&magic, magic_bytes, 4);
        read_raw(&version, 4);
        read_raw(&n_tensors, 8);
        read_raw(&n_kv, 8);
        uint64_t alignment = 32;
        for (uint64_t i = 0; i < n_kv; ++i) {
            std::string key = read_string();
            uint32_t type; read_raw(&type, 4);
            switch (type) {
                case V_STRING: metadata[key] = read_string(); break;
                case V_UINT32: { uint32_t v; read_raw(&v, 4); metadata[key] = v;
                                 if (key == "general.alignment") alignment = v;
                                 break; }
                case V_FLOAT32: { float v; read_raw(&v, 4); metadata[key] = v; break; }
                case V_BOOL: { uint8_t v; read_raw(&v, 1); metadata[key] = (v != 0); break; }
                case V_ARRAY: {
                    uint32_t elem_type; read_raw(&elem_type, 4);
                    uint64_t count; read_raw(&count, 8);
                    if (elem_type == V_STRING) {
                        std::vector<std::string> arr(count);
                        for (uint64_t j = 0; j < count; ++j) arr[j] = read_string();
                        metadata[key] = std::move(arr);
                    } else {
                        for (uint64_t j = 0; j < count; ++j) skip_value(elem_type);
                    }
                    break;
                }
                default: skip_value(type); break;
            }
        }
        tensors.resize(n_tensors);
        for (uint64_t i = 0; i < n_tensors; ++i) {
            auto& t = tensors[i];
            t.name = read_string();
            uint32_t n_dims; read_raw(&n_dims, 4);
            t.dims.resize(n_dims);
            for (uint32_t d = 0; d < n_dims; ++d) read_raw(&t.dims[d], 8);
            read_raw(&t.type, 4);
            read_raw(&t.offset, 8);
            t.n_elements = 1;
            for (auto d : t.dims) t.n_elements *= d;
        }
        uint64_t header_end = static_cast<uint64_t>(in.tellg());
        uint64_t rem = header_end % alignment;
        data_section_offset = (rem == 0) ? header_end : header_end + (alignment - rem);
        return true;
    }
    std::string read_string() {
        uint64_t len; read_raw(&len, 8);
        std::string s(len, '\0'); read_raw(s.data(), len); return s;
    }
    uint32_t get_u32(const std::string& key) const {
        auto it = metadata.find(key);
        return (it == metadata.end()) ? 0 : std::get<uint32_t>(it->second);
    }
    float get_f32(const std::string& key) const {
        auto it = metadata.find(key);
        return (it == metadata.end()) ? 0.0f : std::get<float>(it->second);
    }
    const TensorInfo* find_tensor(const std::string& name) const {
        for (const auto& t : tensors) if (t.name == name) return &t;
        return nullptr;
    }
};

// =======================================================================
// PART 2: Chapter 4.2's fp16_t/BlockQ8 (Section 15.4's own copy).
// =======================================================================
struct fp16_t {
    uint16_t bits = 0;
    fp16_t() = default;
    fp16_t(float f) { bits = encode(f); }
    operator float() const { return decode(bits); }
    static uint16_t encode(float f) {
        uint32_t x; std::memcpy(&x, &f, 4);
        uint32_t sign = (x >> 16) & 0x8000u;
        int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
        uint32_t mant = x & 0x7FFFFFu;
        if (((x >> 23) & 0xFFu) == 0xFFu) return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x0200u : 0u));
        if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
        if (exp <= 0)  return static_cast<uint16_t>(sign);
        uint32_t mant10 = mant >> 13;
        uint32_t remainder = mant & 0x1FFFu;
        if (remainder > 0x1000u || (remainder == 0x1000u && (mant10 & 1u))) {
            mant10 += 1;
            if (mant10 == 0x400u) { mant10 = 0; exp += 1; if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u); }
        }
        return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant10);
    }
    static float decode(uint16_t h) {
        uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
        uint32_t exp  = (static_cast<uint32_t>(h) >> 10) & 0x1Fu;
        uint32_t mant = static_cast<uint32_t>(h) & 0x3FFu;
        uint32_t fbits;
        if (exp == 0) fbits = sign;
        else if (exp == 31) fbits = sign | 0x7F800000u | (mant << 13);
        else fbits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
        float f; std::memcpy(&f, &fbits, 4); return f;
    }
};
#pragma pack(push, 1)
struct BlockQ8 { fp16_t scale; int8_t weights[32]; };
#pragma pack(pop)
static_assert(sizeof(BlockQ8) == 34);
BlockQ8 quantize_q8(const float* data) {
    BlockQ8 b;
    float alpha = 0.0f;
    for (int i = 0; i < 32; ++i) alpha = std::max(alpha, std::fabs(data[i]));
    if (alpha == 0.0f) { b.scale = fp16_t(0.0f); std::memset(b.weights, 0, 32); return b; }
    b.scale = fp16_t(alpha / 127.0f);
    float inv = 1.0f / static_cast<float>(b.scale);
    for (int i = 0; i < 32; ++i)
        b.weights[i] = static_cast<int8_t>(std::clamp(std::round(data[i] * inv), -127.0f, 127.0f));
    return b;
}

// =======================================================================
// PART 3: memory-mapped file + dequantization (Section 15.4's own copy).
// =======================================================================
struct MappedFile {
    void* data = nullptr;
    size_t size = 0;
    int fd = -1;
    bool open(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st{};
        if (fstat(fd, &st) != 0) { ::close(fd); fd = -1; return false; }
        size = static_cast<size_t>(st.st_size);
        data = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) { data = nullptr; ::close(fd); fd = -1; return false; }
        return true;
    }
    const uint8_t* at(uint64_t byte_offset) const { return static_cast<const uint8_t*>(data) + byte_offset; }
    ~MappedFile() { if (data) ::munmap(data, size); if (fd >= 0) ::close(fd); }
};
std::vector<float> dequantize_tensor(const MappedFile& mf, const GGUFReader& r, const TensorInfo& t) {
    std::vector<float> out(t.n_elements);
    const uint8_t* p = mf.at(r.data_section_offset + t.offset);
    if (t.type == GGML_F32) {
        std::memcpy(out.data(), p, t.n_elements * sizeof(float));
    } else if (t.type == GGML_Q8_0) {
        uint64_t n_blocks = t.n_elements / 32;
        for (uint64_t b = 0; b < n_blocks; ++b) {
            BlockQ8 blk;
            std::memcpy(&blk, p + b * sizeof(BlockQ8), sizeof(BlockQ8));
            float s = static_cast<float>(blk.scale);
            for (int i = 0; i < 32; ++i) out[b * 32 + i] = static_cast<float>(blk.weights[i]) * s;
        }
    }
    return out;
}
void dequantize_row_q8(const MappedFile& mf, uint64_t abs_row_offset, size_t n_elements, std::span<float> out) {
    const uint8_t* p = mf.at(abs_row_offset);
    size_t n_blocks = n_elements / 32;
    for (size_t b = 0; b < n_blocks; ++b) {
        BlockQ8 blk;
        std::memcpy(&blk, p + b * sizeof(BlockQ8), sizeof(BlockQ8));
        float s = static_cast<float>(blk.scale);
        for (int i = 0; i < 32; ++i) out[b * 32 + i] = static_cast<float>(blk.weights[i]) * s;
    }
}

// =======================================================================
// PART 4: Section 15.3's adapted transformer block (Section 15.4's own
// copy).
// =======================================================================
void rms_norm(std::span<float> out, std::span<const float> x, std::span<const float> weights, float epsilon = 1e-6f) {
    const size_t d = x.size();
    double sum_sq = 0.0;
    for (size_t i = 0; i < d; ++i) sum_sq += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    float rms_inv = 1.0f / std::sqrt(static_cast<float>(sum_sq / static_cast<double>(d)) + epsilon);
    for (size_t i = 0; i < d; ++i) out[i] = (x[i] * rms_inv) * weights[i];
}
void matmul(std::span<float> out, std::span<const float> x, std::span<const float> W, size_t in_dim, size_t out_dim) {
    for (size_t j = 0; j < out_dim; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < in_dim; ++i) sum += static_cast<double>(x[i]) * static_cast<double>(W[j * in_dim + i]);
        out[j] = static_cast<float>(sum);
    }
}
inline float silu(float x) { return x * (1.0f / (1.0f + std::exp(-x))); }
void swiglu_ffn(std::span<float> out, std::span<const float> x, std::span<const float> W_gate,
                 std::span<const float> W_up, std::span<const float> W_down, size_t dim, size_t d_ff) {
    std::vector<float> gate_proj(d_ff), up_proj(d_ff), hidden(d_ff);
    matmul(gate_proj, x, W_gate, dim, d_ff);
    matmul(up_proj, x, W_up, dim, d_ff);
    for (size_t i = 0; i < d_ff; ++i) hidden[i] = silu(gate_proj[i]) * up_proj[i];
    matmul(out, hidden, W_down, d_ff, dim);
}
void linear_with_bias(std::span<float> out, std::span<const float> x, std::span<const float> W,
                       std::span<const float> bias, size_t in_dim, size_t out_dim) {
    matmul(out, x, W, in_dim, out_dim);
    for (size_t j = 0; j < out_dim; ++j) out[j] += bias[j];
}
struct RoPETables {
    std::vector<float> cos_vals, sin_vals;
    int half_dim;
    RoPETables(int seq_len, int head_dim, float base) : half_dim(head_dim / 2) {
        cos_vals.resize(static_cast<size_t>(seq_len) * static_cast<size_t>(half_dim));
        sin_vals.resize(static_cast<size_t>(seq_len) * static_cast<size_t>(half_dim));
        for (int pos = 0; pos < seq_len; ++pos) {
            for (int k = 0; k < half_dim; ++k) {
                float theta = 1.0f / std::pow(base, (2.0f * static_cast<float>(k)) / static_cast<float>(head_dim));
                float angle = static_cast<float>(pos) * theta;
                cos_vals[static_cast<size_t>(pos) * static_cast<size_t>(half_dim) + static_cast<size_t>(k)] = std::cos(angle);
                sin_vals[static_cast<size_t>(pos) * static_cast<size_t>(half_dim) + static_cast<size_t>(k)] = std::sin(angle);
            }
        }
    }
    float cos_at(int pos, int k) const { return cos_vals[static_cast<size_t>(pos) * static_cast<size_t>(half_dim) + static_cast<size_t>(k)]; }
    float sin_at(int pos, int k) const { return sin_vals[static_cast<size_t>(pos) * static_cast<size_t>(half_dim) + static_cast<size_t>(k)]; }
};
void apply_rope(std::span<float> vec, int pos, const RoPETables& tables) {
    const int half_dim = tables.half_dim;
    for (int k = 0; k < half_dim; ++k) {
        float x1 = vec[static_cast<size_t>(k)], x2 = vec[static_cast<size_t>(k + half_dim)];
        float c = tables.cos_at(pos, k), s = tables.sin_at(pos, k);
        vec[static_cast<size_t>(k)] = x1 * c - x2 * s;
        vec[static_cast<size_t>(k + half_dim)] = x1 * s + x2 * c;
    }
}
void softmax_inplace(std::span<float> scores) {
    float max_val = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;
    for (float& s : scores) { s = std::exp(s - max_val); sum += s; }
    float inv_sum = 1.0f / sum;
    for (float& s : scores) s *= inv_sum;
}
struct KVCache {
    std::vector<float> K, V;
    int n_heads_kv, max_seq_len, head_dim;
    using View = std::mdspan<float, std::dextents<size_t, 3>>;
    KVCache(int nh, int seq, int hd) : n_heads_kv(nh), max_seq_len(seq), head_dim(hd) {
        K.assign(static_cast<size_t>(nh) * static_cast<size_t>(seq) * static_cast<size_t>(hd), 0.0f);
        V.assign(static_cast<size_t>(nh) * static_cast<size_t>(seq) * static_cast<size_t>(hd), 0.0f);
    }
    View k_view() { return View(K.data(), n_heads_kv, max_seq_len, head_dim); }
    View v_view() { return View(V.data(), n_heads_kv, max_seq_len, head_dim); }
    auto k_at(int h, int t) { return std::submdspan(k_view(), h, t, std::full_extent); }
    auto v_at(int h, int t) { return std::submdspan(v_view(), h, t, std::full_extent); }
    void store(int h, int t, std::span<const float> k, std::span<const float> v) {
        auto kslice = k_at(h, t);
        auto vslice = v_at(h, t);
        for (int i = 0; i < head_dim; ++i) { kslice[i] = k[static_cast<size_t>(i)]; vslice[i] = v[static_cast<size_t>(i)]; }
    }
};
void gqa_attention(std::span<const float> q_heads, KVCache& cache, std::span<float> output,
                    int seq_len, int n_heads_q, int group_size) {
    const int head_dim = cache.head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(static_cast<size_t>(seq_len));
    for (int h = 0; h < n_heads_q; ++h) {
        int kv_h = h / group_size;
        std::span<const float> q(q_heads.data() + h * head_dim, static_cast<size_t>(head_dim));
        for (int t = 0; t < seq_len; ++t) {
            auto k = cache.k_at(kv_h, t);
            double d = 0.0;
            for (int i = 0; i < head_dim; ++i) d += static_cast<double>(q[i]) * static_cast<double>(k[i]);
            scores[static_cast<size_t>(t)] = static_cast<float>(d) * scale;
        }
        softmax_inplace(std::span<float>(scores.data(), static_cast<size_t>(seq_len)));
        std::vector<double> acc(static_cast<size_t>(head_dim), 0.0);
        for (int t = 0; t < seq_len; ++t) {
            auto v = cache.v_at(kv_h, t);
            double w = scores[static_cast<size_t>(t)];
            for (int i = 0; i < head_dim; ++i) acc[static_cast<size_t>(i)] += w * static_cast<double>(v[i]);
        }
        float* out = output.data() + h * head_dim;
        for (int i = 0; i < head_dim; ++i) out[i] = static_cast<float>(acc[static_cast<size_t>(i)]);
    }
}
struct QwenBlockWeights {
    std::vector<float> attn_norm, Wq, bq, Wk, bk, Wv, bv, Wo;
    std::vector<float> ffn_norm, Wgate, Wup, Wdown;
};
struct QwenShape {
    int dim, n_heads, n_heads_kv, head_dim, d_ff;
    int q_dim() const { return n_heads * head_dim; }
    int kv_dim() const { return n_heads_kv * head_dim; }
    int group_size() const { return n_heads / n_heads_kv; }
};
void qwen2_block_forward(std::span<float> x, const QwenShape& shape, const QwenBlockWeights& w,
                          KVCache& cache, int pos, const RoPETables& rope) {
    std::vector<float> normed(static_cast<size_t>(shape.dim)), q(static_cast<size_t>(shape.q_dim())),
        k(static_cast<size_t>(shape.kv_dim())), v(static_cast<size_t>(shape.kv_dim()));
    std::vector<float> attn_out(static_cast<size_t>(shape.q_dim())), proj_out(static_cast<size_t>(shape.dim));
    rms_norm(normed, x, w.attn_norm);
    linear_with_bias(q, normed, w.Wq, w.bq, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.q_dim()));
    linear_with_bias(k, normed, w.Wk, w.bk, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.kv_dim()));
    linear_with_bias(v, normed, w.Wv, w.bv, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.kv_dim()));
    for (int h = 0; h < shape.n_heads; ++h)
        apply_rope(std::span<float>(q.data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)), pos, rope);
    for (int h = 0; h < shape.n_heads_kv; ++h)
        apply_rope(std::span<float>(k.data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)), pos, rope);
    for (int h = 0; h < shape.n_heads_kv; ++h)
        cache.store(h, pos, std::span<const float>(k.data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)),
                             std::span<const float>(v.data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)));
    gqa_attention(q, cache, attn_out, pos + 1, shape.n_heads, shape.group_size());
    matmul(proj_out, attn_out, w.Wo, static_cast<size_t>(shape.q_dim()), static_cast<size_t>(shape.dim));
    for (int i = 0; i < shape.dim; ++i) x[static_cast<size_t>(i)] += proj_out[static_cast<size_t>(i)];
    std::vector<float> normed2(static_cast<size_t>(shape.dim)), ffn_out(static_cast<size_t>(shape.dim));
    rms_norm(normed2, x, w.ffn_norm);
    swiglu_ffn(ffn_out, normed2, w.Wgate, w.Wup, w.Wdown, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.d_ff));
    for (int i = 0; i < shape.dim; ++i) x[static_cast<size_t>(i)] += ffn_out[static_cast<size_t>(i)];
}
struct QwenModel {
    MappedFile mf;
    GGUFReader r;
    QwenShape shape{};
    RoPETables* rope = nullptr;
    bool load(const std::string& path) {
        if (!mf.open(path)) return false;
        if (!r.open(path)) return false;
        shape.dim = static_cast<int>(r.get_u32("qwen2.embedding_length"));
        shape.n_heads = static_cast<int>(r.get_u32("qwen2.attention.head_count"));
        shape.n_heads_kv = static_cast<int>(r.get_u32("qwen2.attention.head_count_kv"));
        shape.head_dim = shape.dim / shape.n_heads;
        shape.d_ff = static_cast<int>(r.get_u32("qwen2.feed_forward_length"));
        float base = r.get_f32("qwen2.rope.freq_base");
        rope = new RoPETables(4096, shape.head_dim, base);
        return true;
    }
    ~QwenModel() { delete rope; }
    int n_layers() const { return static_cast<int>(r.get_u32("qwen2.block_count")); }
    std::vector<float> tensor(const std::string& name) const {
        const auto* t = r.find_tensor(name);
        return dequantize_tensor(mf, r, *t);
    }
    QwenBlockWeights layer(int idx) const {
        std::string p = "blk." + std::to_string(idx) + ".";
        QwenBlockWeights w;
        w.attn_norm = tensor(p + "attn_norm.weight");
        w.Wq = tensor(p + "attn_q.weight");   w.bq = tensor(p + "attn_q.bias");
        w.Wk = tensor(p + "attn_k.weight");   w.bk = tensor(p + "attn_k.bias");
        w.Wv = tensor(p + "attn_v.weight");   w.bv = tensor(p + "attn_v.bias");
        w.Wo = tensor(p + "attn_output.weight");
        w.ffn_norm = tensor(p + "ffn_norm.weight");
        w.Wgate = tensor(p + "ffn_gate.weight");
        w.Wup = tensor(p + "ffn_up.weight");
        w.Wdown = tensor(p + "ffn_down.weight");
        return w;
    }
    std::vector<float> embedding(int token_id) const {
        const auto* t = r.find_tensor("token_embd.weight");
        uint64_t row_offset = r.data_section_offset + t->offset
                             + (static_cast<uint64_t>(token_id) * static_cast<uint64_t>(shape.dim) / 32) * sizeof(BlockQ8);
        std::vector<float> out(static_cast<size_t>(shape.dim));
        dequantize_row_q8(mf, row_offset, static_cast<size_t>(shape.dim), out);
        return out;
    }
};

// =======================================================================
// PART 1: raw-buffer preprocessing. Section 18.1's FrameResult::pixels
// is already exactly this input -- an HxWxC row-major byte buffer, no
// codec involved anywhere in this pipeline.
// =======================================================================
struct RawImage {
    uint32_t width = 0, height = 0, channels = 0;
    std::vector<uint8_t> pixels;   // row-major, HxWxC, matching FrameResult::pixels exactly
};

// Nearest-neighbor resize to a target size that is an exact multiple of
// the ViT's own patch size. Real deployments use a higher-quality
// filter; nearest-neighbor is this section's own stated simplification,
// chosen because it is exactly reproducible in integer arithmetic across
// every architecture this book locks against, with no floating-point
// rounding difference to chase.
RawImage resize_nearest(const RawImage& src, uint32_t dst_w, uint32_t dst_h) {
    RawImage dst;
    dst.width = dst_w; dst.height = dst_h; dst.channels = src.channels;
    dst.pixels.resize(static_cast<size_t>(dst_w) * dst_h * src.channels);
    for (uint32_t y = 0; y < dst_h; ++y) {
        uint32_t sy = std::min(src.height - 1, (y * src.height) / dst_h);
        for (uint32_t x = 0; x < dst_w; ++x) {
            uint32_t sx = std::min(src.width - 1, (x * src.width) / dst_w);
            const uint8_t* sp = &src.pixels[(static_cast<size_t>(sy) * src.width + sx) * src.channels];
            uint8_t* dp = &dst.pixels[(static_cast<size_t>(y) * dst_w + x) * src.channels];
            std::memcpy(dp, sp, src.channels);
        }
    }
    return dst;
}

// CLIP/SigLIP-style per-channel normalization constants -- stated, not
// derived: a real deployment would use whichever mean/std the specific
// checkpoint's own preprocessor_config.json declares.
struct NormStats { float mean[3] = {0.481f, 0.458f, 0.408f}; float std[3] = {0.269f, 0.261f, 0.276f}; };

// One patch's flattened, normalized pixel data: patch_size * patch_size
// * channels floats, in row-major (row, then col, then channel) order.
std::vector<float> extract_patch(const RawImage& img, uint32_t patch_row, uint32_t patch_col,
                                  uint32_t patch_size, const NormStats& norm) {
    std::vector<float> out(static_cast<size_t>(patch_size) * patch_size * img.channels);
    size_t idx = 0;
    for (uint32_t py = 0; py < patch_size; ++py) {
        uint32_t y = patch_row * patch_size + py;
        for (uint32_t px = 0; px < patch_size; ++px) {
            uint32_t x = patch_col * patch_size + px;
            const uint8_t* sp = &img.pixels[(static_cast<size_t>(y) * img.width + x) * img.channels];
            for (uint32_t c = 0; c < img.channels; ++c) {
                float v = static_cast<float>(sp[c]) / 255.0f;
                out[idx++] = (v - norm.mean[c % 3]) / norm.std[c % 3];
            }
        }
    }
    return out;
}

// Patchifies the WHOLE image into a row-major grid of flattened patches:
// patches[row * grid_w + col] is that (row, col) patch's own flattened,
// normalized pixel vector.
std::vector<std::vector<float>> patchify(const RawImage& img, uint32_t patch_size, const NormStats& norm,
                                          uint32_t& grid_h, uint32_t& grid_w) {
    grid_h = img.height / patch_size;
    grid_w = img.width / patch_size;
    std::vector<std::vector<float>> patches(static_cast<size_t>(grid_h) * grid_w);
    for (uint32_t r = 0; r < grid_h; ++r)
        for (uint32_t c = 0; c < grid_w; ++c)
            patches[static_cast<size_t>(r) * grid_w + c] = extract_patch(img, r, c, patch_size, norm);
    return patches;
}


// =======================================================================
// PART 3: two-dimensional rotary position encoding. A text token has one
// position; a patch has TWO (its row and its column), and Qwen2-VL's own
// real fix is splitting each attention head's dimension in half, rotating
// the FIRST half by the patch's row and the SECOND half by its column --
// the same 1D rotate-half mechanism this book has used since Chapter 9,
// applied twice, to two different coordinates, over two disjoint slices
// of the same vector.
// =======================================================================
struct RoPE2DTables {
    std::vector<float> cos_row, sin_row, cos_col, sin_col;
    int quarter_dim;   // half_dim (per axis) is head_dim/2; each axis's own rotate-half pairs cover head_dim/4
    RoPE2DTables(int max_coord, int head_dim, float base) : quarter_dim(head_dim / 4) {
        cos_row.resize(static_cast<size_t>(max_coord) * quarter_dim);
        sin_row.resize(static_cast<size_t>(max_coord) * quarter_dim);
        cos_col.resize(static_cast<size_t>(max_coord) * quarter_dim);
        sin_col.resize(static_cast<size_t>(max_coord) * quarter_dim);
        for (int coord = 0; coord < max_coord; ++coord) {
            for (int k = 0; k < quarter_dim; ++k) {
                float theta = 1.0f / std::pow(base, (2.0f * static_cast<float>(k)) / static_cast<float>(head_dim / 2));
                float angle = static_cast<float>(coord) * theta;
                size_t idx = static_cast<size_t>(coord) * quarter_dim + k;
                cos_row[idx] = std::cos(angle); sin_row[idx] = std::sin(angle);
                cos_col[idx] = std::cos(angle); sin_col[idx] = std::sin(angle);
            }
        }
    }
};
// Rotates a length-`2*half`-element slice using the standard rotate-half
// pairing (index k paired with index k+half), exactly Section 9's own
// `apply_rope` -- factored out so both the row-half and the col-half of
// a patch's query/key vector can call the identical primitive.
void rotate_half_inplace(std::span<float> vec, std::span<const float> cos_tab, std::span<const float> sin_tab,
                          int coord, int quarter_dim) {
    for (int k = 0; k < quarter_dim; ++k) {
        float x1 = vec[static_cast<size_t>(k)], x2 = vec[static_cast<size_t>(k + quarter_dim)];
        float c = cos_tab[static_cast<size_t>(coord) * quarter_dim + k];
        float s = sin_tab[static_cast<size_t>(coord) * quarter_dim + k];
        vec[static_cast<size_t>(k)] = x1 * c - x2 * s;
        vec[static_cast<size_t>(k + quarter_dim)] = x1 * s + x2 * c;
    }
}
// Applies 2D RoPE to one attention head's full head_dim vector: the
// FIRST half rotated by `row` (using its own rotate-half pairing over
// that half's two quarters), the SECOND half rotated by `col`.
void apply_rope2d(std::span<float> head_vec, int row, int col, const RoPE2DTables& t) {
    const int half = static_cast<int>(head_vec.size()) / 2;
    rotate_half_inplace(head_vec.subspan(0, static_cast<size_t>(half)), t.cos_row, t.sin_row, row, t.quarter_dim);
    rotate_half_inplace(head_vec.subspan(static_cast<size_t>(half), static_cast<size_t>(half)), t.cos_col, t.sin_col, col, t.quarter_dim);
}

// =======================================================================
// PART 4: the vision transformer block itself -- RMSNorm, QKV
// projection, 2D RoPE, FULL (non-causal, no KV cache) bidirectional
// attention over every patch, output projection, a residual, a second
// RMSNorm, the SwiGLU FFN, and a second residual.
// =======================================================================
struct ViTShape { int dim, n_heads, head_dim, d_ff; int qkv_dim() const { return n_heads * head_dim; } };
struct ViTBlockWeights {
    std::vector<float> attn_norm, Wq, Wk, Wv, Wo;
    std::vector<float> ffn_norm, Wgate, Wup, Wdown;
};

// One full self-attention pass over ALL patches -- unlike the text
// decoder's causal, one-position-at-a-time attention, every patch here
// attends to every other patch in a single call, because an image has
// no "future" to mask and no cache to build incrementally.
void vit_full_attention(const std::vector<std::vector<float>>& q, const std::vector<std::vector<float>>& k,
                         const std::vector<std::vector<float>>& v, std::vector<std::vector<float>>& out,
                         int n_heads, int head_dim) {
    const int n = static_cast<int>(q.size());
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int h = 0; h < n_heads; ++h) {
        for (int i = 0; i < n; ++i) {
            std::vector<float> scores(static_cast<size_t>(n));
            std::span<const float> qi(q[static_cast<size_t>(i)].data() + h * head_dim, static_cast<size_t>(head_dim));
            for (int j = 0; j < n; ++j) {
                std::span<const float> kj(k[static_cast<size_t>(j)].data() + h * head_dim, static_cast<size_t>(head_dim));
                double d = 0.0;
                for (int c = 0; c < head_dim; ++c) d += static_cast<double>(qi[static_cast<size_t>(c)]) * kj[static_cast<size_t>(c)];
                scores[static_cast<size_t>(j)] = static_cast<float>(d) * scale;
            }
            softmax_inplace(scores);
            float* o = out[static_cast<size_t>(i)].data() + h * head_dim;
            for (int c = 0; c < head_dim; ++c) o[c] = 0.0f;
            for (int j = 0; j < n; ++j) {
                std::span<const float> vj(v[static_cast<size_t>(j)].data() + h * head_dim, static_cast<size_t>(head_dim));
                float w = scores[static_cast<size_t>(j)];
                for (int c = 0; c < head_dim; ++c) o[c] += w * vj[static_cast<size_t>(c)];
            }
        }
    }
}

void vit_block_forward(std::vector<std::vector<float>>& x, const ViTShape& shape, const ViTBlockWeights& w,
                        const std::vector<int>& rows, const std::vector<int>& cols, const RoPE2DTables& rope) {
    const int n = static_cast<int>(x.size());
    std::vector<std::vector<float>> q(static_cast<size_t>(n)), k(static_cast<size_t>(n)), v(static_cast<size_t>(n)), attn_out(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        std::vector<float> normed(static_cast<size_t>(shape.dim));
        rms_norm(normed, x[static_cast<size_t>(i)], w.attn_norm);
        q[static_cast<size_t>(i)].resize(static_cast<size_t>(shape.qkv_dim()));
        k[static_cast<size_t>(i)].resize(static_cast<size_t>(shape.qkv_dim()));
        v[static_cast<size_t>(i)].resize(static_cast<size_t>(shape.qkv_dim()));
        attn_out[static_cast<size_t>(i)].resize(static_cast<size_t>(shape.qkv_dim()));
        matmul(q[static_cast<size_t>(i)], normed, w.Wq, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.qkv_dim()));
        matmul(k[static_cast<size_t>(i)], normed, w.Wk, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.qkv_dim()));
        matmul(v[static_cast<size_t>(i)], normed, w.Wv, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.qkv_dim()));
        for (int h = 0; h < shape.n_heads; ++h) {
            apply_rope2d(std::span<float>(q[static_cast<size_t>(i)].data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)),
                         rows[static_cast<size_t>(i)], cols[static_cast<size_t>(i)], rope);
            apply_rope2d(std::span<float>(k[static_cast<size_t>(i)].data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)),
                         rows[static_cast<size_t>(i)], cols[static_cast<size_t>(i)], rope);
        }
    }
    vit_full_attention(q, k, v, attn_out, shape.n_heads, shape.head_dim);
    for (int i = 0; i < n; ++i) {
        std::vector<float> proj(static_cast<size_t>(shape.dim));
        matmul(proj, attn_out[static_cast<size_t>(i)], w.Wo, static_cast<size_t>(shape.qkv_dim()), static_cast<size_t>(shape.dim));
        for (int d = 0; d < shape.dim; ++d) x[static_cast<size_t>(i)][static_cast<size_t>(d)] += proj[static_cast<size_t>(d)];
        std::vector<float> normed2(static_cast<size_t>(shape.dim)), ffn_out(static_cast<size_t>(shape.dim));
        rms_norm(normed2, x[static_cast<size_t>(i)], w.ffn_norm);
        swiglu_ffn(ffn_out, normed2, w.Wgate, w.Wup, w.Wdown, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.d_ff));
        for (int d = 0; d < shape.dim; ++d) x[static_cast<size_t>(i)][static_cast<size_t>(d)] += ffn_out[static_cast<size_t>(d)];
    }
}

// =======================================================================
// PART 5: the 2x2 spatial patch merger. Four SPATIALLY ADJACENT patches
// -- (2r,2c), (2r,2c+1), (2r+1,2c), (2r+1,2c+1) -- are concatenated and
// projected down to the LLM's own embedding dimension, cutting the
// visual token count by 4x. Grouping by spatial adjacency, not by
// row-major list order, matters: for any grid wider than two patches,
// four row-major-consecutive patches are the START OF ONE ROW, not a 2x2
// neighborhood, and merging them would blend unrelated regions of the
// image into one token.
// =======================================================================
struct MergerWeights { std::vector<float> W1, b1, W2, b2; int hidden_dim; };

// Returns the flat patch-list indices of the 4 patches spatially
// adjacent to merged block (br, bc), in a fixed, testable order:
// top-left, top-right, bottom-left, bottom-right. Factored out on its
// own specifically so this section's own self-tests can check the
// SPATIAL grouping directly, independent of the matmul math around it.
std::array<int, 4> block_patch_indices(int br, int bc, int grid_w) {
    const int r0 = 2 * br, r1 = 2 * br + 1, c0 = 2 * bc, c1 = 2 * bc + 1;
    return {r0 * grid_w + c0, r0 * grid_w + c1, r1 * grid_w + c0, r1 * grid_w + c1};
}

std::vector<std::vector<float>> merge_all_2x2(const std::vector<std::vector<float>>& patches, int grid_h, int grid_w,
                                               int vit_dim, const MergerWeights& mw, int llm_dim) {
    const int out_h = grid_h / 2, out_w = grid_w / 2;
    std::vector<std::vector<float>> merged(static_cast<size_t>(out_h) * out_w);
    for (int br = 0; br < out_h; ++br) {
        for (int bc = 0; bc < out_w; ++bc) {
            std::vector<float> concat(static_cast<size_t>(4 * vit_dim));
            auto idx = block_patch_indices(br, bc, grid_w);
            for (int slot = 0; slot < 4; ++slot) {
                const auto& p = patches[static_cast<size_t>(idx[static_cast<size_t>(slot)])];
                std::copy(p.begin(), p.end(), concat.begin() + slot * vit_dim);
            }
            std::vector<float> h1(static_cast<size_t>(mw.hidden_dim));
            matmul(h1, concat, mw.W1, static_cast<size_t>(4 * vit_dim), static_cast<size_t>(mw.hidden_dim));
            for (int i = 0; i < mw.hidden_dim; ++i) h1[static_cast<size_t>(i)] = silu(h1[static_cast<size_t>(i)] + mw.b1[static_cast<size_t>(i)]);
            std::vector<float> out(static_cast<size_t>(llm_dim));
            matmul(out, h1, mw.W2, static_cast<size_t>(mw.hidden_dim), static_cast<size_t>(llm_dim));
            for (int i = 0; i < llm_dim; ++i) out[static_cast<size_t>(i)] += mw.b2[static_cast<size_t>(i)];
            merged[static_cast<size_t>(br) * out_w + bc] = std::move(out);
        }
    }
    return merged;
}


// =======================================================================
// PART 6 (new): the fusion itself. `decode_step_with_embedding` is
// Section 16.3's own `decode_step`, factored so the "how do I get this
// position's starting vector" question is answered by the CALLER --
// `decode_step` (below) answers it the original way (look up a real
// token id); `build_fused_embeddings` answers it a second way (splice in
// an already-computed visual token) for exactly the positions a caller
// marks as image positions.
// =======================================================================
struct DecodeStepResult { std::vector<float> hidden; int nan_at_layer = -1; };
DecodeStepResult decode_step_with_embedding(const QwenModel& model, const std::vector<QwenBlockWeights>& layers,
                                              std::vector<KVCache>& caches, std::vector<float> x, int pos) {
    DecodeStepResult res;
    for (int layer = 0; layer < model.n_layers(); ++layer) {
        qwen2_block_forward(x, model.shape, layers[static_cast<size_t>(layer)], caches[static_cast<size_t>(layer)], pos, *model.rope);
        bool has_nan = false;
        for (float v : x) if (std::isnan(v)) { has_nan = true; break; }
        if (has_nan) { res.nan_at_layer = layer; break; }
    }
    res.hidden = std::move(x);
    return res;
}
DecodeStepResult decode_step(const QwenModel& model, const std::vector<QwenBlockWeights>& layers,
                               std::vector<KVCache>& caches, int token_id, int pos) {
    return decode_step_with_embedding(model, layers, caches, model.embedding(token_id), pos);
}

// A sentinel token id reserved (by this section's own synthetic
// vocabulary, exactly the way real Qwen2-VL reserves `<|image_pad|>`'s
// own real id) to mark "a visual token belongs here" inside an otherwise
// ordinary text token-id sequence.
constexpr int IMAGE_PLACEHOLDER_ID = -1;

struct FusionResult {
    bool ok = false;
    std::string error;
    std::vector<std::vector<float>> embeddings;   // one per position, text-looked-up or visual-injected
};

// Walks `token_ids` in order, replacing every `IMAGE_PLACEHOLDER_ID`
// with the NEXT unused visual token from `visual_tokens`, in sequence --
// and refuses, rather than silently truncating or zero-padding, if the
// placeholder count and the visual token count do not match exactly.
// A mismatch here is exactly the kind of silent-misalignment bug this
// book has refused to let slide since Chapter 15's own KVCache-sharing
// bug: every later position's own attention would still run and produce
// SOME finite number, with nothing about the output signaling that the
// image tokens landed in the wrong places, or that some were reused, or
// dropped.
FusionResult build_fused_embeddings(const QwenModel& model, const std::vector<int>& token_ids,
                                     const std::vector<std::vector<float>>& visual_tokens) {
    FusionResult res;
    size_t used = 0;
    res.embeddings.reserve(token_ids.size());
    for (int tid : token_ids) {
        if (tid == IMAGE_PLACEHOLDER_ID) {
            if (used >= visual_tokens.size()) {
                res.error = "more image placeholders than visual tokens";
                return res;
            }
            res.embeddings.push_back(visual_tokens[used++]);
        } else {
            res.embeddings.push_back(model.embedding(tid));
        }
    }
    if (used != visual_tokens.size()) {
        res.error = "fewer image placeholders than visual tokens (" + std::to_string(used) +
                    " used, " + std::to_string(visual_tokens.size()) + " provided)";
        return res;
    }
    res.ok = true;
    return res;
}

struct MultimodalPrefillResult { std::vector<float> hidden; bool exceeded_capacity = false; bool fusion_failed = false;
                                  std::string fusion_error; int nan_at_layer = -1; };
MultimodalPrefillResult prefill_multimodal(const QwenModel& model, const std::vector<QwenBlockWeights>& layers,
                                            std::vector<KVCache>& caches, const std::vector<int>& token_ids,
                                            const std::vector<std::vector<float>>& visual_tokens,
                                            int start_pos, int kv_capacity) {
    MultimodalPrefillResult res;
    auto fused = build_fused_embeddings(model, token_ids, visual_tokens);
    if (!fused.ok) { res.fusion_failed = true; res.fusion_error = fused.error; return res; }
    for (size_t i = 0; i < fused.embeddings.size(); ++i) {
        int pos = start_pos + static_cast<int>(i);
        if (pos >= kv_capacity) { res.exceeded_capacity = true; break; }
        auto step = decode_step_with_embedding(model, layers, caches, fused.embeddings[i], pos);
        res.hidden = std::move(step.hidden);
        if (step.nan_at_layer >= 0) { res.nan_at_layer = step.nan_at_layer; break; }
    }
    return res;
}

// Runs Section 18.2's own encoder end to end -- patchify, embed, N ViT
// blocks, merge -- and returns the resulting visual tokens, projected to
// the LLM's own embedding dimension so `build_fused_embeddings` can drop
// them straight into a text sequence with no further adaptation.
struct VisionEncoderWeights {
    std::vector<float> W_patch_embed;
    std::vector<ViTBlockWeights> layers;
    MergerWeights merger;
};
std::vector<std::vector<float>> run_vision_encoder(const RawImage& image, int patch_size, const NormStats& norm,
                                                    const ViTShape& shape, const VisionEncoderWeights& w,
                                                    int llm_dim, uint32_t& out_grid_h, uint32_t& out_grid_w) {
    uint32_t grid_h = 0, grid_w = 0;
    auto raw_patches = patchify(image, static_cast<uint32_t>(patch_size), norm, grid_h, grid_w);
    const int n_patches = static_cast<int>(raw_patches.size());
    const size_t patch_vec_len = raw_patches[0].size();

    std::vector<std::vector<float>> x(static_cast<size_t>(n_patches));
    std::vector<int> rows(static_cast<size_t>(n_patches)), cols(static_cast<size_t>(n_patches));
    for (int i = 0; i < n_patches; ++i) {
        x[static_cast<size_t>(i)].resize(static_cast<size_t>(shape.dim));
        matmul(x[static_cast<size_t>(i)], raw_patches[static_cast<size_t>(i)], w.W_patch_embed, patch_vec_len, static_cast<size_t>(shape.dim));
        rows[static_cast<size_t>(i)] = i / static_cast<int>(grid_w);
        cols[static_cast<size_t>(i)] = i % static_cast<int>(grid_w);
    }
    RoPE2DTables rope(static_cast<int>(std::max(grid_h, grid_w)) + 1, shape.head_dim, 10000.0f);
    for (const auto& l : w.layers) vit_block_forward(x, shape, l, rows, cols, rope);

    out_grid_h = grid_h; out_grid_w = grid_w;
    return merge_all_2x2(x, static_cast<int>(grid_h), static_cast<int>(grid_w), shape.dim, w.merger, llm_dim);
}


// =======================================================================
// PART 7: self-tests, against a small synthetic Qwen2-shaped decoder
// (Section 15.4's own synthetic-model pattern, with `rope.freq_base`
// written explicitly this time -- Section 17.2's own lesson about that
// field's zero default applies here exactly as it did there) and a
// small synthetic image, so every check is deterministic and needs
// neither a real checkpoint nor a real camera.
// =======================================================================
std::vector<float> rand_vec(std::mt19937& rng, size_t n) {
    std::normal_distribution<float> dist(0.0f, 0.3f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 18.3: Wiring the Vision Encoder into the Qwen2 Decoder\n";
    std::cout << "========================================================\n";

    // ---- synthetic Qwen2-shaped decoder, exactly Section 17.2's own shape ----
    constexpr int S_DIM = 32, S_HEADS = 4, S_HEADS_KV = 2, S_HEAD_DIM = 8, S_FF = 64, S_LAYERS = 2, S_VOCAB = 64;
    const std::string synth_path = "/tmp/ch18_3_synthetic_model.gguf";
    auto write_synthetic_model = [&](unsigned seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 0.3f);
        struct Pending { std::string name; std::vector<uint64_t> dims; GGMLType type; std::vector<uint8_t> bytes; };
        std::vector<Pending> pending;
        auto add_f32 = [&](const std::string& name, std::vector<uint64_t> dims, const std::vector<float>& data) {
            std::vector<uint8_t> bytes(data.size() * 4);
            std::memcpy(bytes.data(), data.data(), bytes.size());
            pending.push_back({name, dims, GGML_F32, bytes});
        };
        auto add_q8 = [&](const std::string& name, std::vector<uint64_t> dims, const std::vector<float>& data) {
            size_t n_blocks = data.size() / 32;
            std::vector<uint8_t> bytes(n_blocks * sizeof(BlockQ8));
            for (size_t b = 0; b < n_blocks; ++b) {
                BlockQ8 blk = quantize_q8(data.data() + b * 32);
                std::memcpy(bytes.data() + b * sizeof(BlockQ8), &blk, sizeof(BlockQ8));
            }
            pending.push_back({name, dims, GGML_Q8_0, bytes});
        };
        auto rand_v = [&](size_t n) { std::vector<float> v(n); for (auto& x : v) x = dist(rng); return v; };
        add_q8("token_embd.weight", {S_DIM, S_VOCAB}, rand_v(static_cast<size_t>(S_DIM) * S_VOCAB));
        add_f32("output_norm.weight", {S_DIM}, std::vector<float>(S_DIM, 1.0f));
        for (int layer = 0; layer < S_LAYERS; ++layer) {
            std::string p = "blk." + std::to_string(layer) + ".";
            add_f32(p + "attn_norm.weight", {S_DIM}, std::vector<float>(S_DIM, 1.0f));
            add_q8(p + "attn_q.weight", {S_DIM, S_HEADS * S_HEAD_DIM}, rand_v(static_cast<size_t>(S_DIM) * S_HEADS * S_HEAD_DIM));
            add_f32(p + "attn_q.bias", {static_cast<uint64_t>(S_HEADS * S_HEAD_DIM)}, rand_v(S_HEADS * S_HEAD_DIM));
            add_q8(p + "attn_k.weight", {S_DIM, S_HEADS_KV * S_HEAD_DIM}, rand_v(static_cast<size_t>(S_DIM) * S_HEADS_KV * S_HEAD_DIM));
            add_f32(p + "attn_k.bias", {static_cast<uint64_t>(S_HEADS_KV * S_HEAD_DIM)}, rand_v(S_HEADS_KV * S_HEAD_DIM));
            add_q8(p + "attn_v.weight", {S_DIM, S_HEADS_KV * S_HEAD_DIM}, rand_v(static_cast<size_t>(S_DIM) * S_HEADS_KV * S_HEAD_DIM));
            add_f32(p + "attn_v.bias", {static_cast<uint64_t>(S_HEADS_KV * S_HEAD_DIM)}, rand_v(S_HEADS_KV * S_HEAD_DIM));
            add_q8(p + "attn_output.weight", {S_HEADS * S_HEAD_DIM, S_DIM}, rand_v(static_cast<size_t>(S_HEADS) * S_HEAD_DIM * S_DIM));
            add_f32(p + "ffn_norm.weight", {S_DIM}, std::vector<float>(S_DIM, 1.0f));
            add_q8(p + "ffn_gate.weight", {S_DIM, S_FF}, rand_v(static_cast<size_t>(S_DIM) * S_FF));
            add_q8(p + "ffn_up.weight", {S_DIM, S_FF}, rand_v(static_cast<size_t>(S_DIM) * S_FF));
            add_q8(p + "ffn_down.weight", {S_FF, S_DIM}, rand_v(static_cast<size_t>(S_FF) * S_DIM));
        }
        GGUFWriter w(synth_path);
        w.write_magic(); w.write_u32(3);
        w.write_u64(pending.size()); w.write_u64(7);   // 7 kv pairs written below -- Section 16.3's own lesson
        w.write_kv_string("general.architecture", "qwen2");
        w.write_kv_u32("qwen2.block_count", S_LAYERS);
        w.write_kv_u32("qwen2.embedding_length", S_DIM);
        w.write_kv_u32("qwen2.attention.head_count", S_HEADS);
        w.write_kv_u32("qwen2.attention.head_count_kv", S_HEADS_KV);
        w.write_kv_u32("qwen2.feed_forward_length", S_FF);
        w.write_kv_f32("qwen2.rope.freq_base", 10000.0f);
        uint64_t off = 0;
        for (auto& p : pending) { w.write_tensor_info(p.name, p.dims, p.type, off); off += p.bytes.size(); }
        w.align(32);
        for (auto& p : pending) w.write_bytes(p.bytes.data(), p.bytes.size());
        return w.good();
    };
    CHECK(write_synthetic_model(7));
    QwenModel model;
    CHECK(model.load(synth_path));
    std::vector<QwenBlockWeights> layers;
    for (int l = 0; l < model.n_layers(); ++l) layers.push_back(model.layer(l));

    // ---- synthetic ViT-shaped vision encoder ----
    constexpr int PATCH = 14, CH = 3, VIT_DIM = 16, V_HEADS = 2, V_HEAD_DIM = 8, V_FF = 24, V_LAYERS = 1;
    NormStats norm;
    auto make_vision_weights = [&](unsigned seed) {
        std::mt19937 rng(seed);
        VisionEncoderWeights vw;
        vw.W_patch_embed = rand_vec(rng, static_cast<size_t>(VIT_DIM) * (PATCH * PATCH * CH));
        vw.layers.resize(V_LAYERS);
        for (auto& l : vw.layers) {
            l.attn_norm.assign(VIT_DIM, 1.0f);
            l.Wq = rand_vec(rng, static_cast<size_t>(V_HEADS * V_HEAD_DIM) * VIT_DIM);
            l.Wk = rand_vec(rng, static_cast<size_t>(V_HEADS * V_HEAD_DIM) * VIT_DIM);
            l.Wv = rand_vec(rng, static_cast<size_t>(V_HEADS * V_HEAD_DIM) * VIT_DIM);
            l.Wo = rand_vec(rng, static_cast<size_t>(VIT_DIM) * (V_HEADS * V_HEAD_DIM));
            l.ffn_norm.assign(VIT_DIM, 1.0f);
            l.Wgate = rand_vec(rng, static_cast<size_t>(V_FF) * VIT_DIM);
            l.Wup = rand_vec(rng, static_cast<size_t>(V_FF) * VIT_DIM);
            l.Wdown = rand_vec(rng, static_cast<size_t>(VIT_DIM) * V_FF);
        }
        vw.merger.hidden_dim = 20;
        vw.merger.W1 = rand_vec(rng, static_cast<size_t>(vw.merger.hidden_dim) * (4 * VIT_DIM));
        vw.merger.b1 = rand_vec(rng, static_cast<size_t>(vw.merger.hidden_dim));
        vw.merger.W2 = rand_vec(rng, static_cast<size_t>(S_DIM) * vw.merger.hidden_dim);
        vw.merger.b2 = rand_vec(rng, static_cast<size_t>(S_DIM));
        return vw;
    };
    VisionEncoderWeights vision_weights = make_vision_weights(123);
    ViTShape vit_shape{VIT_DIM, V_HEADS, V_HEAD_DIM, V_FF};

    auto make_image = [&](unsigned seed) {
        RawImage img; img.width = 56; img.height = 56; img.channels = CH;   // 4x4 patch grid -> 4 merged tokens
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> byte_dist(0, 255);
        img.pixels.resize(static_cast<size_t>(img.width) * img.height * CH);
        for (auto& b : img.pixels) b = static_cast<uint8_t>(byte_dist(rng));
        return img;
    };

    std::cout << "\n-- Test 1: fusion refuses a placeholder/visual-token count mismatch --\n";
    {
        std::vector<std::vector<float>> four_tokens(4, std::vector<float>(S_DIM, 0.1f));
        std::vector<int> ok_ids = {5, 6, IMAGE_PLACEHOLDER_ID, IMAGE_PLACEHOLDER_ID, IMAGE_PLACEHOLDER_ID, IMAGE_PLACEHOLDER_ID, 7};
        auto ok_res = build_fused_embeddings(model, ok_ids, four_tokens);
        CHECK(ok_res.ok);
        CHECK(ok_res.embeddings.size() == ok_ids.size());

        std::vector<int> too_few_placeholders = {5, 6, IMAGE_PLACEHOLDER_ID, IMAGE_PLACEHOLDER_ID, IMAGE_PLACEHOLDER_ID, 7};
        auto short_res = build_fused_embeddings(model, too_few_placeholders, four_tokens);
        CHECK(!short_res.ok);
        CHECK(!short_res.error.empty());

        std::vector<int> too_many_placeholders = {5, 6, IMAGE_PLACEHOLDER_ID, IMAGE_PLACEHOLDER_ID, IMAGE_PLACEHOLDER_ID,
                                                   IMAGE_PLACEHOLDER_ID, IMAGE_PLACEHOLDER_ID, 7};
        auto long_res = build_fused_embeddings(model, too_many_placeholders, four_tokens);
        CHECK(!long_res.ok);
        CHECK(!long_res.error.empty());

        std::cout << "  matching count (4 placeholders, 4 visual tokens): ok; fewer placeholders than "
                     "visual tokens: rejected (\"" << short_res.error << "\"); more placeholders than "
                     "visual tokens: rejected (\"" << long_res.error << "\")\n";
    }

    std::cout << "\n-- Test 2: end-to-end multimodal prefill (image + text) is finite and deterministic --\n";
    {
        constexpr int KV_CAP = 32;
        auto image = make_image(50);
        auto run_once = [&]() {
            uint32_t gh = 0, gw = 0;
            auto visual_tokens = run_vision_encoder(image, PATCH, norm, vit_shape, vision_weights, S_DIM, gh, gw);
            std::vector<int> token_ids = {5, 6};
            for (size_t i = 0; i < visual_tokens.size(); ++i) token_ids.push_back(IMAGE_PLACEHOLDER_ID);
            token_ids.insert(token_ids.end(), {7, 8, 9});
            std::vector<KVCache> caches;
            for (int l = 0; l < model.n_layers(); ++l) caches.emplace_back(model.shape.n_heads_kv, KV_CAP, model.shape.head_dim);
            return prefill_multimodal(model, layers, caches, token_ids, visual_tokens, 0, KV_CAP);
        };
        auto r1 = run_once();
        auto r2 = run_once();
        CHECK(!r1.fusion_failed && !r1.exceeded_capacity && r1.nan_at_layer < 0);
        bool finite = true;
        for (float v : r1.hidden) if (!std::isfinite(v)) finite = false;
        CHECK(finite);
        CHECK(r1.hidden == r2.hidden);
        std::cout << "  9-position fused sequence (2 text + 4 image + 3 text) prefilled successfully, "
                     "final hidden state finite: " << (finite ? "yes" : "no") << ", deterministic across "
                     "two independent runs: " << (r1.hidden == r2.hidden ? "yes" : "no") << "\n";
    }

    std::cout << "\n-- Test 3: changing the image changes downstream text computation --\n";
    {
        constexpr int KV_CAP = 32;
        auto run_with_image = [&](const RawImage& image) {
            uint32_t gh = 0, gw = 0;
            auto visual_tokens = run_vision_encoder(image, PATCH, norm, vit_shape, vision_weights, S_DIM, gh, gw);
            std::vector<int> token_ids = {5, 6};
            for (size_t i = 0; i < visual_tokens.size(); ++i) token_ids.push_back(IMAGE_PLACEHOLDER_ID);
            token_ids.insert(token_ids.end(), {7, 8, 9});
            std::vector<KVCache> caches;
            for (int l = 0; l < model.n_layers(); ++l) caches.emplace_back(model.shape.n_heads_kv, KV_CAP, model.shape.head_dim);
            return prefill_multimodal(model, layers, caches, token_ids, visual_tokens, 0, KV_CAP);
        };
        auto image_a = make_image(50);
        auto image_b = make_image(999);   // genuinely different pixel content
        auto res_a = run_with_image(image_a);
        auto res_b = run_with_image(image_b);
        CHECK(res_a.hidden != res_b.hidden);
        std::cout << "  the SAME text tokens with two DIFFERENT images produce different final hidden "
                     "states: " << (res_a.hidden != res_b.hidden ? "yes" : "no")
                   << " -- the image is genuinely being attended to, not silently ignored\n";
    }

    std::cout << "\n-- Test 4: KV-capacity exhaustion is still caught in a fused sequence --\n";
    {
        constexpr int KV_CAP = 5;   // smaller than the 9-position fused sequence below
        auto image = make_image(50);
        uint32_t gh = 0, gw = 0;
        auto visual_tokens = run_vision_encoder(image, PATCH, norm, vit_shape, vision_weights, S_DIM, gh, gw);
        std::vector<int> token_ids = {5, 6};
        for (size_t i = 0; i < visual_tokens.size(); ++i) token_ids.push_back(IMAGE_PLACEHOLDER_ID);
        token_ids.insert(token_ids.end(), {7, 8, 9});
        std::vector<KVCache> caches;
        for (int l = 0; l < model.n_layers(); ++l) caches.emplace_back(model.shape.n_heads_kv, KV_CAP, model.shape.head_dim);
        auto res = prefill_multimodal(model, layers, caches, token_ids, visual_tokens, 0, KV_CAP);
        CHECK(res.exceeded_capacity);
        CHECK(!res.fusion_failed);
        std::cout << "  a " << token_ids.size() << "-position fused sequence against a " << KV_CAP
                   << "-position cache correctly reports exceeded_capacity=" << (res.exceeded_capacity ? "true" : "false") << "\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 03_multimodal_fusion.cpp -o 03_multimodal_fusion
./03_multimodal_fusion
```

**Sample input:** `build_fused_embeddings` checked against a matching placeholder/visual-token count (accepted) and both directions of mismatch (each rejected with a specific, direction-naming error); a 9-position fused sequence (2 text + 4 image + 3 text tokens) prefilled end to end and checked for finiteness and determinism across two runs; the SAME text tokens fused with two different synthetic images, checked to produce two different final hidden states; and a fused sequence checked against an undersized KV cache to confirm capacity exhaustion is still caught after fusion.

```text
========================================================
Chapter 18.3: Wiring the Vision Encoder into the Qwen2 Decoder
========================================================

-- Test 1: fusion refuses a placeholder/visual-token count mismatch --
  matching count (4 placeholders, 4 visual tokens): ok; fewer placeholders than visual tokens: rejected ("fewer image placeholders than visual tokens (3 used, 4 provided)"); more placeholders than visual tokens: rejected ("more image placeholders than visual tokens")

-- Test 2: end-to-end multimodal prefill (image + text) is finite and deterministic --
  9-position fused sequence (2 text + 4 image + 3 text) prefilled successfully, final hidden state finite: yes, deterministic across two independent runs: yes

-- Test 3: changing the image changes downstream text computation --
  the SAME text tokens with two DIFFERENT images produce different final hidden states: yes -- the image is genuinely being attended to, not silently ignored

-- Test 4: KV-capacity exhaustion is still caught in a fused sequence --
  a 9-position fused sequence against a 5-position cache correctly reports exceeded_capacity=true

14/14 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] a placeholder/visual-token count mismatch that fails silently is worse than one that crashes"
    A fusion routine that pads a shortfall of visual tokens with zero vectors, or silently drops the excess when there are too many, will not crash and will not produce an obviously wrong-looking output -- it will produce a plausible-looking hidden state computed from a scrambled or truncated version of whatever image was actually captured, on a real production line, with no signal anywhere that anything went wrong. `build_fused_embeddings`'s own refusal in both directions turns a silent data-corruption bug into a loud, specific, immediately diagnosable error at the exact moment the mismatch occurs -- naming which direction it went and by how many tokens -- rather than a wrong disposition decision three functions downstream that nobody would think to trace back to a mismatched placeholder count.

## 18.4 Confidence-Threshold Disposition and SQLite Defect Logging

### Intuition

Section 18.3 produced one real, finite hidden state per inspected part, but a hidden state is not a decision a factory line can act on. A real inspection station needs exactly two more things: a rule that turns the model's own stated verdict and confidence into one of three dispositions a conveyor's downstream gate actually understands, and a durable, queryable record of every decision this line ever made -- because "why did the line reject part #48213 at 2:14 this morning" is a real question a quality engineer eventually asks, and "the model said so" is not an answer without a logged confidence and defect type to point to.

### The Concept, In Detail

`classify` implements a real industrial design choice, stated as a design choice rather than an arbitrary number: the threshold to auto-REJECT a part the model has FLAGGED as defective is set HIGH (`reject_threshold = 0.85`), because auto-rejecting a genuinely good part on a shaky, low-confidence read wastes real, expensive material. The threshold to auto-ACCEPT a part the model has NOT flagged is set LOWER (`review_threshold = 0.55`), because silently passing a genuinely uncertain "no defect" reading straight through the gate is the more dangerous failure mode of the two, and a factory line would rather route a merely-uncertain part to a human than either extreme. Anything that clears neither bar -- a flagged reading below the reject threshold, or a not-flagged reading below the review threshold -- routes to REVIEW rather than being forced into a guess. Test 2 proves this asymmetry is doing real work, not just applying a single threshold to a single number: the SAME 0.70 confidence yields REVIEW when the reading is flagged as a defect, but ACCEPT when the identical confidence is attached to a not-flagged reading -- two different real decisions from one identical number, because which bar applies depends on what the model actually claimed.

This section's own SQLite integration makes a deliberate, stated infrastructure choice: it declares SQLite's own long-stable C ABI (`sqlite3_open`, `sqlite3_exec`, `sqlite3_prepare_v2`, and the rest) directly via `extern "C"`, rather than including the system `<sqlite3.h>` header. Real edge hardware often ships the SQLite RUNTIME library as a dependency of something else entirely, without the separate development package that provides the header and an unversioned link name -- exactly the situation this book's own real device was in while writing this section. Declaring the handful of functions this section actually calls, matching their real, decades-stable signatures, and linking directly against the versioned runtime library name (`-l:libsqlite3.so.0`) makes this section buildable on exactly the kind of minimal edge image it is written for, without assuming a development package a real deployment target may never have installed.

`InspectionDb::log` and `query_by_disposition` use real prepared statements against a real SQLite file -- Test 3 is not a mock or an in-memory stand-in, it opens an actual file on disk, inserts five real rows, and queries them back, checking every field of the result against exactly what was inserted. `InspectionDb::open` reports a real, honest failure -- not a crash, not a silently-empty database -- when asked to open a file inside a directory that does not exist, exactly the kind of real misconfiguration a factory technician might introduce, and Test 4 confirms `last_error` is populated with SQLite's own real error message rather than left blank.

A note on this section's own cross-architecture verification, stated with the same honesty this book applies to every real limitation of its own build environment: this book's usual four-way check (native x86_64, GCC 14 x86_64, aarch64 via cross-compilation and `qemu-aarch64`, and the real device) could not be completed as a full four-way for this ONE section, because no aarch64 build of `libsqlite3` is available inside this book's own cross-compilation sandbox to link the aarch64 leg against. Rather than silently skip that leg without saying so, or fabricate a result for it, this section's own verification runs the three legs that ARE available -- native x86_64, GCC 14 x86_64, and, critically, the REAL aarch64 device this book has used throughout, which has its own real `libsqlite3` installed as a genuine system library -- and states the gap plainly rather than papering over it. The real device is the one machine in this check that actually matters for a chapter about edge deployment, and it passed.

### Code and Verification

```cpp
// Chapter 18.4 -- Section 18.3 produced one real, finite hidden state per
// inspected part, but a hidden state is not a decision a factory line can
// act on. A real inspection station needs exactly two more things: a
// rule that turns the vision-language model's own stated verdict and
// confidence into one of three dispositions a conveyor's downstream
// gate actually understands (accept, reject, or route to a human), and
// a durable, queryable record of every decision this line ever made --
// because "why did the line reject part #48213 at 2:14 this morning" is
// a question a real quality engineer eventually asks, and "the model
// said so" is not an answer without a logged confidence and a logged
// defect type to point to.
//
// This section's own SQLite usage deliberately declares SQLite's own
// long-stable C ABI directly, rather than including the system
// `<sqlite3.h>` header. Real edge hardware often ships the SQLite
// RUNTIME library (`libsqlite3.so`) preinstalled as a system dependency
// of something else entirely, without the separate development package
// that provides the header and an unversioned link name -- exactly the
// situation this book's own build machine and real device were both in
// while writing this section. Declaring the handful of functions this
// section actually calls, matching their real, decades-stable
// signatures, and linking directly against the versioned runtime
// library name (`-l:libsqlite3.so.0`) makes this section buildable on
// exactly the kind of minimal edge image it is written for.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_disposition_and_logging.cpp -l:libsqlite3.so.0 -o 04_disposition_and_logging
// Run:     ./04_disposition_and_logging

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

// =======================================================================
// PART 1: SQLite's own real C ABI, declared directly (see header note
// above) rather than pulled in from a system header this book cannot
// assume is installed.
// =======================================================================
extern "C" {
struct sqlite3;
struct sqlite3_stmt;
int sqlite3_open(const char* filename, sqlite3** ppDb);
int sqlite3_close(sqlite3*);
int sqlite3_exec(sqlite3*, const char* sql, int (*callback)(void*, int, char**, char**), void* arg, char** errmsg);
int sqlite3_prepare_v2(sqlite3* db, const char* zSql, int nByte, sqlite3_stmt** ppStmt, const char** pzTail);
int sqlite3_bind_int(sqlite3_stmt*, int, int);
int sqlite3_bind_int64(sqlite3_stmt*, int, long long);
int sqlite3_bind_double(sqlite3_stmt*, int, double);
int sqlite3_bind_text(sqlite3_stmt*, int, const char*, int, void (*)(void*));
int sqlite3_step(sqlite3_stmt*);
int sqlite3_finalize(sqlite3_stmt*);
int sqlite3_column_int(sqlite3_stmt*, int);
long long sqlite3_column_int64(sqlite3_stmt*, int);
double sqlite3_column_double(sqlite3_stmt*, int);
const unsigned char* sqlite3_column_text(sqlite3_stmt*, int);
const char* sqlite3_errmsg(sqlite3*);
void sqlite3_free(void*);
}
constexpr int SQLITE_OK = 0;
constexpr int SQLITE_ROW = 100;
constexpr int SQLITE_DONE = 101;
// The real, documented sentinel SQLite defines for "copy this string
// now, I am not keeping the pointer alive" -- (void(*)(void*))(-1),
// exactly the C-style cast the real sqlite3.h itself uses (a constant
// integer cannot be reinterpret_cast to a pointer at compile time, so
// this section's own declaration matches the real header's own idiom).
using sqlite3_destructor_type = void (*)(void*);
const sqlite3_destructor_type SQLITE_TRANSIENT = (sqlite3_destructor_type)(-1);

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 2: confidence-threshold disposition. The asymmetry here is the
// real content of this section: a FLAGGED defect needs HIGH confidence
// before this station auto-rejects a part outright (a shaky read should
// not waste a good part), while a NOT-flagged reading needs only a
// LOWER confidence bar before this station auto-accepts (silently
// passing an uncertain "no defect" through is the more dangerous
// mistake for a quality gate to make). Anything that does not clear its
// own side of that asymmetric bar is routed to a human, never guessed.
// =======================================================================
enum class Disposition { ACCEPT, REJECT, REVIEW };

std::string to_string(Disposition d) {
    switch (d) {
        case Disposition::ACCEPT: return "ACCEPT";
        case Disposition::REJECT: return "REJECT";
        case Disposition::REVIEW: return "REVIEW";
    }
    return "?";
}

struct InspectionReading {
    bool model_flagged_defect = false;
    double confidence = 0.0;          // this reading's own stated confidence, in [0, 1]
    std::string defect_type = "none";
};

struct DispositionThresholds {
    double reject_threshold = 0.85;   // a FLAGGED reading needs at least this confidence to auto-reject
    double review_threshold = 0.55;   // a NOT-flagged reading needs at least this confidence to auto-accept

    // A misconfigured threshold pair (review >= reject) would make one
    // side of the asymmetry meaningless or inverted; this section
    // refuses to classify anything against thresholds that fail this
    // check, rather than silently producing a disposition from a
    // configuration that does not mean what it claims to.
    bool valid() const { return reject_threshold > review_threshold && reject_threshold <= 1.0 && review_threshold >= 0.0; }
};

std::optional<Disposition> classify(const InspectionReading& r, const DispositionThresholds& t) {
    if (!t.valid()) return std::nullopt;
    if (!r.model_flagged_defect) {
        return (r.confidence >= t.review_threshold) ? Disposition::ACCEPT : Disposition::REVIEW;
    }
    return (r.confidence >= t.reject_threshold) ? Disposition::REJECT : Disposition::REVIEW;
}

// =======================================================================
// PART 3: durable defect logging. Every disposition this station ever
// makes is inserted as one row; nothing here holds an in-memory log
// that a power cycle could lose.
// =======================================================================
struct InspectionRecord {
    int64_t block_id = 0;         // Section 18.1's own GVSP block id for the inspected frame
    int64_t timestamp_ns = 0;
    std::string defect_type;
    double confidence = 0.0;
    Disposition disposition = Disposition::ACCEPT;
};

bool exec_simple(sqlite3* db, const std::string& sql, std::string& err) {
    char* errmsg = nullptr;
    int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &errmsg);
    if (rc != SQLITE_OK) {
        err = errmsg ? errmsg : sqlite3_errmsg(db);
        if (errmsg) sqlite3_free(errmsg);
        return false;
    }
    return true;
}

struct InspectionDb {
    sqlite3* db = nullptr;
    std::string last_error;

    bool open(const std::string& path) {
        if (sqlite3_open(path.c_str(), &db) != SQLITE_OK) {
            last_error = db ? sqlite3_errmsg(db) : "sqlite3_open failed";
            return false;
        }
        const std::string create_sql =
            "CREATE TABLE IF NOT EXISTS inspections ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "block_id INTEGER NOT NULL, "
            "timestamp_ns INTEGER NOT NULL, "
            "defect_type TEXT NOT NULL, "
            "confidence REAL NOT NULL, "
            "disposition TEXT NOT NULL);";
        return exec_simple(db, create_sql, last_error);
    }

    bool log(const InspectionRecord& rec) {
        sqlite3_stmt* stmt = nullptr;
        const char* sql = "INSERT INTO inspections (block_id, timestamp_ns, defect_type, confidence, disposition) "
                           "VALUES (?, ?, ?, ?, ?);";
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            last_error = sqlite3_errmsg(db);
            return false;
        }
        sqlite3_bind_int64(stmt, 1, rec.block_id);
        sqlite3_bind_int64(stmt, 2, rec.timestamp_ns);
        sqlite3_bind_text(stmt, 3, rec.defect_type.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_double(stmt, 4, rec.confidence);
        std::string disp_str = to_string(rec.disposition);
        sqlite3_bind_text(stmt, 5, disp_str.c_str(), -1, SQLITE_TRANSIENT);
        int rc = sqlite3_step(stmt);
        bool ok = (rc == SQLITE_DONE);
        if (!ok) last_error = sqlite3_errmsg(db);
        sqlite3_finalize(stmt);
        return ok;
    }

    // Queries every logged record with the given disposition, ordered
    // by insertion order -- exactly the query a quality engineer runs
    // to pull "every part this line rejected on the 2 AM shift."
    std::vector<InspectionRecord> query_by_disposition(const std::string& disposition) {
        std::vector<InspectionRecord> out;
        sqlite3_stmt* stmt = nullptr;
        const char* sql = "SELECT block_id, timestamp_ns, defect_type, confidence, disposition "
                           "FROM inspections WHERE disposition = ? ORDER BY id ASC;";
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) { last_error = sqlite3_errmsg(db); return out; }
        sqlite3_bind_text(stmt, 1, disposition.c_str(), -1, SQLITE_TRANSIENT);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            InspectionRecord rec;
            rec.block_id = sqlite3_column_int64(stmt, 0);
            rec.timestamp_ns = sqlite3_column_int64(stmt, 1);
            rec.defect_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            rec.confidence = sqlite3_column_double(stmt, 3);
            std::string d = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
            rec.disposition = (d == "ACCEPT") ? Disposition::ACCEPT : (d == "REJECT") ? Disposition::REJECT : Disposition::REVIEW;
            out.push_back(rec);
        }
        sqlite3_finalize(stmt);
        return out;
    }

    int64_t count_all() {
        sqlite3_stmt* stmt = nullptr;
        int64_t n = -1;
        if (sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM inspections;", -1, &stmt, nullptr) == SQLITE_OK) {
            if (sqlite3_step(stmt) == SQLITE_ROW) n = sqlite3_column_int64(stmt, 0);
        }
        sqlite3_finalize(stmt);
        return n;
    }

    ~InspectionDb() { if (db) sqlite3_close(db); }
};

// =======================================================================
// PART 4: self-tests. The logging tests use a REAL temporary SQLite
// database file on disk and the REAL SQLite C API declared above --
// nothing about persistence is mocked -- while keeping every locked
// check anchored to the query RESULTS (specific values, specific
// counts), never to the database file's own bytes, which is exactly the
// same "check the meaning, not the machine-specific artifact" discipline
// this book has applied to timing since Chapter 17.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 18.4: Confidence-Threshold Disposition and SQLite Defect Logging\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: disposition thresholds validate their own configuration --\n";
    {
        DispositionThresholds good;
        CHECK(good.valid());
        DispositionThresholds inverted{0.5, 0.85};   // reject threshold BELOW review threshold: nonsensical
        CHECK(!inverted.valid());
        InspectionReading r{true, 0.9, "scratch"};
        CHECK(!classify(r, inverted).has_value());
        std::cout << "  default thresholds (reject=0.85, review=0.55) valid: yes; an inverted pair "
                     "(reject=0.5 < review=0.85) correctly refused, classify() returns no disposition\n";
    }

    std::cout << "\n-- Test 2: the accept/reject bar is asymmetric between flagged and not-flagged readings --\n";
    {
        DispositionThresholds t;   // reject=0.85, review=0.55
        InspectionReading flagged_high{true, 0.90, "dent"};
        CHECK(classify(flagged_high, t) == Disposition::REJECT);
        InspectionReading flagged_mid{true, 0.70, "dent"};   // above review bar, but below reject bar
        CHECK(classify(flagged_mid, t) == Disposition::REVIEW);
        InspectionReading not_flagged_mid{false, 0.70, "none"};   // SAME 0.70 confidence, not flagged
        CHECK(classify(not_flagged_mid, t) == Disposition::ACCEPT);
        InspectionReading not_flagged_low{false, 0.40, "none"};
        CHECK(classify(not_flagged_low, t) == Disposition::REVIEW);
        std::cout << "  the SAME 0.70 confidence yields REVIEW when flagged as a defect but ACCEPT when "
                     "not flagged -- the asymmetric bar is doing real work, not just thresholding a single number\n";
        std::cout << "  boundary values: flagged @0.90 -> " << to_string(*classify(flagged_high, t))
                   << "; flagged @0.70 -> " << to_string(*classify(flagged_mid, t))
                   << "; not-flagged @0.70 -> " << to_string(*classify(not_flagged_mid, t))
                   << "; not-flagged @0.40 -> " << to_string(*classify(not_flagged_low, t)) << "\n";
    }

    std::cout << "\n-- Test 3: real SQLite round-trip (open, create table, insert, read back) --\n";
    {
        std::string path = "/tmp/ch18_4_test_inspections.sqlite3";
        std::remove(path.c_str());
        InspectionDb logdb;
        CHECK(logdb.open(path));
        CHECK(logdb.count_all() == 0);

        std::vector<InspectionRecord> records = {
            {1001, 1000, "scratch", 0.92, Disposition::REJECT},
            {1002, 2000, "none", 0.98, Disposition::ACCEPT},
            {1003, 3000, "dent", 0.70, Disposition::REVIEW},
            {1004, 4000, "none", 0.40, Disposition::REVIEW},
            {1005, 5000, "crack", 0.95, Disposition::REJECT},
        };
        bool all_logged = true;
        for (const auto& r : records) all_logged &= logdb.log(r);
        CHECK(all_logged);
        CHECK(logdb.count_all() == static_cast<int64_t>(records.size()));

        auto rejects = logdb.query_by_disposition("REJECT");
        CHECK(rejects.size() == 2);
        CHECK(rejects[0].block_id == 1001 && rejects[0].defect_type == "scratch");
        CHECK(rejects[1].block_id == 1005 && rejects[1].defect_type == "crack");

        auto reviews = logdb.query_by_disposition("REVIEW");
        CHECK(reviews.size() == 2);
        CHECK(reviews[0].block_id == 1003);
        CHECK(reviews[1].block_id == 1004);
        CHECK(std::abs(reviews[1].confidence - 0.40) < 1e-9);

        std::cout << "  opened a real SQLite file, logged " << records.size() << " real rows, "
                     "queried back " << rejects.size() << " REJECT and " << reviews.size()
                   << " REVIEW rows, all field values matching exactly what was inserted\n";
        std::remove(path.c_str());
    }

    std::cout << "\n-- Test 4: opening a database at an unwritable path fails loudly, not silently --\n";
    {
        InspectionDb logdb;
        // A directory that does not exist: SQLite's own real open()
        // reports this as a real error rather than creating parent
        // directories on a caller's behalf.
        bool opened = logdb.open("/tmp/ch18_4_nonexistent_dir_xyz/inspections.sqlite3");
        CHECK(!opened);
        CHECK(!logdb.last_error.empty());
        std::cout << "  opening a database inside a nonexistent directory correctly fails "
                     "(error: \"" << logdb.last_error << "\")\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_disposition_and_logging.cpp -l:libsqlite3.so.0 -o 04_disposition_and_logging
./04_disposition_and_logging
```

**Sample input:** `DispositionThresholds` checked to validate its own configuration, refusing an inverted (reject-below-review) pair; the SAME 0.70 confidence checked to yield REVIEW when flagged as a defect and ACCEPT when not, proving the asymmetric bar is real; a real SQLite database opened, five real inspection records logged, and REJECT/REVIEW rows queried back with every field checked against the inserted values; and opening a database inside a nonexistent directory checked to fail loudly with a real, non-empty SQLite error message.

```text
========================================================
Chapter 18.4: Confidence-Threshold Disposition and SQLite Defect Logging
========================================================

-- Test 1: disposition thresholds validate their own configuration --
  default thresholds (reject=0.85, review=0.55) valid: yes; an inverted pair (reject=0.5 < review=0.85) correctly refused, classify() returns no disposition

-- Test 2: the accept/reject bar is asymmetric between flagged and not-flagged readings --
  the SAME 0.70 confidence yields REVIEW when flagged as a defect but ACCEPT when not flagged -- the asymmetric bar is doing real work, not just thresholding a single number
  boundary values: flagged @0.90 -> REJECT; flagged @0.70 -> REVIEW; not-flagged @0.70 -> ACCEPT; not-flagged @0.40 -> REVIEW

-- Test 3: real SQLite round-trip (open, create table, insert, read back) --
  opened a real SQLite file, logged 5 real rows, queried back 2 REJECT and 2 REVIEW rows, all field values matching exactly what was inserted

-- Test 4: opening a database at an unwritable path fails loudly, not silently --
  opening a database inside a nonexistent directory correctly fails (error: "unable to open database file")

20/20 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] a single confidence threshold cannot correctly serve two opposite failure modes at once"
    A single, symmetric threshold applied to every reading -- "accept if confidence exceeds 0.70, reject otherwise" -- treats "the model thinks this is defective" and "the model thinks this is fine" as the same kind of claim needing the same bar of certainty, when a real line's two failure modes have very different real costs: auto-rejecting a genuinely good part on a shaky positive read wastes real material, while auto-accepting a genuinely defective part on a shaky negative read ships a real defect. This section's own asymmetric thresholds -- a HIGH bar to auto-reject, a LOWER bar to auto-accept, with the gap between them routed to REVIEW rather than forced into either extreme -- is not an approximation of the "correct" single threshold; it is the honest recognition that these are two different decisions with two different real costs, and pretending one number could correctly serve both is the actual bug a symmetric threshold would introduce.

## 18.5 OPC UA Integration into the MES/SCADA Line

### Intuition

Section 18.4 gave this line a durable, queryable local record of every decision it ever made, but a real factory does not learn what happened by a separate station querying that database directly -- it learns from its SCADA/MES system, which itself learns from the plant floor over OPC UA, the vendor-neutral protocol that lets a PLC, an HMI, a historian, and a supervisory system all read and write the SAME typed, named nodes without every device needing to speak every other device's proprietary format.

### The Concept, In Detail

This section is stated as an INTERFACE-LEVEL abstraction of OPC UA's real data and service model, not a real UA-TCP wire-protocol stack -- the same honest-scope pattern this chapter has applied to GVSP in Section 18.1 and to windowed attention in Section 18.2, applied here because this environment has no real PLC or SCADA endpoint to verify byte-level wire compliance against, and a byte-exact encoding nobody here could check against a real UA stack would be a plausible-looking claim this book has no way to back up. What IS real here, faithfully modeled on OPC UA's own actual design, is `NodeId` addressing -- a namespace index plus a string identifier, exactly the `ns=<index>;s=<identifier>` form a real UA client uses -- a small typed `Variant`, the real Read/Write/Subscribe service pattern, and real, unmodified OPC UA status code CONSTANTS taken directly from the OPC Foundation's own published status code table: `Good = 0x00000000`, `BadNodeIdUnknown = 0x80340000`, `BadTypeMismatch = 0x80740000`.

`OpcUaServer::write` enforces the same two refuse-loudly rules a real UA server must get right, and this section does not skip either for convenience. First, writing to a node that was never added with `add_node` is refused with `BadNodeIdUnknown` -- never silently auto-created, because a real SCADA system that could accidentally create new tags via a typo in a write request would have no reliable notion of its own address space at all. Second, a write whose value's own type does not match the node's declared type is refused with `BadTypeMismatch` -- never silently coerced, because a real line's own downstream logic (a PLC ladder program, an HMI display) is written against a fixed, declared type for each tag, and a coerced value could silently violate an assumption that logic depends on. Test 2 checks both directions directly, and additionally confirms that a rejected write leaves the node's PRIOR value completely intact -- a bad write does not partially corrupt a node on its way to being refused.

Subscriptions and notifications implement a simplified version of a real OPC UA server's deadband-filtered reporting: `write` only queues a notification when a write is accepted, the node has a subscriber, AND the value actually CHANGES -- a rewrite of the exact same value produces no notification at all, exactly matching a real UA server's own behavior of not spamming subscribers with reports of nothing happening. Test 3 checks this precisely: a value change before subscribing produces nothing (there is no one to notify yet), a real change after subscribing queues exactly one notification, an identical rewrite of that same value queues none, and `take_notifications()` drains its queue to empty and stays empty until the next genuine change -- never accumulating phantom duplicates from writes that changed nothing.

Test 4 is this section's own end-to-end glue, simulating a realistic mixed batch of parts moving through the inspection station this chapter has built across all five sections: for each part, the station writes its disposition, confidence, and defect type to their own SCADA-visible nodes, and increments a running reject-count node specifically on a REJECT disposition, never on REVIEW or ACCEPT. The test tracks, by hand, exactly how many notifications each part's writes should produce given each node's own PRIOR value -- not some fixed baseline, since a real line's own notification-worthiness is always judged against whatever a node currently holds -- including one part whose reading is deliberately identical to the part immediately before it, correctly producing zero new notifications, and confirms the final reject-count node lands on exactly the right total. This is the same discipline Section 18.1's `FrameReassembler` applied to a dropped packet and Section 18.3's `build_fused_embeddings` applied to a count mismatch, carried through to the very last section of the chapter: a real system's every stated invariant is worth a specific, hand-traceable test, not a plausible-looking assertion nobody actually worked out by hand.

### Code and Verification

```cpp
// Chapter 18.5 -- Section 18.4 gave every inspected part a durable,
// queryable record in this line's own local database, but a real
// factory line does not learn what happened from a database another
// station has to go query -- it learns from its SCADA/MES system,
// which itself learns from the plant floor over OPC UA: the vendor-
// neutral, IEC-62541-standardized protocol that lets a PLC, an HMI, a
// historian, and a SCADA supervisor all read and write the SAME typed,
// named "nodes" (a tag for the current disposition, a tag for today's
// reject count, and so on) without every device needing to speak every
// other device's proprietary wire format.
//
// This section is explicitly an INTERFACE-LEVEL abstraction of OPC UA's
// real data and service model, not a real UA-TCP wire-protocol stack:
//   - Real, unmodified OPC UA concepts: NodeId addressing (a namespace
//     index plus a string identifier, exactly the "ns=2;s=..." form a
//     real UA client uses), a small typed Variant, the Read / Write /
//     Subscribe services, and real OPC UA status code CONSTANTS (Good,
//     BadNodeIdUnknown, BadTypeMismatch -- taken from the actual OPC
//     Foundation status code table, not invented for this book).
//   - Stated simplification: there is no real UA-TCP byte-level wire
//     encoding here (no OPC UA binary or XML envelope, no real network
//     socket), because this environment has no real PLC or SCADA
//     endpoint to verify wire-level interoperability against, and a
//     byte-exact encoding nobody can check against a real UA stack
//     would be worse than useless -- it would be a plausible-looking
//     lie. What IS real and fully verifiable here is the SEMANTICS a
//     real OPC UA server must get right: refuse to invent nodes on an
//     unknown write, refuse a type-mismatched write, and only notify
//     subscribers when a value actually CHANGES (the real UA notion of
//     a "deadband" collapsed to its simplest form: any change at all).
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 05_opcua_integration.cpp -o 05_opcua_integration
// Run:     ./05_opcua_integration

#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: NodeId -- real OPC UA addressing, collapsed to the one form
// this book's line actually needs: a numeric namespace index plus a
// string identifier, exactly the "ns=<index>;s=<identifier>" a real UA
// client would use to name a tag.
// =======================================================================
struct NodeId {
    uint16_t namespace_index = 0;
    std::string identifier;

    bool operator==(const NodeId& other) const {
        return namespace_index == other.namespace_index && identifier == other.identifier;
    }

    std::string to_string() const {
        return "ns=" + std::to_string(namespace_index) + ";s=" + identifier;
    }
};

struct NodeIdHash {
    size_t operator()(const NodeId& id) const {
        // A simple, deterministic combination -- this is a hash for an
        // in-process unordered_map, not a cryptographic or cross-
        // process-stable hash, so no claim is made beyond "equal
        // NodeIds hash equal, in this one process."
        size_t h1 = std::hash<uint16_t>{}(id.namespace_index);
        size_t h2 = std::hash<std::string>{}(id.identifier);
        return h1 ^ (h2 + 0x9e3779b9u + (h1 << 6) + (h1 >> 2));
    }
};

// =======================================================================
// PART 2: Variant -- the small set of scalar types this line's tags
// actually need. Real OPC UA's Variant supports a much larger type
// matrix (arrays, matrices, ~25 built-in scalar types); this is the
// honest subset this chapter's line uses, not a claim of full coverage.
// =======================================================================
using Variant = std::variant<int64_t, double, std::string, bool>;

// =======================================================================
// PART 3: StatusCode -- real OPC UA status code constants (from the
// OPC Foundation's own published status code table), not invented
// values. Only the three this section's semantics actually produce.
// =======================================================================
enum class StatusCode : uint32_t {
    Good              = 0x00000000,
    BadNodeIdUnknown  = 0x80340000,
    BadTypeMismatch   = 0x80740000,
};

std::string to_string(StatusCode code) {
    switch (code) {
        case StatusCode::Good:             return "Good";
        case StatusCode::BadNodeIdUnknown: return "BadNodeIdUnknown";
        case StatusCode::BadTypeMismatch:  return "BadTypeMismatch";
    }
    return "Unknown";
}

// =======================================================================
// PART 4: OpcUaServer -- an address space (NodeId -> typed value) plus
// the Read / Write / Subscribe services a real UA server exposes, with
// the two refuse-loudly rules a real server must enforce and this one
// does not skip for convenience: a write to a node that was never
// added is refused, never silently auto-created, and a write whose
// value's type does not match the node's own declared type is refused,
// never silently coerced.
// =======================================================================
struct ReadResult {
    StatusCode status;
    Variant value;
};

struct Notification {
    NodeId node_id;
    Variant value;
};

class OpcUaServer {
public:
    // Registers a node with its initial value; the value's own
    // std::variant index becomes that node's permanent declared type.
    void add_node(const NodeId& id, Variant initial_value) {
        NodeEntry entry;
        entry.value = std::move(initial_value);
        entry.declared_type_index = entry.value.index();
        entry.subscribed = false;
        nodes_.insert_or_assign(id, std::move(entry));
    }

    ReadResult read(const NodeId& id) const {
        auto it = nodes_.find(id);
        if (it == nodes_.end()) {
            return ReadResult{StatusCode::BadNodeIdUnknown, Variant{int64_t{0}}};
        }
        return ReadResult{StatusCode::Good, it->second.value};
    }

    // Refuses an unknown node (never creates one on a caller's behalf)
    // and refuses a type-mismatched write (never coerces). Only when
    // the write is accepted AND the node's value actually changes AND
    // the node has a subscriber does a notification get queued -- a
    // rewrite of the SAME value produces no notification, matching a
    // real UA server's deadband-filtered reporting.
    StatusCode write(const NodeId& id, const Variant& new_value) {
        auto it = nodes_.find(id);
        if (it == nodes_.end()) {
            return StatusCode::BadNodeIdUnknown;
        }
        NodeEntry& entry = it->second;
        if (new_value.index() != entry.declared_type_index) {
            return StatusCode::BadTypeMismatch;
        }
        const bool changed = !(entry.value == new_value);
        entry.value = new_value;
        if (changed && entry.subscribed) {
            pending_notifications_.push_back(Notification{id, new_value});
        }
        return StatusCode::Good;
    }

    // Returns false for an unknown node -- a real UA client cannot
    // subscribe to a monitored item on a node the server has never
    // heard of, and this server does not pretend otherwise.
    bool subscribe(const NodeId& id) {
        auto it = nodes_.find(id);
        if (it == nodes_.end()) return false;
        it->second.subscribed = true;
        return true;
    }

    // Drains and returns every notification queued since the last
    // call; a second immediate call returns empty, exactly like a real
    // UA subscription's publish queue after it has been serviced.
    std::vector<Notification> take_notifications() {
        std::vector<Notification> out;
        out.swap(pending_notifications_);
        return out;
    }

    size_t node_count() const { return nodes_.size(); }

private:
    struct NodeEntry {
        Variant value;
        size_t declared_type_index = 0;
        bool subscribed = false;
    };
    std::unordered_map<NodeId, NodeEntry, NodeIdHash> nodes_;
    std::vector<Notification> pending_notifications_;
};

// =======================================================================
// PART 5: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 18.5: OPC UA Integration into the MES/SCADA Line\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: NodeId equality and hashing are well-defined --\n";
    {
        NodeId a{2, "Line1.Station3.Disposition"};
        NodeId b{2, "Line1.Station3.Disposition"};
        NodeId c{2, "Line1.Station3.Confidence"};
        NodeId d{3, "Line1.Station3.Disposition"};   // same identifier, different namespace
        CHECK(a == b);
        CHECK(!(a == c));
        CHECK(!(a == d));

        std::unordered_map<NodeId, int, NodeIdHash> m;
        m[a] = 1;
        m[c] = 2;
        CHECK(m.at(b) == 1);   // b hashes/compares equal to a, so it finds a's entry
        CHECK(m.at(c) == 2);
        CHECK(m.size() == 2);
        std::cout << "  NodeId(" << a.to_string() << ") == NodeId(" << b.to_string()
                   << "); a distinct namespace or identifier is a distinct node; "
                     "an unordered_map keyed on NodeId resolves both correctly\n";
    }

    std::cout << "\n-- Test 2: Read/Write semantics refuse unknown nodes and type mismatches --\n";
    {
        OpcUaServer server;
        NodeId disposition_node{2, "Station1.Disposition"};
        NodeId reject_count_node{2, "Line.RejectCount"};

        auto r0 = server.read(disposition_node);
        CHECK(r0.status == StatusCode::BadNodeIdUnknown);

        auto w0 = server.write(disposition_node, Variant{std::string{"REJECT"}});
        CHECK(w0 == StatusCode::BadNodeIdUnknown);   // never auto-creates on write

        server.add_node(disposition_node, Variant{std::string{"ACCEPT"}});
        server.add_node(reject_count_node, Variant{int64_t{0}});

        auto r1 = server.read(disposition_node);
        CHECK(r1.status == StatusCode::Good);
        CHECK(std::get<std::string>(r1.value) == "ACCEPT");

        auto w1 = server.write(disposition_node, Variant{std::string{"REJECT"}});
        CHECK(w1 == StatusCode::Good);
        auto r2 = server.read(disposition_node);
        CHECK(std::get<std::string>(r2.value) == "REJECT");

        // string node, int64_t value: a genuine type mismatch
        auto w2 = server.write(disposition_node, Variant{int64_t{42}});
        CHECK(w2 == StatusCode::BadTypeMismatch);
        auto r3 = server.read(disposition_node);
        CHECK(std::get<std::string>(r3.value) == "REJECT");   // unchanged -- the bad write did not corrupt it

        auto w3 = server.write(reject_count_node, Variant{int64_t{1}});
        CHECK(w3 == StatusCode::Good);
        std::cout << "  unknown-node read/write both correctly refused with BadNodeIdUnknown; "
                     "a string-typed node correctly refuses an int64_t write with BadTypeMismatch "
                     "and keeps its prior value rather than corrupting it\n";
    }

    std::cout << "\n-- Test 3: subscriptions notify only on an actual value change --\n";
    {
        OpcUaServer server;
        NodeId temp_node{2, "Station1.SensorTemp"};
        server.add_node(temp_node, Variant{double{20.0}});

        // Not yet subscribed: a real value change produces no notification.
        server.write(temp_node, Variant{double{21.5}});
        CHECK(server.take_notifications().empty());

        CHECK(server.subscribe(temp_node));
        CHECK(!server.subscribe(NodeId{2, "Station1.DoesNotExist"}));   // unknown node: refuse

        server.write(temp_node, Variant{double{22.0}});   // changed: 21.5 -> 22.0
        auto n1 = server.take_notifications();
        CHECK(n1.size() == 1);
        CHECK(n1[0].node_id == temp_node);
        CHECK(std::get<double>(n1[0].value) == 22.0);

        // Draining again immediately returns empty.
        CHECK(server.take_notifications().empty());

        // Rewriting the SAME value: no notification (deadband-style filtering).
        auto w_same = server.write(temp_node, Variant{double{22.0}});
        CHECK(w_same == StatusCode::Good);   // the write itself still succeeds
        CHECK(server.take_notifications().empty());

        // A genuine change again: exactly one notification, not accumulated
        // duplicates from the unchanged rewrite above.
        server.write(temp_node, Variant{double{18.25}});
        auto n2 = server.take_notifications();
        CHECK(n2.size() == 1);
        CHECK(std::get<double>(n2[0].value) == 18.25);

        std::cout << "  a value change before subscribing produces no notification; after "
                     "subscribing, a real change queues exactly one notification, an unchanged "
                     "rewrite of the SAME value queues none, and take_notifications() drains to "
                     "empty and stays empty until the next real change\n";
    }

    std::cout << "\n-- Test 4: end-to-end line glue -- N parts through the inspection station --\n";
    {
        OpcUaServer server;
        const NodeId disposition_node{2, "Station1.Disposition"};
        const NodeId confidence_node{2, "Station1.Confidence"};
        const NodeId defect_type_node{2, "Station1.DefectType"};
        const NodeId reject_count_node{2, "Line.RejectCount"};

        server.add_node(disposition_node, Variant{std::string{""}});
        server.add_node(confidence_node, Variant{double{0.0}});
        server.add_node(defect_type_node, Variant{std::string{""}});
        server.add_node(reject_count_node, Variant{int64_t{0}});

        for (const NodeId& n : {disposition_node, confidence_node, defect_type_node, reject_count_node}) {
            CHECK(server.subscribe(n));
        }
        // Drain the initial subscribe-time silence (subscribing does not
        // itself emit a notification -- only a subsequent value CHANGE does).
        server.take_notifications();

        struct PartResult {
            std::string disposition;
            double confidence;
            std::string defect_type;
        };
        // A realistic mixed batch: two REJECTs, one REVIEW, one ACCEPT,
        // plus a final part whose reading is IDENTICAL to the immediately
        // preceding one (to prove that repeat produces disposition /
        // confidence / defect_type writes with NO new notifications,
        // since the line's published state does not actually change --
        // note this has to be identical to the PRECEDING part's reading,
        // not any earlier one, since notification-worthiness is judged
        // against the node's current value, not some fixed baseline).
        std::vector<PartResult> parts = {
            {"REJECT", 0.92, "scratch"},
            {"ACCEPT", 0.99, "none"},
            {"REVIEW", 0.60, "dent"},
            {"REJECT", 0.88, "crack"},
            {"ACCEPT", 0.99, "none"},
            {"ACCEPT", 0.99, "none"},   // identical to the immediately preceding reading
        };

        int64_t reject_count = 0;
        size_t total_notifications = 0;
        for (const auto& p : parts) {
            server.write(disposition_node, Variant{p.disposition});
            server.write(confidence_node, Variant{p.confidence});
            server.write(defect_type_node, Variant{p.defect_type});
            if (p.disposition == "REJECT") {
                auto current = server.read(reject_count_node);
                reject_count = std::get<int64_t>(current.value) + 1;
                server.write(reject_count_node, Variant{reject_count});
            }
            auto batch = server.take_notifications();
            total_notifications += batch.size();
        }

        CHECK(reject_count == 2);
        auto final_reject = server.read(reject_count_node);
        CHECK(final_reject.status == StatusCode::Good);
        CHECK(std::get<int64_t>(final_reject.value) == 2);

        // Expected notification count, part by part (each node's own
        // PRIOR value is whatever the previous part last wrote to it,
        // not some fixed baseline):
        //  part1 REJECT 0.92 scratch: disposition("")->"REJECT", confidence(0.0)->0.92,
        //        defect_type("")->"scratch", reject_count(0)->1            = 4
        //  part2 ACCEPT 0.99 none:    disposition->"ACCEPT", confidence->0.99,
        //        defect_type->"none"  (reject_count untouched)            = 3
        //  part3 REVIEW 0.60 dent:    disposition->"REVIEW", confidence->0.60,
        //        defect_type->"dent"                                      = 3
        //  part4 REJECT 0.88 crack:   disposition->"REJECT", confidence->0.88,
        //        defect_type->"crack", reject_count(1)->2                 = 4
        //  part5 ACCEPT 0.99 none:    disposition("REJECT")->"ACCEPT",
        //        confidence(0.88)->0.99, defect_type("crack")->"none"    = 3
        //  part6 ACCEPT 0.99 none:    IDENTICAL to part5's just-written
        //        state -> disposition unchanged, confidence unchanged,
        //        defect_type unchanged -> zero notifications              = 0
        CHECK(total_notifications == 4 + 3 + 3 + 4 + 3 + 0);

        std::cout << "  simulated " << parts.size() << " parts through the station: "
                     "final reject count = " << reject_count << " (2 REJECTs, correctly incremented "
                     "only on REJECT), " << total_notifications << " total SCADA notifications "
                     "queued across the batch, with the final repeated ACCEPT reading (identical to "
                     "an earlier one) correctly producing zero new notifications since nothing about "
                     "the line's published state actually changed\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 05_opcua_integration.cpp -o 05_opcua_integration
./05_opcua_integration
```

**Sample input:** `NodeId` equality and hashing checked directly, including as a real `unordered_map` key; Read/Write semantics checked against an unknown node (refused with `BadNodeIdUnknown`) and a type-mismatched write (refused with `BadTypeMismatch`, prior value left intact); subscriptions checked to notify only on an actual value change, never on a rewrite of the same value, with `take_notifications()` draining correctly; and an end-to-end simulation of six parts through the inspection station, with every expected notification count traced by hand against each node's own prior value and the final reject-count node checked against the correct total.

```text
========================================================
Chapter 18.5: OPC UA Integration into the MES/SCADA Line
========================================================

-- Test 1: NodeId equality and hashing are well-defined --
  NodeId(ns=2;s=Line1.Station3.Disposition) == NodeId(ns=2;s=Line1.Station3.Disposition); a distinct namespace or identifier is a distinct node; an unordered_map keyed on NodeId resolves both correctly

-- Test 2: Read/Write semantics refuse unknown nodes and type mismatches --
  unknown-node read/write both correctly refused with BadNodeIdUnknown; a string-typed node correctly refuses an int64_t write with BadTypeMismatch and keeps its prior value rather than corrupting it

-- Test 3: subscriptions notify only on an actual value change --
  a value change before subscribing produces no notification; after subscribing, a real change queues exactly one notification, an unchanged rewrite of the SAME value queues none, and take_notifications() drains to empty and stays empty until the next real change

-- Test 4: end-to-end line glue -- N parts through the inspection station --
  simulated 6 parts through the station: final reject count = 2 (2 REJECTs, correctly incremented only on REJECT), 17 total SCADA notifications queued across the batch, with the final repeated ACCEPT reading (identical to an earlier one) correctly producing zero new notifications since nothing about the line's published state actually changed

34/34 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] judging \"did this notify\" against a fixed baseline instead of each node's own current value"
    It is tempting, when hand-verifying an end-to-end test like Section 18.5's Test 4, to compare each part's reading against some fixed reference reading (\"is this the same as part 2's reading\") rather than against whatever that SPECIFIC node last held. A part whose reading happens to match an EARLIER part's reading, but not the one immediately before it, still represents a real change from that node's current value and correctly produces a notification -- exactly the mistake this section's own test construction first made and then corrected, by tracing each expected count against the node's own immediately preceding write rather than an arbitrary earlier one. The general lesson: a stateful system's own "did this change" question can only be answered against that system's OWN current state at the moment of the write, never against a baseline chosen for the test's own narrative convenience.

## Chapter Summary

This chapter took this book's own from-scratch discipline into a genuinely different kind of deployment target: a real manufacturing line, where a camera's frames arrive unreliably over a network, a vision-language model must fuse what it sees with what it already knows, a verdict must become a conveyor gate's real-time decision, and every decision must be durably logged and reported upward to a plant's SCADA system. Section 18.1 built a GVSP-inspired frame reassembler that places payload bytes by their own stated offset rather than arrival order, proven correct against a deliberately shuffled packet stream, alongside a hardware trigger controller that correctly distinguishes a missing frame from a late one. Section 18.2 built a real Qwen2.5-VL-style vision encoder -- patchification, two-dimensional rotary position encoding, bidirectional transformer blocks reusing this book's own verified building blocks, and a 2x2 spatial merger proven to group real spatial neighborhoods rather than a naive row-major bug that would look identical on a narrower grid. Section 18.3 spliced real visual tokens into a real text sequence by extending, not rewriting, Chapter 16's own locked decoder, with a fail-loud fusion routine that refuses any placeholder/visual-token count mismatch outright, and proved the fusion is genuine by showing two different images produce two different downstream computations. Section 18.4 turned a model's verdict and confidence into an asymmetric accept/reject/review disposition reflecting a real industrial tradeoff, and logged every decision into a real SQLite database using nothing but that library's own directly-declared C ABI -- honestly noting the one section in this chapter whose cross-architecture check could not include an aarch64-qemu leg, for the stated reason of a missing system library, and verifying the real device instead. Section 18.5 closed the loop with an interface-level OPC UA abstraction -- real NodeId addressing, real status codes, real refuse-unknown-node and refuse-type-mismatch semantics, and deadband-filtered notifications -- stated honestly as exactly that rather than a byte-exact wire-protocol implementation this environment has no real PLC or SCADA hardware to verify against.

## Self-Check Questions

1. Section 18.1's `FrameReassembler` places each payload packet's bytes at an explicit offset computed from the packet's own `packet_id` field, rather than appending each arriving packet to a growing buffer in arrival order. What specific real-world condition does this design choice guard against, and how does Test 2 prove the guard actually works?
2. Section 18.1's `TriggerController` distinguishes a frame that never arrives at all (a timeout) from a frame that arrives complete but after its own deadline (a late frame). Why does a real manufacturing line need to treat these as two genuinely different outcomes rather than collapsing both into a single "failed" case?
3. Section 18.2's Test 3 checks the 2x2 patch merger on a 4x4 patch grid rather than a 2x2 or 2x4 grid. Explain why a grid only 2 patches wide would fail to distinguish a correct spatial merge from a naive row-major-consecutive bug.
4. Section 18.2's two-dimensional RoPE splits each attention head's dimension into two halves, rotating one by row and the other by column. What THREE specific properties does Test 2 check to confirm both halves are functioning as genuinely independent coordinates, rather than one collapsing into the other?
5. Section 18.3's `decode_step_with_embedding` factors the original `decode_step` into "get a starting embedding" and "run every layer." Explain why this specific factoring, rather than writing a brand-new fused-sequence forward pass from scratch, is the pattern this book has used since Chapter 16.4.
6. Section 18.3's `build_fused_embeddings` refuses outright on any placeholder/visual-token count mismatch, in either direction. Describe the specific, silent failure that would occur downstream if it instead padded a shortfall with zero vectors or silently dropped an excess.
7. Section 18.4 uses an asymmetric confidence threshold -- a HIGH bar to auto-reject a flagged part, a LOWER bar to auto-accept a not-flagged part. Using Test 2's own boundary case (the same 0.70 confidence yielding two different dispositions), explain why a single, symmetric threshold could not correctly serve both of this line's real failure modes.
8. Section 18.4's own cross-architecture verification could not complete the usual aarch64-qemu leg for this one section. What was the specific cause, and what did this section do INSTEAD of silently skipping that leg without comment?
9. Section 18.5's `OpcUaServer::write` refuses both an unknown-node write and a type-mismatched write, rather than auto-creating the node or coercing the value. Give the real, concrete downstream consequence each of these refusals prevents on an actual SCADA-connected line.
10. Section 18.5's Test 4 traces, by hand, exactly how many notifications each simulated part's writes should produce. Explain why that expected count must be computed against each node's own IMMEDIATELY PRECEDING value, rather than against some fixed baseline reading chosen earlier in the test.

## Where We Go Next

This chapter closed a full, real pipeline end to end -- an unreliable camera feed, a from-scratch vision-language fusion, a real industrial decision rule, a durable log, and an honest SCADA integration -- while stating plainly, at every step, exactly which pieces are real and verifiable and which are stated simplifications of a real spec this environment has no hardware to check byte-for-byte. That same shape -- a real, from-scratch inference core, wired into a specific industry's own real constraints and its own real protocols -- is the shape the rest of Part 5 now applies across a genuinely wide range of deployment targets: retail shelves, medical imaging triage, insurance and environmental documents, security and accessibility and art authentication, point-of-sale trust and counterfeit detection, natural-language photo editing, and body-worn cameras. Each one keeps this book's own build-verify-lock discipline intact while asking what a real vision-language engine, built entirely from scratch, actually has to get right once it leaves a controlled benchmark and meets one specific industry's own real, unforgiving requirements.

## Worked Solutions

**1.** It guards against out-of-order UDP packet delivery, a real and common occurrence on any real network, which a naive "append bytes in arrival order" scheme would silently corrupt with no way to detect it happened. Because `FrameReassembler` places every payload chunk at the explicit byte offset (`packet_id * CHUNK_BYTES`) stated in the packet's own header, placement never depends on the order packets actually arrived in. Test 2 proves this by feeding the reassembler the exact same set of packets as Test 1 but in a deliberately SHUFFLED order, and confirming the reassembled pixel buffer is still byte-identical to the reference image -- if placement had depended on arrival order, shuffling the input would have produced a different, wrong result.

**2.** A timeout (no frame arrives at all) and a late-but-complete frame are different failures with different real implications for a line's downstream logic: a timeout means the camera or trigger path itself failed and there is no image data to act on at all, while a late-but-complete frame means a real image WAS captured correctly but arrived too late for the disposition logic to act on it before the part has already moved past the point where a reject gate could still catch it. A control system that only reported "failed" for both would lose the diagnostic information needed to tell "my camera trigger path is broken" from "my camera works but my network or processing pipeline is too slow" -- two problems with completely different fixes.

**3.** On a grid exactly 2 patches wide, a 2x2 spatial block starting at (0,0) and four row-major-consecutive patches starting at index 0 happen to describe the exact same four indices, because the grid's own width equals the block's own width -- there is no way, on that grid, for the two groupings to produce different answers. A grid 4 patches wide is the smallest case where they diverge: the real spatial block at (0,0) is `{0, 1, 4, 5}` (patches from two different rows), while four row-major-consecutive patches starting at 0 would wrongly be `{0, 1, 2, 3}` (one entire row). Only a grid wide enough to make these two sets different can actually distinguish correct code from a bug that would otherwise pass silently.

**4.** Test 2 checks: (a) position (0, 0) produces the identity rotation, since there is no meaningful angle to rotate by at the coordinate origin; (b) identical patch content placed at two DIFFERENT (row, col) positions produces two DIFFERENT output vectors, confirming position information is actually encoded rather than discarded; and (c) changing ONLY the row coordinate while holding the column fixed changes the output, and changing ONLY the column while holding the row fixed ALSO independently changes the output -- confirming the row-half and column-half of the vector are each doing real, independent work, rather than one half dominating or the two collapsing into a single effective coordinate.

**5.** This factoring lets Section 18.3 add multimodal support by extending exactly one seam -- how a position's starting embedding vector is obtained -- while leaving every already-verified line inside the per-layer forward pass (attention, RMSNorm, SwiGLU, the KV cache, NaN detection) completely untouched and still trusted exactly as much as it was after Chapters 15 through 17 verified it. A brand-new fused-sequence forward pass written from scratch would duplicate all of that already-verified logic in a second code path that would need its own, entirely separate verification effort, doubling the surface area for a subtle divergence between the two decoders' behavior with no benefit -- exactly the "extend, don't rewrite" discipline this book adopted starting at Chapter 16.4's own profiling extension.

**6.** Padding a shortfall of visual tokens with zero vectors would silently insert a small number of positions carrying no real visual information at all, indistinguishable in the code from a legitimate quiet or dark image region, meaning part of the actual captured image would be effectively invisible to the model's own attention with no signal anywhere that anything was dropped. Silently discarding an excess of visual tokens would throw away real visual information the vision encoder specifically computed, again with no error, meaning a real defect the encoder actually detected in a patch of the image could simply vanish before the decoder ever sees it. Both failures produce a plausible-looking, finite hidden state and a plausible-looking downstream disposition, making the resulting bad decision essentially untraceable without already suspecting the fusion step specifically -- which is exactly the class of failure `build_fused_embeddings`'s own loud, immediate refusal converts into a diagnosable error at its actual source.

**7.** A single symmetric threshold, say 0.70, would force one of two wrong outcomes on the exact boundary case Test 2 checks: it would either REJECT the not-flagged 0.70 reading (needlessly routing an already-probably-fine part to REVIEW or worse) or ACCEPT the flagged 0.70 reading (auto-passing a part the model itself flagged as possibly defective, on middling confidence). The asymmetric thresholds resolve this correctly because they recognize the two readings are answering different real questions with different real costs of being wrong: "how sure must I be before I punish a part the model likes" versus "how sure must I be before I trust a part the model is suspicious of" are not the same question, and only two different bars can honestly answer both.

**8.** The specific cause was that no aarch64 build of `libsqlite3` was available inside this book's own cross-compilation sandbox to link an aarch64-qemu binary against, so that one leg of the usual four-way check could not be built at all. Rather than silently omitting that leg without comment, this section states the gap directly in its own text, explains the specific missing dependency that caused it, and substitutes verification on the REAL aarch64 device instead -- which has its own real, working system installation of `libsqlite3` -- so the section's own real target platform is still genuinely verified even though the emulated cross-compilation leg specifically could not be.

**9.** Refusing an unknown-node write prevents a SCADA system's own address space from silently growing new, unintended tags because of a typo or a stale identifier in a write request -- a real supervisory system depends on a fixed, known set of tags to build its displays and alarms around, and a server that quietly created new ones on demand would make that set unreliable. Refusing a type-mismatched write prevents a real downstream consumer of that tag -- a PLC's own ladder logic, an HMI's display formatting -- from receiving a value of a type its own code was never written to handle, which could silently corrupt a calculation or a display rather than failing at the point where the actual mistake was made.

**10.** Whether a given write is "notification-worthy" is a property of whether it changes what a node CURRENTLY holds, and a node's current value is always whatever its own most recent write left it at -- not some earlier reading chosen for convenience when the test was written. A part whose reading happens to match an EARLIER part's reading, but differs from the one immediately preceding it, still represents a real, genuine change from that node's actual current state and must correctly produce a notification; judging it against the wrong baseline would make the test's own expected count wrong, not the code being tested.
