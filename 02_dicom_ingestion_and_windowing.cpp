// Chapter 20.2 -- A shelf photograph from Chapter 19 arrives as an
// ordinary raw pixel buffer; a medical scan does not. It arrives as a
// real, standardized DICOM file -- Digital Imaging and Communications in
// Medicine -- carrying its own pixel data alongside real metadata (the
// image's own dimensions, how many bits each pixel occupies, and a
// WINDOW CENTER and WINDOW WIDTH the scanner or radiologist intends the
// image to be viewed through) inside one self-describing binary
// container. Before Section 20.1's own triage machinery -- or Section
// 18.2's vision encoder, unmodified since Chapter 18 -- can see a single
// pixel, that container has to be parsed for real, and the raw sample
// values it holds have to be converted into a real, viewable grayscale
// image using the exact windowing math a radiology workstation itself
// applies.
//
// A note on this section's own real, stated scope, in the same honest
// voice this book has used for every external format it has ever
// implemented a real subset of (GGUF in Chapter 5, GVSP in Chapter 18.1):
// this section implements Explicit VR Little Endian, the single most
// common DICOM transfer syntax, and reads only the handful of real,
// standard data elements a windowed-extraction pipeline actually needs
// (Rows, Columns, BitsAllocated, WindowCenter, WindowWidth, PixelData).
// It does not implement Implicit VR, big-endian transfer syntaxes, or
// compressed pixel data (JPEG, JPEG-LS, JPEG2000 transfer syntaxes),
// and it assumes 16-bit signed grayscale samples -- the common case for
// CT and MR pixel data expressed in Hounsfield-unit-like values -- not
// every real DICOM photometric interpretation. A real clinical PACS
// pipeline has to handle far more of the standard than this; this
// section implements, correctly and verifiably, the real, specific
// slice of it that this chapter's own windowing math depends on.
//
// The windowing formula itself is the exact linear VOI LUT function the
// real DICOM standard defines (PS3.3, C.11.2.1.2): a raw sample at or
// below `center - 0.5 - (width-1)/2` clips to the display minimum, a
// sample above `center - 0.5 + (width-1)/2` clips to the display
// maximum, and everything between maps linearly -- continuously, with
// no discontinuity at either boundary, which this section's own tests
// verify with hand-computable numbers rather than trusting the formula
// on faith.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 02_dicom_ingestion_and_windowing.cpp -o 02_dicom_ingestion_and_windowing
// Run:     ./02_dicom_ingestion_and_windowing

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: the real Explicit VR Little Endian element format.
// =======================================================================
struct Tag {
    uint16_t group;
    uint16_t element;
    bool operator==(const Tag& o) const { return group == o.group && element == o.element; }
};
struct TagHash {
    size_t operator()(const Tag& t) const { return (static_cast<size_t>(t.group) << 16) | t.element; }
};

// The real, standard tags this section's own pipeline needs.
constexpr Tag TAG_ROWS{0x0028, 0x0010};
constexpr Tag TAG_COLUMNS{0x0028, 0x0011};
constexpr Tag TAG_BITS_ALLOCATED{0x0028, 0x0100};
constexpr Tag TAG_WINDOW_CENTER{0x0028, 0x1050};
constexpr Tag TAG_WINDOW_WIDTH{0x0028, 0x1051};
constexpr Tag TAG_PIXEL_DATA{0x7FE0, 0x0010};

// The real DICOM rule: a VR from this set is encoded with 2 reserved
// bytes followed by a 4-byte length; every other (short-form) VR is
// encoded with a 2-byte length directly. This is the real baseline set
// -- a handful of newer VRs from later DICOM editions are not included,
// consistent with this section's own stated scope above.
bool is_long_form_vr(const std::string& vr) {
    static const std::set<std::string> long_form = {"OB", "OW", "OF", "SQ", "UT", "UN"};
    return long_form.count(vr) != 0;
}

struct DataElement {
    Tag tag{};
    std::string vr;
    std::vector<uint8_t> value;
};

// =======================================================================
// PART 2: DicomWriter -- builds a real, byte-accurate synthetic DICOM
// file for this section's own self-tests. Nothing about this class is
// part of the real ingestion pipeline; it exists so the reader below can
// be tested against bytes this section's own code fully controls and
// can verify by construction, exactly as Chapter 5's GGUFWriter did for
// Chapter 15's real GGUF reader.
// =======================================================================
class DicomWriter {
public:
    explicit DicomWriter(const std::string& path) : out_(path, std::ios::binary) {}

    void write_preamble_and_magic() {
        std::vector<uint8_t> preamble(128, 0);
        out_.write(reinterpret_cast<const char*>(preamble.data()), static_cast<std::streamsize>(preamble.size()));
        out_.write("DICM", 4);
    }

    void write_corrupt_magic() {
        std::vector<uint8_t> preamble(128, 0);
        out_.write(reinterpret_cast<const char*>(preamble.data()), static_cast<std::streamsize>(preamble.size()));
        out_.write("XXXX", 4);
    }

    // A real string-VR value is padded to even length with a trailing
    // space per the DICOM standard's own real rule; this writer applies
    // that rule itself so the reader's own trimming can be tested
    // against a genuinely padded value, not one this test fixture
    // conveniently avoided needing to pad in the first place.
    void write_element(Tag tag, const std::string& vr, std::vector<uint8_t> value) {
        if (value.size() % 2 != 0) value.push_back(static_cast<uint8_t>(' '));
        write_u16(tag.group);
        write_u16(tag.element);
        out_.write(vr.data(), 2);
        if (is_long_form_vr(vr)) {
            write_u16(0);   // reserved
            write_u32(static_cast<uint32_t>(value.size()));
        } else {
            write_u16(static_cast<uint16_t>(value.size()));
        }
        if (!value.empty()) out_.write(reinterpret_cast<const char*>(value.data()), static_cast<std::streamsize>(value.size()));
    }

    void write_element_us(Tag tag, uint16_t v) {
        std::vector<uint8_t> bytes(2);
        std::memcpy(bytes.data(), &v, 2);
        write_element(tag, "US", bytes);
    }

    void write_element_ds(Tag tag, const std::string& decimal_string) {
        write_element(tag, "DS", std::vector<uint8_t>(decimal_string.begin(), decimal_string.end()));
    }

    void write_pixel_data_16bit(const std::vector<int16_t>& samples) {
        std::vector<uint8_t> bytes(samples.size() * 2);
        std::memcpy(bytes.data(), samples.data(), bytes.size());
        write_element(TAG_PIXEL_DATA, "OW", bytes);
    }

    bool good() const { return out_.good(); }

    // Explicitly flushes and closes the underlying file. A test fixture
    // that only relies on the destructor to close the file risks reading
    // it back before that destructor has actually run -- this section's
    // own tests call `close()` explicitly, right after writing every
    // element, specifically to avoid that real, easy-to-miss ordering bug.
    void close() { out_.close(); }

private:
    void write_u16(uint16_t v) { out_.write(reinterpret_cast<const char*>(&v), 2); }
    void write_u32(uint32_t v) { out_.write(reinterpret_cast<const char*>(&v), 4); }
    std::ofstream out_;
};

// =======================================================================
// PART 3: DicomReader -- the real ingestion code under test.
// =======================================================================
class DicomReader {
public:
    bool open(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) { last_error_ = "could not open file: " + path; return false; }
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        return parse(bytes);
    }

    std::optional<uint16_t> get_us(Tag tag) const {
        auto it = elements_.find(tag);
        if (it == elements_.end() || it->second.value.size() < 2) return std::nullopt;
        uint16_t v = 0;
        std::memcpy(&v, it->second.value.data(), 2);
        return v;
    }

    // DS (Decimal String) values may be multi-valued, backslash-
    // separated -- this section's own stated scope reads only the
    // FIRST value, which is sufficient for a single-frame window
    // center/width and stated honestly as not handling genuinely
    // per-frame multi-valued windowing.
    std::optional<double> get_ds(Tag tag) const {
        auto it = elements_.find(tag);
        if (it == elements_.end()) return std::nullopt;
        std::string raw(it->second.value.begin(), it->second.value.end());
        size_t backslash = raw.find('\\');
        std::string first = (backslash == std::string::npos) ? raw : raw.substr(0, backslash);
        // Trim the real DICOM padding characters (trailing space or NUL).
        while (!first.empty() && (first.back() == ' ' || first.back() == '\0')) first.pop_back();
        while (!first.empty() && (first.front() == ' ')) first.erase(first.begin());
        if (first.empty()) return std::nullopt;
        double result = 0.0;
        auto parsed = std::from_chars(first.data(), first.data() + first.size(), result);
        if (parsed.ec != std::errc() || parsed.ptr != first.data() + first.size()) return std::nullopt;
        return result;
    }

    const std::vector<uint8_t>* get_raw(Tag tag) const {
        auto it = elements_.find(tag);
        return (it == elements_.end()) ? nullptr : &it->second.value;
    }

    const std::string& last_error() const { return last_error_; }

private:
    bool parse(const std::vector<uint8_t>& bytes) {
        if (bytes.size() < 132) { last_error_ = "file too small to contain a preamble and DICM magic"; return false; }
        if (std::string(bytes.begin() + 128, bytes.begin() + 132) != "DICM") {
            last_error_ = "missing 'DICM' magic at offset 128 -- not a valid DICOM file (or an unsupported transfer syntax)";
            return false;
        }
        size_t pos = 132;
        while (pos + 8 <= bytes.size()) {
            Tag tag{};
            std::memcpy(&tag.group, &bytes[pos], 2);
            std::memcpy(&tag.element, &bytes[pos + 2], 2);
            pos += 4;
            std::string vr(bytes.begin() + static_cast<long>(pos), bytes.begin() + static_cast<long>(pos) + 2);
            pos += 2;

            uint32_t length = 0;
            if (is_long_form_vr(vr)) {
                if (pos + 6 > bytes.size()) { last_error_ = "truncated long-form VR header"; return false; }
                pos += 2;   // reserved
                std::memcpy(&length, &bytes[pos], 4);
                pos += 4;
            } else {
                if (pos + 2 > bytes.size()) { last_error_ = "truncated short-form VR header"; return false; }
                uint16_t short_len = 0;
                std::memcpy(&short_len, &bytes[pos], 2);
                length = short_len;
                pos += 2;
            }
            if (pos + length > bytes.size()) { last_error_ = "element value runs past end of file"; return false; }

            DataElement el;
            el.tag = tag;
            el.vr = vr;
            el.value.assign(bytes.begin() + static_cast<long>(pos), bytes.begin() + static_cast<long>(pos) + length);
            elements_[tag] = std::move(el);
            pos += length;
        }
        return true;
    }

    std::unordered_map<Tag, DataElement, TagHash> elements_;
    std::string last_error_;
};

// =======================================================================
// PART 4: real DICOM linear VOI LUT windowing (PS3.3 C.11.2.1.2), and
// a real windowed-image extractor built on top of the reader above.
// =======================================================================
double apply_window(double x, double center, double width, double ymin = 0.0, double ymax = 255.0) {
    double lower = center - 0.5 - (width - 1.0) / 2.0;
    double upper = center - 0.5 + (width - 1.0) / 2.0;
    if (x <= lower) return ymin;
    if (x > upper) return ymax;
    return ((x - (center - 0.5)) / (width - 1.0) + 0.5) * (ymax - ymin) + ymin;
}

struct WindowedImage {
    bool ok = false;
    std::string error;
    uint16_t rows = 0;
    uint16_t columns = 0;
    std::vector<uint8_t> pixels;   // row-major, one byte per pixel
};

WindowedImage extract_windowed_image(const DicomReader& reader) {
    WindowedImage img;
    auto rows = reader.get_us(TAG_ROWS);
    auto cols = reader.get_us(TAG_COLUMNS);
    auto bits = reader.get_us(TAG_BITS_ALLOCATED);
    auto center = reader.get_ds(TAG_WINDOW_CENTER);
    auto width = reader.get_ds(TAG_WINDOW_WIDTH);
    const auto* pixel_data = reader.get_raw(TAG_PIXEL_DATA);

    if (!rows || !cols || !bits || !center || !width || !pixel_data) {
        img.error = "missing one or more required elements (Rows, Columns, BitsAllocated, WindowCenter, WindowWidth, PixelData)";
        return img;
    }
    if (*bits != 16) {
        img.error = "unsupported BitsAllocated (" + std::to_string(*bits) + ") -- this section only implements 16-bit grayscale";
        return img;
    }
    size_t expected_samples = static_cast<size_t>(*rows) * static_cast<size_t>(*cols);
    if (pixel_data->size() != expected_samples * 2) {
        img.error = "PixelData size does not match Rows*Columns*2 bytes -- refusing to extract from a mismatched buffer";
        return img;
    }

    img.rows = *rows;
    img.columns = *cols;
    img.pixels.resize(expected_samples);
    for (size_t i = 0; i < expected_samples; ++i) {
        int16_t raw = 0;
        std::memcpy(&raw, pixel_data->data() + i * 2, 2);
        double windowed = apply_window(static_cast<double>(raw), *center, *width);
        windowed = std::clamp(windowed, 0.0, 255.0);
        img.pixels[i] = static_cast<uint8_t>(std::lround(windowed));
    }
    img.ok = true;
    return img;
}

// =======================================================================
// PART 5: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 20.2: DICOM Ingestion and Windowed Image Extraction\n";
    std::cout << "========================================================\n";

    const std::string synth_path = "/tmp/ch20_2_synthetic.dcm";

    std::cout << "\n-- Test 1: a real synthetic DICOM file round-trips its own metadata exactly --\n";
    {
        DicomWriter w(synth_path);
        w.write_preamble_and_magic();
        w.write_element_us(TAG_ROWS, 4);
        w.write_element_us(TAG_COLUMNS, 4);
        w.write_element_us(TAG_BITS_ALLOCATED, 16);
        w.write_element_ds(TAG_WINDOW_CENTER, "2048");
        w.write_element_ds(TAG_WINDOW_WIDTH, "4096");
        std::vector<int16_t> samples(16, 0);
        w.write_pixel_data_16bit(samples);
        w.close();
        CHECK(w.good());

        DicomReader r;
        CHECK(r.open(synth_path));
        CHECK(r.get_us(TAG_ROWS).value() == 4);
        CHECK(r.get_us(TAG_COLUMNS).value() == 4);
        CHECK(r.get_us(TAG_BITS_ALLOCATED).value() == 16);
        CHECK(r.get_ds(TAG_WINDOW_CENTER).value() == 2048.0);
        CHECK(r.get_ds(TAG_WINDOW_WIDTH).value() == 4096.0);
        std::cout << "  a synthetic 4x4, 16-bit DICOM file round-trips Rows, Columns, BitsAllocated, "
                     "WindowCenter, and WindowWidth exactly as written\n";
    }

    std::cout << "\n-- Test 2: the real linear VOI LUT windowing formula matches hand-computed values exactly --\n";
    {
        // center=2048, width=4096 -> lower bound = 0.0, upper bound = 4095.0 exactly.
        CHECK(apply_window(0.0, 2048, 4096) == 0.0);        // exactly at the lower clip boundary
        CHECK(apply_window(-100.0, 2048, 4096) == 0.0);     // well below -- clipped
        CHECK(std::abs(apply_window(2047.5, 2048, 4096) - 127.5) < 1e-9);   // the formula's own true midpoint
        CHECK(std::abs(apply_window(4095.0, 2048, 4096) - 255.0) < 1e-9);   // exactly at the upper clip boundary
        CHECK(apply_window(5000.0, 2048, 4096) == 255.0);   // well above -- clipped
        std::cout << "  window(center=2048, width=4096): x=0 -> 0 (lower boundary), x=-100 -> 0 "
                     "(clipped), x=2047.5 -> 127.5 (the formula's true midpoint), x=4095 -> 255 "
                     "(upper boundary), x=5000 -> 255 (clipped) -- all matching hand-computed values exactly\n";
    }

    std::cout << "\n-- Test 3: extracting a windowed image from a real file produces exactly the expected bytes --\n";
    {
        DicomWriter w(synth_path);
        w.write_preamble_and_magic();
        w.write_element_us(TAG_ROWS, 1);
        w.write_element_us(TAG_COLUMNS, 5);
        w.write_element_us(TAG_BITS_ALLOCATED, 16);
        w.write_element_ds(TAG_WINDOW_CENTER, "2048");
        w.write_element_ds(TAG_WINDOW_WIDTH, "4096");
        // Five known raw samples spanning clip-low, near-low, near-mid, near-high, clip-high.
        std::vector<int16_t> samples = {-100, 0, 2048, 4095, 5000};
        w.write_pixel_data_16bit(samples);
        w.close();
        CHECK(w.good());

        DicomReader r;
        CHECK(r.open(synth_path));
        auto img = extract_windowed_image(r);
        CHECK(img.ok);
        CHECK(img.rows == 1 && img.columns == 5);
        CHECK(img.pixels.size() == 5);
        CHECK(img.pixels[0] == 0);     // -100: clipped low
        CHECK(img.pixels[1] == 0);     // 0: exactly the lower boundary
        CHECK(img.pixels[4] == 255);   // 5000: clipped high
        CHECK(img.pixels[3] == 255);   // 4095: exactly the upper boundary
        // pixel[2] (raw=2048) rounds from 127.53...  to 128.
        CHECK(img.pixels[2] == 128);
        std::cout << "  a 1x5 synthetic image with raw samples {-100, 0, 2048, 4095, 5000} extracts "
                     "to windowed bytes {" << static_cast<int>(img.pixels[0]) << ", "
                   << static_cast<int>(img.pixels[1]) << ", " << static_cast<int>(img.pixels[2]) << ", "
                   << static_cast<int>(img.pixels[3]) << ", " << static_cast<int>(img.pixels[4])
                   << "} -- clipped, boundary, and near-midpoint values all correct\n";
    }

    std::cout << "\n-- Test 4: a file with the wrong magic bytes is refused, not crashed on --\n";
    {
        DicomWriter w(synth_path);
        w.write_corrupt_magic();
        w.close();
        CHECK(w.good());
        DicomReader r;
        bool opened = r.open(synth_path);
        CHECK(!opened);
        CHECK(r.last_error().find("DICM") != std::string::npos);
        std::cout << "  a file with corrupted magic bytes at offset 128 is correctly refused (\""
                   << r.last_error() << "\") rather than parsed as if it were valid\n";
    }

    std::cout << "\n-- Test 5: a padded, odd-length DS value and a multi-valued DS both parse correctly --\n";
    {
        DicomWriter w(synth_path);
        w.write_preamble_and_magic();
        // "500" is 3 bytes -- an odd length the writer must pad to 4 with
        // a trailing space, exactly the real DICOM rule this test checks
        // the reader correctly reverses.
        w.write_element_ds(TAG_WINDOW_WIDTH, "500");
        // A real multi-valued DS: this section's own stated scope reads
        // only the first value.
        w.write_element_ds(TAG_WINDOW_CENTER, "1024\\2048\\4096");
        w.write_element_us(TAG_ROWS, 1);
        w.write_element_us(TAG_COLUMNS, 1);
        w.write_element_us(TAG_BITS_ALLOCATED, 16);
        w.write_pixel_data_16bit({0});
        w.close();
        CHECK(w.good());

        DicomReader r;
        CHECK(r.open(synth_path));
        CHECK(r.get_ds(TAG_WINDOW_WIDTH).value() == 500.0);
        CHECK(r.get_ds(TAG_WINDOW_CENTER).value() == 1024.0);   // first of the three backslash-separated values
        std::cout << "  a space-padded odd-length DS value (\"500\" padded to 4 bytes) correctly "
                     "parses to 500.0, and a multi-valued DS (\"1024\\2048\\4096\") correctly reads "
                     "only its first value, 1024.0\n";
    }

    std::remove(synth_path.c_str());

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
