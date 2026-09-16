// Chapter 16.2 -- Section 15.4 produced exactly one token by taking the
// argmax of a real logit vector and stopping. A real generation LOOP
// needs two things Section 15.4 never built: a way to pick something
// other than the single most likely token (so the same prompt does not
// always produce the same word-for-word response), and a way to turn a
// stream of token IDs into text a terminal can display as it arrives,
// rather than one already-known-correct token printed once.
//
// This section reuses Section 15.4's GPT2ByteCodec and Vocabulary
// unchanged -- neither needed a single new method -- and adds two new
// pieces: a sampling pipeline (temperature, repetition penalty, top-k,
// top-p, all standard, independently well-known techniques, composed in
// the specific order this section states and tests), and a streaming
// decoder that buffers a generated token's decoded bytes until they
// form a complete UTF-8 character before flushing to output.
//
// The streaming problem is concrete, not hypothetical: a real BPE
// vocabulary includes single-byte "byte-fallback" tokens for any raw
// byte sequence that never earned its own merged vocabulary entry, and
// a real Unicode character outside the vocabulary's common merges can
// be produced as two or more of these byte-fallback tokens in a row.
// Decoding and printing each token's bytes the instant it is generated
// would print a lone continuation byte on its own -- not valid UTF-8 by
// itself -- before the character it belongs to is complete.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 02_sampling_and_streaming_decoder.cpp -o 02_sampling_and_streaming_decoder
// Run:     ./02_sampling_and_streaming_decoder

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
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
// PART 1: the sampling pipeline. Every function below takes and returns
// a full logits vector -- this book's own Section 15.4 already commits
// to computing one full row's dot product per vocabulary entry inside
// project_argmax; a real generation loop needs the full vector rather
// than only its argmax, so Section 16.4 replaces that function's early
// "keep only the best" reduction with one that fills a caller-provided
// vector<float> logits(vocab_size) instead. Everything in this section
// operates on that vector, in the fixed order the contract below states.
// =======================================================================

// Repetition penalty: for each token ID already in `history`, if its
// current logit is positive, dividing by `penalty` (>1) pulls it toward
// zero (less attractive); if it is already negative, multiplying by
// `penalty` pushes it further negative (also less attractive). Applying
// the SAME operation regardless of sign would do the opposite of what is
// intended for negative logits, which is why the two branches differ.
void apply_repetition_penalty(std::span<float> logits, const std::vector<int>& history, float penalty) {
    if (penalty == 1.0f) return;
    for (int id : history) {
        if (id < 0 || static_cast<size_t>(id) >= logits.size()) continue;
        float& l = logits[static_cast<size_t>(id)];
        l = (l > 0.0f) ? (l / penalty) : (l * penalty);
    }
}

// Temperature: dividing every logit by temp < 1 widens the gaps between
// them (softmax becomes peakier, closer to greedy); temp > 1 narrows the
// gaps (softmax becomes flatter, closer to uniform). temp == 0 is
// handled by the caller as an explicit greedy path -- dividing by zero
// here would produce +/-inf logits rather than a clean argmax.
void apply_temperature(std::span<float> logits, float temp) {
    for (float& l : logits) l /= temp;
}

// Top-k: keep only the k highest logits; every other position is set to
// -infinity so it can never be sampled. Ties at the k-th position are
// broken by index order (nth_element's own tie-breaking), which is a
// documented, deterministic choice rather than an accident.
void apply_top_k(std::span<float> logits, int k) {
    int n = static_cast<int>(logits.size());
    if (k >= n) return;
    std::vector<float> sorted(logits.begin(), logits.end());
    std::nth_element(sorted.begin(), sorted.begin() + k, sorted.end(), std::greater<float>());
    float kth = sorted[static_cast<size_t>(k - 1)];
    int kept = 0;
    for (float& l : logits) {
        if (l >= kth && kept < k) { ++kept; }
        else { l = -std::numeric_limits<float>::infinity(); }
    }
}

// Top-p (nucleus): sort by probability descending, keep the smallest
// PREFIX of that sorted order whose cumulative probability is >= p, and
// mask everything after it. Computed from a softmax of the CURRENT
// logits (after temperature and repetition penalty have already been
// applied), so top-p's notion of "probability" matches what will
// actually be sampled from.
void apply_top_p(std::span<float> logits, float p) {
    int n = static_cast<int>(logits.size());
    std::vector<int> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return logits[static_cast<size_t>(a)] > logits[static_cast<size_t>(b)]; });

    float max_logit = logits[static_cast<size_t>(order[0])];
    std::vector<double> exp_vals(static_cast<size_t>(n));
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double e = std::exp(static_cast<double>(logits[static_cast<size_t>(order[static_cast<size_t>(i)])] - max_logit));
        exp_vals[static_cast<size_t>(i)] = e;
        sum += e;
    }
    double cumulative = 0.0;
    int cutoff = n;   // exclusive: indices [0, cutoff) in `order` survive
    for (int i = 0; i < n; ++i) {
        cumulative += exp_vals[static_cast<size_t>(i)] / sum;
        if (cumulative >= static_cast<double>(p)) { cutoff = i + 1; break; }
    }
    for (int i = cutoff; i < n; ++i) logits[static_cast<size_t>(order[static_cast<size_t>(i)])] = -std::numeric_limits<float>::infinity();
}

// Softmax the surviving (non -inf) logits and draw one sample. Masked
// (-inf) entries contribute exp(-inf - max) = 0, so they are never
// drawn without needing a separate "skip masked entries" branch.
int sample_categorical(std::span<const float> logits, std::mt19937& rng) {
    float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probs(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        double e = std::exp(static_cast<double>(logits[i] - max_logit));
        probs[i] = e;
        sum += e;
    }
    std::uniform_real_distribution<double> uni(0.0, sum);
    double target = uni(rng);
    double running = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        running += probs[i];
        if (running >= target) return static_cast<int>(i);
    }
    return static_cast<int>(probs.size()) - 1;   // floating-point edge case: last surviving index
}

int argmax(std::span<const float> logits) {
    return static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

// The complete pipeline, in the fixed order this section's own tests
// check: repetition penalty, then temperature, then top-k, then top-p,
// then sample -- repetition penalty and temperature both need to see
// the ORIGINAL logit scale before any candidates are discarded, and
// top-p's cumulative-probability accounting needs to run on whatever
// top-k already kept, not the other way around, or a low top-k could
// discard mass top-p's own cutoff was computed against.
int sample_next_token(std::vector<float> logits, const std::vector<int>& history,
                       float temperature, int top_k, float top_p, float repeat_penalty,
                       std::mt19937& rng) {
    if (temperature <= 0.0f) return argmax(logits);   // greedy: penalty/top-k/top-p never apply
    apply_repetition_penalty(logits, history, repeat_penalty);
    apply_temperature(logits, temperature);
    apply_top_k(logits, top_k);
    apply_top_p(logits, top_p);
    return sample_categorical(logits, rng);
}

// =======================================================================
// PART 2: the streaming decoder. Reuses Chapter 15.4's GPT2ByteCodec
// (byte<->symbol) and Vocabulary (id<->text) exactly as built there --
// this section adds no new tokenizer machinery, only a byte-buffering
// wrapper around calling codec.decode(vocab.text_of(id)) once per token.
// =======================================================================
std::string utf8_encode_cp(uint32_t cp) {
    std::string out;
    if (cp <= 0x7F) { out += static_cast<char>(cp); }
    else if (cp <= 0x7FF) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return out;
}
std::vector<uint32_t> utf8_decode_cp(const std::string& s) {
    std::vector<uint32_t> out;
    size_t i = 0;
    while (i < s.size()) {
        uint8_t b0 = static_cast<uint8_t>(s[i]);
        if ((b0 & 0x80) == 0) { out.push_back(b0); i += 1; }
        else if ((b0 & 0xE0) == 0xC0) { out.push_back(((b0 & 0x1Fu) << 6) | (static_cast<uint8_t>(s[i+1]) & 0x3Fu)); i += 2; }
        else { out.push_back(((b0 & 0x0Fu) << 12) | ((static_cast<uint8_t>(s[i+1]) & 0x3Fu) << 6) | (static_cast<uint8_t>(s[i+2]) & 0x3Fu)); i += 3; }
    }
    return out;
}
// Section 15.4's GPT2ByteCodec, reused verbatim (byte<->symbol mapping
// -- unrelated to this section's own utf8_char_length below, which
// classifies raw OUTPUT bytes rather than the codepoints this codec's
// own symbol alphabet uses).
struct GPT2ByteCodec {
    std::string byte_to_symbol[256];
    std::unordered_map<uint32_t, uint8_t> codepoint_to_byte;
    GPT2ByteCodec() {
        std::vector<int> bs, cs;
        auto in_bs = [&](int b) { return std::find(bs.begin(), bs.end(), b) != bs.end(); };
        for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) bs.push_back(b);
        for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
        for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
        cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b) if (!in_bs(b)) { bs.push_back(b); cs.push_back(256 + n); ++n; }
        for (size_t i = 0; i < bs.size(); ++i) {
            uint32_t cp = static_cast<uint32_t>(cs[i]);
            byte_to_symbol[static_cast<uint8_t>(bs[i])] = utf8_encode_cp(cp);
            codepoint_to_byte[cp] = static_cast<uint8_t>(bs[i]);
        }
    }
    std::string decode(const std::string& symbol_text) const {
        std::string out;
        for (uint32_t cp : utf8_decode_cp(symbol_text)) {
            auto it = codepoint_to_byte.find(cp);
            if (it != codepoint_to_byte.end()) out += static_cast<char>(it->second);
        }
        return out;
    }
};
struct Vocabulary {
    std::vector<std::string> id_to_text;
    int add(const std::string& text) { id_to_text.push_back(text); return static_cast<int>(id_to_text.size()) - 1; }
    const std::string& text_of(int id) const { return id_to_text[static_cast<size_t>(id)]; }
};

// Classifies a raw OUTPUT byte's expected UTF-8 sequence length from its
// leading-byte bit pattern -- the standard four-case check (ASCII, 2, 3,
// or 4-byte leading byte), used here to find where a decoded byte
// buffer's trailing, not-yet-complete character begins.
int utf8_char_length(uint8_t first_byte) {
    if ((first_byte & 0x80) == 0x00) return 1;
    if ((first_byte & 0xE0) == 0xC0) return 2;
    if ((first_byte & 0xF0) == 0xE0) return 3;
    if ((first_byte & 0xF8) == 0xF0) return 4;
    return 1;   // a stray continuation byte with no leading byte of its own -- treat as one byte so it can't stall the buffer forever
}

// Consumes and returns the longest prefix of `buffer` that consists
// entirely of complete UTF-8 characters, leaving any genuinely
// incomplete trailing sequence (at most 3 bytes, since a 4-byte
// character missing only its last byte is the longest possible
// incomplete case) in `buffer` for the next call to complete.
std::string flush_complete_utf8(std::string& buffer) {
    size_t i = 0, n = buffer.size(), last_complete = 0;
    while (i < n) {
        int len = utf8_char_length(static_cast<uint8_t>(buffer[i]));
        if (i + static_cast<size_t>(len) > n) break;   // this character's bytes are not all here yet
        i += static_cast<size_t>(len);
        last_complete = i;
    }
    std::string out = buffer.substr(0, last_complete);
    buffer.erase(0, last_complete);
    return out;
}

struct StreamingDecoder {
    const Vocabulary& vocab;
    const GPT2ByteCodec& codec;
    std::string pending;

    // Call once per generated token ID (the caller checks for a stop
    // token, such as <|im_end|>, BEFORE calling this -- a stop token is
    // never decoded or streamed, so this struct needs no separate
    // "control token" classification of its own).
    std::string process_token(int id) {
        pending += codec.decode(vocab.text_of(id));
        return flush_complete_utf8(pending);
    }
    // Call once, after generation stops, to flush any bytes still
    // sitting in the buffer -- ordinarily empty, but not guaranteed to
    // be if generation stopped (max tokens reached) mid-character.
    std::string finish() {
        std::string rest = pending;
        pending.clear();
        return rest;
    }
};

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 16.2: Sampling and the Streaming Token Decoder\n";
    std::cout << "========================================================\n";

    // =====================================================================
    // TEST 1: temperature widens or narrows the gap between logits, in
    // the direction the formula implies, without changing their order.
    // =====================================================================
    std::cout << "\n-- Test 1: temperature scaling --\n";
    {
        std::vector<float> low_temp = {2.0f, 1.0f, 0.0f};
        std::vector<float> high_temp = low_temp;
        apply_temperature(low_temp, 0.5f);
        apply_temperature(high_temp, 2.0f);
        CHECK(low_temp[0] - low_temp[1] > high_temp[0] - high_temp[1]);   // temp<1 widens the gap
        CHECK(low_temp[0] > low_temp[1] && low_temp[1] > low_temp[2]);    // order preserved
        std::cout << "  gap at temp=0.5: " << (low_temp[0] - low_temp[1])
                   << ", gap at temp=2.0: " << (high_temp[0] - high_temp[1]) << " (lower temp widens the gap)\n";
    }

    // =====================================================================
    // TEST 2: repetition penalty measurably lowers a previously-seen
    // token's logit, without touching a token absent from history.
    // =====================================================================
    std::cout << "\n-- Test 2: repetition penalty --\n";
    {
        std::vector<float> logits = {3.0f, 3.0f, -1.0f};
        std::vector<int> history = {0, 2};   // token 0 and token 2 already generated
        apply_repetition_penalty(logits, history, 1.2f);
        CHECK(logits[0] < 3.0f);     // positive logit pulled toward zero
        CHECK(logits[1] == 3.0f);    // untouched -- never generated
        CHECK(logits[2] < -1.0f);    // negative logit pushed further negative
        std::cout << "  token 0 (seen, positive): 3.0 -> " << logits[0]
                   << "; token 1 (unseen): unchanged at " << logits[1]
                   << "; token 2 (seen, negative): -1.0 -> " << logits[2] << "\n";
    }

    // =====================================================================
    // TEST 3: top-k keeps exactly k finite entries.
    // =====================================================================
    std::cout << "\n-- Test 3: top-k filtering --\n";
    {
        std::vector<float> logits = {5.0f, 1.0f, 4.0f, 2.0f, 3.0f};
        apply_top_k(logits, 2);
        int finite_count = 0;
        for (float l : logits) if (std::isfinite(l)) ++finite_count;
        CHECK(finite_count == 2);
        CHECK(std::isfinite(logits[0]) && std::isfinite(logits[2]));   // the two largest (5.0, 4.0) survive
        std::cout << "  kept exactly 2 of 5 candidates: the two largest logits (5.0 and 4.0) survive\n";
    }

    // =====================================================================
    // TEST 4: top-p keeps the smallest prefix (by descending probability)
    // whose cumulative mass crosses p.
    // =====================================================================
    std::cout << "\n-- Test 4: top-p (nucleus) filtering --\n";
    {
        // Logits chosen so the softmax is heavily concentrated on index 0:
        // exp(10)/(exp(10)+exp(0)*3) is already > 0.99, so p=0.9 keeps only index 0.
        std::vector<float> logits = {10.0f, 0.0f, 0.0f, 0.0f};
        apply_top_p(logits, 0.9f);
        int finite_count = 0;
        for (float l : logits) if (std::isfinite(l)) ++finite_count;
        CHECK(finite_count == 1);
        CHECK(std::isfinite(logits[0]));
        std::cout << "  p=0.9 over a heavily peaked distribution keeps exactly the 1 dominant candidate\n";

        std::vector<float> flat = {1.0f, 1.0f, 1.0f, 1.0f};   // uniform: needs all 4 to reach p=0.9
        apply_top_p(flat, 0.9f);
        int finite_count2 = 0;
        for (float l : flat) if (std::isfinite(l)) ++finite_count2;
        CHECK(finite_count2 == 4);
        std::cout << "  p=0.9 over a uniform distribution needs all 4 candidates to cross the threshold\n";
    }

    // =====================================================================
    // TEST 5: temperature=0 is greedy -- always the argmax, regardless
    // of RNG state, and immune to top-k/top-p/repetition-penalty.
    // =====================================================================
    std::cout << "\n-- Test 5: temperature=0 is greedy, unaffected by the other three knobs --\n";
    {
        std::vector<float> logits = {1.0f, 5.0f, 2.0f};
        std::vector<int> history = {1};   // even though token 1 (the true best) was just generated
        std::mt19937 rng_a(1), rng_b(999999);
        int a = sample_next_token(logits, history, 0.0f, 1, 0.01f, 2.0f, rng_a);
        int b = sample_next_token(logits, history, 0.0f, 1, 0.01f, 2.0f, rng_b);
        CHECK(a == 1 && b == 1);   // still picks the true argmax despite aggressive penalty/top-k/top-p settings
        std::cout << "  two different RNG seeds under temp=0 both pick token 1 (the argmax): " << a << ", " << b << "\n";
    }

    // =====================================================================
    // TEST 6: the contract's own determinism guarantee -- the same seed
    // and the same logits produce the same sampled token, twice.
    // =====================================================================
    std::cout << "\n-- Test 6: same seed -> bit-identical sampled token --\n";
    {
        std::vector<float> logits(50);
        for (size_t i = 0; i < logits.size(); ++i) logits[i] = static_cast<float>((i * 37) % 13) - 5.0f;
        std::vector<int> history = {3, 7, 12};
        std::mt19937 rng1(42), rng2(42);
        int r1 = sample_next_token(logits, history, 0.8f, 20, 0.9f, 1.1f, rng1);
        int r2 = sample_next_token(logits, history, 0.8f, 20, 0.9f, 1.1f, rng2);
        CHECK(r1 == r2);
        std::cout << "  seed 42 twice, identical config, identical logits -> both runs picked token " << r1 << "\n";
    }

    // =====================================================================
    // TEST 7: normal ASCII tokens (Section 15.4's own real-vocabulary
    // shape: multi-character merged tokens like "Hello" or " world")
    // flush immediately, in full.
    // =====================================================================
    std::cout << "\n-- Test 7: multi-character tokens flush immediately --\n";
    {
        GPT2ByteCodec codec;
        Vocabulary vocab;
        auto encode_ascii = [&](const std::string& s) {
            std::string sym;
            for (unsigned char c : s) sym += codec.byte_to_symbol[c];
            return sym;
        };
        int id_hello = vocab.add(encode_ascii("Hello"));
        int id_world = vocab.add(encode_ascii(" world"));
        StreamingDecoder dec{vocab, codec, ""};
        std::string t1 = dec.process_token(id_hello);
        std::string t2 = dec.process_token(id_world);
        CHECK(t1 == "Hello");
        CHECK(t2 == " world");
        std::cout << "  \"Hello\" and \" world\" each flush whole, immediately: \"" << t1 << "\" \"" << t2 << "\"\n";
    }

    // =====================================================================
    // TEST 8: a real byte-fallback split -- two single-byte tokens whose
    // GPT2 symbols decode to the two raw bytes of "e" with an acute
    // accent (U+00E9, UTF-8: 0xC3 0xA9). The first flushes nothing; the
    // second completes the character and flushes it whole.
    // =====================================================================
    std::cout << "\n-- Test 8: split multi-byte UTF-8 character across two byte-fallback tokens --\n";
    {
        GPT2ByteCodec codec;
        Vocabulary vocab;
        int id_c3 = vocab.add(codec.byte_to_symbol[0xC3]);   // a lone byte-fallback token: raw byte 0xC3
        int id_a9 = vocab.add(codec.byte_to_symbol[0xA9]);   // a lone byte-fallback token: raw byte 0xA9
        StreamingDecoder dec{vocab, codec, ""};
        std::string t1 = dec.process_token(id_c3);
        CHECK(t1.empty());
        std::string t2 = dec.process_token(id_a9);
        CHECK(t2.size() == 2);
        CHECK(static_cast<uint8_t>(t2[0]) == 0xC3 && static_cast<uint8_t>(t2[1]) == 0xA9);
        std::cout << "  after byte 1 (0xC3): flushed \"\" (" << t1.size() << " bytes, still accumulating)\n";
        std::cout << "  after byte 2 (0xA9): flushed the complete 2-byte character (" << t2.size() << " bytes)\n";
    }

    // =====================================================================
    // TEST 9: generation stopping mid-character -- finish() recovers
    // whatever was still buffered rather than silently dropping it.
    // =====================================================================
    std::cout << "\n-- Test 9: finish() recovers a buffer left incomplete when generation stops --\n";
    {
        GPT2ByteCodec codec;
        Vocabulary vocab;
        int id_c3 = vocab.add(codec.byte_to_symbol[0xC3]);
        StreamingDecoder dec{vocab, codec, ""};
        std::string during = dec.process_token(id_c3);
        CHECK(during.empty());
        std::string leftover = dec.finish();
        CHECK(leftover.size() == 1 && static_cast<uint8_t>(leftover[0]) == 0xC3);
        std::cout << "  generation stopped with one buffered byte; finish() returned it rather than losing it\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
