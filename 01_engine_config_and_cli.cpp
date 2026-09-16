// Chapter 16.1 -- Chapter 15 produced one real, correct token from a
// real forward pass. This chapter turns that single computation into a
// complete, self-contained production binary: a command line a reader
// can actually type, a generation loop that produces more than one
// token, a multi-turn conversation, and the error handling and
// diagnostics a program someone else runs needs that a single self-test
// section never did.
//
// This section builds the config struct and argument parser every other
// section in this chapter reads from. A production binary's contract is
// only as good as the promises its own flags make, so this section
// states that contract explicitly and keeps the parser honest about it:
// every flag here maps to something Sections 16.2-16.4 actually
// implement, and nothing this book has not built is offered as an
// option. Concretely, that means no `-t <threads>` flag: Chapter 8
// built a real thread pool, but it was never wired into the real Qwen2
// forward pass Chapter 15 built (that forward pass is Section 15.4's
// single-threaded, double-accumulating code, unchanged here), so
// offering a thread-count flag that silently does nothing would be
// exactly the kind of unearned promise this book's own contract-first
// convention exists to rule out. Likewise no `--kv-bits`: Chapters 13
// and 14 built a TurboQuant-compressed KV cache, but against a
// synthetic cache model, never against Section 15.4's real per-layer
// `KVCache`; this chapter's cache stays plain `float`, sized by a
// `--kv-cap` capacity instead, and Chapter 17's own job (per this
// book's Table of Contents) is measuring what integrating that
// compression into a real model would actually cost and save.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_engine_config_and_cli.cpp -o 01_engine_config_and_cli
// Run:     ./01_engine_config_and_cli

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// ---------------------------------------------------------------------
// The single source of truth for every parameter this chapter's
// subsystems need. Every later section receives this by const
// reference rather than reading individual globals or re-parsing
// anything -- the standard fix for the class of bug where the same
// parameter ends up defined in two places with two different values.
// ---------------------------------------------------------------------
struct EngineConfig {
    // Model
    std::string model_path;

    // Conversation content
    std::string prompt;
    std::string system_message =
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
    // Section 15.4's own real-file finding: this is the real GGUF's own
    // embedded chat_template's default when no system message is given.
    // Making it the CLI's own default (rather than leaving it implicit)
    // means a reader who overrides --system sees exactly what changed.

    // Generation
    int max_new_tokens = 64;
    float temperature = 0.8f;
    int top_k = 40;
    float top_p = 0.95f;
    float repeat_penalty = 1.1f;
    unsigned seed = 42;

    // KV cache
    int kv_capacity = 512;   // preallocated positions per layer; Section 16.3

    // Output
    bool verbose = false;
};

// ---------------------------------------------------------------------
// Returns true on success. On failure, prints a specific message to
// stderr and returns false -- the caller (Section 16.4's main) is
// responsible for turning that into exit code 1, per this chapter's own
// error-handling convention (Section 16.3).
// ---------------------------------------------------------------------
bool parse_args(int argc, const char* const* argv, EngineConfig& cfg, std::ostream& err) {
    if (argc < 2) {
        err << "Usage: " << argv[0] << " <model.gguf> -p \"prompt\" [options]\n\n"
            << "Options:\n"
            << "  -p \"text\"          Prompt text (required)\n"
            << "  --system \"text\"    System message (default: the real GGUF's own\n"
            << "                     chat-template default -- see Section 15.4)\n"
            << "  -n N               Max new tokens to generate (default: 64)\n"
            << "  --temp F           Temperature (default: 0.8)\n"
            << "  --top-k N          Top-k sampling (default: 40)\n"
            << "  --top-p F          Top-p nucleus sampling (default: 0.95)\n"
            << "  --repeat-pen F     Repetition penalty (default: 1.1)\n"
            << "  --seed N           RNG seed (default: 42)\n"
            << "  --kv-cap N         KV cache positions to preallocate (default: 512)\n"
            << "  --verbose          Print per-token stats\n";
        return false;
    }
    cfg.model_path = argv[1];

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) return nullptr;
            return argv[++i];
        };
        if (arg == "-p") {
            const char* v = next();
            if (!v) { err << "Error: -p requires a value\n"; return false; }
            cfg.prompt = v;
        } else if (arg == "--system") {
            const char* v = next();
            if (!v) { err << "Error: --system requires a value\n"; return false; }
            cfg.system_message = v;
        } else if (arg == "-n") {
            const char* v = next();
            if (!v) { err << "Error: -n requires a value\n"; return false; }
            cfg.max_new_tokens = std::atoi(v);
        } else if (arg == "--temp") {
            const char* v = next();
            if (!v) { err << "Error: --temp requires a value\n"; return false; }
            cfg.temperature = std::strtof(v, nullptr);
        } else if (arg == "--top-k") {
            const char* v = next();
            if (!v) { err << "Error: --top-k requires a value\n"; return false; }
            cfg.top_k = std::atoi(v);
        } else if (arg == "--top-p") {
            const char* v = next();
            if (!v) { err << "Error: --top-p requires a value\n"; return false; }
            cfg.top_p = std::strtof(v, nullptr);
        } else if (arg == "--repeat-pen") {
            const char* v = next();
            if (!v) { err << "Error: --repeat-pen requires a value\n"; return false; }
            cfg.repeat_penalty = std::strtof(v, nullptr);
        } else if (arg == "--seed") {
            const char* v = next();
            if (!v) { err << "Error: --seed requires a value\n"; return false; }
            cfg.seed = static_cast<unsigned>(std::atoi(v));
        } else if (arg == "--kv-cap") {
            const char* v = next();
            if (!v) { err << "Error: --kv-cap requires a value\n"; return false; }
            cfg.kv_capacity = std::atoi(v);
        } else if (arg == "--verbose") {
            cfg.verbose = true;
        } else {
            err << "Error: unknown argument: " << arg << "\n";
            return false;
        }
    }

    // -- Validation. Each check reports the SPECIFIC bound violated,
    // matching Section 16.3's error-handling convention of a diagnostic
    // message a user can act on rather than a bare "invalid arguments". --
    if (cfg.prompt.empty()) { err << "Error: prompt is required (-p \"text\")\n"; return false; }
    if (cfg.max_new_tokens < 1 || cfg.max_new_tokens > 4096) {
        err << "Error: -n must be between 1 and 4096\n"; return false;
    }
    if (cfg.temperature < 0.0f || cfg.temperature > 2.0f) {
        err << "Error: --temp must be between 0.0 and 2.0\n"; return false;
    }
    if (cfg.top_k < 1) { err << "Error: --top-k must be at least 1\n"; return false; }
    if (cfg.top_p <= 0.0f || cfg.top_p > 1.0f) {
        err << "Error: --top-p must be in (0.0, 1.0]\n"; return false;
    }
    if (cfg.repeat_penalty < 1.0f || cfg.repeat_penalty > 2.0f) {
        err << "Error: --repeat-pen must be between 1.0 and 2.0\n"; return false;
    }
    if (cfg.kv_capacity < 1) { err << "Error: --kv-cap must be at least 1\n"; return false; }

    return true;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 16.1: Engine Configuration and the Command Line\n";
    std::cout << "========================================================\n";

    // =====================================================================
    // TEST 1: every flag parses to the field it names.
    // =====================================================================
    std::cout << "\n-- Test 1: every flag parses correctly --\n";
    {
        const char* argv[] = {
            "./engine", "model.gguf", "-p", "What is the capital of France?",
            "--system", "Be terse.", "-n", "40", "--temp", "0.7",
            "--top-k", "50", "--top-p", "0.9", "--repeat-pen", "1.2",
            "--seed", "123", "--kv-cap", "256", "--verbose"
        };
        int argc = static_cast<int>(sizeof(argv) / sizeof(argv[0]));
        EngineConfig cfg;
        bool ok = parse_args(argc, argv, cfg, std::cerr);
        CHECK(ok);
        CHECK(cfg.model_path == "model.gguf");
        CHECK(cfg.prompt == "What is the capital of France?");
        CHECK(cfg.system_message == "Be terse.");
        CHECK(cfg.max_new_tokens == 40);
        CHECK(cfg.temperature == 0.7f);
        CHECK(cfg.top_k == 50);
        CHECK(cfg.top_p == 0.9f);
        CHECK(cfg.repeat_penalty == 1.2f);
        CHECK(cfg.seed == 123u);
        CHECK(cfg.kv_capacity == 256);
        CHECK(cfg.verbose == true);
        std::cout << "  all 11 fields parsed to the value their flag specified\n";
    }

    // =====================================================================
    // TEST 2: unspecified flags fall back to the documented defaults,
    // including the real GGUF's own chat-template default system message.
    // =====================================================================
    std::cout << "\n-- Test 2: defaults --\n";
    {
        const char* argv[] = {"./engine", "model.gguf", "-p", "Hi"};
        EngineConfig cfg;
        bool ok = parse_args(4, argv, cfg, std::cerr);
        CHECK(ok);
        CHECK(cfg.max_new_tokens == 64);
        CHECK(cfg.temperature == 0.8f);
        CHECK(cfg.top_k == 40);
        CHECK(cfg.top_p == 0.95f);
        CHECK(cfg.repeat_penalty == 1.1f);
        CHECK(cfg.seed == 42u);
        CHECK(cfg.kv_capacity == 512);
        CHECK(cfg.verbose == false);
        CHECK(cfg.system_message == "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.");
        std::cout << "  defaults match Section 15.4's own real-file chat-template default\n";
    }

    // =====================================================================
    // TEST 3: each validation bound rejects the value just past it, with
    // a message naming the specific flag -- not a generic "bad arguments".
    // =====================================================================
    std::cout << "\n-- Test 3: validation rejects out-of-range values individually --\n";
    {
        struct Case { std::vector<const char*> argv; const char* which; };
        std::vector<Case> cases = {
            {{"./engine", "m.gguf"}, "missing prompt"},
            {{"./engine", "m.gguf", "-p", "Hi", "-n", "0"}, "-n too low"},
            {{"./engine", "m.gguf", "-p", "Hi", "-n", "999999"}, "-n too high"},
            {{"./engine", "m.gguf", "-p", "Hi", "--temp", "5.0"}, "--temp too high"},
            {{"./engine", "m.gguf", "-p", "Hi", "--top-p", "1.5"}, "--top-p too high"},
            {{"./engine", "m.gguf", "-p", "Hi", "--repeat-pen", "0.5"}, "--repeat-pen too low"},
            {{"./engine", "m.gguf", "-p", "Hi", "--kv-cap", "0"}, "--kv-cap too low"},
            {{"./engine", "m.gguf", "-p", "Hi", "--frobnicate"}, "unknown flag"},
        };
        int rejected = 0;
        for (auto& c : cases) {
            EngineConfig cfg;
            std::ostringstream discard;
            bool ok = parse_args(static_cast<int>(c.argv.size()), c.argv.data(), cfg, discard);
            CHECK(!ok);
            if (!ok) ++rejected;
        }
        std::cout << "  " << rejected << "/" << cases.size() << " invalid configurations correctly rejected\n";
    }

    // =====================================================================
    // TEST 4: hardware_concurrency is deliberately NOT read anywhere in
    // this config -- confirmed by grep-level inspection being replaced
    // with a positive statement: no field named "thread" exists at all,
    // so there is no flag whose value this binary could silently ignore.
    // =====================================================================
    std::cout << "\n-- Test 4: no unearned thread-count flag --\n";
    {
        const char* argv[] = {"./engine", "m.gguf", "-p", "Hi", "-t", "4"};
        EngineConfig cfg;
        std::ostringstream discard;
        bool ok = parse_args(6, argv, cfg, discard);
        CHECK(!ok);   // "-t" is not a recognized flag -- rejected, not silently accepted and ignored
        std::cout << "  \"-t 4\" (this book's other worked examples use -t for threads) is rejected outright:\n";
        std::cout << "  this chapter's real generation loop is Section 15.4's single-threaded code,\n";
        std::cout << "  so no flag exists that could look like it enables threading and silently not\n";
        std::cout << "  perform it\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
