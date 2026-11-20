"""Exercise malformed CLI inputs and repeated-token sampling in both engine copies."""
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CLI_TESTS = r"""
#include <cassert>
#include <limits>
int main() {
    auto parse = [](const char* flag, const char* value) {
        const char* args[] = {"engine", "model.gguf", "-p", "hello", flag, value};
        EngineConfig config;
        std::ostringstream error;
        return parse_args(6, args, config, error);
    };
    for (const char* flag : {"--temp", "--top-p", "--repeat-pen"}) {
        for (const char* value : {"nan", "NaN", "inf", "-inf", "1.0junk", "", "1e999"}) {
            assert(!parse(flag, value));
        }
    }
    for (const char* flag : {"-n", "--top-k", "--kv-cap", "--seed"}) {
        for (const char* value : {"12junk", "", "999999999999999999999999", "1.2"}) {
            assert(!parse(flag, value));
        }
    }
    assert(!parse("--seed", "-1"));
    assert(parse("--temp", "0"));
    assert(parse("--temp", "0.8"));
    assert(parse("--top-p", "1"));
    assert(parse("--seed", "4294967295"));
    assert(parse("-n", "4096"));
}
"""
PENALTY_TESTS = r"""
#include <cassert>
int main() {
    std::vector<float> logits = {8.0f, -8.0f, 1.0f};
    apply_repetition_penalty(logits, {0, 0, 1, 1, -1, 99}, 2.0f);
    assert(logits[0] == 4.0f);
    assert(logits[1] == -16.0f);
    assert(logits[2] == 1.0f);
}
"""
with tempfile.TemporaryDirectory() as directory:
    work = Path(directory)
    for source, checks, needs_mdspan in [
        ("01_engine_config_and_cli.cpp", CLI_TESTS, False),
        ("02_sampling_and_streaming_decoder.cpp", PENALTY_TESTS, False),
        ("04_production_engine.cpp", CLI_TESTS, True),
        ("04_production_engine.cpp", PENALTY_TESTS, True),
    ]:
        harness = work / "regression.cpp"
        harness.write_text(
            '#define main book_example_main\n'
            f'#include "{(ROOT / source).as_posix()}"\n'
            '#undef main\n' + checks
        )
        command = [os.environ.get("CXX", "g++"), "-std=c++23", "-O2", "-Wall", "-Wextra"]
        if needs_mdspan:
            command += [
                "-DMDSPAN_IMPL_STANDARD_NAMESPACE=std",
                "-DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental",
                f"-I{ROOT / '_vendor_mdspan/include'}",
            ]
        binary = work / "regression"
        subprocess.run(command + [str(harness), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)
        print(f"PASS {source}: {'CLI' if checks == CLI_TESTS else 'repetition penalty'}")
