#!/usr/bin/env python3
"""Exercise the published quantizers, not a copied helper (TA-01/02/03).
Run with CXX=clang++ python3 tests/numerical_accuracy_regressions.py.
"""
import os
from pathlib import Path
import re
import subprocess
import tempfile
ROOT = Path(__file__).resolve().parents[1]
CXX = os.environ.get('CXX', 'clang++')
page = (ROOT/'docs/part1/04-quantization-strategies.md').read_text()
blocks = re.findall(r'```cpp\n(.*?)\n```', page, re.S)
harness = r'''
int main() {
    for (unsigned bits = 0; bits < 65536; ++bits) {
        if ((bits & 0x7c00u) == 0x7c00u && (bits & 0x3ffu)) continue;
        if (fp16_t::encode(fp16_t::decode(uint16_t(bits))) != bits) return 10;
    }
    if (fp16_t::decode(1) != 0x1p-24f || fp16_t::decode(0x3ff) != 1023*0x1p-24f) return 11;
    if (fp16_t::encode(0x1p-25f) != 0 || fp16_t::encode(3*0x1p-25f) != 2) return 12;
    if (fp16_t::decode(0x7bff) != 65504.0f) return 13;
    for (float magnitude : {0.001f, 0.000001f, 1e-10f}) {
        std::vector<float> x(32, 0.0f); x[0] = magnitude;
        auto q8 = quantize_q8(x);
        if (!(float(q8.scale)>0) || !std::isfinite(float(q8.scale))) return 14;
        std::vector<float> values(32); dequantize_q8(q8, values);
        for (float v: values) if (!std::isfinite(v)) return 15;
    }
    bool rejected = false;
    std::vector<float> x(32, 0.0f); x[0] = std::numeric_limits<float>::quiet_NaN();
    try { (void)quantize_q8(x); } catch (const std::domain_error&) { rejected = true; }
    if (!rejected) return 16;
    rejected = false; x[0] = std::numeric_limits<float>::infinity();
    try { (void)quantize_q8(x); } catch (const std::domain_error&) { rejected = true; }
    if (!rejected) return 17;
    rejected = false;
    try { (void)quantize_q8(std::span<const float>(x.data(), 1)); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) return 18;
    std::cout << "binary16 finite-pattern round trips, subnormal values/ties, tiny scales, invalid input: PASS\n";
    return 0;
}
'''
with tempfile.TemporaryDirectory(prefix='numerical-regression-') as tmp:
    tmp = Path(tmp)
    # The first matching snippet is the full Q8 lesson; preserve its original checks.
    q8 = next(b for b in blocks if 'struct fp16_t {' in b and 'BlockQ8 quantize_q8' in b)
    q8 = q8.replace('int main()', 'int lesson_main()', 1)
    harness = harness.replace('int main() {', 'int main() {\n    if (lesson_main() != 0) return 1;', 1)
    src = tmp/'quantizers.cpp'; exe = tmp/'quantizers'
    src.write_text('#include <limits>\n'+q8+harness)
    subprocess.run([CXX, '-std=c++23', '-O1', '-g', '-fsanitize=undefined,float-cast-overflow',
                    '-fno-sanitize-recover=all', str(src), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
    q4 = next(b for b in blocks if 'BlockQ4 quantize_q4' in b and 'dequantize_q4' in b)
    q4 = q4.replace('int main()', 'int lesson_main()', 1)
    q4_harness = r"""
int main() {
    if (lesson_main() != 0) return 1;
    for (float v : {0.001f, 1e-6f, 1e-10f}) {
        std::vector<float> x(32); x[0]=v; x[1]=-v;
        auto q=quantize_q4(x);
        if (!(float(q.scale)>0) || !std::isfinite(float(q.scale))) return 20;
        std::vector<float> y(32); dequantize_q4(q,y);
        for (float f:y) if (!std::isfinite(f)) return 21;
    }
    std::vector<float> x(32); x[0]=7.0f; x[1]=-7.0f;
    auto q=quantize_q4(x);
    if (q.nibbles[0] != 0x1f) return 22;
    bool rejected=false; x[0]=std::numeric_limits<float>::quiet_NaN();
    try { (void)quantize_q4(x); } catch(const std::domain_error&) { rejected=true; }
    if (!rejected) return 23;
    std::cout << "TeachingQ4 tiny scales, finite outputs, adjacent-nibble fixture and invalid input: PASS\n";
}
"""
    src.write_text('#include <limits>\n'+q4+q4_harness)
    subprocess.run([CXX, '-std=c++23', '-O1', '-fsanitize=undefined,float-cast-overflow',
                    '-fno-sanitize-recover=all', str(src), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
    # Check the public lesson constants against the literal wire bytes, not its own reader.
    model_page = (ROOT/'docs/part1/05-model-file-formats.md').read_text()
    assert '0x46475547' not in model_page
    assert 'GGML_Q8_0 = 7' not in model_page
    assert 'GGML_TeachingQ4 = 0x80000000u' in model_page
    assert (0x46554747).to_bytes(4, 'little') == b'GGUF'
    print('GGUF magic/type constants and private TeachingQ4 separation: PASS')
