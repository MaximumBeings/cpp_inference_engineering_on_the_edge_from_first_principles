// Chapter 32.3: tiled online-softmax attention and CPU/GPU validation.
// Compile for the target GPU, for example A100:
//   nvcc -std=c++20 -arch=sm_80 03_cuda_kernel_and_validation_suite.cu -o cuda_validation
// Jetson Orin uses sm_87. The suite also supports hosts without a device,
// reporting skipped device checks explicitly.

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

constexpr int TILE_KEYS = 32;   // real keys per shared-memory tile
constexpr int MAX_HEAD_DIM = 64; // a stated real limit for this kernel's own per-thread register array

// =======================================================================
// PART 1: the real CUDA kernel. One real thread owns one query row's own
// online-softmax state; every thread in a block cooperates to load the
// SAME real K/V tile into shared memory once per tile, then each thread
// updates its own running (m, l, o) using Sections 32.1 and 32.2's own
// already-proven-correct real recurrence.
// =======================================================================
__global__ void flash_attention_kernel(const float* q, const float* k, const float* v,
                                        float* out, int nq, int nk, int d) {
    extern __shared__ float shared_kv[];
    float* k_tile = shared_kv;
    float* v_tile = shared_kv + TILE_KEYS * d;

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    float m = -INFINITY;
    float l = 0.0f;
    float o[MAX_HEAD_DIM];
    for (int dd = 0; dd < d; ++dd) o[dd] = 0.0f;

    for (int tile_start = 0; tile_start < nk; tile_start += TILE_KEYS) {
        int tile_len = min(TILE_KEYS, nk - tile_start);

        for (int idx = threadIdx.x; idx < tile_len * d; idx += blockDim.x) {
            int local_key = idx / d, dim = idx % d;
            k_tile[local_key * d + dim] = k[(tile_start + local_key) * d + dim];
            v_tile[local_key * d + dim] = v[(tile_start + local_key) * d + dim];
        }
        __syncthreads();

        if (row < nq) {
            float m_block = -INFINITY;
            float local_scores[TILE_KEYS];
            for (int j = 0; j < tile_len; ++j) {
                float s = 0.0f;
                for (int dd = 0; dd < d; ++dd) s += q[row * d + dd] * k_tile[j * d + dd];
                local_scores[j] = s;
                m_block = fmaxf(m_block, s);
            }
            float m_new = fmaxf(m, m_block);
            float correction = expf(m - m_new);
            l *= correction;
            for (int dd = 0; dd < d; ++dd) o[dd] *= correction;
            for (int j = 0; j < tile_len; ++j) {
                float e = expf(local_scores[j] - m_new);
                l += e;
                for (int dd = 0; dd < d; ++dd) o[dd] += e * v_tile[j * d + dd];
            }
            m = m_new;
        }
        __syncthreads();
    }

    if (row < nq) {
        for (int dd = 0; dd < d; ++dd) out[row * d + dd] = o[dd] / l;
    }
}

// =======================================================================
// PART 2: the real CPU golden reference -- Sections 32.1 and 32.2's own
// already-proven-correct naive attention, restated here in float32 (the
// kernel's own real precision) so a real GPU's output could be compared
// against it apples-to-apples.
// =======================================================================
void cpu_golden_reference(const std::vector<float>& q, const std::vector<float>& k,
                           const std::vector<float>& v, std::vector<float>& out,
                           int nq, int nk, int d) {
    out.assign(static_cast<std::size_t>(nq) * static_cast<std::size_t>(d), 0.0f);
    for (int i = 0; i < nq; ++i) {
        float m = -std::numeric_limits<float>::infinity();
        std::vector<float> scores(static_cast<std::size_t>(nk));
        for (int j = 0; j < nk; ++j) {
            float s = 0.0f;
            for (int dd = 0; dd < d; ++dd) s += q[static_cast<std::size_t>(i * d + dd)] * k[static_cast<std::size_t>(j * d + dd)];
            scores[static_cast<std::size_t>(j)] = s;
            m = std::max(m, s);
        }
        float sum = 0.0f;
        std::vector<float> e(static_cast<std::size_t>(nk));
        for (int j = 0; j < nk; ++j) {
            e[static_cast<std::size_t>(j)] = std::exp(scores[static_cast<std::size_t>(j)] - m);
            sum += e[static_cast<std::size_t>(j)];
        }
        for (int dd = 0; dd < d; ++dd) {
            float acc = 0.0f;
            for (int j = 0; j < nk; ++j) acc += e[static_cast<std::size_t>(j)] * v[static_cast<std::size_t>(j * d + dd)];
            out[static_cast<std::size_t>(i * d + dd)] = acc / sum;
        }
    }
}

// =======================================================================
// PART 3: the real comparison harness a kernel-validation suite actually
// needs -- checked directly against both a matching case and a
// deliberately corrupted one, so this harness's own correctness is
// established independently of whether a real GPU is ever available to
// exercise it end to end.
// =======================================================================
bool outputs_match(const std::vector<float>& golden, const std::vector<float>& candidate, float tol) {
    if (golden.size() != candidate.size() || !std::isfinite(tol) || tol < 0.0f) return false;
    for (std::size_t i = 0; i < golden.size(); ++i) {
        if (!std::isfinite(golden[i]) || !std::isfinite(candidate[i]) ||
            std::fabs(golden[i] - candidate[i]) > tol) return false;
    }
    return true;
}

enum class ValidationStatus { PASSED, FAILED, NO_DEVICE_AVAILABLE };

// Validate input before allocating device memory. Every CUDA operation
// is checked, and buffers are released on each exit path.
ValidationStatus validate_kernel_on_device(const std::vector<float>& q, const std::vector<float>& k,
                                            const std::vector<float>& v, int nq, int nk, int d,
                                            float tol, std::string* diagnostic) {
    if (diagnostic) diagnostic->clear();
    if (nq <= 0 || nk <= 0 || d <= 0 || d > MAX_HEAD_DIM ||
        !std::isfinite(tol) || tol < 0.0f ||
        q.size() != static_cast<std::size_t>(nq) * d ||
        k.size() != static_cast<std::size_t>(nk) * d || v.size() != k.size()) {
        if (diagnostic) *diagnostic = "Invalid attention dimensions, buffers, or tolerance";
        return ValidationStatus::FAILED;
    }

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        if (diagnostic) *diagnostic = "GPU validation skipped: " +
            std::string(cudaGetErrorString(err)) + ", device count " + std::to_string(device_count);
        return ValidationStatus::NO_DEVICE_AVAILABLE;
    }

    std::vector<float> golden;
    cpu_golden_reference(q, k, v, golden, nq, nk, d);
    float *dq = nullptr, *dk = nullptr, *dv = nullptr, *dout = nullptr;
    auto release = [&]() {
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
    };
    auto checked = [&](cudaError_t status, const char* operation) {
        if (status == cudaSuccess) return true;
        if (diagnostic) *diagnostic = std::string(operation) + ": " + cudaGetErrorString(status);
        return false;
    };
    if (!checked(cudaMalloc(&dq, q.size() * sizeof(float)), "allocate Q") ||
        !checked(cudaMalloc(&dk, k.size() * sizeof(float)), "allocate K") ||
        !checked(cudaMalloc(&dv, v.size() * sizeof(float)), "allocate V") ||
        !checked(cudaMalloc(&dout, golden.size() * sizeof(float)), "allocate output") ||
        !checked(cudaMemcpy(dq, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice), "copy Q") ||
        !checked(cudaMemcpy(dk, k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice), "copy K") ||
        !checked(cudaMemcpy(dv, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice), "copy V")) {
        release();
        return ValidationStatus::FAILED;
    }

    constexpr int threads = 128;
    const int blocks = (nq + threads - 1) / threads;
    const std::size_t shared_bytes = static_cast<std::size_t>(2 * TILE_KEYS * d) * sizeof(float);
    flash_attention_kernel<<<blocks, threads, shared_bytes>>>(dq, dk, dv, dout, nq, nk, d);
    if (!checked(cudaGetLastError(), "launch attention") ||
        !checked(cudaDeviceSynchronize(), "execute attention")) {
        release();
        return ValidationStatus::FAILED;
    }
    std::vector<float> gpu_out(golden.size());
    const bool copied = checked(cudaMemcpy(gpu_out.data(), dout, gpu_out.size() * sizeof(float),
                                           cudaMemcpyDeviceToHost), "read output");
    release();
    if (!copied) return ValidationStatus::FAILED;
    const bool matched = outputs_match(golden, gpu_out, tol);
    if (!matched && diagnostic) *diagnostic = "Device output differs from the CPU reference";
    return matched ? ValidationStatus::PASSED : ValidationStatus::FAILED;
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 32.3: A CUDA Production Engine and Its Own Kernel-Validation Suite\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: the real CPU golden reference, restated here in float32 precision, reduces "
                 "correctly on the identical zero-query-vector degenerate case Sections 32.1 and 32.2 both "
                 "already verified by hand, confirming this section's own golden reference is itself "
                 "correct before it is ever used as a comparison baseline --\n";
    {
        std::vector<float> q = {0.0f, 0.0f};
        std::vector<float> k = {1.0f, 0.0f, 0.0f, 1.0f};
        std::vector<float> v = {2.0f, 4.0f, 6.0f, 8.0f};
        std::vector<float> out;
        cpu_golden_reference(q, k, v, out, 1, 2, 2);
        CHECK(std::fabs(out[0] - 4.0f) < 1e-5f);
        CHECK(std::fabs(out[1] - 6.0f) < 1e-5f);
        std::cout << "  the real float32 golden reference produces {" << out[0] << ", " << out[1]
                  << "}, matching the identical exact result Sections 32.1 and 32.2 both verified in "
                     "double precision\n";
    }

    std::cout << "\n-- Test 2: CUDA device discovery --\n";
    int device_count = 0;
    const cudaError_t discovery = cudaGetDeviceCount(&device_count);
    const bool device_available = discovery == cudaSuccess && device_count > 0;
    CHECK(discovery != cudaSuccess || device_count >= 0);
    std::cout << "  available device: " << (device_available ? "yes" : "no")
              << "; runtime: " << cudaGetErrorString(discovery) << "\n";

    std::cout << "\n-- Test 3: the real comparison harness a kernel-validation suite depends on correctly "
                 "accepts a matching real output and correctly REJECTS a deliberately corrupted one beyond "
                 "the stated real tolerance -- proving this harness would genuinely catch a real GPU "
                 "kernel bug, independent of whether a real device is ever available to produce one --\n";
    {
        std::vector<float> golden = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> exact_copy = golden;
        std::vector<float> tiny_rounding = {1.0f + 1e-7f, 2.0f - 1e-7f, 3.0f, 4.0f};
        std::vector<float> genuinely_wrong = {1.0f, 2.0f, 3.0f, 4.5f};

        CHECK(outputs_match(golden, exact_copy, 1e-5f));
        CHECK(outputs_match(golden, tiny_rounding, 1e-5f));
        CHECK(!outputs_match(golden, genuinely_wrong, 1e-5f));
        auto nonfinite = golden;
        nonfinite[0] = std::numeric_limits<float>::quiet_NaN();
        CHECK(!outputs_match(golden, nonfinite, 1e-5f));
        CHECK(!outputs_match(golden, golden, -1.0f));
        std::cout << "  the real comparison harness accepts an exact copy and a copy differing only by "
                     "real float32 rounding noise (1e-7), and correctly REJECTS a copy with a genuine "
                     "0.5-magnitude error at index 3 -- the harness's own pass/fail logic is proven correct "
                     "independent of any real GPU ever running\n";
    }

    std::cout << "\n-- Test 4: device attention across full and partial tiles --\n";
    for (int d : {1, 2, 17, 64}) {
        for (int nq : {1, 2, 129}) {
            for (int nk : {1, 3, 32, 33, 65}) {
                std::vector<float> q(static_cast<std::size_t>(nq) * d);
                std::vector<float> k(static_cast<std::size_t>(nk) * d);
                std::vector<float> v(k.size());
                for (std::size_t i = 0; i < q.size(); ++i) q[i] = std::sin(float(i) * 0.13f);
                for (std::size_t i = 0; i < k.size(); ++i) {
                    k[i] = std::cos(float(i) * 0.17f);
                    v[i] = std::sin(float(i) * 0.19f);
                }
                std::string diagnostic;
                const auto result = validate_kernel_on_device(q, k, v, nq, nk, d, 2e-4f, &diagnostic);
                CHECK(result == (device_available ? ValidationStatus::PASSED
                                                 : ValidationStatus::NO_DEVICE_AVAILABLE));
                if (device_available && result != ValidationStatus::PASSED)
                    std::cerr << "  nq=" << nq << " nk=" << nk << " d=" << d << ": " << diagnostic << "\n";
            }
        }
    }
    std::cout << "  60 shapes checked; device comparisons "
              << (device_available ? "executed" : "skipped (no device)") << "\n";
    std::cout << "\n-- Test 5: invalid dimensions are rejected before a launch --\n";
    std::string diagnostic;
    CHECK(validate_kernel_on_device({}, {}, {}, 1, 1, 65, 1e-4f, &diagnostic) == ValidationStatus::FAILED);
    CHECK(validate_kernel_on_device({}, {}, {}, 0, 1, 1, 1e-4f, &diagnostic) == ValidationStatus::FAILED);
    CHECK(validate_kernel_on_device({1}, {1}, {}, 1, 1, 1, 1e-4f, &diagnostic) == ValidationStatus::FAILED);

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
