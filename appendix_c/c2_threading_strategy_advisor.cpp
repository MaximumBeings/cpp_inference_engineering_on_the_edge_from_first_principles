// Appendix C.2 -- A Threading and Parallelization Strategy Advisor.
//
// The real, hard-won lesson across Chapters 8-11 is not "should I use
// threads" -- it is "which partition axis keeps the result correct and
// reproducible, and how many threads are actually worth adding." This file
// restates those chapters' own real formulas and partitioning rules as one
// callable advisor, changing none of them:
//
//   - Chapter 30.1's own crossover-batch-size formula, reused verbatim,
//     decides whether a linear layer at a given batch size is memory-bound
//     or compute-bound -- which in turn decides whether shrinking bytes
//     (quantization) or shrinking FLOPs (a faster kernel) is the lever
//     that actually helps.
//   - Chapter 10.1's own Amdahl's-law speedup and ceiling formulas, reused
//     verbatim, decide how many threads are still worth adding before a
//     workload's own serial fraction dominates the result.
//   - Chapter 10.3/11.1's own row-parallel-over-output-rows rule, Chapter
//     11.3's own group-aligned head-parallel rule, and Chapter 11.4/11.5's
//     own fixed-block-reduction rule decide WHICH axis to partition an
//     operation across, each one keeping this book's own bit-for-bit or
//     thread-count-independent reproducibility guarantee that a naive
//     partition of the same operation would silently break.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 c2_threading_strategy_advisor.cpp -o c2_threading_strategy_advisor
// Run:     ./c2_threading_strategy_advisor

#include <cmath>
#include <iostream>
#include <string>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// -- Part 1: Chapter 30.1's own crossover-batch-size formula, reused
// verbatim (AI(M) = M/2 for a weight-dominated GEMM, solved against the
// machine's own ridge point). --
enum class ComputeRegime { MEMORY_BOUND, COMPUTE_BOUND };

double crossover_batch_size(double ridge_point) {
    return 2.0 * ridge_point;  // Chapter 30.1: solving AI(M) = M/2 = ridge_point for M
}

ComputeRegime classify_batch(double batch_size, double ridge_point) {
    return batch_size < crossover_batch_size(ridge_point) ? ComputeRegime::MEMORY_BOUND
                                                            : ComputeRegime::COMPUTE_BOUND;
}

// -- Part 2: Chapter 10.1's own Amdahl's-law formulas, reused verbatim. --
double amdahl_speedup(double serial_fraction, int n_threads) {
    double total_relative_time = serial_fraction + (1.0 - serial_fraction) / n_threads;
    return 1.0 / total_relative_time;
}

double amdahl_ceiling(double serial_fraction) {
    return 1.0 / serial_fraction;  // the limit as n_threads -> infinity
}

// -- Part 3: which axis to partition an operation across. --
enum class OperationKind { MATMUL_ROW_OUTPUT, ATTENTION_HEADS_GQA, ELEMENTWISE_REDUCTION };
enum class PartitionStrategy { ROW_PARALLEL, GROUP_ALIGNED_HEAD_PARALLEL, FIXED_BLOCK_REDUCTION };

std::string to_string(PartitionStrategy s) {
    switch (s) {
        case PartitionStrategy::ROW_PARALLEL: return "ROW_PARALLEL";
        case PartitionStrategy::GROUP_ALIGNED_HEAD_PARALLEL: return "GROUP_ALIGNED_HEAD_PARALLEL";
        case PartitionStrategy::FIXED_BLOCK_REDUCTION: return "FIXED_BLOCK_REDUCTION";
    }
    return "UNKNOWN";
}

struct ThreadingRecommendation {
    PartitionStrategy strategy;
    std::string reason;
};

ThreadingRecommendation recommend_partition_strategy(OperationKind kind) {
    switch (kind) {
        case OperationKind::MATMUL_ROW_OUTPUT:
            return {PartitionStrategy::ROW_PARALLEL,
                    "Chapter 10.3/11.1: partitioning disjoint OUTPUT rows needs no combination step "
                    "and reproduces the serial result bit-for-bit; never combine a column-parallel "
                    "split's partial sums with a shared atomic -- well-defined is not the same as "
                    "reproducible run-to-run"};
        case OperationKind::ATTENTION_HEADS_GQA:
            return {PartitionStrategy::GROUP_ALIGNED_HEAD_PARALLEL,
                    "Chapter 11.3: partitioning query heads by raw index can split a single KV-group "
                    "across two threads, forcing that KV head's cache rows to be read redundantly by "
                    "more than one thread -- assign whole KV groups to threads instead"};
        case OperationKind::ELEMENTWISE_REDUCTION:
            return {PartitionStrategy::FIXED_BLOCK_REDUCTION,
                    "Chapter 11.4/11.5: thread-indexed chunking of a reduction like RMSNorm produces a "
                    "different bit pattern for every thread count tested; fixed-size blocks independent "
                    "of thread count are required, and this one reduction's own reproducibility gates "
                    "the entire pipeline's -- not merely its own phase's"};
    }
    return {PartitionStrategy::ROW_PARALLEL, "unreachable"};
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "Appendix C.2: The Threading and Parallelization Strategy Advisor\n";
    std::cout << "=========================================================\n\n";

    // -- Test 1: Chapter 30.1's own worked example reproduced exactly --
    // a machine with 1000 GFLOP/s peak compute and 25 GB/s peak bandwidth
    // has ridge point 40.0 FLOPs/byte, so crossover batch size is 80.0:
    // batch 79 is still memory-bound, batch 81 is already compute-bound. --
    {
        double ridge_point = 1000.0 / 25.0;  // GFLOPs/s / GB/s = FLOPs/byte
        double crossover = crossover_batch_size(ridge_point);
        std::cout << "-- Test 1: ridge_point=" << ridge_point << " -- crossover_batch_size=" << crossover << " --\n";
        CHECK(ridge_point == 40.0);
        CHECK(crossover == 80.0);
        CHECK(classify_batch(79.0, ridge_point) == ComputeRegime::MEMORY_BOUND);
        CHECK(classify_batch(81.0, ridge_point) == ComputeRegime::COMPUTE_BOUND);
    }

    // -- Test 2: Chapter 10.1's own worked example reproduced exactly --
    // a 5% serial fraction gives roughly 5.93x speedup at 8 threads, far
    // short of the true, unreachable ceiling of 20x. --
    {
        double serial_fraction = 0.05;
        double speedup_8 = amdahl_speedup(serial_fraction, 8);
        double ceiling = amdahl_ceiling(serial_fraction);
        std::cout << "-- Test 2: serial_fraction=" << serial_fraction << " -- speedup@8=" << speedup_8
                  << ", ceiling=" << ceiling << " --\n";
        CHECK(ceiling == 20.0);
        CHECK(std::abs(speedup_8 - 5.925925925925926) < 1e-9);
        CHECK(speedup_8 < ceiling);
    }

    // -- Test 3: the ceiling is a true limit -- even a millon threads on
    // the same 5% serial fraction never reaches it, only approaches it. --
    {
        double serial_fraction = 0.05;
        double speedup_huge = amdahl_speedup(serial_fraction, 1000000);
        double ceiling = amdahl_ceiling(serial_fraction);
        std::cout << "-- Test 3: speedup@1000000 threads=" << speedup_huge << " (ceiling=" << ceiling << ") --\n";
        CHECK(speedup_huge < ceiling);
        CHECK(speedup_huge > 19.99);
    }

    // -- Tests 4-6: the three real partition rules, reproduced exactly. --
    {
        auto matmul = recommend_partition_strategy(OperationKind::MATMUL_ROW_OUTPUT);
        std::cout << "-- Test 4: matmul output rows -- strategy=" << to_string(matmul.strategy) << " --\n";
        CHECK(matmul.strategy == PartitionStrategy::ROW_PARALLEL);
    }
    {
        auto attn = recommend_partition_strategy(OperationKind::ATTENTION_HEADS_GQA);
        std::cout << "-- Test 5: GQA attention heads -- strategy=" << to_string(attn.strategy) << " --\n";
        CHECK(attn.strategy == PartitionStrategy::GROUP_ALIGNED_HEAD_PARALLEL);
        CHECK(attn.reason.find("redundant") != std::string::npos);
    }
    {
        auto reduce = recommend_partition_strategy(OperationKind::ELEMENTWISE_REDUCTION);
        std::cout << "-- Test 6: elementwise reduction (e.g. RMSNorm) -- strategy=" << to_string(reduce.strategy) << " --\n";
        CHECK(reduce.strategy == PartitionStrategy::FIXED_BLOCK_REDUCTION);
        CHECK(reduce.reason.find("thread count") != std::string::npos);
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
