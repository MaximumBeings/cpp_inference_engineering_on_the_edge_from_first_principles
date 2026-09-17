// Chapter 26.3 -- Chapter 3 implemented RoPE as a working piece of a
// real computational graph; this section derives WHY it actually works.
// A 2D rotation matrix is a real orthogonal transformation, and
// composing the transpose of one rotation with another real rotation
// yields exactly a THIRD rotation by the real angle DIFFERENCE -- a
// clean trigonometric identity. Applying a position-dependent rotation
// to a query and a key before taking their real dot product is what
// turns that identity into RoPE's own central, load-bearing property:
// the resulting dot product depends only on the real relative position
// between the two tokens, never on their real absolute positions.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_rope_rotation_math.cpp -o 03_rope_rotation_math
// Run:     ./03_rope_rotation_math

#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

bool near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) < eps; }

// =======================================================================
// PART 1: a real 2D vector and a real 2x2 rotation matrix, built from
// the standard rotation-matrix formula.
// =======================================================================
struct Vec2 {
    double x = 0.0, y = 0.0;
};

struct Mat2 {
    double a = 0.0, b = 0.0, c = 0.0, d = 0.0;  // [[a, b], [c, d]]
};

Mat2 rotation_matrix(double theta) {
    double ct = std::cos(theta), st = std::sin(theta);
    return {ct, -st, st, ct};
}

Vec2 apply(const Mat2& m, const Vec2& v) {
    return {m.a * v.x + m.b * v.y, m.c * v.x + m.d * v.y};
}

Mat2 transpose(const Mat2& m) { return {m.a, m.c, m.b, m.d}; }

Mat2 matmul(const Mat2& p, const Mat2& q) {
    return {
        p.a * q.a + p.b * q.c, p.a * q.b + p.b * q.d,
        p.c * q.a + p.d * q.c, p.c * q.b + p.d * q.d,
    };
}

double dot(const Vec2& u, const Vec2& v) { return u.x * v.x + u.y * v.y; }

bool mat_near(const Mat2& p, const Mat2& q, double eps = 1e-9) {
    return near(p.a, q.a, eps) && near(p.b, q.b, eps) && near(p.c, q.c, eps) && near(p.d, q.d, eps);
}

// =======================================================================
// PART 2: RoPE itself -- rotate a 2D query/key subvector by an angle
// that scales linearly with the token's own real absolute position.
// =======================================================================
Vec2 rope_rotate(const Vec2& v, int64_t position, double theta_base) {
    return apply(rotation_matrix(static_cast<double>(position) * theta_base), v);
}

// =======================================================================
// PART 3: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 26.3: RoPE's Rotation Math\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: the rotation matrix itself matches exact, hand-verifiable real angles --\n";
    {
        Vec2 v{1.0, 0.0};
        Vec2 rotated_zero = apply(rotation_matrix(0.0), v);
        CHECK(near(rotated_zero.x, 1.0) && near(rotated_zero.y, 0.0));

        Vec2 rotated_90 = apply(rotation_matrix(std::numbers::pi / 2.0), v);
        CHECK(near(rotated_90.x, 0.0, 1e-9) && near(rotated_90.y, 1.0, 1e-9));

        Vec2 rotated_180 = apply(rotation_matrix(std::numbers::pi), v);
        CHECK(near(rotated_180.x, -1.0, 1e-9) && near(rotated_180.y, 0.0, 1e-9));
        std::cout << "  rotating (1, 0) by a real angle of 0 leaves it exactly (1, 0); rotating it by "
                     "exactly pi/2 radians (90 degrees) maps it to exactly (0, 1); rotating it by "
                     "exactly pi radians (180 degrees) maps it to exactly (-1, 0)\n";
    }

    std::cout << "\n-- Test 2: a real trigonometric identity -- R(a) transposed, composed with R(b), "
                 "equals exactly R(b - a), checked as a genuine 2x2 matrix equality across several real "
                 "angle pairs --\n";
    {
        struct Pair { double a, b; };
        std::vector<Pair> pairs = {
            {0.3, 0.9}, {1.2, 0.4}, {std::numbers::pi / 4.0, std::numbers::pi / 3.0}, {2.5, 2.5},
        };
        for (const auto& p : pairs) {
            Mat2 lhs = matmul(transpose(rotation_matrix(p.a)), rotation_matrix(p.b));
            Mat2 rhs = rotation_matrix(p.b - p.a);
            CHECK(mat_near(lhs, rhs, 1e-9));
        }
        std::cout << "  across 4 real angle pairs -- including a = b, where the identity predicts "
                     "exactly the identity matrix R(0) -- R(a)^T * R(b) matches R(b - a) exactly, "
                     "confirming rotation composition really does reduce to a single real angle "
                     "subtraction\n";
    }

    std::cout << "\n-- Test 3: RoPE's own central property -- the real dot product of a rotated query "
                 "and a rotated key depends ONLY on their real relative position, never on their "
                 "absolute positions -- verified directly across multiple genuinely different absolute "
                 "position pairs sharing the identical real offset --\n";
    {
        Vec2 q{1.3, -0.7}, k{0.4, 2.1};
        double theta_base = 0.05;

        // Three different absolute (m, n) pairs, all sharing the identical real relative offset (m - n) = 2.
        double dot_5_3 = dot(rope_rotate(q, 5, theta_base), rope_rotate(k, 3, theta_base));
        double dot_10_8 = dot(rope_rotate(q, 10, theta_base), rope_rotate(k, 8, theta_base));
        double dot_2_0 = dot(rope_rotate(q, 2, theta_base), rope_rotate(k, 0, theta_base));

        CHECK(near(dot_5_3, dot_10_8, 1e-9));
        CHECK(near(dot_10_8, dot_2_0, 1e-9));

        // A genuinely different real relative offset produces a genuinely different real dot product.
        double dot_5_2 = dot(rope_rotate(q, 5, theta_base), rope_rotate(k, 2, theta_base));  // offset = 3
        CHECK(!near(dot_5_2, dot_5_3, 1e-6));
        std::cout << "  position pairs (5, 3), (10, 8), and (2, 0) -- three genuinely different real "
                     "absolute positions, all sharing the identical real relative offset of 2 -- produce "
                     "an identical real rotated dot product to within 1e-9; the pair (5, 2), sharing a "
                     "genuinely different offset of 3, produces a measurably different real value\n";
    }

    std::cout << "\n-- Test 4: the multi-position rotated dot product from Test 3 matches the abstract "
                 "identity from Test 2 applied directly -- q^T * R(theta * (n - m)) * k -- bridging the "
                 "concrete RoPE computation to the real rotation-composition identity that explains it "
                 "--\n";
    {
        Vec2 q{1.3, -0.7}, k{0.4, 2.1};
        double theta_base = 0.05;
        int64_t m = 5, n = 3;

        double rope_dot = dot(rope_rotate(q, m, theta_base), rope_rotate(k, n, theta_base));
        Vec2 k_via_identity = apply(rotation_matrix(theta_base * static_cast<double>(n - m)), k);
        double identity_dot = dot(q, k_via_identity);

        CHECK(near(rope_dot, identity_dot, 1e-9));
        std::cout << "  computing dot(R(theta*5) q, R(theta*3) k) directly, and computing "
                     "dot(q, R(theta*(3-5)) k) via the rotation-composition identity from Test 2, agree "
                     "to within 1e-9 -- confirming RoPE's own real relative-position property is not a "
                     "coincidence but a direct, provable consequence of the rotation matrix's own real "
                     "orthogonality\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
