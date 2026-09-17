// Chapter 26.3 -- A US check states its own amount TWICE: once as digits
// ("$1,234.56") and once spelled out in words ("One thousand two hundred
// thirty-four and 56/100 dollars"). This redundancy exists specifically
// so that altering the numerals alone -- the classic "check washing"
// fraud, chemically removing and rewriting the amount -- does not
// silently succeed, provided something actually cross-checks the two
// fields against each other. This section builds a real, from-scratch
// English number-word parser that reconstructs the numeric value the
// words field actually claims, and cross-validates it against the
// numerals field.
//
// A note on this section's own honest scope: `parse_amount_words`
// implements the specific real US-check convention (a whole-dollar
// amount in words, followed by "and NN/100" for cents) -- it does not
// attempt to parse arbitrary English number phrases (currencies with
// their own different word-order conventions, or numbers phrased with
// "and" placed differently), the same deliberately narrow, honestly
// stated scope Chapter 12's own tokenizer applied to its own real
// vocabulary rather than claiming to handle every language.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_amount_in_words_cross_validation.cpp -o 03_amount_in_words_cross_validation
// Run:     ./03_amount_in_words_cross_validation

#include <algorithm>
#include <cctype>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: a real, from-scratch parser turning US check-convention words
// ("one thousand two hundred thirty-four and 56/100") into an exact cent
// count -- the same integer-cents discipline Chapter 4's own quantization
// work used to avoid floating-point drift on a monetary value.
// =======================================================================
const std::map<std::string, int> UNITS = {
    {"zero", 0}, {"one", 1}, {"two", 2}, {"three", 3}, {"four", 4},
    {"five", 5}, {"six", 6}, {"seven", 7}, {"eight", 8}, {"nine", 9},
    {"ten", 10}, {"eleven", 11}, {"twelve", 12}, {"thirteen", 13},
    {"fourteen", 14}, {"fifteen", 15}, {"sixteen", 16}, {"seventeen", 17},
    {"eighteen", 18}, {"nineteen", 19},
};
const std::map<std::string, int> TENS = {
    {"twenty", 20}, {"thirty", 30}, {"forty", 40}, {"fifty", 50},
    {"sixty", 60}, {"seventy", 70}, {"eighty", 80}, {"ninety", 90},
};

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::vector<std::string> tokenize_words(const std::string& phrase) {
    std::vector<std::string> tokens;
    std::string current;
    for (char c : phrase) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '/') {
            current += c;
        } else if (c == '-') {
            // "thirty-four" splits into two number words joined by a hyphen
            if (!current.empty()) { tokens.push_back(current); current.clear(); }
        } else {
            if (!current.empty()) { tokens.push_back(current); current.clear(); }
        }
    }
    if (!current.empty()) tokens.push_back(current);
    return tokens;
}

// Parses the WHOLE-DOLLAR portion of a words phrase (everything up to,
// but not including, "and NN/100" or "dollars"). Returns std::nullopt on
// any token this parser does not recognize -- an honest failure, never a
// silent zero.
std::optional<long long> parse_dollar_words(const std::vector<std::string>& tokens) {
    long long total = 0;
    long long current_group = 0;
    bool saw_any = false;
    for (const std::string& raw : tokens) {
        std::string tok = to_lower(raw);
        if (tok == "and" || tok == "dollars" || tok == "dollar") continue;
        if (tok == "hundred") {
            if (current_group == 0) return std::nullopt;
            current_group *= 100;
            saw_any = true;
        } else if (tok == "thousand") {
            if (current_group == 0) return std::nullopt;
            total += current_group * 1000;
            current_group = 0;
            saw_any = true;
        } else if (auto it = UNITS.find(tok); it != UNITS.end()) {
            current_group += it->second;
            saw_any = true;
        } else if (auto it2 = TENS.find(tok); it2 != TENS.end()) {
            current_group += it2->second;
            saw_any = true;
        } else {
            return std::nullopt;  // an unrecognized token: honest failure, not a silent guess
        }
    }
    total += current_group;
    if (!saw_any) return std::nullopt;
    return total;
}

struct ParsedAmount {
    bool valid = false;
    long long total_cents = 0;
};

// Parses the full check-convention phrase, including the "and NN/100"
// cents suffix if present (cents default to 0 if the phrase states none).
ParsedAmount parse_amount_words(const std::string& phrase) {
    std::string lower = to_lower(phrase);
    std::size_t slash_pos = lower.find('/');
    long long cents = 0;
    std::string dollar_part = phrase;
    if (slash_pos != std::string::npos) {
        // Walk backward from the slash to find the start of the "NN" cents
        // numerator, then forward past "/100" to split the phrase.
        std::size_t num_start = slash_pos;
        while (num_start > 0 && std::isdigit(static_cast<unsigned char>(lower[num_start - 1]))) num_start--;
        std::string cents_str = lower.substr(num_start, slash_pos - num_start);
        if (cents_str.empty()) return ParsedAmount{false, 0};
        cents = std::stoll(cents_str);
        dollar_part = phrase.substr(0, num_start);
    }
    std::vector<std::string> tokens = tokenize_words(dollar_part);
    std::optional<long long> dollars = parse_dollar_words(tokens);
    if (!dollars.has_value() || cents < 0 || cents > 99) return ParsedAmount{false, 0};
    return ParsedAmount{true, *dollars * 100 + cents};
}

long long numerals_to_cents(double amount) {
    // Round to the nearest cent using integer arithmetic on a scaled
    // value -- the same avoid-floating-point-drift discipline used
    // wherever this book compares a monetary or quantized value exactly.
    return static_cast<long long>(amount * 100.0 + (amount >= 0 ? 0.5 : -0.5));
}

struct CrossCheckResult {
    bool matches = false;
    bool words_parse_failed = false;
    long long numerals_cents = 0;
    long long words_cents = 0;
};

CrossCheckResult cross_check(double numerals_amount, const std::string& words_phrase) {
    long long numerals_cents = numerals_to_cents(numerals_amount);
    ParsedAmount parsed = parse_amount_words(words_phrase);
    if (!parsed.valid) return CrossCheckResult{false, true, numerals_cents, 0};
    return CrossCheckResult{numerals_cents == parsed.total_cents, false, numerals_cents, parsed.total_cents};
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 26.3: Amount-in-Words Cross-Validation\n";
    std::cout << "========================================================\n\n";

    // -- Test 1: a correctly matching check -- numerals and words agree
    // exactly, down to the cent. --
    CrossCheckResult r1 = cross_check(1234.56, "One thousand two hundred thirty-four and 56/100 dollars");
    std::cout << "-- Test 1: $1234.56 vs. \"One thousand two hundred thirty-four and 56/100 dollars\" -- numerals = "
              << r1.numerals_cents << " cents, words = " << r1.words_cents << " cents, match: "
              << (r1.matches ? "YES" : "NO") << " --\n";
    CHECK(r1.matches);
    CHECK(r1.numerals_cents == 123456);
    CHECK(r1.words_cents == 123456);

    // -- Test 2: the classic check-washing fraud -- the numerals field
    // altered (say, from $1,234.56 to $9,234.56) while the words field is
    // left untouched. The mismatch is caught exactly, naming both values. --
    CrossCheckResult r2 = cross_check(9234.56, "One thousand two hundred thirty-four and 56/100 dollars");
    std::cout << "-- Test 2: $9234.56 (numerals altered) vs. the SAME unaltered words field -- numerals = "
              << r2.numerals_cents << " cents, words = " << r2.words_cents << " cents, match: "
              << (r2.matches ? "YES (WRONG)" : "NO (correctly caught)") << " --\n";
    CHECK(!r2.matches);

    // -- Test 3: the reverse alteration -- the words field altered while
    // numerals are left untouched -- is caught by the identical logic,
    // since the check compares the two fields symmetrically. --
    CrossCheckResult r3 = cross_check(1234.56, "Nine thousand two hundred thirty-four and 56/100 dollars");
    std::cout << "-- Test 3: $1234.56 vs. \"Nine thousand two hundred thirty-four and 56/100 dollars\" (words altered) -- match: "
              << (r3.matches ? "YES (WRONG)" : "NO (correctly caught)") << " --\n";
    CHECK(!r3.matches);

    // -- Test 4: a whole-dollar amount with no stated cents defaults to
    // zero cents, matching a numerals amount of exactly $500.00. --
    CrossCheckResult r4 = cross_check(500.00, "Five hundred dollars");
    std::cout << "-- Test 4: $500.00 vs. \"Five hundred dollars\" (no cents stated) -- words = "
              << r4.words_cents << " cents, match: " << (r4.matches ? "YES" : "NO") << " --\n";
    CHECK(r4.matches);
    CHECK(r4.words_cents == 50000);

    // -- Test 5: a genuinely unparseable words field (containing a token
    // this narrow, stated parser does not recognize) is reported as an
    // honest parse failure, never silently treated as a match or a
    // silent zero. --
    CrossCheckResult r5 = cross_check(1234.56, "A gazillion dollars");
    std::cout << "-- Test 5: \"A gazillion dollars\" (an unrecognized token) -- words_parse_failed: "
              << (r5.words_parse_failed ? "YES (correctly reported)" : "NO (WRONG)") << " --\n";
    CHECK(r5.words_parse_failed);
    CHECK(!r5.matches);

    // -- Test 6: an amount just below the "hundred" boundary (99 dollars,
    // no hundreds/thousands grouping at all) parses correctly, confirming
    // the parser handles the plain-units case, not only multi-group
    // amounts. --
    CrossCheckResult r6 = cross_check(99.00, "Ninety-nine dollars");
    std::cout << "-- Test 6: $99.00 vs. \"Ninety-nine dollars\" -- words = " << r6.words_cents
              << " cents, match: " << (r6.matches ? "YES" : "NO") << " --\n";
    CHECK(r6.matches);
    CHECK(r6.words_cents == 9900);

    // -- Test 7: a real, larger multi-thousand amount with a nonzero
    // hundreds group inside the thousands group -- confirming the parser
    // correctly resets its own running group after each "thousand". --
    CrossCheckResult r7 = cross_check(42305.75, "Forty-two thousand three hundred five and 75/100 dollars");
    std::cout << "-- Test 7: $42305.75 vs. \"Forty-two thousand three hundred five and 75/100 dollars\" -- words = "
              << r7.words_cents << " cents, match: " << (r7.matches ? "YES" : "NO") << " --\n";
    CHECK(r7.matches);
    CHECK(r7.words_cents == 4230575);

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
