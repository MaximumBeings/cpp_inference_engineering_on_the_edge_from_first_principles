// Chapter 23.4 -- A consignment counter's own real-time authentication
// screen cannot inspect an item the way a trained human expert eventually
// will, but it CAN check three real, computable structural properties
// against a brand's own stated specification: whether a serial number's
// own digit portion satisfies a real, well-known check-digit algorithm
// (the Luhn algorithm, the same real algorithm that validates credit-card
// numbers), whether a measured weight falls within a real stated spec
// range, and whether a hardware finish is among a real stated valid list.
// This section's own central honesty discipline, consistent with every
// structural-override check since Chapter 20: a single hard-failing
// check is NEVER outvoted by two other checks that happen to pass.
//
// A note on this section's own honest scope: `authenticate_item` never
// returns a verdict named "AUTHENTIC" anywhere in this file. Its best
// possible outcome is named `PASSES_STRUCTURAL_SCREENING` -- an honest,
// narrow claim that three specific, real, checkable properties held, not
// a claim that a trained human expert's own final authentication is no
// longer needed, exactly the same honest-scope discipline Section 22.3's
// provenance validator applied to its own chronological consistency
// checks.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_luxury_goods_authentication_engine.cpp -o 04_luxury_goods_authentication_engine
// Run:     ./04_luxury_goods_authentication_engine

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: the real, well-known Luhn check-digit algorithm, the same
// algorithm that validates credit-card numbers -- this section's own
// stated brand uses it as its own serial-number check-digit scheme.
// =======================================================================
bool luhn_valid(const std::string& digits) {
    int sum = 0;
    bool double_it = false;
    for (auto it = digits.rbegin(); it != digits.rend(); ++it) {
        int d = *it - '0';
        if (double_it) {
            d *= 2;
            if (d > 9) d -= 9;
        }
        sum += d;
        double_it = !double_it;
    }
    return sum % 10 == 0;
}

// =======================================================================
// PART 2: this section's own stated serial-number format -- 2 uppercase
// factory-code letters followed by an 11-digit Luhn-valid number -- and
// its structural validator.
// =======================================================================
constexpr int SERIAL_PREFIX_LEN = 2;
constexpr int SERIAL_DIGITS_LEN = 11;
constexpr int SERIAL_TOTAL_LEN = SERIAL_PREFIX_LEN + SERIAL_DIGITS_LEN;

bool is_upper_alpha(char c) { return c >= 'A' && c <= 'Z'; }
bool is_ascii_digit(char c) { return c >= '0' && c <= '9'; }

struct SerialValidationResult {
    bool valid_format = false;
    bool valid_checksum = false;
    std::string reason;
};

SerialValidationResult validate_serial(const std::string& serial) {
    SerialValidationResult r;
    if (static_cast<int>(serial.size()) != SERIAL_TOTAL_LEN) {
        r.reason = "serial length " + std::to_string(serial.size()) + " does not match the required " +
                   std::to_string(SERIAL_TOTAL_LEN) + " characters";
        return r;
    }
    for (int i = 0; i < SERIAL_PREFIX_LEN; i++) {
        if (!is_upper_alpha(serial[static_cast<std::size_t>(i)])) {
            r.reason = "serial's own first " + std::to_string(SERIAL_PREFIX_LEN) +
                       " characters must be uppercase factory-code letters";
            return r;
        }
    }
    std::string digits = serial.substr(SERIAL_PREFIX_LEN);
    for (char c : digits) {
        if (!is_ascii_digit(c)) {
            r.reason = "serial's own digit portion contains a non-digit character";
            return r;
        }
    }
    r.valid_format = true;
    r.valid_checksum = luhn_valid(digits);
    if (!r.valid_checksum) {
        r.reason = "serial's own digit portion fails the brand's stated Luhn check-digit scheme";
    }
    return r;
}

// =======================================================================
// PART 3: the hardware spec, the item submission, and the combined
// authentication engine.
// =======================================================================
struct HardwareSpec {
    double weight_grams_min = 0.0, weight_grams_max = 0.0;
    std::vector<std::string> valid_finishes;
};

struct ItemSubmission {
    std::string item_id;
    std::string serial_number;
    double measured_weight_grams = 0.0;
    std::string hardware_finish;
};

enum class AuthenticationVerdict {
    REJECTED_INVALID_SERIAL,
    REJECTED_WEIGHT_OUT_OF_RANGE,
    FLAGGED_FOR_EXPERT_REVIEW,
    PASSES_STRUCTURAL_SCREENING,
};

struct AuthenticationResult {
    AuthenticationVerdict verdict;
    std::string reason;
};

AuthenticationResult authenticate_item(const ItemSubmission& item, const HardwareSpec& spec) {
    auto serial_result = validate_serial(item.serial_number);
    if (!serial_result.valid_format || !serial_result.valid_checksum) {
        return {AuthenticationVerdict::REJECTED_INVALID_SERIAL, serial_result.reason};
    }

    if (item.measured_weight_grams < spec.weight_grams_min || item.measured_weight_grams > spec.weight_grams_max) {
        return {AuthenticationVerdict::REJECTED_WEIGHT_OUT_OF_RANGE,
                "measured weight " + std::to_string(item.measured_weight_grams) +
                    "g is outside the real spec range [" + std::to_string(spec.weight_grams_min) + "g, " +
                    std::to_string(spec.weight_grams_max) + "g]"};
    }

    bool finish_valid = std::find(spec.valid_finishes.begin(), spec.valid_finishes.end(), item.hardware_finish) !=
                         spec.valid_finishes.end();
    if (!finish_valid) {
        return {AuthenticationVerdict::FLAGGED_FOR_EXPERT_REVIEW,
                "hardware finish \"" + item.hardware_finish +
                    "\" is not among this item's own real, stated valid finishes"};
    }

    return {AuthenticationVerdict::PASSES_STRUCTURAL_SCREENING,
            "serial checksum valid, weight in spec, hardware finish recognized -- passes this counter's "
            "own structural screen; final authentication remains a human expert's own call"};
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 23.4: A Real-Time Luxury-Goods Authentication Engine\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: the real Luhn algorithm matches its own famous exact reference value "
                 "(79927398713 is Luhn-valid), and a single altered digit breaks it --\n";
    {
        CHECK(luhn_valid("79927398713"));
        CHECK(!luhn_valid("79927398714"));
        CHECK(luhn_valid("4532015112830366"));
        CHECK(!luhn_valid("4532015112830367"));
        std::cout << "  \"79927398713,\" the Luhn algorithm's own famous reference test number, validates "
                     "exactly as expected; changing its own last digit by 1 breaks the checksum; the "
                     "same holds for a second, independent real 16-digit Luhn-valid reference number\n";
    }

    std::cout << "\n-- Test 2: this section's own stated serial format is checked structurally -- wrong "
                 "length, a lowercase factory code, and a non-digit in the numeric portion are each "
                 "flagged by their own specific, named reason --\n";
    {
        auto wrong_length = validate_serial("FR7992739871");
        auto lowercase_prefix = validate_serial("fr79927398713");
        auto non_digit = validate_serial("FR7992739871A");
        CHECK(!wrong_length.valid_format);
        CHECK(wrong_length.reason.find("length") != std::string::npos);
        CHECK(!lowercase_prefix.valid_format);
        CHECK(lowercase_prefix.reason.find("uppercase") != std::string::npos);
        CHECK(!non_digit.valid_format);
        CHECK(non_digit.reason.find("non-digit") != std::string::npos);
        std::cout << "  a 12-character serial (one short of the required 13) is flagged for its own "
                     "exact length; a lowercase factory-code prefix is flagged by name; a numeric "
                     "portion containing a letter is flagged by name -- each a specific, different "
                     "structural reason\n";
    }

    std::cout << "\n-- Test 3: a serial with valid structural format but a failing checksum is flagged "
                 "distinctly from a structurally invalid one --\n";
    {
        auto valid = validate_serial("FR79927398713");
        auto bad_checksum = validate_serial("FR79927398714");
        CHECK(valid.valid_format && valid.valid_checksum);
        CHECK(bad_checksum.valid_format && !bad_checksum.valid_checksum);
        CHECK(bad_checksum.reason.find("Luhn") != std::string::npos);
        std::cout << "  \"FR79927398713\" passes both structural format and checksum; "
                     "\"FR79927398714\" passes the structural format (right length, right character "
                     "classes) but fails the Luhn checksum specifically, named by its own reason\n";
    }

    std::cout << "\n-- Test 4: an item with a fully valid serial, weight, and hardware finish passes "
                 "structural screening -- an honest, narrow claim, never named \"authentic\" --\n";
    {
        HardwareSpec spec{180.0, 220.0, {"Gold-Tone", "Silver-Tone", "Gunmetal"}};
        ItemSubmission item{"ITEM-1", "FR79927398713", 200.0, "Gold-Tone"};
        auto result = authenticate_item(item, spec);
        CHECK(result.verdict == AuthenticationVerdict::PASSES_STRUCTURAL_SCREENING);
        CHECK(result.reason.find("human expert") != std::string::npos);
        std::cout << "  a valid serial, a 200g weight inside the real [180g, 220g] spec, and a "
                     "recognized \"Gold-Tone\" finish together produce PASSES_STRUCTURAL_SCREENING -- "
                     "whose own reason text explicitly states final authentication remains a human "
                     "expert's own call\n";
    }

    std::cout << "\n-- Test 5: an invalid serial checksum alone forces REJECTED_INVALID_SERIAL even when "
                 "weight and hardware finish both pass -- one hard-failing check is never outvoted by "
                 "two passing ones --\n";
    {
        HardwareSpec spec{180.0, 220.0, {"Gold-Tone", "Silver-Tone", "Gunmetal"}};
        ItemSubmission item{"ITEM-2", "FR79927398714", 200.0, "Gold-Tone"};
        auto result = authenticate_item(item, spec);
        CHECK(result.verdict == AuthenticationVerdict::REJECTED_INVALID_SERIAL);
        std::cout << "  the identical 200g \"Gold-Tone\" item, with only its own serial's checksum "
                     "digit altered, is REJECTED_INVALID_SERIAL outright -- a perfectly in-spec weight "
                     "and a perfectly recognized finish never override one specific failing checksum\n";
    }

    std::cout << "\n-- Test 6: a weight outside the real spec range is rejected with the exact range "
                 "named, even with a fully valid serial --\n";
    {
        HardwareSpec spec{180.0, 220.0, {"Gold-Tone", "Silver-Tone", "Gunmetal"}};
        ItemSubmission item{"ITEM-3", "FR79927398713", 260.0, "Gold-Tone"};
        auto result = authenticate_item(item, spec);
        CHECK(result.verdict == AuthenticationVerdict::REJECTED_WEIGHT_OUT_OF_RANGE);
        CHECK(result.reason.find("180") != std::string::npos && result.reason.find("220") != std::string::npos);
        std::cout << "  a 260g item with an otherwise perfectly valid serial is rejected, naming the "
                     "real [180g, 220g] spec range it falls outside of\n";
    }

    std::cout << "\n-- Test 7: an unrecognized hardware finish is flagged for expert review, not silently "
                 "passed, even when the serial and weight both look fine --\n";
    {
        HardwareSpec spec{180.0, 220.0, {"Gold-Tone", "Silver-Tone", "Gunmetal"}};
        ItemSubmission item{"ITEM-4", "FR79927398713", 200.0, "Rose-Gold-Tone"};
        auto result = authenticate_item(item, spec);
        CHECK(result.verdict == AuthenticationVerdict::FLAGGED_FOR_EXPERT_REVIEW);
        CHECK(result.reason.find("Rose-Gold-Tone") != std::string::npos);
        std::cout << "  a valid serial and an in-spec 200g weight both look fine, but \"Rose-Gold-Tone\" "
                     "is not among this item's own 3 stated valid finishes, so the item is flagged for "
                     "expert review rather than silently passed on the strength of its other 2 checks\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
