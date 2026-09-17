// Chapter 27.3 -- A real, well-documented finding in financial text
// analysis (Loughran and McDonald, "When Is a Liability Not a
// Liability? Textual Analysis, Dictionaries, and 10-Ks," Journal of
// Finance, 2011) is that a GENERIC English sentiment word list badly
// misclassifies financial text: ordinary business vocabulary such as
// "tax," "liability," "cost," and "debt" is treated as negative by a
// generic dictionary, even though these words describe completely
// routine, expected line items in real financial disclosures and
// headlines, not bad news. The real fix Loughran-McDonald published is a
// FINANCE-SPECIFIC sentiment lexicon, deliberately built to exclude that
// routine vocabulary from its own negative word list.
//
// This section builds a small, explicitly illustrative lexicon modeled on
// that real published methodology -- not the real published word list
// itself, which spans several thousand words across seven real
// categories. This section's own COMMON TRAP box returns to exactly that
// scope limitation.
//
// A second real, standard technique from lexicon-based sentiment analysis
// is included as well: simple negation handling. A negation word (such as
// "not") flips the polarity of the very next lexicon word it precedes --
// "not profitable" should score as negative, not positive, even though
// "profitable" alone is a real positive-lexicon word.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_financial_news_sentiment_scoring.cpp -o 03_financial_news_sentiment_scoring
// Run:     ./03_financial_news_sentiment_scoring

#include <cctype>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// Lowercases a word and strips any leading/trailing punctuation, so that
// "Profit!", "profit,", and "profit" all tokenize identically.
static std::string clean_word(const std::string& raw) {
    std::string w;
    for (char c : raw) {
        if (std::isalpha(static_cast<unsigned char>(c))) {
            w += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    return w;
}

static std::vector<std::string> tokenize_headline(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string raw;
    while (iss >> raw) {
        std::string cleaned = clean_word(raw);
        if (!cleaned.empty()) tokens.push_back(cleaned);
    }
    return tokens;
}

struct SentimentResult {
    int positive_hits = 0;
    int negative_hits = 0;
    int score = 0;
    std::string label;
    std::vector<std::string> matched_positive;
    std::vector<std::string> matched_negative;
};

// A finance-specific negative lexicon deliberately EXCLUDES routine
// business vocabulary (tax, liability, cost, debt) that a generic
// English lexicon would flag as negative -- this is the real
// Loughran-McDonald finding this section's own Test 5 demonstrates.
static const std::set<std::string> POS_LEXICON = {
    "profit", "profitable", "growth", "surge", "beat", "exceeded", "record", "strong", "upgrade",
};
static const std::set<std::string> NEG_LEXICON = {
    "loss", "losses", "decline", "miss", "missed", "downgrade", "weak", "lawsuit", "bankruptcy", "fraud",
};
// Words a GENERIC (non-finance-specific) sentiment lexicon would mark
// negative, that the real finance-specific methodology deliberately
// excludes from NEG_LEXICON above -- used only by Test 5 to demonstrate
// the contrast, never merged into NEG_LEXICON itself.
static const std::set<std::string> GENERIC_NEGATIVE_EXTRA = {
    "tax", "liability", "liabilities", "cost", "costs", "debt",
};
static const std::set<std::string> NEGATION_WORDS = {"not", "no", "never"};

// Negation applies only to the SINGLE token immediately following a
// negation word -- a deliberately narrow, stated scope, not an attempt
// to track negation across an entire clause.
static SentimentResult score_sentiment(const std::vector<std::string>& tokens,
                                        const std::set<std::string>& pos_lexicon,
                                        const std::set<std::string>& neg_lexicon) {
    SentimentResult r;
    bool negate_next = false;
    for (const auto& tok : tokens) {
        if (NEGATION_WORDS.contains(tok)) {
            negate_next = true;
            continue;
        }
        const bool is_pos = pos_lexicon.contains(tok);
        const bool is_neg = neg_lexicon.contains(tok);
        if (is_pos) {
            if (negate_next) {
                r.negative_hits++;
                r.matched_negative.push_back(tok + " (negated)");
            } else {
                r.positive_hits++;
                r.matched_positive.push_back(tok);
            }
        } else if (is_neg) {
            if (negate_next) {
                r.positive_hits++;
                r.matched_positive.push_back(tok + " (negated)");
            } else {
                r.negative_hits++;
                r.matched_negative.push_back(tok);
            }
        }
        negate_next = false;
    }
    r.score = r.positive_hits - r.negative_hits;
    r.label = (r.score > 0) ? "POSITIVE" : (r.score < 0) ? "NEGATIVE" : "NEUTRAL";
    return r;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 27.3: Financial News Sentiment Scoring\n";
    std::cout << "========================================================\n\n";

    // -- Test 1: a clearly positive headline -- every matched word is a
    // real finance-positive term, none negative. --
    {
        auto tokens = tokenize_headline("Company reports record profit and strong growth");
        auto r = score_sentiment(tokens, POS_LEXICON, NEG_LEXICON);
        std::cout << "-- Test 1: \"record profit ... strong growth\" -- positive_hits=" << r.positive_hits
                  << ", negative_hits=" << r.negative_hits << ", score=" << r.score << ", label=" << r.label
                  << " --\n";
        CHECK(r.positive_hits == 4);
        CHECK(r.negative_hits == 0);
        CHECK(r.score == 4);
        CHECK(r.label == "POSITIVE");
    }

    // -- Test 2: a clearly negative headline. --
    {
        auto tokens = tokenize_headline("Company reports steep decline and missed earnings amid lawsuit");
        auto r = score_sentiment(tokens, POS_LEXICON, NEG_LEXICON);
        std::cout << "-- Test 2: \"decline ... missed earnings ... lawsuit\" -- positive_hits="
                  << r.positive_hits << ", negative_hits=" << r.negative_hits << ", score=" << r.score
                  << ", label=" << r.label << " --\n";
        CHECK(r.positive_hits == 0);
        CHECK(r.negative_hits == 3);
        CHECK(r.score == -3);
        CHECK(r.label == "NEGATIVE");
    }

    // -- Test 3: an evenly mixed headline -- equal positive and negative
    // hits must net to an honest NEUTRAL, not a tie-break in either
    // direction. --
    {
        auto tokens = tokenize_headline("profit and growth offset by loss and decline");
        auto r = score_sentiment(tokens, POS_LEXICON, NEG_LEXICON);
        std::cout << "-- Test 3: \"profit and growth offset by loss and decline\" -- positive_hits="
                  << r.positive_hits << ", negative_hits=" << r.negative_hits << ", score=" << r.score
                  << ", label=" << r.label << " --\n";
        CHECK(r.positive_hits == 2);
        CHECK(r.negative_hits == 2);
        CHECK(r.score == 0);
        CHECK(r.label == "NEUTRAL");
    }

    // -- Test 4: negation flips a positive word's own contribution to
    // negative -- confirmed against the SAME word scored without a
    // preceding negation, to isolate the negation logic's own effect. --
    {
        auto negated_tokens = tokenize_headline("not profitable this quarter");
        auto negated = score_sentiment(negated_tokens, POS_LEXICON, NEG_LEXICON);
        auto plain_tokens = tokenize_headline("profitable this quarter");
        auto plain = score_sentiment(plain_tokens, POS_LEXICON, NEG_LEXICON);
        std::cout << "-- Test 4: \"not profitable this quarter\" -> score=" << negated.score << " ("
                  << negated.label << "), vs. \"profitable this quarter\" -> score=" << plain.score << " ("
                  << plain.label << ") --\n";
        CHECK(negated.score == -1);
        CHECK(negated.label == "NEGATIVE");
        CHECK(plain.score == 1);
        CHECK(plain.label == "POSITIVE");
    }

    // -- Test 5: the real Loughran-McDonald finding, demonstrated
    // directly. "tax" and "cost" are routine finance vocabulary, excluded
    // from this section's own finance-specific NEG_LEXICON -- scoring
    // NEUTRAL here -- but scoring NEGATIVE under a stated GENERIC lexicon
    // that (like a real general-purpose English sentiment dictionary)
    // does treat them as negative words. --
    {
        auto tokens = tokenize_headline("The company reported higher tax and cost this quarter");
        auto finance = score_sentiment(tokens, POS_LEXICON, NEG_LEXICON);
        std::set<std::string> generic_neg = NEG_LEXICON;
        generic_neg.insert(GENERIC_NEGATIVE_EXTRA.begin(), GENERIC_NEGATIVE_EXTRA.end());
        auto generic = score_sentiment(tokens, POS_LEXICON, generic_neg);
        std::cout << "-- Test 5: \"higher tax and cost\" -- finance-lexicon score=" << finance.score << " ("
                  << finance.label << "), generic-lexicon score=" << generic.score << " (" << generic.label
                  << ") --\n";
        CHECK(finance.score == 0);
        CHECK(finance.label == "NEUTRAL");
        CHECK(generic.score == -2);
        CHECK(generic.label == "NEGATIVE");
    }

    // -- Test 6: tokenizer robustness -- mixed case and trailing
    // punctuation attached directly to each word must still match the
    // lowercase, punctuation-free lexicon entries. --
    {
        auto tokens = tokenize_headline("Profit! Growth... Exceeded expectations.");
        auto r = score_sentiment(tokens, POS_LEXICON, NEG_LEXICON);
        std::cout << "-- Test 6: \"Profit! Growth... Exceeded expectations.\" -- positive_hits="
                  << r.positive_hits << ", score=" << r.score << ", label=" << r.label << " --\n";
        CHECK(r.positive_hits == 3);
        CHECK(r.score == 3);
        CHECK(r.label == "POSITIVE");
    }

    // -- Test 7: a headline containing no lexicon words at all must
    // report an honest, explicit zero -- NEUTRAL with empty matched
    // lists -- never a crash or a fabricated nonzero score. --
    {
        auto tokens = tokenize_headline("The quarterly meeting was held Tuesday");
        auto r = score_sentiment(tokens, POS_LEXICON, NEG_LEXICON);
        std::cout << "-- Test 7: \"The quarterly meeting was held Tuesday\" -- positive_hits="
                  << r.positive_hits << ", negative_hits=" << r.negative_hits << ", score=" << r.score
                  << ", label=" << r.label << " --\n";
        CHECK(r.positive_hits == 0);
        CHECK(r.negative_hits == 0);
        CHECK(r.score == 0);
        CHECK(r.label == "NEUTRAL");
        CHECK(r.matched_positive.empty());
        CHECK(r.matched_negative.empty());
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
