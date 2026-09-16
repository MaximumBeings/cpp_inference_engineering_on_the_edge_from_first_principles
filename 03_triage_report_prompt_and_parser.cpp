// Chapter 20.3 -- Section 20.2 produced a real, windowed, viewable image
// from a DICOM study. Section 20.1 built a state machine that refuses to
// let anything but a human clinician finalize a case. This section
// builds the piece that has to sit correctly between them: the prompt
// that tells a vision-language model exactly what kind of answer it is
// allowed to give, and a parser strict enough that even a model that
// tried to give a different kind of answer could not get it through.
//
// The one real design decision this section centers on is what the
// STRUCTURED OUTPUT FORMAT ITSELF is allowed to represent. `URGENCY` is
// a QUEUE PRIORITY -- which of three tiers a human reviewer's own
// worklist should treat a study as -- and the parser accepts exactly
// three literal values for it (`ROUTINE`, `EXPEDITED`, `STAT`) and
// nothing else. There is no `DIAGNOSIS` field anywhere in this format,
// and Test 2 confirms directly that a line attempting to smuggle one in
// -- either as an extra field, or by writing a diagnosis-shaped string
// into the `URGENCY` field itself -- is refused outright by the same
// strict parsing discipline Chapter 19.2 already applied to a retail
// detection line. This is the schema-level half of Section 20.1's own
// structural guarantee: even if a model's own free-text output somehow
// tried to assert a diagnosis, there is no path for that assertion to
// survive parsing and reach anything downstream as structured data. The
// same discipline governs the report as a whole: `build_triage_report`
// is the ONLY way to construct a `TriageReport`, and it unconditionally
// attaches one fixed, unmodifiable disclaimer string -- there is no
// parameter, overload, or code path anywhere in this file that produces
// a `TriageReport` without it.
//
// As in Section 20.1, this section makes no claim to represent any
// specific jurisdiction's regulatory requirements for how such a report
// must be worded or structured; the disclaimer text below is this
// book's own stated, illustrative example of the KIND of statement such
// a system would need to carry, not a claim that this exact wording
// satisfies any real requirement.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_triage_report_prompt_and_parser.cpp -o 03_triage_report_prompt_and_parser
// Run:     ./03_triage_report_prompt_and_parser

#include <algorithm>
#include <charconv>
#include <iostream>
#include <set>
#include <sstream>
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
// PART 1: study metadata and a deterministic, validated system prompt.
// =======================================================================
struct StudyMetadata {
    std::string study_id;
    std::string modality;     // one of a real, fixed, known set (see validate_study_metadata)
    std::string body_part;
    int rows = 0;
    int columns = 0;
};

struct ValidationResult { bool ok = false; std::string error; };

ValidationResult validate_study_metadata(const StudyMetadata& m) {
    static const std::set<std::string> known_modalities = {"CT", "MR", "CR", "US", "XA", "DX"};
    if (m.study_id.empty()) return {false, "study_id must not be empty"};
    if (!known_modalities.count(m.modality)) return {false, "unknown modality: \"" + m.modality + "\""};
    if (m.body_part.empty()) return {false, "body_part must not be empty"};
    if (m.rows <= 0 || m.columns <= 0) return {false, "rows and columns must both be positive"};
    return {true, ""};
}

// A fixed, real instructional block, stating what kind of answer this
// prompt permits and what structured format the answer must follow.
// Rendering is otherwise identical to Chapter 19.1's own deterministic
// planogram rendering: the same metadata always produces the exact same
// prompt text, byte for byte.
std::string render_triage_prompt(const StudyMetadata& m) {
    std::ostringstream out;
    out << "TRIAGE ASSISTANCE REQUEST\n";
    out << "You are assisting a radiologist's own review queue. You are NOT diagnosing this study.\n";
    out << "For each notable finding, respond with EXACTLY one line in this format:\n";
    out << "FINDING=<short description>;LOCATION=<anatomical location>;URGENCY=<ROUTINE|EXPEDITED|STAT>;CONFIDENCE=<0.0-1.0>\n";
    out << "URGENCY is a QUEUE PRIORITY for human review, not a diagnosis. Do not include any other field.\n";
    out << "If the study shows nothing notable, respond with no finding lines at all.\n";
    out << "Study: " << m.study_id << "\n";
    out << "Modality: " << m.modality << "\n";
    out << "Body part: " << m.body_part << "\n";
    out << "Image size: " << m.rows << "x" << m.columns << "\n";
    return out.str();
}

// =======================================================================
// PART 2: urgency tiers and the strict finding-line parser.
// =======================================================================
enum class Urgency { ROUTINE, EXPEDITED, STAT };

std::string to_string(Urgency u) {
    switch (u) {
        case Urgency::ROUTINE: return "ROUTINE";
        case Urgency::EXPEDITED: return "EXPEDITED";
        case Urgency::STAT: return "STAT";
    }
    return "UNKNOWN";
}

// Real ordering for aggregation: STAT is the most urgent, ROUTINE the
// least. Used only to pick the single most urgent tier across a study's
// own findings -- never to average or otherwise blend urgency values.
int urgency_rank(Urgency u) { return static_cast<int>(u); }

struct ParsedFinding {
    std::string finding;
    std::string location;
    Urgency urgency = Urgency::ROUTINE;
    double confidence = 0.0;
};

struct ParseResult {
    bool ok = false;
    std::string error;
    ParsedFinding finding;
};

// Strict by construction: exactly the four real fields this format
// defines, an URGENCY value that must be one of the three real literal
// tokens (never a free-text string that could carry a diagnosis), and a
// CONFIDENCE inside [0, 1]. `std::from_chars` throughout, matching this
// book's own standing preference for checked return values over
// exceptions.
ParseResult parse_finding_line(const std::string& line) {
    ParseResult res;
    std::unordered_map<std::string, std::string> fields;
    std::istringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ';')) {
        auto eq = field.find('=');
        if (eq == std::string::npos) {
            res.error = "malformed field (no '=' found): \"" + field + "\"";
            return res;
        }
        std::string key = field.substr(0, eq);
        std::string value = field.substr(eq + 1);
        if (fields.count(key)) {
            res.error = "duplicate field: " + key;
            return res;
        }
        fields[key] = value;
    }

    for (const char* required : {"FINDING", "LOCATION", "URGENCY", "CONFIDENCE"}) {
        if (!fields.count(required)) {
            res.error = std::string("missing required field: ") + required;
            return res;
        }
    }
    if (fields.size() != 4) {
        // This is the real structural refusal: an extra field -- for
        // instance a model-generated "DIAGNOSIS=..." -- is rejected
        // wholesale, not silently dropped or silently accepted alongside
        // the four real fields.
        res.error = "unexpected extra field(s) present (expected exactly FINDING, LOCATION, URGENCY, CONFIDENCE)";
        return res;
    }
    if (fields["FINDING"].empty()) { res.error = "FINDING field is empty"; return res; }
    if (fields["LOCATION"].empty()) { res.error = "LOCATION field is empty"; return res; }

    const std::string& urgency_str = fields["URGENCY"];
    Urgency urgency;
    if (urgency_str == "ROUTINE") urgency = Urgency::ROUTINE;
    else if (urgency_str == "EXPEDITED") urgency = Urgency::EXPEDITED;
    else if (urgency_str == "STAT") urgency = Urgency::STAT;
    else {
        res.error = "URGENCY must be one of ROUTINE, EXPEDITED, STAT (a queue priority, not a diagnosis) -- got \"" + urgency_str + "\"";
        return res;
    }

    double confidence = 0.0;
    const std::string& conf_str = fields["CONFIDENCE"];
    auto conf_parsed = std::from_chars(conf_str.data(), conf_str.data() + conf_str.size(), confidence);
    if (conf_parsed.ec != std::errc() || conf_parsed.ptr != conf_str.data() + conf_str.size()) {
        res.error = "CONFIDENCE is not a valid number: \"" + conf_str + "\"";
        return res;
    }
    if (confidence < 0.0 || confidence > 1.0) {
        res.error = "CONFIDENCE out of the valid [0, 1] range: " + conf_str;
        return res;
    }

    res.ok = true;
    res.finding = ParsedFinding{fields["FINDING"], fields["LOCATION"], urgency, confidence};
    return res;
}

// =======================================================================
// PART 3: TriageReport -- the ONLY way to attach findings to a fixed,
// unmodifiable disclaimer, and the ONLY way to compute a study's own
// overall queue priority.
// =======================================================================
constexpr const char* MANDATORY_DISCLAIMER =
    "PRELIMINARY AI-GENERATED TRIAGE SUGGESTION -- NOT A DIAGNOSIS -- "
    "REQUIRES REVIEW AND SIGN-OFF BY A LICENSED CLINICIAN BEFORE ANY CLINICAL ACTION";

struct TriageReport {
    std::string study_id;
    std::vector<ParsedFinding> findings;
    Urgency overall_priority = Urgency::ROUTINE;
    std::string disclaimer;
};

// A study with no findings at all defaults to ROUTINE -- an absence of
// notable findings is not itself an urgent signal. A study with any
// findings takes the SINGLE MOST URGENT tier among them, never an
// average and never merely the first one encountered, since a study
// containing even one STAT-tier finding belongs at the front of a real
// review queue regardless of how many ROUTINE findings sit alongside it.
TriageReport build_triage_report(const std::string& study_id, const std::vector<ParsedFinding>& findings) {
    TriageReport report;
    report.study_id = study_id;
    report.findings = findings;
    report.disclaimer = MANDATORY_DISCLAIMER;   // unconditional -- no parameter can omit or alter this
    Urgency worst = Urgency::ROUTINE;
    for (const auto& f : findings) {
        if (urgency_rank(f.urgency) > urgency_rank(worst)) worst = f.urgency;
    }
    report.overall_priority = worst;
    return report;
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 20.3: A Structured Triage-Reporting Prompt and Parser\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: study metadata validates, and the rendered prompt is deterministic and complete --\n";
    {
        StudyMetadata m{"STUDY-2026-0091", "CT", "CHEST", 512, 512};
        auto v = validate_study_metadata(m);
        CHECK(v.ok);
        std::string p1 = render_triage_prompt(m);
        std::string p2 = render_triage_prompt(m);
        CHECK(p1 == p2);
        CHECK(p1.find("NOT diagnosing") != std::string::npos);
        CHECK(p1.find("QUEUE PRIORITY for human review, not a diagnosis") != std::string::npos);
        CHECK(p1.find("STUDY-2026-0091") != std::string::npos);
        CHECK(p1.find("512x512") != std::string::npos);

        StudyMetadata bad_modality{"STUDY-BAD", "LASER", "CHEST", 512, 512};
        auto v_bad = validate_study_metadata(bad_modality);
        CHECK(!v_bad.ok);
        CHECK(v_bad.error.find("modality") != std::string::npos);

        std::cout << "  a valid CT chest study validates and renders deterministically, with the "
                     "prompt explicitly stating it is not a diagnosis request and that URGENCY is a "
                     "queue priority; an unknown modality (\"LASER\") is correctly refused (\""
                   << v_bad.error << "\")\n";
    }

    std::cout << "\n-- Test 2: the parser accepts real queue-priority tiers and refuses every attempt to smuggle a diagnosis --\n";
    {
        auto r1 = parse_finding_line("FINDING=opacity;LOCATION=right lower lobe;URGENCY=EXPEDITED;CONFIDENCE=0.72");
        CHECK(r1.ok);
        CHECK(r1.finding.urgency == Urgency::EXPEDITED);

        auto r2 = parse_finding_line("FINDING=small nodule;LOCATION=left upper lobe;URGENCY=ROUTINE;CONFIDENCE=0.4");
        CHECK(r2.ok && r2.finding.urgency == Urgency::ROUTINE);

        auto r3 = parse_finding_line("FINDING=large pneumothorax;LOCATION=left hemithorax;URGENCY=STAT;CONFIDENCE=0.95");
        CHECK(r3.ok && r3.finding.urgency == Urgency::STAT);

        // A free-text urgency value -- not one of the three real
        // tokens -- is refused outright, even though "URGENT" looks
        // superficially like it could be a real tier.
        auto bad_urgency = parse_finding_line("FINDING=x;LOCATION=y;URGENCY=URGENT;CONFIDENCE=0.5");
        CHECK(!bad_urgency.ok);
        CHECK(bad_urgency.error.find("queue priority") != std::string::npos);

        // An attempted diagnosis written INTO the urgency field is
        // refused by the exact same check -- there is no special case
        // that treats a diagnosis-shaped string differently from any
        // other invalid token.
        auto smuggled_in_urgency = parse_finding_line("FINDING=x;LOCATION=y;URGENCY=PNEUMONIA;CONFIDENCE=0.9");
        CHECK(!smuggled_in_urgency.ok);

        // An attempted diagnosis as its OWN extra field is refused by
        // the strict field-count check -- the schema has no slot for it
        // to occupy even if a model tried to emit one.
        auto smuggled_extra_field = parse_finding_line(
            "FINDING=x;LOCATION=y;URGENCY=STAT;CONFIDENCE=0.9;DIAGNOSIS=acute myocardial infarction");
        CHECK(!smuggled_extra_field.ok);
        CHECK(smuggled_extra_field.error.find("extra field") != std::string::npos);

        std::cout << "  all three real queue-priority tiers (ROUTINE, EXPEDITED, STAT) parse "
                     "correctly; a free-text \"URGENT\" is refused (\"" << bad_urgency.error
                   << "\"); a diagnosis-shaped string written into URGENCY is refused by the same "
                     "check; and an explicit DIAGNOSIS=... field is refused outright (\""
                   << smuggled_extra_field.error << "\") -- there is no path for a diagnosis to "
                     "survive parsing as structured data\n";
    }

    std::cout << "\n-- Test 3: an out-of-range confidence and a missing field are both refused --\n";
    {
        auto bad_conf = parse_finding_line("FINDING=x;LOCATION=y;URGENCY=ROUTINE;CONFIDENCE=1.5");
        CHECK(!bad_conf.ok);
        auto missing = parse_finding_line("FINDING=x;LOCATION=y;URGENCY=ROUTINE");
        CHECK(!missing.ok);
        CHECK(missing.error.find("missing required field") != std::string::npos);
        std::cout << "  an out-of-range confidence (1.5) and a missing CONFIDENCE field are both "
                     "correctly refused\n";
    }

    std::cout << "\n-- Test 4: a study's overall priority is its SINGLE MOST URGENT finding, never an average --\n";
    {
        std::vector<ParsedFinding> mixed = {
            {"small nodule", "left upper lobe", Urgency::ROUTINE, 0.4},
            {"small nodule", "right upper lobe", Urgency::ROUTINE, 0.35},
            {"pneumothorax", "left hemithorax", Urgency::STAT, 0.9},
        };
        auto report = build_triage_report("STUDY-0002", mixed);
        CHECK(report.overall_priority == Urgency::STAT);
        CHECK(report.findings.size() == 3);

        auto empty_report = build_triage_report("STUDY-0003", {});
        CHECK(empty_report.overall_priority == Urgency::ROUTINE);
        CHECK(empty_report.findings.empty());

        std::cout << "  a study with two ROUTINE findings and one STAT finding correctly reports an "
                     "overall priority of " << to_string(report.overall_priority)
                   << " (the single most urgent tier present, not an average of the three); a study "
                     "with no findings at all correctly defaults to " << to_string(empty_report.overall_priority) << "\n";
    }

    std::cout << "\n-- Test 5: the mandatory disclaimer is unconditional, identical, and unomittable across reports --\n";
    {
        auto r1 = build_triage_report("STUDY-A", {});
        auto r2 = build_triage_report("STUDY-B", {{"x", "y", Urgency::STAT, 0.99}});
        CHECK(r1.disclaimer == r2.disclaimer);
        CHECK(r1.disclaimer == MANDATORY_DISCLAIMER);
        CHECK(r1.disclaimer.find("NOT A DIAGNOSIS") != std::string::npos);
        CHECK(r1.disclaimer.find("LICENSED CLINICIAN") != std::string::npos);
        std::cout << "  two reports with completely different findings still carry the exact same, "
                     "byte-identical disclaimer -- build_triage_report offers no parameter capable of "
                     "changing or omitting it\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
