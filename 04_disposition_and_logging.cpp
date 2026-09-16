// Chapter 18.4 -- Section 18.3 produced one real, finite hidden state per
// inspected part, but a hidden state is not a decision a factory line can
// act on. A real inspection station needs exactly two more things: a
// rule that turns the vision-language model's own stated verdict and
// confidence into one of three dispositions a conveyor's downstream
// gate actually understands (accept, reject, or route to a human), and
// a durable, queryable record of every decision this line ever made --
// because "why did the line reject part #48213 at 2:14 this morning" is
// a question a real quality engineer eventually asks, and "the model
// said so" is not an answer without a logged confidence and a logged
// defect type to point to.
//
// This section's own SQLite usage deliberately declares SQLite's own
// long-stable C ABI directly, rather than including the system
// `<sqlite3.h>` header. Real edge hardware often ships the SQLite
// RUNTIME library (`libsqlite3.so`) preinstalled as a system dependency
// of something else entirely, without the separate development package
// that provides the header and an unversioned link name -- exactly the
// situation this book's own build machine and real device were both in
// while writing this section. Declaring the handful of functions this
// section actually calls, matching their real, decades-stable
// signatures, and linking directly against the versioned runtime
// library name (`-l:libsqlite3.so.0`) makes this section buildable on
// exactly the kind of minimal edge image it is written for.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_disposition_and_logging.cpp -l:libsqlite3.so.0 -o 04_disposition_and_logging
// Run:     ./04_disposition_and_logging

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

// =======================================================================
// PART 1: SQLite's own real C ABI, declared directly (see header note
// above) rather than pulled in from a system header this book cannot
// assume is installed.
// =======================================================================
extern "C" {
struct sqlite3;
struct sqlite3_stmt;
int sqlite3_open(const char* filename, sqlite3** ppDb);
int sqlite3_close(sqlite3*);
int sqlite3_exec(sqlite3*, const char* sql, int (*callback)(void*, int, char**, char**), void* arg, char** errmsg);
int sqlite3_prepare_v2(sqlite3* db, const char* zSql, int nByte, sqlite3_stmt** ppStmt, const char** pzTail);
int sqlite3_bind_int(sqlite3_stmt*, int, int);
int sqlite3_bind_int64(sqlite3_stmt*, int, long long);
int sqlite3_bind_double(sqlite3_stmt*, int, double);
int sqlite3_bind_text(sqlite3_stmt*, int, const char*, int, void (*)(void*));
int sqlite3_step(sqlite3_stmt*);
int sqlite3_finalize(sqlite3_stmt*);
int sqlite3_column_int(sqlite3_stmt*, int);
long long sqlite3_column_int64(sqlite3_stmt*, int);
double sqlite3_column_double(sqlite3_stmt*, int);
const unsigned char* sqlite3_column_text(sqlite3_stmt*, int);
const char* sqlite3_errmsg(sqlite3*);
void sqlite3_free(void*);
}
constexpr int SQLITE_OK = 0;
constexpr int SQLITE_ROW = 100;
constexpr int SQLITE_DONE = 101;
// The real, documented sentinel SQLite defines for "copy this string
// now, I am not keeping the pointer alive" -- (void(*)(void*))(-1),
// exactly the C-style cast the real sqlite3.h itself uses (a constant
// integer cannot be reinterpret_cast to a pointer at compile time, so
// this section's own declaration matches the real header's own idiom).
using sqlite3_destructor_type = void (*)(void*);
const sqlite3_destructor_type SQLITE_TRANSIENT = (sqlite3_destructor_type)(-1);

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 2: confidence-threshold disposition. The asymmetry here is the
// real content of this section: a FLAGGED defect needs HIGH confidence
// before this station auto-rejects a part outright (a shaky read should
// not waste a good part), while a NOT-flagged reading needs only a
// LOWER confidence bar before this station auto-accepts (silently
// passing an uncertain "no defect" through is the more dangerous
// mistake for a quality gate to make). Anything that does not clear its
// own side of that asymmetric bar is routed to a human, never guessed.
// =======================================================================
enum class Disposition { ACCEPT, REJECT, REVIEW };

std::string to_string(Disposition d) {
    switch (d) {
        case Disposition::ACCEPT: return "ACCEPT";
        case Disposition::REJECT: return "REJECT";
        case Disposition::REVIEW: return "REVIEW";
    }
    return "?";
}

struct InspectionReading {
    bool model_flagged_defect = false;
    double confidence = 0.0;          // this reading's own stated confidence, in [0, 1]
    std::string defect_type = "none";
};

struct DispositionThresholds {
    double reject_threshold = 0.85;   // a FLAGGED reading needs at least this confidence to auto-reject
    double review_threshold = 0.55;   // a NOT-flagged reading needs at least this confidence to auto-accept

    // A misconfigured threshold pair (review >= reject) would make one
    // side of the asymmetry meaningless or inverted; this section
    // refuses to classify anything against thresholds that fail this
    // check, rather than silently producing a disposition from a
    // configuration that does not mean what it claims to.
    bool valid() const { return reject_threshold > review_threshold && reject_threshold <= 1.0 && review_threshold >= 0.0; }
};

std::optional<Disposition> classify(const InspectionReading& r, const DispositionThresholds& t) {
    if (!t.valid()) return std::nullopt;
    if (!r.model_flagged_defect) {
        return (r.confidence >= t.review_threshold) ? Disposition::ACCEPT : Disposition::REVIEW;
    }
    return (r.confidence >= t.reject_threshold) ? Disposition::REJECT : Disposition::REVIEW;
}

// =======================================================================
// PART 3: durable defect logging. Every disposition this station ever
// makes is inserted as one row; nothing here holds an in-memory log
// that a power cycle could lose.
// =======================================================================
struct InspectionRecord {
    int64_t block_id = 0;         // Section 18.1's own GVSP block id for the inspected frame
    int64_t timestamp_ns = 0;
    std::string defect_type;
    double confidence = 0.0;
    Disposition disposition = Disposition::ACCEPT;
};

bool exec_simple(sqlite3* db, const std::string& sql, std::string& err) {
    char* errmsg = nullptr;
    int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &errmsg);
    if (rc != SQLITE_OK) {
        err = errmsg ? errmsg : sqlite3_errmsg(db);
        if (errmsg) sqlite3_free(errmsg);
        return false;
    }
    return true;
}

struct InspectionDb {
    sqlite3* db = nullptr;
    std::string last_error;

    bool open(const std::string& path) {
        if (sqlite3_open(path.c_str(), &db) != SQLITE_OK) {
            last_error = db ? sqlite3_errmsg(db) : "sqlite3_open failed";
            return false;
        }
        const std::string create_sql =
            "CREATE TABLE IF NOT EXISTS inspections ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "block_id INTEGER NOT NULL, "
            "timestamp_ns INTEGER NOT NULL, "
            "defect_type TEXT NOT NULL, "
            "confidence REAL NOT NULL, "
            "disposition TEXT NOT NULL);";
        return exec_simple(db, create_sql, last_error);
    }

    bool log(const InspectionRecord& rec) {
        sqlite3_stmt* stmt = nullptr;
        const char* sql = "INSERT INTO inspections (block_id, timestamp_ns, defect_type, confidence, disposition) "
                           "VALUES (?, ?, ?, ?, ?);";
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            last_error = sqlite3_errmsg(db);
            return false;
        }
        sqlite3_bind_int64(stmt, 1, rec.block_id);
        sqlite3_bind_int64(stmt, 2, rec.timestamp_ns);
        sqlite3_bind_text(stmt, 3, rec.defect_type.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_double(stmt, 4, rec.confidence);
        std::string disp_str = to_string(rec.disposition);
        sqlite3_bind_text(stmt, 5, disp_str.c_str(), -1, SQLITE_TRANSIENT);
        int rc = sqlite3_step(stmt);
        bool ok = (rc == SQLITE_DONE);
        if (!ok) last_error = sqlite3_errmsg(db);
        sqlite3_finalize(stmt);
        return ok;
    }

    // Queries every logged record with the given disposition, ordered
    // by insertion order -- exactly the query a quality engineer runs
    // to pull "every part this line rejected on the 2 AM shift."
    std::vector<InspectionRecord> query_by_disposition(const std::string& disposition) {
        std::vector<InspectionRecord> out;
        sqlite3_stmt* stmt = nullptr;
        const char* sql = "SELECT block_id, timestamp_ns, defect_type, confidence, disposition "
                           "FROM inspections WHERE disposition = ? ORDER BY id ASC;";
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) { last_error = sqlite3_errmsg(db); return out; }
        sqlite3_bind_text(stmt, 1, disposition.c_str(), -1, SQLITE_TRANSIENT);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            InspectionRecord rec;
            rec.block_id = sqlite3_column_int64(stmt, 0);
            rec.timestamp_ns = sqlite3_column_int64(stmt, 1);
            rec.defect_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            rec.confidence = sqlite3_column_double(stmt, 3);
            std::string d = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
            rec.disposition = (d == "ACCEPT") ? Disposition::ACCEPT : (d == "REJECT") ? Disposition::REJECT : Disposition::REVIEW;
            out.push_back(rec);
        }
        sqlite3_finalize(stmt);
        return out;
    }

    int64_t count_all() {
        sqlite3_stmt* stmt = nullptr;
        int64_t n = -1;
        if (sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM inspections;", -1, &stmt, nullptr) == SQLITE_OK) {
            if (sqlite3_step(stmt) == SQLITE_ROW) n = sqlite3_column_int64(stmt, 0);
        }
        sqlite3_finalize(stmt);
        return n;
    }

    ~InspectionDb() { if (db) sqlite3_close(db); }
};

// =======================================================================
// PART 4: self-tests. The logging tests use a REAL temporary SQLite
// database file on disk and the REAL SQLite C API declared above --
// nothing about persistence is mocked -- while keeping every locked
// check anchored to the query RESULTS (specific values, specific
// counts), never to the database file's own bytes, which is exactly the
// same "check the meaning, not the machine-specific artifact" discipline
// this book has applied to timing since Chapter 17.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 18.4: Confidence-Threshold Disposition and SQLite Defect Logging\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: disposition thresholds validate their own configuration --\n";
    {
        DispositionThresholds good;
        CHECK(good.valid());
        DispositionThresholds inverted{0.5, 0.85};   // reject threshold BELOW review threshold: nonsensical
        CHECK(!inverted.valid());
        InspectionReading r{true, 0.9, "scratch"};
        CHECK(!classify(r, inverted).has_value());
        std::cout << "  default thresholds (reject=0.85, review=0.55) valid: yes; an inverted pair "
                     "(reject=0.5 < review=0.85) correctly refused, classify() returns no disposition\n";
    }

    std::cout << "\n-- Test 2: the accept/reject bar is asymmetric between flagged and not-flagged readings --\n";
    {
        DispositionThresholds t;   // reject=0.85, review=0.55
        InspectionReading flagged_high{true, 0.90, "dent"};
        CHECK(classify(flagged_high, t) == Disposition::REJECT);
        InspectionReading flagged_mid{true, 0.70, "dent"};   // above review bar, but below reject bar
        CHECK(classify(flagged_mid, t) == Disposition::REVIEW);
        InspectionReading not_flagged_mid{false, 0.70, "none"};   // SAME 0.70 confidence, not flagged
        CHECK(classify(not_flagged_mid, t) == Disposition::ACCEPT);
        InspectionReading not_flagged_low{false, 0.40, "none"};
        CHECK(classify(not_flagged_low, t) == Disposition::REVIEW);
        std::cout << "  the SAME 0.70 confidence yields REVIEW when flagged as a defect but ACCEPT when "
                     "not flagged -- the asymmetric bar is doing real work, not just thresholding a single number\n";
        std::cout << "  boundary values: flagged @0.90 -> " << to_string(*classify(flagged_high, t))
                   << "; flagged @0.70 -> " << to_string(*classify(flagged_mid, t))
                   << "; not-flagged @0.70 -> " << to_string(*classify(not_flagged_mid, t))
                   << "; not-flagged @0.40 -> " << to_string(*classify(not_flagged_low, t)) << "\n";
    }

    std::cout << "\n-- Test 3: real SQLite round-trip (open, create table, insert, read back) --\n";
    {
        std::string path = "/tmp/ch18_4_test_inspections.sqlite3";
        std::remove(path.c_str());
        InspectionDb logdb;
        CHECK(logdb.open(path));
        CHECK(logdb.count_all() == 0);

        std::vector<InspectionRecord> records = {
            {1001, 1000, "scratch", 0.92, Disposition::REJECT},
            {1002, 2000, "none", 0.98, Disposition::ACCEPT},
            {1003, 3000, "dent", 0.70, Disposition::REVIEW},
            {1004, 4000, "none", 0.40, Disposition::REVIEW},
            {1005, 5000, "crack", 0.95, Disposition::REJECT},
        };
        bool all_logged = true;
        for (const auto& r : records) all_logged &= logdb.log(r);
        CHECK(all_logged);
        CHECK(logdb.count_all() == static_cast<int64_t>(records.size()));

        auto rejects = logdb.query_by_disposition("REJECT");
        CHECK(rejects.size() == 2);
        CHECK(rejects[0].block_id == 1001 && rejects[0].defect_type == "scratch");
        CHECK(rejects[1].block_id == 1005 && rejects[1].defect_type == "crack");

        auto reviews = logdb.query_by_disposition("REVIEW");
        CHECK(reviews.size() == 2);
        CHECK(reviews[0].block_id == 1003);
        CHECK(reviews[1].block_id == 1004);
        CHECK(std::abs(reviews[1].confidence - 0.40) < 1e-9);

        std::cout << "  opened a real SQLite file, logged " << records.size() << " real rows, "
                     "queried back " << rejects.size() << " REJECT and " << reviews.size()
                   << " REVIEW rows, all field values matching exactly what was inserted\n";
        std::remove(path.c_str());
    }

    std::cout << "\n-- Test 4: opening a database at an unwritable path fails loudly, not silently --\n";
    {
        InspectionDb logdb;
        // A directory that does not exist: SQLite's own real open()
        // reports this as a real error rather than creating parent
        // directories on a caller's behalf.
        bool opened = logdb.open("/tmp/ch18_4_nonexistent_dir_xyz/inspections.sqlite3");
        CHECK(!opened);
        CHECK(!logdb.last_error.empty());
        std::cout << "  opening a database inside a nonexistent directory correctly fails "
                     "(error: \"" << logdb.last_error << "\")\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
