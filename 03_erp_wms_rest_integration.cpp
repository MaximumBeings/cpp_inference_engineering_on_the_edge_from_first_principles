// Chapter 19.3 -- Section 19.2's batch processor can now tell, for a
// single photo, whether a shelf slot looks correctly stocked. None of
// that is useful to an actual retailer until it reaches the systems that
// already run the store: an ERP tracking on-hand inventory, a WMS
// tracking warehouse stock, both reachable over the same real interface
// nearly every modern retail backend actually exposes -- a REST API
// exchanging JSON over HTTP's own real methods and status codes.
//
// This section is stated as an INTERFACE-LEVEL abstraction, in the same
// honest voice Chapter 18.5 applied to OPC UA: there is no real ERP or
// WMS endpoint reachable from this environment to integrate against, so
// this section builds its own small, from-scratch JSON serializer and a
// synthetic backend implementing REAL REST semantics -- real HTTP status
// codes, real optimistic-concurrency conflict detection, real malformed-
// request rejection -- standing in for a genuine ERP/WMS system. What
// this section does NOT claim is compliance with any one real vendor's
// actual API shape (SAP, NetSuite, and a dozen others each expose their
// own real, incompatible endpoint conventions); what it DOES teach
// faithfully is the general REST discipline every one of those real
// systems' own integrations still has to get right.
//
// The one piece of REST discipline this section centers on is
// optimistic concurrency: a real inventory record updated by an ERP
// integration is also being updated by cashiers ringing up sales, other
// stores' own systems, and human stock takes, all at once. A PUT that
// blindly overwrites whatever is currently stored risks a genuine LOST
// UPDATE -- two concurrent writers each believe they are applying the
// authoritative new value, and whichever one's PUT lands second silently
// erases the other's. This section's own `expected_version` parameter
// (the same real idea as an HTTP `If-Match` header carrying an ETag) is
// how `SyntheticErpBackend::put_stock_level` refuses that silent
// overwrite: a PUT whose caller does not currently hold the record's own
// latest version is rejected with a real `409 Conflict`, never quietly
// applied over a value the caller never actually saw.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_erp_wms_rest_integration.cpp -o 03_erp_wms_rest_integration
// Run:     ./03_erp_wms_rest_integration

#include <cstdint>
#include <iomanip>
#include <iostream>
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
// PART 1: a small, from-scratch JSON value type and serializer -- the
// real subset this section's own request and response bodies need
// (null, bool, integer, double, string, and a string-keyed object with
// insertion order preserved for deterministic output), with real string
// escaping so a SKU id or a ticket description containing a quote,
// backslash, or newline still serializes to valid JSON.
// =======================================================================
enum class JsonType { Null, Bool, Int, Double, String, Object };

struct JsonValue {
    JsonType type = JsonType::Null;
    bool b = false;
    int64_t i = 0;
    double d = 0.0;
    std::string s;
    std::vector<std::pair<std::string, JsonValue>> obj;

    static JsonValue make_bool(bool v) { JsonValue j; j.type = JsonType::Bool; j.b = v; return j; }
    static JsonValue make_int(int64_t v) { JsonValue j; j.type = JsonType::Int; j.i = v; return j; }
    static JsonValue make_double(double v) { JsonValue j; j.type = JsonType::Double; j.d = v; return j; }
    static JsonValue make_string(std::string v) { JsonValue j; j.type = JsonType::String; j.s = std::move(v); return j; }
    static JsonValue make_object(std::vector<std::pair<std::string, JsonValue>> fields) {
        JsonValue j; j.type = JsonType::Object; j.obj = std::move(fields); return j;
    }
};

std::string escape_json_string(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            case '\r': out += "\\r"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    std::ostringstream esc;
                    esc << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
                    out += esc.str();
                } else {
                    out += c;
                }
        }
    }
    return out;
}

std::string serialize_json(const JsonValue& v) {
    switch (v.type) {
        case JsonType::Null: return "null";
        case JsonType::Bool: return v.b ? "true" : "false";
        case JsonType::Int: return std::to_string(v.i);
        case JsonType::Double: {
            std::ostringstream out;
            out << v.d;
            return out.str();
        }
        case JsonType::String: return "\"" + escape_json_string(v.s) + "\"";
        case JsonType::Object: {
            std::ostringstream out;
            out << "{";
            for (size_t i = 0; i < v.obj.size(); ++i) {
                if (i) out << ",";
                out << "\"" << escape_json_string(v.obj[i].first) << "\":" << serialize_json(v.obj[i].second);
            }
            out << "}";
            return out.str();
        }
    }
    return "null";
}

// =======================================================================
// PART 2: real HTTP method and status modeling -- real, standard status
// codes (200, 201, 400, 404, 409), not invented placeholders.
// =======================================================================
enum class HttpStatus : int {
    OK = 200,
    Created = 201,
    BadRequest = 400,
    NotFound = 404,
    Conflict = 409,
};

struct HttpResponse {
    HttpStatus status;
    JsonValue body;
};

// =======================================================================
// PART 3: SyntheticErpBackend -- a synthetic REST endpoint implementing
// real REST semantics, standing in for a genuine ERP/WMS system this
// environment has no way to reach. See the header note above for why
// optimistic concurrency (expected_version) is this section's own real
// centerpiece.
// =======================================================================
struct StockRecord {
    int64_t quantity = 0;
    uint64_t version = 0;
};

class SyntheticErpBackend {
public:
    HttpResponse get_stock_level(const std::string& store_id, const std::string& sku_id) const {
        auto it = records_.find(key(store_id, sku_id));
        if (it == records_.end()) {
            return HttpResponse{HttpStatus::NotFound,
                                 JsonValue::make_object({{"error", JsonValue::make_string("no stock record for this store/sku")}})};
        }
        return HttpResponse{HttpStatus::OK, JsonValue::make_object({
            {"store_id", JsonValue::make_string(store_id)},
            {"sku_id", JsonValue::make_string(sku_id)},
            {"quantity", JsonValue::make_int(it->second.quantity)},
            {"version", JsonValue::make_int(static_cast<int64_t>(it->second.version))},
        })};
    }

    // Sets the ABSOLUTE quantity for a (store, sku) pair. `expected_version`
    // is the caller's own belief about the record's CURRENT version --
    // 0 for "I believe no record exists yet." A caller whose belief does
    // not match the record's real current version is refused with a real
    // 409 Conflict rather than silently overwriting a value it never
    // actually observed; the stored record is left completely untouched
    // by a rejected write. A malformed request -- an empty sku_id, or a
    // negative quantity, which no real on-hand inventory count can be --
    // is refused with a real 400 Bad Request, also leaving any existing
    // record untouched.
    HttpResponse put_stock_level(const std::string& store_id, const std::string& sku_id,
                                  int64_t quantity, uint64_t expected_version) {
        if (sku_id.empty()) {
            return HttpResponse{HttpStatus::BadRequest,
                                 JsonValue::make_object({{"error", JsonValue::make_string("sku_id must not be empty")}})};
        }
        if (quantity < 0) {
            return HttpResponse{HttpStatus::BadRequest,
                                 JsonValue::make_object({{"error", JsonValue::make_string("quantity must not be negative")}})};
        }
        const std::string k = key(store_id, sku_id);
        auto it = records_.find(k);
        uint64_t current_version = (it == records_.end()) ? 0 : it->second.version;
        if (expected_version != current_version) {
            return HttpResponse{HttpStatus::Conflict, JsonValue::make_object({
                {"error", JsonValue::make_string("version mismatch -- record was modified since it was last read")},
                {"current_version", JsonValue::make_int(static_cast<int64_t>(current_version))},
            })};
        }
        StockRecord rec{quantity, current_version + 1};
        records_[k] = rec;
        return HttpResponse{HttpStatus::OK, JsonValue::make_object({
            {"store_id", JsonValue::make_string(store_id)},
            {"sku_id", JsonValue::make_string(sku_id)},
            {"quantity", JsonValue::make_int(quantity)},
            {"version", JsonValue::make_int(static_cast<int64_t>(rec.version))},
        })};
    }

    // Always creates a genuinely NEW ticket -- unlike put_stock_level,
    // there is no meaningful "expected version" for a POST that creates
    // a new resource each time it is called, which is exactly the real
    // REST distinction between an idempotent PUT and a non-idempotent
    // POST this section's own self-tests check directly.
    HttpResponse create_shrinkage_ticket(const std::string& store_id, const std::string& sku_id,
                                          const std::string& description) {
        if (description.empty()) {
            return HttpResponse{HttpStatus::BadRequest,
                                 JsonValue::make_object({{"error", JsonValue::make_string("description must not be empty")}})};
        }
        ++next_ticket_seq_;
        std::ostringstream ticket_id;
        ticket_id << "TCKT-" << std::setw(6) << std::setfill('0') << next_ticket_seq_;
        tickets_.push_back(Ticket{ticket_id.str(), store_id, sku_id, description});
        return HttpResponse{HttpStatus::Created, JsonValue::make_object({
            {"ticket_id", JsonValue::make_string(ticket_id.str())},
            {"store_id", JsonValue::make_string(store_id)},
            {"sku_id", JsonValue::make_string(sku_id)},
        })};
    }

    size_t ticket_count() const { return tickets_.size(); }

private:
    struct Ticket { std::string ticket_id, store_id, sku_id, description; };
    static std::string key(const std::string& store_id, const std::string& sku_id) { return store_id + "|" + sku_id; }

    std::unordered_map<std::string, StockRecord> records_;
    std::vector<Ticket> tickets_;
    uint64_t next_ticket_seq_ = 0;
};

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 19.3: REST Integration into ERP/WMS Systems\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: the JSON serializer escapes real special characters and preserves field order --\n";
    {
        std::string tricky = "line1\nquote:\"back\\slash";
        std::string escaped = escape_json_string(tricky);
        CHECK(escaped.find('\n') == std::string::npos);   // the raw newline must not survive unescaped
        CHECK(escaped.find("\\n") != std::string::npos);
        CHECK(escaped.find("\\\"") != std::string::npos);
        CHECK(escaped.find("\\\\") != std::string::npos);

        JsonValue obj = JsonValue::make_object({
            {"a", JsonValue::make_int(1)},
            {"b", JsonValue::make_string("x")},
            {"c", JsonValue::make_bool(true)},
        });
        std::string rendered = serialize_json(obj);
        CHECK(rendered == "{\"a\":1,\"b\":\"x\",\"c\":true}");
        std::cout << "  a string with a newline, a quote, and a backslash escapes to \"" << escaped
                   << "\"; a 3-field object serializes to the exact expected string with field order preserved: "
                   << rendered << "\n";
    }

    std::cout << "\n-- Test 2: GET on an unknown record is a real 404, and a first PUT creates it --\n";
    {
        SyntheticErpBackend backend;
        auto miss = backend.get_stock_level("store-042", "SKU-1001");
        CHECK(miss.status == HttpStatus::NotFound);

        auto created = backend.put_stock_level("store-042", "SKU-1001", 50, /*expected_version=*/0);
        CHECK(created.status == HttpStatus::OK);
        auto found = backend.get_stock_level("store-042", "SKU-1001");
        CHECK(found.status == HttpStatus::OK);
        CHECK(found.body.obj[2].second.i == 50);   // quantity field
        CHECK(found.body.obj[3].second.i == 1);    // version field, now 1

        std::cout << "  GET before any record exists returns 404; a PUT with expected_version=0 "
                     "correctly creates the record at version 1 with quantity 50\n";
    }

    std::cout << "\n-- Test 3: setting the same quantity again with the current version is idempotent in effect --\n";
    {
        SyntheticErpBackend backend;
        backend.put_stock_level("store-042", "SKU-1001", 50, 0);   // version becomes 1
        auto repeat = backend.put_stock_level("store-042", "SKU-1001", 50, 1);   // same quantity, correct current version
        CHECK(repeat.status == HttpStatus::OK);
        auto after = backend.get_stock_level("store-042", "SKU-1001");
        CHECK(after.body.obj[2].second.i == 50);   // the business-observable quantity is unchanged
        CHECK(after.body.obj[3].second.i == 2);    // the bookkeeping version still advances -- it is not business state
        std::cout << "  PUTting the same quantity (50) a second time, using the version the first PUT "
                     "returned, succeeds and leaves the observable quantity unchanged at 50 -- the "
                     "version counter itself advances to 2, but that is bookkeeping, not the business "
                     "value a retailer's own inventory dashboard would show\n";
    }

    std::cout << "\n-- Test 4: a stale expected_version is refused with 409, and never applied --\n";
    {
        SyntheticErpBackend backend;
        backend.put_stock_level("store-042", "SKU-1001", 50, 0);   // version becomes 1
        auto stale = backend.put_stock_level("store-042", "SKU-1001", 999, /*expected_version=*/0);   // stale: real version is 1
        CHECK(stale.status == HttpStatus::Conflict);
        auto after = backend.get_stock_level("store-042", "SKU-1001");
        CHECK(after.body.obj[2].second.i == 50);   // the stale write's quantity (999) must NOT have landed
        CHECK(after.body.obj[3].second.i == 1);    // version must still be 1, not bumped by the rejected write
        std::cout << "  a PUT with a stale expected_version (0, when the record's real version is "
                     "already 1) is correctly rejected with 409 Conflict, and the record's stored "
                     "quantity remains 50 -- the rejected write's own quantity (999) never landed, "
                     "which is exactly the lost-update this section's optimistic concurrency exists "
                     "to prevent\n";
    }

    std::cout << "\n-- Test 5: malformed requests are refused with 400, and never change stored state --\n";
    {
        SyntheticErpBackend backend;
        backend.put_stock_level("store-042", "SKU-1001", 50, 0);

        auto empty_sku = backend.put_stock_level("store-042", "", 10, 0);
        CHECK(empty_sku.status == HttpStatus::BadRequest);

        auto negative_qty = backend.put_stock_level("store-042", "SKU-1001", -5, 1);
        CHECK(negative_qty.status == HttpStatus::BadRequest);

        auto after = backend.get_stock_level("store-042", "SKU-1001");
        CHECK(after.body.obj[2].second.i == 50);   // untouched by either rejected request
        CHECK(after.body.obj[3].second.i == 1);

        std::cout << "  an empty sku_id and a negative quantity are both refused with 400 Bad Request, "
                     "and the existing record's quantity (50) and version (1) are completely "
                     "unaffected by either rejected request\n";
    }

    std::cout << "\n-- Test 6: ticket creation is a real, non-idempotent POST -- each call makes a new ticket --\n";
    {
        SyntheticErpBackend backend;
        auto t1 = backend.create_shrinkage_ticket("store-042", "SKU-1001", "3 consecutive empty-slot observations");
        CHECK(t1.status == HttpStatus::Created);
        auto t2 = backend.create_shrinkage_ticket("store-042", "SKU-1001", "3 consecutive empty-slot observations");
        CHECK(t2.status == HttpStatus::Created);
        CHECK(t1.body.obj[0].second.s != t2.body.obj[0].second.s);   // two DIFFERENT ticket ids from identical inputs
        CHECK(backend.ticket_count() == 2);

        auto bad_ticket = backend.create_shrinkage_ticket("store-042", "SKU-1001", "");
        CHECK(bad_ticket.status == HttpStatus::BadRequest);
        CHECK(backend.ticket_count() == 2);   // the rejected request created no ticket

        std::cout << "  two POSTs with IDENTICAL inputs produce two DIFFERENT ticket ids (\""
                   << t1.body.obj[0].second.s << "\" and \"" << t2.body.obj[0].second.s
                   << "\") -- unlike put_stock_level's idempotent PUT, a POST is not expected to "
                     "collapse repeated calls into one outcome; an empty description is refused with "
                     "400 and correctly creates no ticket at all\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
