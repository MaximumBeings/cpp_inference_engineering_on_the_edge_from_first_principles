// Chapter 18.5 -- Section 18.4 gave every inspected part a durable,
// queryable record in this line's own local database, but a real
// factory line does not learn what happened from a database another
// station has to go query -- it learns from its SCADA/MES system,
// which itself learns from the plant floor over OPC UA: the vendor-
// neutral, IEC-62541-standardized protocol that lets a PLC, an HMI, a
// historian, and a SCADA supervisor all read and write the SAME typed,
// named "nodes" (a tag for the current disposition, a tag for today's
// reject count, and so on) without every device needing to speak every
// other device's proprietary wire format.
//
// This section is explicitly an INTERFACE-LEVEL abstraction of OPC UA's
// real data and service model, not a real UA-TCP wire-protocol stack:
//   - Real, unmodified OPC UA concepts: NodeId addressing (a namespace
//     index plus a string identifier, exactly the "ns=2;s=..." form a
//     real UA client uses), a small typed Variant, the Read / Write /
//     Subscribe services, and real OPC UA status code CONSTANTS (Good,
//     BadNodeIdUnknown, BadTypeMismatch -- taken from the actual OPC
//     Foundation status code table, not invented for this book).
//   - Stated simplification: there is no real UA-TCP byte-level wire
//     encoding here (no OPC UA binary or XML envelope, no real network
//     socket), because this environment has no real PLC or SCADA
//     endpoint to verify wire-level interoperability against, and a
//     byte-exact encoding nobody can check against a real UA stack
//     would be worse than useless -- it would be a plausible-looking
//     lie. What IS real and fully verifiable here is the SEMANTICS a
//     real OPC UA server must get right: refuse to invent nodes on an
//     unknown write, refuse a type-mismatched write, and only notify
//     subscribers when a value actually CHANGES (the real UA notion of
//     a "deadband" collapsed to its simplest form: any change at all).
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 05_opcua_integration.cpp -o 05_opcua_integration
// Run:     ./05_opcua_integration

#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: NodeId -- real OPC UA addressing, collapsed to the one form
// this book's line actually needs: a numeric namespace index plus a
// string identifier, exactly the "ns=<index>;s=<identifier>" a real UA
// client would use to name a tag.
// =======================================================================
struct NodeId {
    uint16_t namespace_index = 0;
    std::string identifier;

    bool operator==(const NodeId& other) const {
        return namespace_index == other.namespace_index && identifier == other.identifier;
    }

    std::string to_string() const {
        return "ns=" + std::to_string(namespace_index) + ";s=" + identifier;
    }
};

struct NodeIdHash {
    size_t operator()(const NodeId& id) const {
        // A simple, deterministic combination -- this is a hash for an
        // in-process unordered_map, not a cryptographic or cross-
        // process-stable hash, so no claim is made beyond "equal
        // NodeIds hash equal, in this one process."
        size_t h1 = std::hash<uint16_t>{}(id.namespace_index);
        size_t h2 = std::hash<std::string>{}(id.identifier);
        return h1 ^ (h2 + 0x9e3779b9u + (h1 << 6) + (h1 >> 2));
    }
};

// =======================================================================
// PART 2: Variant -- the small set of scalar types this line's tags
// actually need. Real OPC UA's Variant supports a much larger type
// matrix (arrays, matrices, ~25 built-in scalar types); this is the
// honest subset this chapter's line uses, not a claim of full coverage.
// =======================================================================
using Variant = std::variant<int64_t, double, std::string, bool>;

// =======================================================================
// PART 3: StatusCode -- real OPC UA status code constants (from the
// OPC Foundation's own published status code table), not invented
// values. Only the three this section's semantics actually produce.
// =======================================================================
enum class StatusCode : uint32_t {
    Good              = 0x00000000,
    BadNodeIdUnknown  = 0x80340000,
    BadTypeMismatch   = 0x80740000,
};

std::string to_string(StatusCode code) {
    switch (code) {
        case StatusCode::Good:             return "Good";
        case StatusCode::BadNodeIdUnknown: return "BadNodeIdUnknown";
        case StatusCode::BadTypeMismatch:  return "BadTypeMismatch";
    }
    return "Unknown";
}

// =======================================================================
// PART 4: OpcUaServer -- an address space (NodeId -> typed value) plus
// the Read / Write / Subscribe services a real UA server exposes, with
// the two refuse-loudly rules a real server must enforce and this one
// does not skip for convenience: a write to a node that was never
// added is refused, never silently auto-created, and a write whose
// value's type does not match the node's own declared type is refused,
// never silently coerced.
// =======================================================================
struct ReadResult {
    StatusCode status;
    Variant value;
};

struct Notification {
    NodeId node_id;
    Variant value;
};

class OpcUaServer {
public:
    // Registers a node with its initial value; the value's own
    // std::variant index becomes that node's permanent declared type.
    void add_node(const NodeId& id, Variant initial_value) {
        NodeEntry entry;
        entry.value = std::move(initial_value);
        entry.declared_type_index = entry.value.index();
        entry.subscribed = false;
        nodes_.insert_or_assign(id, std::move(entry));
    }

    ReadResult read(const NodeId& id) const {
        auto it = nodes_.find(id);
        if (it == nodes_.end()) {
            return ReadResult{StatusCode::BadNodeIdUnknown, Variant{int64_t{0}}};
        }
        return ReadResult{StatusCode::Good, it->second.value};
    }

    // Refuses an unknown node (never creates one on a caller's behalf)
    // and refuses a type-mismatched write (never coerces). Only when
    // the write is accepted AND the node's value actually changes AND
    // the node has a subscriber does a notification get queued -- a
    // rewrite of the SAME value produces no notification, matching a
    // real UA server's deadband-filtered reporting.
    StatusCode write(const NodeId& id, const Variant& new_value) {
        auto it = nodes_.find(id);
        if (it == nodes_.end()) {
            return StatusCode::BadNodeIdUnknown;
        }
        NodeEntry& entry = it->second;
        if (new_value.index() != entry.declared_type_index) {
            return StatusCode::BadTypeMismatch;
        }
        const bool changed = !(entry.value == new_value);
        entry.value = new_value;
        if (changed && entry.subscribed) {
            pending_notifications_.push_back(Notification{id, new_value});
        }
        return StatusCode::Good;
    }

    // Returns false for an unknown node -- a real UA client cannot
    // subscribe to a monitored item on a node the server has never
    // heard of, and this server does not pretend otherwise.
    bool subscribe(const NodeId& id) {
        auto it = nodes_.find(id);
        if (it == nodes_.end()) return false;
        it->second.subscribed = true;
        return true;
    }

    // Drains and returns every notification queued since the last
    // call; a second immediate call returns empty, exactly like a real
    // UA subscription's publish queue after it has been serviced.
    std::vector<Notification> take_notifications() {
        std::vector<Notification> out;
        out.swap(pending_notifications_);
        return out;
    }

    size_t node_count() const { return nodes_.size(); }

private:
    struct NodeEntry {
        Variant value;
        size_t declared_type_index = 0;
        bool subscribed = false;
    };
    std::unordered_map<NodeId, NodeEntry, NodeIdHash> nodes_;
    std::vector<Notification> pending_notifications_;
};

// =======================================================================
// PART 5: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 18.5: OPC UA Integration into the MES/SCADA Line\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: NodeId equality and hashing are well-defined --\n";
    {
        NodeId a{2, "Line1.Station3.Disposition"};
        NodeId b{2, "Line1.Station3.Disposition"};
        NodeId c{2, "Line1.Station3.Confidence"};
        NodeId d{3, "Line1.Station3.Disposition"};   // same identifier, different namespace
        CHECK(a == b);
        CHECK(!(a == c));
        CHECK(!(a == d));

        std::unordered_map<NodeId, int, NodeIdHash> m;
        m[a] = 1;
        m[c] = 2;
        CHECK(m.at(b) == 1);   // b hashes/compares equal to a, so it finds a's entry
        CHECK(m.at(c) == 2);
        CHECK(m.size() == 2);
        std::cout << "  NodeId(" << a.to_string() << ") == NodeId(" << b.to_string()
                   << "); a distinct namespace or identifier is a distinct node; "
                     "an unordered_map keyed on NodeId resolves both correctly\n";
    }

    std::cout << "\n-- Test 2: Read/Write semantics refuse unknown nodes and type mismatches --\n";
    {
        OpcUaServer server;
        NodeId disposition_node{2, "Station1.Disposition"};
        NodeId reject_count_node{2, "Line.RejectCount"};

        auto r0 = server.read(disposition_node);
        CHECK(r0.status == StatusCode::BadNodeIdUnknown);

        auto w0 = server.write(disposition_node, Variant{std::string{"REJECT"}});
        CHECK(w0 == StatusCode::BadNodeIdUnknown);   // never auto-creates on write

        server.add_node(disposition_node, Variant{std::string{"ACCEPT"}});
        server.add_node(reject_count_node, Variant{int64_t{0}});

        auto r1 = server.read(disposition_node);
        CHECK(r1.status == StatusCode::Good);
        CHECK(std::get<std::string>(r1.value) == "ACCEPT");

        auto w1 = server.write(disposition_node, Variant{std::string{"REJECT"}});
        CHECK(w1 == StatusCode::Good);
        auto r2 = server.read(disposition_node);
        CHECK(std::get<std::string>(r2.value) == "REJECT");

        // string node, int64_t value: a genuine type mismatch
        auto w2 = server.write(disposition_node, Variant{int64_t{42}});
        CHECK(w2 == StatusCode::BadTypeMismatch);
        auto r3 = server.read(disposition_node);
        CHECK(std::get<std::string>(r3.value) == "REJECT");   // unchanged -- the bad write did not corrupt it

        auto w3 = server.write(reject_count_node, Variant{int64_t{1}});
        CHECK(w3 == StatusCode::Good);
        std::cout << "  unknown-node read/write both correctly refused with BadNodeIdUnknown; "
                     "a string-typed node correctly refuses an int64_t write with BadTypeMismatch "
                     "and keeps its prior value rather than corrupting it\n";
    }

    std::cout << "\n-- Test 3: subscriptions notify only on an actual value change --\n";
    {
        OpcUaServer server;
        NodeId temp_node{2, "Station1.SensorTemp"};
        server.add_node(temp_node, Variant{double{20.0}});

        // Not yet subscribed: a real value change produces no notification.
        server.write(temp_node, Variant{double{21.5}});
        CHECK(server.take_notifications().empty());

        CHECK(server.subscribe(temp_node));
        CHECK(!server.subscribe(NodeId{2, "Station1.DoesNotExist"}));   // unknown node: refuse

        server.write(temp_node, Variant{double{22.0}});   // changed: 21.5 -> 22.0
        auto n1 = server.take_notifications();
        CHECK(n1.size() == 1);
        CHECK(n1[0].node_id == temp_node);
        CHECK(std::get<double>(n1[0].value) == 22.0);

        // Draining again immediately returns empty.
        CHECK(server.take_notifications().empty());

        // Rewriting the SAME value: no notification (deadband-style filtering).
        auto w_same = server.write(temp_node, Variant{double{22.0}});
        CHECK(w_same == StatusCode::Good);   // the write itself still succeeds
        CHECK(server.take_notifications().empty());

        // A genuine change again: exactly one notification, not accumulated
        // duplicates from the unchanged rewrite above.
        server.write(temp_node, Variant{double{18.25}});
        auto n2 = server.take_notifications();
        CHECK(n2.size() == 1);
        CHECK(std::get<double>(n2[0].value) == 18.25);

        std::cout << "  a value change before subscribing produces no notification; after "
                     "subscribing, a real change queues exactly one notification, an unchanged "
                     "rewrite of the SAME value queues none, and take_notifications() drains to "
                     "empty and stays empty until the next real change\n";
    }

    std::cout << "\n-- Test 4: end-to-end line glue -- N parts through the inspection station --\n";
    {
        OpcUaServer server;
        const NodeId disposition_node{2, "Station1.Disposition"};
        const NodeId confidence_node{2, "Station1.Confidence"};
        const NodeId defect_type_node{2, "Station1.DefectType"};
        const NodeId reject_count_node{2, "Line.RejectCount"};

        server.add_node(disposition_node, Variant{std::string{""}});
        server.add_node(confidence_node, Variant{double{0.0}});
        server.add_node(defect_type_node, Variant{std::string{""}});
        server.add_node(reject_count_node, Variant{int64_t{0}});

        for (const NodeId& n : {disposition_node, confidence_node, defect_type_node, reject_count_node}) {
            CHECK(server.subscribe(n));
        }
        // Drain the initial subscribe-time silence (subscribing does not
        // itself emit a notification -- only a subsequent value CHANGE does).
        server.take_notifications();

        struct PartResult {
            std::string disposition;
            double confidence;
            std::string defect_type;
        };
        // A realistic mixed batch: two REJECTs, one REVIEW, one ACCEPT,
        // plus a final part whose reading is IDENTICAL to the immediately
        // preceding one (to prove that repeat produces disposition /
        // confidence / defect_type writes with NO new notifications,
        // since the line's published state does not actually change --
        // note this has to be identical to the PRECEDING part's reading,
        // not any earlier one, since notification-worthiness is judged
        // against the node's current value, not some fixed baseline).
        std::vector<PartResult> parts = {
            {"REJECT", 0.92, "scratch"},
            {"ACCEPT", 0.99, "none"},
            {"REVIEW", 0.60, "dent"},
            {"REJECT", 0.88, "crack"},
            {"ACCEPT", 0.99, "none"},
            {"ACCEPT", 0.99, "none"},   // identical to the immediately preceding reading
        };

        int64_t reject_count = 0;
        size_t total_notifications = 0;
        for (const auto& p : parts) {
            server.write(disposition_node, Variant{p.disposition});
            server.write(confidence_node, Variant{p.confidence});
            server.write(defect_type_node, Variant{p.defect_type});
            if (p.disposition == "REJECT") {
                auto current = server.read(reject_count_node);
                reject_count = std::get<int64_t>(current.value) + 1;
                server.write(reject_count_node, Variant{reject_count});
            }
            auto batch = server.take_notifications();
            total_notifications += batch.size();
        }

        CHECK(reject_count == 2);
        auto final_reject = server.read(reject_count_node);
        CHECK(final_reject.status == StatusCode::Good);
        CHECK(std::get<int64_t>(final_reject.value) == 2);

        // Expected notification count, part by part (each node's own
        // PRIOR value is whatever the previous part last wrote to it,
        // not some fixed baseline):
        //  part1 REJECT 0.92 scratch: disposition("")->"REJECT", confidence(0.0)->0.92,
        //        defect_type("")->"scratch", reject_count(0)->1            = 4
        //  part2 ACCEPT 0.99 none:    disposition->"ACCEPT", confidence->0.99,
        //        defect_type->"none"  (reject_count untouched)            = 3
        //  part3 REVIEW 0.60 dent:    disposition->"REVIEW", confidence->0.60,
        //        defect_type->"dent"                                      = 3
        //  part4 REJECT 0.88 crack:   disposition->"REJECT", confidence->0.88,
        //        defect_type->"crack", reject_count(1)->2                 = 4
        //  part5 ACCEPT 0.99 none:    disposition("REJECT")->"ACCEPT",
        //        confidence(0.88)->0.99, defect_type("crack")->"none"    = 3
        //  part6 ACCEPT 0.99 none:    IDENTICAL to part5's just-written
        //        state -> disposition unchanged, confidence unchanged,
        //        defect_type unchanged -> zero notifications              = 0
        CHECK(total_notifications == 4 + 3 + 3 + 4 + 3 + 0);

        std::cout << "  simulated " << parts.size() << " parts through the station: "
                     "final reject count = " << reject_count << " (2 REJECTs, correctly incremented "
                     "only on REJECT), " << total_notifications << " total SCADA notifications "
                     "queued across the batch, with the final repeated ACCEPT reading (identical to "
                     "an earlier one) correctly producing zero new notifications since nothing about "
                     "the line's published state actually changed\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
