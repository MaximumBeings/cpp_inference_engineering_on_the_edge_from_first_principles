// Chapter 20.4 -- Section 20.1 built a state machine that makes a single
// case's own human sign-off structural rather than advisory. A real
// picture archiving and communication system (PACS) worklist has to
// enforce a second, equally real property across an entire QUEUE of
// cases at once: when two radiologists are both looking at the same
// worklist, at most one of them may actually be working a given case,
// and once someone signs a case off, nothing -- not a network retry, not
// a second reviewer clicking "claim" a moment too late -- may silently
// stomp on that decision. Section 19.3's `put_stock_level` already
// solved a version of this for warehouse inventory with a real
// optimistic-concurrency check (`expected_version`, a real HTTP 409 on a
// stale write); this section reuses exactly that idea, but the
// resource being protected from a lost update is not an inventory
// count, it is which human is accountable for a patient's imaging study.
//
// A note on scope, stated as directly as every other section in this
// chapter states it: this is NOT legal or regulatory advice, this
// section does not model any real PACS vendor's actual wire protocol
// (DICOM's real Modality Worklist and Structured Report services are
// far larger than what is useful to build from scratch here), and
// nothing in this file should be read as a claim that a real clinical
// deployment could be built from this interface alone. What this
// section builds is the one piece worth building from scratch: a
// worklist whose claim-and-sign-off discipline is enforced by the type
// itself, exactly like Section 20.1's `TriageCase`, extended across a
// whole queue of cases with real optimistic concurrency and a real,
// append-only audit trail per case.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_pacs_worklist_integration.cpp -o 04_pacs_worklist_integration
// Run:     ./04_pacs_worklist_integration

#include <algorithm>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: queue priority. Repeated from Section 20.3's own definition --
// every file in this book stands alone -- with the same meaning: a
// worklist ordering hint, never a diagnosis, never averaged.
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

int urgency_rank(Urgency u) { return static_cast<int>(u); }

// =======================================================================
// PART 2: the per-case state machine.
// =======================================================================
enum class WorklistState { UNASSIGNED, ASSIGNED, IN_REVIEW, FINALIZED, REJECTED };

std::string to_string(WorklistState s) {
    switch (s) {
        case WorklistState::UNASSIGNED: return "UNASSIGNED";
        case WorklistState::ASSIGNED: return "ASSIGNED";
        case WorklistState::IN_REVIEW: return "IN_REVIEW";
        case WorklistState::FINALIZED: return "FINALIZED";
        case WorklistState::REJECTED: return "REJECTED";
    }
    return "UNKNOWN";
}

struct AuditEntry {
    int tick = 0;
    std::string actor;
    std::string action;
};

// `WorklistEntry` is one study sitting in the queue. Every field that
// matters for accountability -- who claimed it, what state it is in,
// the full history of who did what and when -- is private, and every
// transition is guarded by two checks together: the state must be the
// one, specific state that transition is allowed from, AND (for every
// transition after the initial claim) the caller must be the SAME
// radiologist who is already accountable for the case. A second
// radiologist cannot begin reviewing, finalize, or reject a case they
// never claimed, even if the first radiologist simply forgot to finish
// it -- exactly the kind of silent hand-off this section exists to rule
// out structurally rather than leave to policy.
class WorklistEntry {
public:
    struct Result { bool ok = false; std::string error; };

    WorklistEntry(std::string accession_number, std::string study_id, Urgency priority)
        : accession_number_(std::move(accession_number)),
          study_id_(std::move(study_id)),
          priority_(priority) {
        log(0, "system", "added to worklist, priority " + to_string(priority_));
    }

    // The one operation with real optimistic concurrency: it is a
    // compare-and-set on `state_ == UNASSIGNED`, not a blind write. Two
    // radiologists racing to claim the same accession number will not
    // both succeed -- the first call to reach this check wins, and the
    // second is refused with an explicit conflict message that names
    // who already holds the case, exactly as Section 19.3's PUT refused
    // a stale write with a named current version rather than silently
    // overwriting it.
    Result claim(const std::string& radiologist_id, int tick) {
        if (radiologist_id.empty()) {
            return fail("claim requires a non-empty radiologist_id");
        }
        if (state_ != WorklistState::UNASSIGNED) {
            return fail("conflict: accession " + accession_number_ + " is already " +
                         to_string(state_) + (claimed_by_ ? (" (claimed by \"" + *claimed_by_ + "\")") : ""));
        }
        claimed_by_ = radiologist_id;
        state_ = WorklistState::ASSIGNED;
        log(tick, radiologist_id, "claimed");
        return ok();
    }

    Result begin_review(const std::string& radiologist_id, int tick) {
        if (state_ != WorklistState::ASSIGNED) {
            return fail("begin_review requires state ASSIGNED, current state is " + to_string(state_));
        }
        if (!claimed_by_ || radiologist_id != *claimed_by_) {
            return fail("begin_review requires the radiologist who claimed the case (\"" +
                         claimed_by_.value_or("<none>") + "\"), not \"" + radiologist_id + "\"");
        }
        state_ = WorklistState::IN_REVIEW;
        log(tick, radiologist_id, "began review");
        return ok();
    }

    // The only path to FINALIZED. Requires IN_REVIEW, requires the same
    // radiologist who has been accountable for the case since `claim`,
    // and requires a real, non-empty report summary -- an empty summary
    // is refused just as loudly as the wrong reviewer, for the same
    // reason Section 20.1 refused an empty clinician_id: a finalized
    // case with nothing recorded about what was decided is not really
    // finalized.
    Result finalize(const std::string& radiologist_id, int tick, const std::string& report_summary) {
        if (state_ != WorklistState::IN_REVIEW) {
            return fail("finalize requires state IN_REVIEW, current state is " + to_string(state_));
        }
        if (!claimed_by_ || radiologist_id != *claimed_by_) {
            return fail("finalize requires the radiologist who claimed the case (\"" +
                         claimed_by_.value_or("<none>") + "\"), not \"" + radiologist_id + "\"");
        }
        if (report_summary.empty()) {
            return fail("finalize requires a non-empty report_summary");
        }
        report_summary_ = report_summary;
        state_ = WorklistState::FINALIZED;
        log(tick, radiologist_id, "FINALIZED: " + report_summary);
        return ok();
    }

    // A radiologist declining to read a case they claimed is a real,
    // expected outcome (equipment mismatch, wrong study routed, needs a
    // subspecialist) -- allowed from either ASSIGNED or IN_REVIEW, never
    // from FINALIZED, and always recorded by name and reason.
    Result reject(const std::string& radiologist_id, int tick, const std::string& reason) {
        if (state_ != WorklistState::ASSIGNED && state_ != WorklistState::IN_REVIEW) {
            return fail("reject requires state ASSIGNED or IN_REVIEW, current state is " + to_string(state_));
        }
        if (!claimed_by_ || radiologist_id != *claimed_by_) {
            return fail("reject requires the radiologist who claimed the case (\"" +
                         claimed_by_.value_or("<none>") + "\"), not \"" + radiologist_id + "\"");
        }
        state_ = WorklistState::REJECTED;
        log(tick, radiologist_id, "rejected: " + reason);
        return ok();
    }

    const std::string& accession_number() const { return accession_number_; }
    const std::string& study_id() const { return study_id_; }
    Urgency priority() const { return priority_; }
    WorklistState state() const { return state_; }
    const std::optional<std::string>& claimed_by() const { return claimed_by_; }
    const std::optional<std::string>& report_summary() const { return report_summary_; }
    const std::vector<AuditEntry>& audit_log() const { return audit_log_; }

private:
    Result fail(std::string error) { return Result{false, std::move(error)}; }
    Result ok() { return Result{true, ""}; }
    void log(int tick, std::string actor, std::string action) {
        audit_log_.push_back(AuditEntry{tick, std::move(actor), std::move(action)});
    }

    std::string accession_number_;
    std::string study_id_;
    Urgency priority_;
    WorklistState state_ = WorklistState::UNASSIGNED;
    std::optional<std::string> claimed_by_;
    std::optional<std::string> report_summary_;
    std::vector<AuditEntry> audit_log_;
};

// =======================================================================
// PART 3: the worklist itself -- a queue of `WorklistEntry` objects,
// insertion-ordered, with a priority view over the still-pending ones.
// =======================================================================
class PacsWorklist {
public:
    struct Result { bool ok = false; std::string error; };

    Result add_case(const std::string& accession_number, const std::string& study_id, Urgency priority) {
        if (accession_number.empty()) {
            return {false, "add_case requires a non-empty accession_number"};
        }
        if (find_index(accession_number) != -1) {
            return {false, "conflict: accession " + accession_number + " already exists on this worklist"};
        }
        entries_.emplace_back(accession_number, study_id, priority);
        return {true, ""};
    }

    WorklistEntry* find(const std::string& accession_number) {
        int idx = find_index(accession_number);
        return idx == -1 ? nullptr : &entries_[static_cast<size_t>(idx)];
    }

    // Pending cases (still UNASSIGNED), ordered STAT first, then
    // EXPEDITED, then ROUTINE -- and, within a single priority tier, in
    // the same order they were added. `std::stable_sort` is not an
    // arbitrary choice here: without it, two STAT studies added minutes
    // apart could be reordered arbitrarily every time this view is
    // recomputed, which is exactly the kind of queue-fairness bug a real
    // worklist cannot tolerate.
    std::vector<std::string> pending_by_priority() const {
        std::vector<const WorklistEntry*> pending;
        for (const auto& e : entries_) {
            if (e.state() == WorklistState::UNASSIGNED) pending.push_back(&e);
        }
        std::stable_sort(pending.begin(), pending.end(), [](const WorklistEntry* a, const WorklistEntry* b) {
            return urgency_rank(a->priority()) > urgency_rank(b->priority());
        });
        std::vector<std::string> result;
        result.reserve(pending.size());
        for (const auto* e : pending) result.push_back(e->accession_number());
        return result;
    }

    size_t size() const { return entries_.size(); }

private:
    int find_index(const std::string& accession_number) const {
        for (size_t i = 0; i < entries_.size(); i++) {
            if (entries_[i].accession_number() == accession_number) return static_cast<int>(i);
        }
        return -1;
    }

    std::vector<WorklistEntry> entries_;
};

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 20.4: PACS Worklist Integration with a Human-in-the-Loop Queue\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: pending cases surface STAT first, then EXPEDITED, then ROUTINE, "
                 "FIFO within a tier --\n";
    {
        PacsWorklist wl;
        auto r1 = wl.add_case("ACC-0001", "STUDY-CHEST-CT", Urgency::ROUTINE);
        auto r2 = wl.add_case("ACC-0002", "STUDY-HEAD-CT", Urgency::STAT);
        auto r3 = wl.add_case("ACC-0003", "STUDY-ABD-XR", Urgency::EXPEDITED);
        auto r4 = wl.add_case("ACC-0004", "STUDY-HEAD-CT-2", Urgency::STAT);
        auto dup = wl.add_case("ACC-0001", "STUDY-DUP", Urgency::STAT);
        CHECK(r1.ok && r2.ok && r3.ok && r4.ok);
        CHECK(!dup.ok);
        CHECK(wl.size() == 4);
        auto order = wl.pending_by_priority();
        std::vector<std::string> expected = {"ACC-0002", "ACC-0004", "ACC-0003", "ACC-0001"};
        CHECK(order == expected);
        std::cout << "  4 cases added (duplicate accession correctly refused); pending order is "
                     "ACC-0002, ACC-0004 (both STAT, insertion order preserved), ACC-0003 "
                     "(EXPEDITED), ACC-0001 (ROUTINE)\n";
    }

    std::cout << "\n-- Test 2: claiming is real optimistic concurrency -- the first radiologist "
                 "to claim a case wins, and a second claim is refused by name, not silently "
                 "overwritten --\n";
    {
        PacsWorklist wl;
        wl.add_case("ACC-0100", "STUDY-CT", Urgency::STAT);
        auto* e = wl.find("ACC-0100");
        auto first = e->claim("dr-alvarez", 1);
        auto second = e->claim("dr-nakamura", 2);
        auto empty_id = e->claim("", 3);
        CHECK(first.ok);
        CHECK(!second.ok);
        CHECK(second.error.find("dr-alvarez") != std::string::npos);
        CHECK(!empty_id.ok);
        CHECK(e->state() == WorklistState::ASSIGNED);
        CHECK(e->claimed_by().value() == "dr-alvarez");
        std::cout << "  dr-alvarez's claim succeeds; dr-nakamura's later claim on the same "
                     "accession is refused and the refusal names dr-alvarez as the current "
                     "holder; an empty radiologist_id is refused outright\n";
    }

    std::cout << "\n-- Test 3: only the radiologist who claimed a case may advance or finalize "
                 "it, and finalize refuses an empty report --\n";
    {
        PacsWorklist wl;
        wl.add_case("ACC-0200", "STUDY-MR", Urgency::EXPEDITED);
        auto* e = wl.find("ACC-0200");
        e->claim("dr-alvarez", 1);
        auto wrong_reviewer = e->begin_review("dr-nakamura", 2);
        auto right_reviewer = e->begin_review("dr-alvarez", 3);
        auto empty_report = e->finalize("dr-alvarez", 4, "");
        auto wrong_finalizer = e->finalize("dr-nakamura", 5, "no acute findings");
        auto real_finalize = e->finalize("dr-alvarez", 6, "no acute findings, follow-up in 6 months");
        CHECK(!wrong_reviewer.ok);
        CHECK(right_reviewer.ok);
        CHECK(!empty_report.ok);
        CHECK(!wrong_finalizer.ok);
        CHECK(real_finalize.ok);
        CHECK(e->state() == WorklistState::FINALIZED);
        CHECK(e->report_summary().value() == "no acute findings, follow-up in 6 months");
        std::cout << "  dr-nakamura cannot begin review or finalize a case claimed by "
                     "dr-alvarez; an empty report summary is refused even from the correct "
                     "reviewer; the real finalize by dr-alvarez succeeds\n";
    }

    std::cout << "\n-- Test 4: a finalized case accepts no further transitions of any kind, and "
                 "rejection is a real, distinct, always-attributed outcome --\n";
    {
        PacsWorklist wl;
        wl.add_case("ACC-0300", "STUDY-CT", Urgency::ROUTINE);
        auto* e1 = wl.find("ACC-0300");
        e1->claim("dr-alvarez", 1);
        e1->begin_review("dr-alvarez", 2);
        e1->finalize("dr-alvarez", 3, "no acute findings");
        auto reclaim = e1->claim("dr-nakamura", 4);
        auto rereview = e1->begin_review("dr-alvarez", 5);
        auto refinalize = e1->finalize("dr-alvarez", 6, "different finding entirely");
        auto reject_after_finalize = e1->reject("dr-alvarez", 7, "changed my mind");
        CHECK(!reclaim.ok && !rereview.ok && !refinalize.ok && !reject_after_finalize.ok);
        CHECK(e1->state() == WorklistState::FINALIZED);
        CHECK(e1->report_summary().value() == "no acute findings");

        wl.add_case("ACC-0301", "STUDY-XR", Urgency::STAT);
        auto* e2 = wl.find("ACC-0301");
        e2->claim("dr-nakamura", 10);
        auto rejection = e2->reject("dr-nakamura", 11, "wrong modality routed, needs re-acquisition");
        CHECK(rejection.ok);
        CHECK(e2->state() == WorklistState::REJECTED);
        std::cout << "  every attempt to reclaim, re-review, re-finalize, or reject a already-"
                     "FINALIZED case is refused and its original report is untouched; a "
                     "separate case is correctly moved to REJECTED with its reason recorded\n";
    }

    std::cout << "\n-- Test 5: the per-case audit trail is complete, ordered, and attributed --\n";
    {
        PacsWorklist wl;
        wl.add_case("ACC-0400", "STUDY-CT", Urgency::EXPEDITED);
        auto* e = wl.find("ACC-0400");
        e->claim("dr-alvarez", 5);
        e->begin_review("dr-alvarez", 6);
        e->finalize("dr-alvarez", 7, "no acute findings");
        const auto& log = e->audit_log();
        CHECK(log.size() == 4);
        CHECK(log[0].actor == "system" && log[0].action.find("added to worklist") != std::string::npos);
        CHECK(log[1].tick == 5 && log[1].actor == "dr-alvarez" && log[1].action == "claimed");
        CHECK(log[2].tick == 6 && log[2].actor == "dr-alvarez" && log[2].action == "began review");
        CHECK(log[3].tick == 7 && log[3].actor == "dr-alvarez" &&
              log[3].action.find("FINALIZED") != std::string::npos);
        std::cout << "  the audit trail for ACC-0400 records all 4 events in order, each with "
                     "the correct tick and actor: system add, dr-alvarez's claim, dr-alvarez's "
                     "review start, and dr-alvarez's finalization\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    if (g_passed == g_tests) {
        std::cout << "ALL CHECKS PASSED\n";
        return 0;
    }
    std::cout << "SOME CHECKS FAILED\n";
    return 1;
}
