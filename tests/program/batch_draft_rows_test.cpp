// batch_draft_rows: how --batch-mtp splits the verify window's rows over the slots (synthetic: no model, no GPU).
#include "strata/program/batch_draft_rows.hpp"

#include <cstdio>
#include <initializer_list>

using namespace strata::program::batch_draft_rows;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

/// A window's slots as batch_step sees them: up to kVerifyMaxT rows over the active slots.
struct Slot {
    bool active;
    int offered;      // drafts this slot has offered so far (0 = no history yet)
    int accepted;     // drafts the target took of those
};

/// The rows each slot gets; the sum is what run_slot_rows will be handed.
struct Plan {
    int rows[8];
    int count;        // slots planned (active ones only)
    int total_rows;
};

/// The policy under test (the header's), wrapped over the plain arrays the test writes by hand.
Plan plan_of(std::initializer_list<Slot> slots, int capacity) {
    Plan p{{}, 0, 0};
    const int n = (int) slots.size();
    bool active[8];
    int offered[8], accepted[8];
    int at = 0;
    for (const Slot& s : slots) {
        active[at] = s.active;
        offered[at] = s.offered;
        accepted[at] = s.accepted;
        if (s.active) ++p.count;                  // the plan counts the slots it plans, as batch_step does
        ++at;
    }
    strata::program::batch_draft_rows::split(p.rows, n, active, offered, accepted, capacity);
    int sum = 0;
    for (int i = 0; i < n; ++i) sum += p.rows[i];
    p.total_rows = sum;
    return p;
}

int main() {
    // Without acceptance history every active slot gets the shape the 0.1.x window had: its token row plus one
    // proposal row.  Nothing changes until a slot has earned a different share.
    {
        const Plan p = plan_of({{true, 0, 0}, {true, 0, 0}, {true, 0, 0}, {true, 0, 0}}, 8);
        CHECK(p.total_rows == 8);
        CHECK(p.rows[0] == 2 && p.rows[1] == 2 && p.rows[2] == 2 && p.rows[3] == 2);
    }
    // Three active slots: the old code still gave every active slot two rows (6 of 8, two rows idle).  The plan
    // may hand the spare rows to slots whose proposals the target has been taking - and only to those.
    {
        const Plan p = plan_of({{true, 100, 58}, {true, 100, 58}, {true, 100, 58}}, 8);
        CHECK(p.total_rows == 8);                        // the idle rows go to work, the window stays full
        CHECK(p.rows[0] >= 2 && p.rows[1] >= 2 && p.rows[2] >= 2);
    }
    // A slot whose proposals are never taken loses its proposal row first: the token row is its floor.
    {
        const Plan high = plan_of({{true, 100, 90}, {true, 100, 0}}, 8);
        CHECK(high.rows[0] >= 3);                        // the accepted-90% slot grows past its two rows
        CHECK(high.rows[1] == 1);                        // the accepted-0% slot is down to its token row
        CHECK(high.rows[0] + high.rows[1] <= 8);
    }
    // Acceptance alone does not buy rows: a slot with one lucky acceptance is not tilted towards like a
    // 90% one - tiny histories read as "unknown", the shape stays the default.
    {
        const Plan p = plan_of({{true, 2, 2}, {true, 0, 0}, {true, 0, 0}, {true, 0, 0}}, 8);
        CHECK(p.rows[0] == 2 && p.rows[1] == 2 && p.rows[2] == 2 && p.rows[3] == 2);
    }
    // Every active slot keeps at least its token row, whatever its acceptance - the slot must make progress
    // (a conversation's slot is its own; starving it for a neighbour's acceptance would stall that session).
    {
        const Plan p = plan_of({{true, 100, 95}, {true, 100, 0}, {true, 100, 0}, {true, 100, 0}}, 8);
        CHECK(p.rows[0] >= 1 && p.rows[1] >= 1 && p.rows[2] >= 1 && p.rows[3] >= 1);
        CHECK(p.rows[0] + p.rows[1] + p.rows[2] + p.rows[3] <= 8);
        CHECK(p.rows[0] > p.rows[1]);                    // and the strong slot still gets the surplus
    }
    // More active slots than one draft each fits: the window's capacity binds, later slots keep only their
    // token row (the rotation batch_step already runs decides which slots make the window).
    {
        const Plan p = plan_of({{true, 100, 0}, {true, 100, 0}, {true, 100, 0}, {true, 100, 0},
                                {true, 100, 0}, {true, 100, 0}, {true, 100, 0}, {true, 100, 0}}, 8);
        int sum = 0;
        for (int i = 0; i < 8; ++i) sum += p.rows[i];
        CHECK(sum <= 8);
        CHECK(p.count == 8);
        for (int i = 0; i < 8; ++i) CHECK(p.rows[i] == 1);
    }
    // Inactive slots take nothing, and an empty window is an empty plan.
    {
        const Plan p = plan_of({{false, 100, 90}, {false, 100, 90}}, 8);
        CHECK(p.count == 0 && p.total_rows == 0);
    }
    // One slot alone: it may use the whole window when its acceptance earns it.
    {
        const Plan strong = plan_of({{true, 100, 95}}, 8);
        CHECK(strong.rows[0] >= 2);
        const Plan unknown = plan_of({{true, 0, 0}}, 8);
        CHECK(unknown.rows[0] == 2);                     // no history: the default two rows, as ever
    }
    // A capacity below two rows per slot still serves every active slot once.
    {
        const Plan p = plan_of({{true, 100, 90}, {true, 100, 90}, {true, 100, 90}}, 3);
        CHECK(p.count == 3);
        CHECK(p.rows[0] + p.rows[1] + p.rows[2] == 3);
    }
    // A small capacity still distributes every row it has: the plan never leaves rows on the table when a
    // strong slot could use them (a capacity of 4 -- --spec 4 -- is not a reason to run 1/1/2 when 2/1/1
    // serves the same count; the exact split follows the shares)
    {
        const Plan p = plan_of({{true, 100, 95}, {true, 100, 0}, {true, 100, 0}}, 4);
        CHECK(p.total_rows == 4);
        CHECK(p.rows[0] == 2);                           // the 95% slot buys the one surplus row
        CHECK(p.rows[1] == 1 && p.rows[2] == 1);         // the refused-everything slots keep their token rows
    }
    if (fails == 0) std::printf("batch_draft_rows_test: ok\n");
    return fails == 0 ? 0 : 1;
}
