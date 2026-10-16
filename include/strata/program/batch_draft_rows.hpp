// include/strata/program/batch_draft_rows.hpp - how --batch-mtp splits the verify window's rows over the slots.
//
// A --batch-mtp verify window holds kVerifyMaxT (8) rows.  Until now every active slot took exactly two of them
// (its token row plus one proposal), and a window with fewer than four active slots left rows idle.  The slots
// differ, though: a slot whose proposals the target has been taking is worth more rows than one whose proposals
// are always refused - a refused proposal costs its row in the window and the drafter's step for nothing.
//
// split() hands the window's rows out.  Every active slot first keeps its token row - a slot's row is the
// conversation's progress; starving it for a neighbour's acceptance would stall that session.  The proposal rows
// go by the slot's record: a share read from too few offers is "unknown" and keeps the one proposal row the
// 0.1.x window gave every slot; a slot the target has been accepting from buys surplus rows first, a slot whose
// proposals are all refused loses its proposal row (its drafts are noise); one lucky acceptance out of two is
// a 100% share and must not tilt the plan.  A row on top of many is worth less than a first spare row.  The rows
// over the active slots sum to at most `capacity`; the order the caller hands rows to the verifier is unchanged.
//
// Pure (no GPU, no environment read): generate.cpp passes in what it knows.
#pragma once

#include <cstdint>

namespace strata::program::batch_draft_rows {

/// Offers below this read as "no usable history": the slot keeps its default proposal row and buys no surplus
/// rows.  (The acceptance study's steady shares sat far above chance over hundreds of windows; 8 offers are
/// enough to tell a drafter that works from one that guesses.)
inline constexpr int kMinOffers = 8;

/// The default shape: every active slot's token row plus one proposal row, the window the 0.1.x code ran.
inline constexpr int kBaseRows = 2;

/// How the window's `capacity` rows go to `count` slots.  `active[i]` marks the slots planned, in the order the
/// caller hands rows to the verifier (batch_step's rotation order - the plan never reorders).  `offered[i]` /
/// `accepted[i]` are the slot's proposal tallies so far (the BSlot counters; 0 offers = no history).  `rows[i]`
/// gets 0 for an inactive slot, at least 1 for an active one, and the rows over all slots sum to at most
/// `capacity`.
inline void split(int* rows, int count, const bool* active, const int* offered, const int* accepted,
                  int capacity) {
    for (int i = 0; i < count; ++i) rows[i] = active[i] ? 1 : 0;   // the floor: every active slot progresses
    int used = 0;
    for (int i = 0; i < count; ++i) used += rows[i];
    for (int i = 0; i < count && used < capacity; ++i) {           // the default shape, one proposal row each,
        if (active[i] && !(offered[i] >= kMinOffers && accepted[i] == 0)) {   // a refused-everything slot: none
            ++rows[i];
            ++used;
        }
    }
    while (used < capacity) {                                      // the surplus: best marginal value first
        int best = -1;
        int64_t best_value = -1;
        for (int i = 0; i < count; ++i) {
            if (!active[i] || rows[i] < kBaseRows) continue;       // first row on top of the default shape
            const bool known = offered[i] >= kMinOffers;
            if (!known) continue;                                  // unknown histories buy no surplus rows
            // a stronger share first, and the next row is worth less the more the slot already has:
            // value = share^2 / rows on a 1..1023 scale - exact arithmetic, and a barely-over-chance
            // share still ranks below a strong one at many rows
            const int64_t share = (int64_t) (accepted[i] + 1) * 1024 / (int64_t) (offered[i] + 2);
            const int64_t value = share * share / (int64_t) rows[i];
            if (value > best_value) { best_value = value; best = i; }
        }
        if (best < 0) break;
        ++rows[best];
        ++used;
    }
}

}  // namespace strata::program::batch_draft_rows
