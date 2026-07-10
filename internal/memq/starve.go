package memq

import (
	"fmt"
	"strconv"
	"strings"
)

// This file is the deterministic anti-starvation credit (#4021): the NACL/Scissorhands
// retention-diversity AXIS realized without their RNG. memq keeps strictly by rank and
// trims at the cutline, so the same borderline cell is dropped every pass; the credit
// tracks a per-cell "below the cutline for N consecutive passes" counter in the OPEN
// Attrs bag and, once a non-durable, unreferenced cell has sat JUST below the cutline
// for K passes, retains it for ONE pass — bounded, reproducible, no sampling. The
// upstream mechanism (seeded-RNG eviction) is explicitly rejected: determinism is the
// audit contract, so the same (cells, query, caps) still yields a byte-identical Result.

// StarveAttr is the Attrs key carrying a cell's consecutive-below-cutline pass counter.
// It rides the OPEN attribute bag (forward-compatible, drop-unknown), so no Cell schema
// change is needed and any backend that round-trips Attrs persists it for free.
const StarveAttr = "memq.starve"

// StarveReason is the closed-vocabulary verdict name stamped when the credit retains a
// perennially-below-cutline cell (#4021). Like OverflowReason, it is a TYPED event a
// consumer keys off — never free text.
const StarveReason = "MEMORY_STARVE_CREDIT"

// StarveUpdate proposes one cell's new counter value. memq core never writes a backend
// (the same rung-1 honest scope as reclassify), so Run REPORTS the updates and the
// store applies them between passes (MemStore.ApplyStarveUpdates; a recall-image
// backend would persist Attrs the same way).
type StarveUpdate struct {
	ID      string `json:"id"`
	Passes  int    `json:"passes"`            // new consecutive-below-cutline count to persist
	Granted bool   `json:"granted,omitempty"` // the one-pass survival credit fired (counter reset)
}

// StarveReport is the typed anti-starvation verdict for one Run (#4021): the K
// threshold, every proposed counter update, and the ID of the cell (if any) the credit
// retained this pass. Emitted only when at least one counter changes — an idle pass
// leaves Result.Starve nil (no advisory spam), mirroring IndexOverflow. A second
// cutline op in the same pipeline appends its updates and overwrites K (last op wins),
// the same way Overflow keeps the last budget verdict.
type StarveReport struct {
	Reason  string         `json:"reason"` // always StarveReason
	K       int            `json:"k"`
	Granted string         `json:"granted,omitempty"` // cell retained past the cutline this pass
	Updates []StarveUpdate `json:"updates,omitempty"`
}

// starveCount reads a cell's persisted counter. A missing, non-numeric, or negative
// value fails closed to 0 — a malformed attr never manufactures a credit.
func starveCount(c Cell) int {
	n, err := strconv.Atoi(strings.TrimSpace(c.Attrs[StarveAttr]))
	if err != nil || n < 0 {
		return 0
	}
	return n
}

// starveEligible reports whether a cell may earn the credit: non-durable (a durable
// cell needs no anti-starvation help — the render driver always admits it),
// unreferenced (refcount==0; a referenced cell survives through its referrers), and
// admissible (a sealed/tombstoned cell must never earn retention).
func starveEligible(c Cell, refcount int) bool {
	return !c.Sealed && !c.Tombstoned && refcount == 0 &&
		NormDurability(c.Durability) != DurabilityDurable
}

// withStarveAttr returns a copy of the cell with the counter set, cloning the Attrs map
// so the backend's snapshot is never mutated through the shared map (Run stays a read).
func withStarveAttr(c Cell, n int) Cell {
	attrs := make(map[string]string, len(c.Attrs)+1)
	for k, v := range c.Attrs {
		attrs[k] = v
	}
	attrs[StarveAttr] = strconv.Itoa(n)
	c.Attrs = attrs
	return c
}

// applyStarveCredit post-processes one cutline (limit/budget) split. Deterministic, no
// RNG: every cell below the cutline advances its consecutive-passes counter by one; a
// kept cell's nonzero counter resets (its streak broke). The single candidate for the
// credit is cut[0] — the cell JUST below the cutline — and only when it is eligible and
// its streak has reached k is it retained, appended after the kept prefix (its natural
// rank position, so the kept ranking is otherwise unchanged) with its counter reset: a
// bounded ONE-PASS survival credit, at most one cell per op. rest is the cut minus any
// granted cell (the overflow verdict must reflect the post-credit truth). updates lists
// only counters whose persisted value changes — kept order first, then cut order — so
// the report inherits the pipeline's total order and stays deterministic.
func applyStarveCredit(kept, cut []Cell, k int, refcount map[string]int) (out, rest []Cell, updates []StarveUpdate) {
	out = kept
	for _, c := range kept {
		if starveCount(c) != 0 {
			updates = append(updates, StarveUpdate{ID: c.ID, Passes: 0})
		}
	}
	for i, c := range cut {
		streak := starveCount(c) + 1
		if i == 0 && streak >= k && starveEligible(c, refcount[c.ID]) {
			out = append(out, withStarveAttr(c, 0))
			updates = append(updates, StarveUpdate{ID: c.ID, Passes: 0, Granted: true})
			continue
		}
		rest = append(rest, c)
		updates = append(updates, StarveUpdate{ID: c.ID, Passes: streak})
	}
	return out, rest, updates
}

// recordStarve folds one op's updates into the Result's typed report and returns the
// human-readable step-note fragment. An empty update set records nothing (Result.Starve
// stays nil when the pass is idle), so the opt-out AND the idle opt-in path both stay
// free of advisory churn.
func recordStarve(res *Result, k int, updates []StarveUpdate) string {
	if len(updates) == 0 {
		return ""
	}
	if res.Starve == nil {
		res.Starve = &StarveReport{Reason: StarveReason}
	}
	res.Starve.K = k
	res.Starve.Updates = append(res.Starve.Updates, updates...)
	for _, u := range updates {
		if u.Granted {
			res.Starve.Granted = u.ID
			return fmt.Sprintf("%s: %s retained past the cutline after %d consecutive below-cutline pass(es)", StarveReason, u.ID, k)
		}
	}
	return fmt.Sprintf("starve: %d counter update(s) proposed", len(updates))
}

// ApplyStarveUpdates persists proposed counter updates onto the page table's Attrs —
// the write-back half of the credit, kept on the store (not the executor) so Run stays
// a pure read. Unknown IDs are skipped; it returns how many cells were updated. The
// Attrs map is cloned per write (snapshots handed out by Cells share maps with the
// table, and a past Result must not observe a later pass's counter).
func (m *MemStore) ApplyStarveUpdates(updates []StarveUpdate) int {
	applied := 0
	for _, u := range updates {
		for i := range m.cells {
			if m.cells[i].ID != u.ID {
				continue
			}
			attrs := make(map[string]string, len(m.cells[i].Attrs)+1)
			for key, v := range m.cells[i].Attrs {
				attrs[key] = v
			}
			attrs[StarveAttr] = strconv.Itoa(u.Passes)
			m.cells[i].Attrs = attrs
			applied++
			break
		}
	}
	return applied
}
