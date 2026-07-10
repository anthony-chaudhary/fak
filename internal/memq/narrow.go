package memq

import (
	"fmt"
	"sort"
	"strings"
)

// OpNarrow (#4019) is the opt-in SECOND compression axis, borrowed from ThinK's
// query-driven channel pruning: where OpBudget optimizes the item-COUNT axis (a cell
// is atomic to applyBudget — kept whole or tail-dropped whole), narrow shrinks a
// retained cell's WIDTH, so a budget can be met by slimming kept cells instead of
// only dropping them. The two axes compose multiplicatively: chain narrow
// immediately before budget (Params.NarrowBytes does exactly that for render and
// compact).
//
// A memq cell never carries body bytes (they stay behind the trust gate), so the
// width narrow acts on is the cell's SAFE projected surface: the Descriptor plus the
// open Attrs bag. Attrs fields are memq's channels: each is scored by query-term
// overlap (ThinK scores channels by query relevance before pruning) and the
// least-relevant drop first, until the cell fits the per-cell byte cap; only when
// every field is gone and the cell is still over cap is the Descriptor re-headLined
// at the remaining allowance. Body-segment narrowing at render is the rung-2
// follow-on — this pass never pages bytes in.
//
// The whole pass is deterministic (explicit total-order tie-breaks, no map-order
// dependence, no RNG) and read-only: the backend snapshot is never mutated (a
// narrowed cell carries a fresh Attrs map), and the folded remainder rides an audit
// Effect exactly as OpDedup records its collapsed siblings.

// cellWidth is the cell's SAFE projected width in bytes: the Descriptor plus every
// Attrs field's key+value footprint. Summation over the map is order-independent,
// so the metric is deterministic.
func cellWidth(c Cell) int64 {
	w := int64(len(c.Descriptor))
	for k, v := range c.Attrs {
		w += int64(len(k) + len(v))
	}
	return w
}

// applyNarrow narrows every over-wide, non-sealed, non-tombstoned cell in the
// working set to the per-cell byte cap (0 = unbounded = pass-through, mirroring
// OpBudget). A cell already within cap passes through byte-identical. Narrowed cells
// are recorded on ONE audit Effect naming the folded remainder (dropped field keys
// in drop order, plus a "desc~" marker when the Descriptor was trimmed) — the
// OpDedup discipline. Sealed/tombstoned cells are left untouched: narrow never edits
// what the trust gate already refuses.
func applyNarrow(res *Result, work []Cell, cap int64, qterms []string) ([]Cell, string) {
	if cap <= 0 {
		return work, ""
	}
	kept := make([]Cell, 0, len(work))
	var narrowedIDs []string
	var remainder []string
	var reclaimed int64
	for _, c := range work {
		if c.Sealed || c.Tombstoned || cellWidth(c) <= cap {
			kept = append(kept, c)
			continue
		}
		nc, dropped, trimmed := narrowCell(c, cap, qterms)
		reclaimed += c.Bytes - nc.Bytes
		parts := make([]string, 0, len(dropped)+1)
		for _, k := range dropped {
			parts = append(parts, "-"+k)
		}
		if trimmed {
			parts = append(parts, "desc~")
		}
		remainder = append(remainder, fmt.Sprintf("%s{%s}", c.ID, strings.Join(parts, ",")))
		narrowedIDs = append(narrowedIDs, c.ID)
		kept = append(kept, nc)
	}
	if len(narrowedIDs) == 0 {
		return kept, ""
	}
	sort.Strings(narrowedIDs)
	sort.Strings(remainder) // entries lead with the cell ID, so this matches Cells order
	res.Effects = append(res.Effects, Effect{
		Kind:  OpNarrow,
		Cells: narrowedIDs,
		Note: fmt.Sprintf("%d cell(s) narrowed to <=%d-byte width (read-only; %d byte(s) reclaimed; folded remainder: %s)",
			len(narrowedIDs), cap, reclaimed, narrowRemainder(remainder)),
	})
	res.Stats.Narrowed = len(narrowedIDs)
	return kept, fmt.Sprintf("narrowed %d cell(s) to <=%d-byte width", len(narrowedIDs), cap)
}

// narrowCell projects ONE over-wide cell down to cap. Attrs fields are kept
// most-relevant-first: sorted by descending query-term overlap, then ascending
// footprint (between equally relevant fields the big boilerplate one drops first),
// then ascending key — a total order, so the result never depends on map iteration.
// Fields drop from the tail until the cell fits; if every field is gone and the cell
// is still over cap, the Descriptor is re-headLined at the remaining allowance. The
// input cell is never mutated (its Attrs map is shared with the backend snapshot); a
// narrowed cell carries a fresh map. Bytes is lowered to the narrowed width — narrow
// never inflates a cell's cost proxy.
func narrowCell(c Cell, cap int64, qterms []string) (nc Cell, dropped []string, trimmed bool) {
	type field struct {
		key, val string
		score    int
		size     int64
	}
	fields := make([]field, 0, len(c.Attrs))
	for k, v := range c.Attrs {
		fields = append(fields, field{key: k, val: v, score: overlap(qterms, tokenize(k+" "+v)), size: int64(len(k) + len(v))})
	}
	sort.Slice(fields, func(i, j int) bool {
		if fields[i].score != fields[j].score {
			return fields[i].score > fields[j].score // most relevant keeps first
		}
		if fields[i].size != fields[j].size {
			return fields[i].size < fields[j].size // among equals, the big one drops first
		}
		return fields[i].key < fields[j].key
	})
	width := cellWidth(c)
	for len(fields) > 0 && width > cap {
		last := fields[len(fields)-1]
		fields = fields[:len(fields)-1]
		width -= last.size
		dropped = append(dropped, last.key)
	}
	nc = c
	if len(dropped) > 0 {
		attrs := make(map[string]string, len(fields))
		for _, f := range fields {
			attrs[f.key] = f.val
		}
		if len(attrs) == 0 {
			attrs = nil
		}
		nc.Attrs = attrs
	}
	if width > cap {
		// Every droppable field is gone and the Descriptor alone still overflows:
		// re-headLine it at its allowance (the cap minus the kept-field width).
		allow := cap - (width - int64(len(nc.Descriptor)))
		if allow < 0 {
			allow = 0
		}
		nc.Descriptor = headLine([]byte(nc.Descriptor), int(allow))
		trimmed = true
		width = cellWidth(nc)
	}
	if width < nc.Bytes {
		nc.Bytes = width
	}
	return nc, dropped, trimmed
}

// narrowRemainder bounds the per-cell folded-remainder list for the audit note, the
// same way overflowNames bounds the overflow note. The Effect.Cells list stays full.
func narrowRemainder(entries []string) string {
	const max = 5
	if len(entries) > max {
		entries = append(entries[:max:max], fmt.Sprintf("+%d more", len(entries)-max))
	}
	return strings.Join(entries, "; ")
}

// insertNarrowBeforeBudget returns ops with an OpNarrow (per-cell width cap = bytes)
// inserted immediately before the FIRST OpBudget — the compose order the width axis
// requires (narrow kept cells, then meter the count axis). A pipeline with no
// OpBudget is returned unchanged: narrow without a budget axis has nothing to
// compose with, and a driver must stay byte-identical when the knob is unset.
func insertNarrowBeforeBudget(ops []Op, bytes int64) []Op {
	for i, op := range ops {
		if op.Kind != OpBudget {
			continue
		}
		out := make([]Op, 0, len(ops)+1)
		out = append(out, ops[:i]...)
		out = append(out, Op{Kind: OpNarrow, Bytes: bytes})
		out = append(out, ops[i:]...)
		return out
	}
	return ops
}
