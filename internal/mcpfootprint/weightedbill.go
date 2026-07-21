package mcpfootprint

// weightedbill.go — the SECOND axis of the footprint bill (#5304, follow-on from
// #2866's Footprint Ladder doctrine).
//
// floorgate.go and mcpfootprint.go price the FIRST axis: the marginal per-call token
// cost each always-sent tool schema adds to every API request. That answers "how
// expensive is this tool to keep resident" but not "how often is it actually reached
// for". The doctrine's decision variable is a product of both:
//
//   footprint bill(tool) = marginal prefix tokens(tool) x invocation frequency(tool)
//
// With frequency pinned at the conservative 1.0 always-resident fallback, the ladder
// cannot tell a cheap-and-hot tool from an expensive-and-cold one — exactly the
// comparison the deferral advice turns on. This file computes the weighted bill so a
// cheap-but-frequent item can outrank an expensive-but-rare one.
//
// It is a pure JOIN over two inputs already in hand — the priced per-tool marginal
// costs (agent.ToolFootprint.Tokens) and an observed per-tool frequency read from the
// usage feed — so it stays deterministic and offline: no live model call, no
// wall-clock, no network. The frequency values are ESTIMATED-token weights, never a
// provider-billed count (Law A2: every value carries its provenance).

import (
	"sort"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// FrequencyFallback is the conservative invocation frequency assigned to a tool with
// no telemetry: 1.0, the always-resident worst case. A brand-new tool is scored at its
// maximum possible bill first, so it is never under-billed by an absent measurement —
// the same fallback the floor gate already pins for un-observed tools.
const FrequencyFallback = 1.0

// WeightedItem is one tool's frequency-weighted footprint contribution: its marginal
// per-call cost and its observed invocation frequency, multiplied into the total this
// item adds to the weighted bill. Contribution is the ranking key — a low-marginal
// item invoked often can carry a larger Contribution than a high-marginal item invoked
// rarely.
type WeightedItem struct {
	Name         string  `json:"name"`
	Marginal     int     `json:"marginal"`     // marginal per-call cost, in ESTIMATED tokens (the first axis)
	Frequency    float64 `json:"frequency"`    // observed invocation frequency (the second axis); FrequencyFallback when un-observed
	Contribution float64 `json:"contribution"` // Marginal x Frequency — this item's share of the weighted bill
}

// WeightedBill joins the priced per-tool marginal costs with an observed frequency
// table and returns the frequency-weighted footprint bill, ranked by total
// contribution largest-first (ties broken by name so the order is deterministic).
//
// A tool ABSENT from the frequency table keeps FrequencyFallback (1.0), so an
// un-observed tool is billed at worst case. A tool PRESENT with a zero frequency
// contributes zero — an observed-idle tool is distinct from an un-observed one.
//
// Degenerate inputs fail CLOSED to a safe zero rather than crediting the bill: a
// negative marginal cost or a negative frequency is a broken measurement and yields a
// zero Contribution (the offending negative is still surfaced in the item's fields so
// the breakage is visible). An empty item set yields an empty bill.
func WeightedBill(items []agent.ToolFootprint, freq map[string]float64) []WeightedItem {
	out := make([]WeightedItem, 0, len(items))
	for _, it := range items {
		f := FrequencyFallback
		if v, ok := freq[it.Name]; ok {
			f = v
		}
		item := WeightedItem{
			Name:      it.Name,
			Marginal:  it.Tokens,
			Frequency: f,
		}
		// Fail closed: a negative axis is broken input, not a credit against the bill.
		if it.Tokens >= 0 && f >= 0 {
			item.Contribution = float64(it.Tokens) * f
		}
		out = append(out, item)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Contribution != out[j].Contribution {
			return out[i].Contribution > out[j].Contribution
		}
		return out[i].Name < out[j].Name
	})
	return out
}

// TotalBill sums the per-item contributions — the whole frequency-weighted floor, the
// single number the weighted read rolls up to. It is derived from the same
// non-negative contributions WeightedBill produced, so a broken (negative-axis) item
// adds its safe zero and never drags the total below the sum of the healthy items.
func TotalBill(bill []WeightedItem) float64 {
	var sum float64
	for _, it := range bill {
		sum += it.Contribution
	}
	return sum
}
