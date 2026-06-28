package gateway

import (
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type memoryFitRow struct {
	Scope         string
	WantBytes     int64
	BudgetBytes   int64
	MarginBytes   int64
	CapacityKnown bool
	FreeKnown     bool
}

func modelLoadMemoryFitRows(plan []ModelLoadMemoryDemand, caps []ModelLoadMemoryCapacity, headroom float64) []memoryFitRow {
	wants := fitWants(plan, func(r ModelLoadMemoryDemand) (string, int64) { return r.Scope, r.Bytes })
	capByScope := fitCapsByScope(caps, func(c ModelLoadMemoryCapacity) (string, int64, int64, bool, bool) {
		return c.Scope, c.TotalBytes, c.FreeBytes, c.Known, c.FreeKnown
	})
	return memoryFitRows(wants, capByScope, headroom)
}

func requestMemoryFitRows(plan []agent.RequestMemoryDemand, caps []agent.RequestMemoryCapacity, headroom float64) []memoryFitRow {
	wants := fitWants(plan, func(r agent.RequestMemoryDemand) (string, int64) { return r.Scope, r.Bytes })
	capByScope := fitCapsByScope(caps, func(c agent.RequestMemoryCapacity) (string, int64, int64, bool, bool) {
		return c.Scope, c.TotalBytes, c.FreeBytes, c.Known, c.FreeKnown
	})
	return memoryFitRows(wants, capByScope, headroom)
}

// fitWants sums positive demand bytes per scope. Shared by the model-load and
// request-memory fit paths, which carry structurally identical demand rows.
func fitWants[T any](plan []T, of func(T) (scope string, bytes int64)) map[string]int64 {
	wants := map[string]int64{}
	for _, row := range plan {
		scope, bytes := of(row)
		if bytes <= 0 {
			continue
		}
		scope = modelLoadScope(scope)
		wants[scope] = addFitBytes(wants[scope], bytes)
	}
	return wants
}

// fitCapsByScope indexes capacity rows by scope. FreeKnown is gated on Known to
// match the prior inline behavior on both fit paths.
func fitCapsByScope[T any](caps []T, of func(T) (scope string, total, free int64, known, freeKnown bool)) map[string]fitCapacity {
	capByScope := map[string]fitCapacity{}
	for _, cap := range caps {
		scope, total, free, known, freeKnown := of(cap)
		scope = modelLoadScope(scope)
		capByScope[scope] = fitCapacity{total: total, free: free, known: known, freeKnown: known && freeKnown}
	}
	return capByScope
}

type fitCapacity struct {
	total     int64
	free      int64
	known     bool
	freeKnown bool
}

func memoryFitRows(wants map[string]int64, caps map[string]fitCapacity, headroom float64) []memoryFitRow {
	seen := map[string]bool{}
	scopes := make([]string, 0, len(wants)+len(caps))
	for scope := range wants {
		scope = modelLoadScope(scope)
		if !seen[scope] {
			seen[scope] = true
			scopes = append(scopes, scope)
		}
	}
	for scope := range caps {
		scope = modelLoadScope(scope)
		if !seen[scope] {
			seen[scope] = true
			scopes = append(scopes, scope)
		}
	}
	sort.SliceStable(scopes, func(i, j int) bool { return strings.Compare(scopes[i], scopes[j]) < 0 })
	out := make([]memoryFitRow, 0, len(scopes))
	for _, scope := range scopes {
		row := memoryFitRow{Scope: scope, WantBytes: wants[scope]}
		if cap, ok := caps[scope]; ok && cap.known {
			row.CapacityKnown = true
			row.FreeKnown = cap.freeKnown
			budget := cap.total
			if cap.freeKnown {
				budget = cap.free
			}
			row.BudgetBytes = applyFitHeadroom(budget, headroom)
			row.MarginBytes = row.BudgetBytes - row.WantBytes
		}
		out = append(out, row)
	}
	return out
}

func addFitBytes(a, b int64) int64 {
	const maxInt64 = int64(^uint64(0) >> 1)
	if b <= 0 {
		return a
	}
	if a > maxInt64-b {
		return maxInt64
	}
	return a + b
}

func applyFitHeadroom(bytes int64, headroom float64) int64 {
	if bytes <= 0 {
		return 0
	}
	if headroom <= 0 || headroom >= 1 {
		return bytes
	}
	return int64(float64(bytes) * (1 - headroom))
}
