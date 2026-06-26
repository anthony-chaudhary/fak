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
	wants := map[string]int64{}
	for _, row := range plan {
		if row.Bytes <= 0 {
			continue
		}
		scope := modelLoadScope(row.Scope)
		wants[scope] = addFitBytes(wants[scope], row.Bytes)
	}
	capByScope := map[string]fitCapacity{}
	for _, cap := range caps {
		scope := modelLoadScope(cap.Scope)
		capByScope[scope] = fitCapacity{total: cap.TotalBytes, free: cap.FreeBytes, known: cap.Known, freeKnown: cap.Known && cap.FreeKnown}
	}
	return memoryFitRows(wants, capByScope, headroom)
}

func requestMemoryFitRows(plan []agent.RequestMemoryDemand, caps []agent.RequestMemoryCapacity, headroom float64) []memoryFitRow {
	wants := map[string]int64{}
	for _, row := range plan {
		if row.Bytes <= 0 {
			continue
		}
		scope := modelLoadScope(row.Scope)
		wants[scope] = addFitBytes(wants[scope], row.Bytes)
	}
	capByScope := map[string]fitCapacity{}
	for _, cap := range caps {
		scope := modelLoadScope(cap.Scope)
		capByScope[scope] = fitCapacity{total: cap.TotalBytes, free: cap.FreeBytes, known: cap.Known, freeKnown: cap.Known && cap.FreeKnown}
	}
	return memoryFitRows(wants, capByScope, headroom)
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
