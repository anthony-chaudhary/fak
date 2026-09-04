package deepseekv4moe

import (
	"errors"
	"sort"
)

var (
	// ErrInvalidTraceCapacity indicates the cache capacity given to trace simulation is non-positive.
	ErrInvalidTraceCapacity = errors.New("deepseekv4moe: expert-cache trace capacity must be positive")
	// ErrInvalidTraceShape indicates dimension parameters are non-positive or top-k exceeds expert count.
	ErrInvalidTraceShape = errors.New("deepseekv4moe: expert-cache trace shape is invalid")
	// ErrInvalidTraceRoute indicates a route has an invalid layer index, expert index, or top-k width.
	ErrInvalidTraceRoute = errors.New("deepseekv4moe: expert-cache route is invalid")
	// ErrDuplicateTraceExpert indicates duplicate expert IDs were selected within a single route.
	ErrDuplicateTraceExpert = errors.New("deepseekv4moe: duplicate expert in cache route")
)

// ExpertRoute is one layer's routed-expert selection. Expert IDs must be unique
// within a route because a top-k router cannot select the same expert twice.
type ExpertRoute struct {
	Layer   int
	Experts []int
}

// ExpertGroup identifies one independently pageable routed (layer, expert)
// weight group.
type ExpertGroup struct {
	Layer  int
	Expert int
}

// ExpertCacheTrace reports deterministic control-plane behavior for a bounded
// routed-expert LRU. It is weight-free evidence: it does not model transfer or
// dequant latency, GPU execution, I/O bandwidth, output parity, or throughput.
type ExpertCacheTrace struct {
	PageIns       int64
	Hits          int64
	Evictions     int64
	PeakResident  int
	FinalResident []ExpertGroup // sorted by layer then expert for stable evidence
}

type traceEntry struct {
	group ExpertGroup
	last  uint64
}

// SimulateExpertCache runs routes through a deterministic least-recently-used
// routed-group cache. Each selected group is admitted before its simulated use,
// and observed residency never exceeds capacity.
func SimulateExpertCache(routes []ExpertRoute, capacity, layers, expertsPerLayer, topK int) (ExpertCacheTrace, error) {
	if capacity <= 0 {
		return ExpertCacheTrace{}, ErrInvalidTraceCapacity
	}
	if layers <= 0 || expertsPerLayer <= 0 || topK <= 0 || topK > expertsPerLayer {
		return ExpertCacheTrace{}, ErrInvalidTraceShape
	}

	resident := make(map[ExpertGroup]traceEntry, capacity)
	var out ExpertCacheTrace
	var clock uint64
	for _, route := range routes {
		if route.Layer < 0 || route.Layer >= layers || len(route.Experts) != topK {
			return ExpertCacheTrace{}, ErrInvalidTraceRoute
		}
		seen := make(map[int]struct{}, topK)
		for _, expert := range route.Experts {
			if expert < 0 || expert >= expertsPerLayer {
				return ExpertCacheTrace{}, ErrInvalidTraceRoute
			}
			if _, duplicate := seen[expert]; duplicate {
				return ExpertCacheTrace{}, ErrDuplicateTraceExpert
			}
			seen[expert] = struct{}{}
		}

		for _, expert := range route.Experts {
			clock++
			group := ExpertGroup{Layer: route.Layer, Expert: expert}
			if entry, ok := resident[group]; ok {
				out.Hits++
				entry.last = clock
				resident[group] = entry
				continue
			}
			out.PageIns++
			if len(resident) == capacity {
				var victim ExpertGroup
				var oldest uint64
				first := true
				for candidate, entry := range resident {
					if first || entry.last < oldest || (entry.last == oldest && groupLess(candidate, victim)) {
						victim, oldest, first = candidate, entry.last, false
					}
				}
				delete(resident, victim)
				out.Evictions++
			}
			resident[group] = traceEntry{group: group, last: clock}
			if len(resident) > out.PeakResident {
				out.PeakResident = len(resident)
			}
		}
	}

	out.FinalResident = make([]ExpertGroup, 0, len(resident))
	for group := range resident {
		out.FinalResident = append(out.FinalResident, group)
	}
	sort.Slice(out.FinalResident, func(i, j int) bool { return groupLess(out.FinalResident[i], out.FinalResident[j]) })
	return out, nil
}

func groupLess(a, b ExpertGroup) bool {
	return a.Layer < b.Layer || (a.Layer == b.Layer && a.Expert < b.Expert)
}
