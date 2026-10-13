package deepseekv4moe

import "sort"

// SimulateExpertCacheTrace converts a routed-expert trace into the flat unit-sized
// CachePolicyStream and replays it through the shipped LRU arm. It is the real-trace
// adapter for the policy bench: the returned ExpertCacheTrace is exactly what an LRU
// replay of the derived stream witnesses (trace.Hits == CachePolicyRow.Hits for
// CachePolicyLRU), so the conversion is identity-preserving and the final resident set is
// carried out unchanged in sorted order.
//
// NOTE the argument order differs from SimulateExpertCache(routes, capacity, layers,
// expertsPerLayer, topK) in cache_trace.go: here capacity is LAST so the shape
// (layers, expertsPerLayer, topK) reads together, matching how a caller derives it from a
// trace header. Callers should prefer this adapter for real traces and SimulateExpertCache
// for shape-major synthetic routes.
//
// Validation mirrors SimulateExpertCache: capacity must be positive
// (ErrInvalidTraceCapacity); layers/expertsPerLayer/topK must be positive and
// topK <= expertsPerLayer (ErrInvalidTraceShape); every route must name an in-range layer
// with exactly topK in-range, non-duplicated experts (ErrInvalidTraceRoute,
// ErrDuplicateTraceExpert).
func SimulateExpertCacheTrace(routes []ExpertRoute, layers, expertsPerLayer, topK, capacity int) (ExpertCacheTrace, error) {
	if capacity <= 0 {
		return ExpertCacheTrace{}, ErrInvalidTraceCapacity
	}
	if layers <= 0 || expertsPerLayer <= 0 || topK <= 0 || topK > expertsPerLayer {
		return ExpertCacheTrace{}, ErrInvalidTraceShape
	}

	stream := CachePolicyStream{Name: "expert-cache-trace"}
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
			stream.Groups = append(stream.Groups, ExpertGroup{Layer: route.Layer, Expert: expert})
		}
	}

	// Identity witness: the LRU row over the derived stream supplies this trace's hits and
	// page-ins. The residency pass below reuses the same pickLeastByKey victim rule, so the
	// two views never drift.
	hits, misses, evictions := replayOnlineLRU(stream.Groups, capacity)
	peak, finalResident := expertCacheLRUResidency(stream.Groups, capacity)

	return ExpertCacheTrace{
		PageIns:       int64(misses),
		Hits:          int64(hits),
		Evictions:     int64(evictions),
		PeakResident:  peak,
		FinalResident: finalResident,
	}, nil
}

// expertCacheLRUResidency replays the same LRU policy as replayOnlineLRU to report the
// peak resident count and the sorted final resident set. It shares pickLeastByKey so the
// victim choice is bit-for-bit the shipped reference arm's.
func expertCacheLRUResidency(groups []ExpertGroup, capacity int) (peak int, final []ExpertGroup) {
	resident := make(map[ExpertGroup]residentEntry, capacity)
	var clock uint64
	for _, group := range groups {
		clock++
		if entry, ok := resident[group]; ok {
			entry.lastUse = clock
			resident[group] = entry
			continue
		}
		if len(resident) == capacity {
			victim := pickLeastByKey(resident, func(e residentEntry) uint64 { return e.lastUse })
			delete(resident, victim)
		}
		resident[group] = residentEntry{group: group, lastUse: clock}
		if len(resident) > peak {
			peak = len(resident)
		}
	}
	final = make([]ExpertGroup, 0, len(resident))
	for group := range resident {
		final = append(final, group)
	}
	sort.Slice(final, func(i, j int) bool { return groupLess(final[i], final[j]) })
	return peak, final
}
