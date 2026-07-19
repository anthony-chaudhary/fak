package deepseekv4moe

// BatchStep is one decode step's routed top-k selection for every agent in a
// batch at one layer. PerAgent[b] lists the topK expert ids agent b selected;
// expert ids must be unique within one agent's selection because a top-k
// router cannot select the same expert twice, while overlap ACROSS agents is
// the coalescing opportunity this simulator measures.
type BatchStep struct {
	Layer    int
	PerAgent [][]int
}

// BatchCoalesceTrace reports deterministic cross-agent expert-coalescing
// behavior for a bounded routed-expert LRU. B agents advance one decode step
// together and each distinct (layer, expert) group is streamed once per step,
// however many agents in the batch routed to it. Like ExpertCacheTrace it is
// weight-free evidence: it does not model transfer or dequant latency, GPU
// execution, I/O bandwidth, output parity, or throughput.
type BatchCoalesceTrace struct {
	Hits             int64   // resident-set hits over distinct (layer, expert) groups
	Misses           int64   // resident-set misses (page-ins) over distinct groups
	DistinctStreamed int64   // sum over steps of |U_step|, the coalesced stream (== Hits+Misses)
	NaiveStreamed    int64   // sum over steps of agents*topK, B independent un-coalesced streams
	CoalesceRatio    float64 // NaiveStreamed / DistinctStreamed == mean (B*K)/U(B)
	PeakResident     int     // max observed residency, never exceeds capacity
	PeakStepUnion    int     // max |U_step|; the one-step working set admission must fit
}

// SimulateExpertCacheBatch runs batch steps through the same deterministic
// least-recently-used routed-group cache as SimulateExpertCache. Within one
// step the distinct union of every agent's selection is processed in
// first-touch order (agent-major, selection order within an agent), so the
// agents==1 case reduces bit-identically to the single-stream simulator:
// Hits, Misses (PageIns), and PeakResident match it exactly.
func SimulateExpertCacheBatch(steps []BatchStep, capacity, layers, expertsPerLayer, topK, agents int) (BatchCoalesceTrace, error) {
	if capacity <= 0 {
		return BatchCoalesceTrace{}, ErrInvalidTraceCapacity
	}
	if layers <= 0 || expertsPerLayer <= 0 || topK <= 0 || topK > expertsPerLayer || agents <= 0 {
		return BatchCoalesceTrace{}, ErrInvalidTraceShape
	}

	resident := make(map[ExpertGroup]traceEntry, capacity)
	agentSeen := make(map[int]struct{}, topK)
	stepSeen := make(map[int]struct{}, topK*agents)
	var out BatchCoalesceTrace
	var clock uint64
	for _, step := range steps {
		if step.Layer < 0 || step.Layer >= layers || len(step.PerAgent) != agents {
			return BatchCoalesceTrace{}, ErrInvalidTraceRoute
		}
		clear(stepSeen)
		stepUnion := 0
		for _, selection := range step.PerAgent {
			if len(selection) != topK {
				return BatchCoalesceTrace{}, ErrInvalidTraceRoute
			}
			clear(agentSeen)
			for _, expert := range selection {
				if expert < 0 || expert >= expertsPerLayer {
					return BatchCoalesceTrace{}, ErrInvalidTraceRoute
				}
				if _, duplicate := agentSeen[expert]; duplicate {
					return BatchCoalesceTrace{}, ErrDuplicateTraceExpert
				}
				agentSeen[expert] = struct{}{}

				out.NaiveStreamed++
				if _, coalesced := stepSeen[expert]; coalesced {
					continue // already streamed for a peer agent this step
				}
				stepSeen[expert] = struct{}{}
				stepUnion++
				out.DistinctStreamed++

				clock++
				group := ExpertGroup{Layer: step.Layer, Expert: expert}
				if entry, ok := resident[group]; ok {
					out.Hits++
					entry.last = clock
					resident[group] = entry
					continue
				}
				out.Misses++
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
				}
				resident[group] = traceEntry{group: group, last: clock}
				if len(resident) > out.PeakResident {
					out.PeakResident = len(resident)
				}
			}
		}
		if stepUnion > out.PeakStepUnion {
			out.PeakStepUnion = stepUnion
		}
	}
	if out.DistinctStreamed > 0 {
		out.CoalesceRatio = float64(out.NaiveStreamed) / float64(out.DistinctStreamed)
	}
	return out, nil
}

// ExpertCacheBatchAdmission joins static byte admission to deterministic
// cross-agent coalescing evidence. It remains weight-free: it proves neither
// model I/O nor dequant/GPU execution, output parity, latency, cache hit
// rate, or throughput.
type ExpertCacheBatchAdmission struct {
	Plan  ExpertCachePlan
	Trace BatchCoalesceTrace
}

// AdmitExpertCacheBatchTrace computes a byte-bounded routed-group capacity and
// then simulates the supplied batch steps under that exact capacity. One full
// B-agent step's distinct union must fit concurrently: a plan that cannot even
// hold a fully-overlapped step (topK groups) fails before simulation, and a
// step whose observed union exceeds capacity fails closed after it, mirroring
// AdmitExpertCacheTrace's ErrRouteWorkingSetExceedsCache discipline.
func AdmitExpertCacheBatchTrace(totalHBMBytes, nonRoutedResidentBytes, runtimeReserveBytes, routedExpertGroupBytes int64, layers, expertsPerLayer, topK, agents int, steps []BatchStep) (ExpertCacheBatchAdmission, error) {
	plan, err := PlanExpertCache(totalHBMBytes, nonRoutedResidentBytes, runtimeReserveBytes, routedExpertGroupBytes, int64(layers), int64(expertsPerLayer))
	if err != nil {
		return ExpertCacheBatchAdmission{}, err
	}
	if plan.ResidentRoutedGroups < int64(topK) {
		return ExpertCacheBatchAdmission{}, ErrRouteWorkingSetExceedsCache
	}
	trace, err := SimulateExpertCacheBatch(steps, int(plan.ResidentRoutedGroups), layers, expertsPerLayer, topK, agents)
	if err != nil {
		return ExpertCacheBatchAdmission{}, err
	}
	if trace.PeakStepUnion > int(plan.ResidentRoutedGroups) || trace.PeakResident > int(plan.ResidentRoutedGroups) {
		return ExpertCacheBatchAdmission{}, ErrRouteWorkingSetExceedsCache
	}
	return ExpertCacheBatchAdmission{Plan: plan, Trace: trace}, nil
}
