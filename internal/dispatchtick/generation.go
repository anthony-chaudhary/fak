package dispatchtick

import "strings"

// Generation bucket names, matching the GitHub label taxonomy (gen/now, gen/next,
// gen/second-next, gen/future) and the selection contract in
// docs/notes/GENERATION-DISPATCH-SELECTION-2026-06-30.md.
const (
	GenNow          = "gen/now"
	GenNext         = "gen/next"
	GenSecondNext   = "gen/second-next"
	GenFuture       = "gen/future"
	GenUnclassified = "unclassified"
)

var generationLabels = map[string]string{
	"gen/now":         GenNow,
	"gen/next":        GenNext,
	"gen/second-next": GenSecondNext,
	"gen/future":      GenFuture,
}

// GenerationBucket classifies an issue's generation horizon from its labels: exactly
// one of gen/now, gen/next, gen/second-next, gen/future, or GenUnclassified when the
// issue carries none of the four labels. If an issue carries more than one (a
// labeling mistake), the nearest horizon wins so the issue never falls further out
// of the schedule than its nearest-term label already put it.
func GenerationBucket(labels []string) string {
	found := ""
	for _, label := range labels {
		bucket, ok := generationLabels[strings.TrimSpace(label)]
		if !ok {
			continue
		}
		if found == "" || generationRank(bucket) < generationRank(found) {
			found = bucket
		}
	}
	if found == "" {
		return GenUnclassified
	}
	return found
}

func generationRank(bucket string) int {
	switch bucket {
	case GenNow:
		return 0
	case GenNext:
		return 1
	case GenSecondNext:
		return 2
	case GenFuture:
		return 3
	default:
		return 4
	}
}

// GenerationHeldByDefault reports whether a bucket sits outside the automatic
// dispatch window absent an explicit horizon request, per the Default Window rule:
// gen/now and gen/next launch by default; gen/second-next, gen/future, and
// unclassified issues are held so long-horizon research never consumes an
// immediate ship slot by accident.
func GenerationHeldByDefault(bucket string) bool {
	switch bucket {
	case GenNow, GenNext:
		return false
	default:
		return true
	}
}

// GenerationEligible reports whether an issue in the given generation bucket may be
// dispatched under the requested horizon filter. horizon is "" (the automatic
// default window), "now", "next", "second-next", "future", or "all" -- mirroring
// the --generation flag in the selection contract. An unrecognized horizon string
// falls back to the default window rather than silently admitting everything.
func GenerationEligible(bucket, horizon string) bool {
	switch strings.ToLower(strings.TrimSpace(horizon)) {
	case "now":
		return bucket == GenNow
	case "next":
		return bucket == GenNext
	case "second-next":
		return bucket == GenSecondNext
	case "future":
		return bucket == GenFuture
	case "all":
		// Every classified generation is admitted; unclassified issues stay held
		// for classification repair even under an explicit "all" request.
		return bucket != GenUnclassified
	default:
		return !GenerationHeldByDefault(bucket)
	}
}

// GenerationCandidate is one orderable issue plus the generation-eligibility facts
// the picker needs: its priority weight (see OrderLaneCandidates) and its
// classified generation bucket.
type GenerationCandidate struct {
	Number     int
	Weight     int
	Generation string
}

// GenerationGateActive reports whether generation-based filtering should apply at all: only
// when the caller supplied an explicit horizon, or at least one candidate carries a real
// (non-empty, classified) generation label. Most of a repo's backlog carries no gen/* label,
// so the gate must stay OFF for an all-unclassified candidate set -- otherwise a picker that
// wires this in would silently hold the entire ordinary backlog the moment nothing in it
// happens to be generation-labeled, per the Core Rule that generation is a scheduling hint,
// not a queue silo.
func GenerationGateActive(cands []GenerationCandidate, horizon string) bool {
	if strings.TrimSpace(horizon) != "" {
		return true
	}
	for _, c := range cands {
		if isKnownGenerationBucket(c.Generation) {
			return true
		}
	}
	return false
}

func isKnownGenerationBucket(bucket string) bool {
	switch bucket {
	case GenNow, GenNext, GenSecondNext, GenFuture:
		return true
	default:
		return false
	}
}

// OrderEligibleGenerationCandidates filters cands to the issues eligible under
// horizon, then orders the survivors exactly as OrderLaneCandidates does --
// priority weight first, then the oldest/newest number tiebreak. Held candidates
// (e.g. gen/second-next and gen/future under the default window) are dropped
// entirely rather than merely sorted last, so a caller's launchable set never
// silently includes a horizon it did not ask for. When GenerationGateActive is
// false (no explicit horizon and no candidate carries a real generation label),
// every candidate passes through unfiltered -- the ordinary, generation-blind
// backlog keeps flowing exactly as OrderLaneCandidates already ordered it.
func OrderEligibleGenerationCandidates(cands []GenerationCandidate, horizon string, preferNewest bool) []int {
	all := make([]LaneCandidate, len(cands))
	for i, c := range cands {
		all[i] = LaneCandidate{Number: c.Number, Weight: c.Weight}
	}
	if !GenerationGateActive(cands, horizon) {
		return OrderLaneCandidates(all, preferNewest)
	}
	eligible := make([]LaneCandidate, 0, len(cands))
	for _, c := range cands {
		if GenerationEligible(c.Generation, horizon) {
			eligible = append(eligible, LaneCandidate{Number: c.Number, Weight: c.Weight})
		}
	}
	return OrderLaneCandidates(eligible, preferNewest)
}
