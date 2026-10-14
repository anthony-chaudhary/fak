package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// v41RoutedPrefetchMisses returns the checkpoint-served projections of a layer's picks that the
// routed-expert ring does not hold, in pick order. Resident and non-checkpoint weights are left
// out, because the page-in loop serves them without a checkpoint read.
func (s *Session) v41RoutedPrefetchMisses(names [][3]string) []string {
	if s == nil || s.M == nil || s.M.expertCheckpoint == nil {
		return nil
	}
	misses := make([]string, 0, 3*len(names))
	for _, triple := range names {
		for _, name := range triple {
			w, ok := s.resolveExpertWeight(name)
			if !ok || w.ck == nil {
				continue
			}
			if r := s.routedExpertRing(name); r != nil {
				done := s.ringEnter(r)
				resident := r.isResident(w.ck.key)
				done()
				if resident {
					continue
				}
			}
			misses = append(misses, name)
		}
	}
	return misses
}

// prefetchV41RoutedPicks starts the checkpoint reads for a layer's ring misses together, so the
// serial page-in loop that follows consumes reads already in flight or finished.
func (s *Session) prefetchV41RoutedPicks(names [][3]string) int {
	misses := s.v41RoutedPrefetchMisses(names)
	if len(misses) == 0 {
		return 0
	}
	return s.M.expertCheckpoint.Prefetch(misses)
}

// readAheadV41RoutedPicks hints only current mapped checkpoint misses after the
// complete demand set fits the ring. Caller holds its span and resident demand
// holds. Thus requests are bounded by one admitted set, with no cache expansion,
// speculative layer union, payload allocation, or background lifetime to drain.
func (s *Session) readAheadV41RoutedPicks(r *pagedRing, names [][3]string) {
	if s == nil || s.M == nil || s.M.expertCheckpoint == nil || r == nil ||
		s.ExpertPrefetch == ExpertPrefetchOnDemand || s.M.Cfg.ActGeluTanh || s.M.Cfg.ActGeluErf {
		return
	}
	// Complete metadata-only admission precedes every hint, including a late
	// down-dtype or gate/up-bias refusal. Partly resident source layouts retain
	// their original path; ring-resident checkpoint weights are skipped below.
	for _, triple := range names {
		for _, name := range triple {
			w, ok := s.resolveExpertWeight(name)
			if !ok || w.ck == nil || (w.ck.dt != compute.Q2_K && w.ck.dt != compute.Q3_K) ||
				!compute.BackendSupportsDeviceWeightDtype(s.Backend, w.ck.dt) ||
				s.M.has(name[:len(name)-len("weight")]+"bias") {
				return
			}
		}
	}
	seen := make(map[string]bool, 3*len(names))
	for _, triple := range names {
		for _, name := range triple {
			if seen[name] {
				continue
			}
			seen[name] = true
			w, _ := s.resolveExpertWeight(name)
			if !r.isResident(w.ck.key) {
				s.M.expertCheckpoint.readAhead(name)
			}
		}
	}
}
