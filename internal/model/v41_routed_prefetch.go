package model

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
