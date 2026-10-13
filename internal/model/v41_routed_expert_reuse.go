package model

// v41HoldResidentExpertTriple protects the current demand's resident members
// before its first miss can evict a later projection. The ordinary acquisition
// still stages and holds gate/up/down in order. This changes neither the ring
// budget nor payload reads, and declining the protection preserves staging's
// original failure path when the complete triple cannot fit.
func (s *Session) v41HoldResidentExpertTriple(gate, up, down string) func() {
	// Cold ring initialization and budget validation belong to the original
	// projection admission, after its bias/dtype checks. There is nothing to
	// protect until a session already has routed-expert residency.
	if s == nil || s.expertRing == nil {
		return func() {}
	}
	r := s.expertRing
	done := s.ringEnter(r)
	defer done()
	names := [3]string{gate, up, down}
	var held [3]string
	n := 0
	for _, name := range names {
		if w, ok := s.resolveExpertWeight(name); ok && r.isResident(w.halKey()) {
			held[n] = w.halKey()
			n++
		}
	}
	if n == 0 || !s.v41RoutedLayerRingFits(r, [][3]string{names}) {
		return func() {}
	}
	for _, key := range held[:n] {
		r.hold(key)
	}
	return func() {
		// A selected staging failure may have cancelled the shared span and
		// detached the session. Release against the original ring, as ordinary
		// projection holds do, without depending on s.expertRing still being live.
		done := s.ringEnter(r)
		defer done()
		for n > 0 {
			n--
			r.release(held[n])
		}
	}
}
