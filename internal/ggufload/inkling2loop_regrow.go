package ggufload

// dropCaptures marks every graph the session captured for recapture, so none is
// replayed into buffers that were freed or replaced.
func (s *inkling2Resident) dropCaptures() {
	s.graphReady, s.warm = false, false
	s.verifyWarm, s.verifyReady, s.keepReady = false, false, false
	s.tailWarm, s.tailReady, s.tailKeepReady = [inkling2SpecMaxDraft + 1]bool{}, [inkling2SpecMaxDraft + 1]bool{}, [inkling2SpecMaxDraft + 1]bool{}
	if d := s.draft; d != nil {
		d.warm, d.ready = false, false
	}
}

// release frees buffers the session allocated before close.
func (s *inkling2Resident) release(refs ...ink2Ref) {
	for _, r := range refs {
		if r.T.Buf() == nil {
			continue
		}
		for i, t := range s.owned {
			if t.Buf() == r.T.Buf() {
				s.be.Free(t)
				s.owned = append(s.owned[:i], s.owned[i+1:]...)
				break
			}
		}
	}
}
