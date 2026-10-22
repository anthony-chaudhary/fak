package model

// v41EngramSourceReadOwner is an optional lifetime capability, not evidence of
// digest verification, concurrent-read safety, or eligibility for look-ahead.
// A successful caller must join its reads before releasing the source hold.
type v41EngramSourceReadOwner interface {
	V41EngramHoldSourceReads() (release func(), closing <-chan struct{}, ok bool)
}

func v41EngramForwardSourceHold(source any) (func(), <-chan struct{}, bool) {
	owner, ok := source.(v41EngramSourceReadOwner)
	if !ok {
		return nil, nil, false
	}
	return owner.V41EngramHoldSourceReads()
}

// V41EngramHoldSourceReads delegates to this prepared reader's exact backing
// owner. It does not acquire a hold during construction or ordinary row reads.
func (s *v41PackedEngramRowSource) V41EngramHoldSourceReads() (func(), <-chan struct{}, bool) {
	if s == nil {
		return nil, nil, false
	}
	return v41EngramForwardSourceHold(s.reader)
}

// V41EngramHoldSourceReads preserves the underlying source's lifetime contract
// independently of this wrapper's artifact binding.
func (s *v41VerifiedEngramRowSource) V41EngramHoldSourceReads() (func(), <-chan struct{}, bool) {
	if s == nil {
		return nil, nil, false
	}
	return v41EngramForwardSourceHold(s.inner)
}

// V41EngramHoldSourceReads retains the original encoded-row owner rather than
// one transient ReaderAt returned for an individual encoded extent.
func (s *v41EngramQ2KRowSource) V41EngramHoldSourceReads() (func(), <-chan struct{}, bool) {
	if s == nil {
		return nil, nil, false
	}
	return v41EngramForwardSourceHold(s.enc)
}
