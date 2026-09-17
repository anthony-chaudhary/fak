package model

import (
	"math/rand"
	"testing"
)

// keyedSampler is the per-request RNG-identity capability the spec-decode
// rejection sampler must expose so a request's verdict coin is keyed to the
// request, not the physical batch slot (#13146).
type keyedSampler interface {
	SetRequestKey(requestID string)
	KeyedOn() bool
}

// TestSpecDecodeSamplerExposesPerRequestCoin performs the red-then-green symptom
// check for #13146 on code paths that compile at BOTH the parent and the fix.
//
// It asserts the required guarantee structurally, then behaviourally:
//
//	STRUCTURAL: the sampler must be keyable per request. At the parent commit
//	  RejectionSampler has no SetRequestKey, so the assertion fails -> RED.
//	  At the fix the assertion holds -> the capability exists.
//
//	BEHAVIOURAL: once keyed, one request's draws must not perturb a co-resident
//	  request's verdict sequence. The parent's only randomness is a shared
//	  randSource, so two requests sharing it interfere; the keyed path decouples
//	  them.
func TestSpecDecodeSamplerExposesPerRequestCoin(t *testing.T) {
	pTarget := []float32{0.4, 0.6}
	pDraft := []float32{0.8, 0.2}

	newKeyed := func(t *testing.T, requestID string, src *rand.Rand) keyedSampler {
		t.Helper()
		s := NewRejectionSampler(src.Float32)
		ks, ok := any(s).(keyedSampler)
		if !ok {
			t.Fatalf("RejectionSampler does not expose a per-request keyed coin (SetRequestKey/KeyedOn); the verdict coin remains bound to the shared source, not the request (#13146)")
		}
		ks.SetRequestKey(requestID)
		if !ks.KeyedOn() {
			t.Fatalf("SetRequestKey(%q) did not arm a keyed stream", requestID)
		}
		return ks
	}

	verdict := func(s any) bool {
		rs := s.(*RejectionSampler)
		return rs.VerifyToken(0, pTarget, pDraft, -1).Accepted
	}

	sequence := func(interleavePeer bool) []bool {
		src := rand.New(rand.NewSource(7))
		b := newKeyed(t, "req-B", src)
		var a keyedSampler
		if interleavePeer {
			a = newKeyed(t, "req-A", src)
		}
		out := make([]bool, 8)
		for i := range out {
			if a != nil {
				verdict(a)
			}
			out[i] = verdict(b)
		}
		return out
	}

	alone := sequence(false)
	inter := sequence(true)

	differ := 0
	for i := range alone {
		if alone[i] != inter[i] {
			differ++
		}
	}
	if differ != 0 {
		t.Fatalf("co-resident request perturbed the keyed verdict stream in %d/8 steps (alone=%v inter=%v): the coin is keyed to the shared slot, not the request", differ, alone, inter)
	}
}
