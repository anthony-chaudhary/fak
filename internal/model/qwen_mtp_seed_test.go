package model

import (
	"math/rand"
	"testing"
)

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

	newKeyed := func(requestID string, src *rand.Rand) *RejectionSampler {
		t.Helper()
		s := NewRejectionSampler(src.Float32)
		keyedRequest(t, s, requestID)
		return s
	}

	sequence := func(interleavePeer bool) []bool {
		src := rand.New(rand.NewSource(7))
		b := newKeyed("req-B", src)
		var a *RejectionSampler
		if interleavePeer {
			a = newKeyed("req-A", src)
		}
		out := make([]bool, 8)
		for i := range out {
			if a != nil {
				a.VerifyToken(0, pTarget, pDraft, -1)
			}
			out[i] = b.VerifyToken(0, pTarget, pDraft, -1).Accepted
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
