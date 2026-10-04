package model

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// q4k_prefill_gemm_observation.go — the per-prefill Q4_K GEMM route observation (#13694).
// Every resident Q4_K prefill dispatch records which kernel actually served its
// q4_k-majority projections, so a short-prompt TTFT sweep point can be attributed to the
// GEMM that ran (the inkernel_chat line's gemm= key) instead of inferred from the
// process opt-ins. The Metal labels come from the typed metalgemm requested/executed
// identity (metal_q4k_on.go); the CPU and token-loop labels are recorded by the
// dispatchers that take those routes. Generation owns a Session serially, so the
// observation is a plain session field.

// Q4KPrefillGEMM* are the non-Metal route labels a prefill observation can carry. The
// Metal kernels report their own executed identity (scalar, mm32, m5, gemv, ...).
const (
	// Q4KPrefillGEMMNone: no Q4_K prefill dispatch has run since the last reset (a full
	// prefix hit, or a model with no resident Q4_K projections).
	Q4KPrefillGEMMNone = "none"
	// Q4KPrefillGEMMCPU: the batched prefill served its Q4_K projections with CPU q4kGemm.
	Q4KPrefillGEMMCPU = "cpu"
	// Q4KPrefillGEMMTokenLoop: the prompt ran one full decode forward per token.
	Q4KPrefillGEMMTokenLoop = "token_loop"
)

// q4kPrefillGEMMMaxLabels bounds the distinct routes one observation keeps, so a mixed
// prefill cannot grow the log line without bound.
const q4kPrefillGEMMMaxLabels = 4

// qwen35HybridMetalResidentPrefill reports whether this session is the backend-nil
// resident-Q4_K Metal session whose batched hybrid prefill is valid and amortizes at any
// prompt of two or more tokens. An admitted native Metal sequence owner proves the same
// lane without re-probing the device.
func (s *Session) qwen35HybridMetalResidentPrefill() bool {
	if s == nil || s.Backend != nil || !s.Q4K || !s.MetalQ4K {
		return false
	}
	if s.qwen35HAL != nil && s.qwen35HAL.sequenceAccepted {
		return true
	}
	return metalgemm.Available()
}

// observeQ4KPrefillGEMM records one prefill dispatch's route label. Distinct labels are
// kept in first-seen order up to q4kPrefillGEMMMaxLabels; an empty label is ignored.
func (s *Session) observeQ4KPrefillGEMM(label string) {
	if s == nil || label == "" {
		return
	}
	for _, seen := range s.q4kPrefillGEMMLabels {
		if seen == label {
			return
		}
	}
	if len(s.q4kPrefillGEMMLabels) >= q4kPrefillGEMMMaxLabels {
		return
	}
	// Full-slice expression: always reallocate so a copied Session never shares backing.
	n := len(s.q4kPrefillGEMMLabels)
	s.q4kPrefillGEMMLabels = append(s.q4kPrefillGEMMLabels[:n:n], label)
}

// ResetQ4KPrefillGEMMObservation clears the observation; the serving planner calls it
// immediately before a request's prefill so the readback covers only that prefill.
func (s *Session) ResetQ4KPrefillGEMMObservation() {
	if s == nil {
		return
	}
	s.q4kPrefillGEMMLabels = nil
}

// Q4KPrefillGEMMObservation returns the Q4_K prefill routes observed since the last reset,
// joined with "+" in first-seen order, or Q4KPrefillGEMMNone when nothing ran.
func (s *Session) Q4KPrefillGEMMObservation() string {
	if s == nil || len(s.q4kPrefillGEMMLabels) == 0 {
		return Q4KPrefillGEMMNone
	}
	return strings.Join(s.q4kPrefillGEMMLabels, "+")
}
