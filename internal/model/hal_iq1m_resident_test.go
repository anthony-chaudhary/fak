package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// iq1mMixedExpertModel keeps the existing IQ2_XS gate/up fixture and replaces
// only down with finite-scale, native IQ1_M bytes from the pinned host fixture.
func iq1mMixedExpertModel(t *testing.T) (*Model, [3]string) {
	t.Helper()
	m, names := iq2ResidentExpertModel(t, kindIQ2XS)
	m.kqw[names[2]] = &kQuantTensor{
		out: 256, in: 256, nblk: 1, kind: kindIQ1M,
		raw: makeIQ12Raw(kindIQ1M, 256, 256, 0x1a2b3c4d),
	}
	return m, names
}

// This is a real consumer/CPU-recorded operator witness, not CUDA qualification.
// It compiles on the frozen IQ2 parent: that parent falls back to CPU and fails
// the positive operator/upload assertions rather than failing to build.
// fak-test:runtime fast est=100ms lane=default
func TestIQ1MResidentExpertBackendRoute(t *testing.T) {
	if SupportsHALKQuant(kindIQ1M) || SupportsHALKQuant(kindIQ2XS) {
		t.Fatal("backend-specific admission changed the generic HAL registry")
	}
	t.Run("mixed-cold-warm", func(t *testing.T) {
		m, names := iq1mMixedExpertModel(t)
		rec := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
		be := &dtypeAdmissionBackend{expertHALRecordingBackend: rec, dtypes: map[compute.Dtype]bool{
			compute.F32: true, compute.IQ2_XS: true, compute.IQ1_M: true,
		}}
		var budget int64
		for _, name := range names {
			budget += kQuantResidentBytes(m.kqw[name])
		}
		// Two 74-byte IQ2_XS and one 56-byte IQ1_M super-block per row.
		if budget != 256*(2*74+56) {
			t.Fatalf("mixed packed budget=%d", budget)
		}
		s := &Session{M: m, Backend: be, halW: map[string]compute.Tensor{}, ExpertRingBytes: budget}
		t.Cleanup(s.Close)
		for token := 0; token < 2; token++ {
			x := q2kExpertInput(256)
			for i := range x {
				x[i] *= float32(token + 1)
			}
			want := expertSwiGLU(m, 0, 0, x, residentKernel{m})
			got := expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})
			routedExpertParity(t, "CPU-recorded mixed IQ1_M resident route", got, want)
			if rec.matmuls != 3*(token+1) || rec.swiglu != token+1 || rec.uploads[compute.IQ2_XS] != 2 || rec.uploads[compute.IQ1_M] != 1 {
				t.Fatalf("token=%d operators/reuse: matmul=%d swiglu=%d uploads=%v", token, rec.matmuls, rec.swiglu, rec.uploads)
			}
			st := s.ExpertRing()
			if !st.Enabled || st.PageIns != 3 || st.ResidentBytes != budget || st.PeakBytes > budget || (token == 1 && st.Hits < 3) {
				t.Fatalf("token=%d packed ring=%+v budget=%d", token, st, budget)
			}
		}
	})
	for _, mode := range []string{"marker-only", "explicit-down-refusal", "unaligned-down-row"} {
		t.Run(mode, func(t *testing.T) {
			m, names := iq1mMixedExpertModel(t)
			rec := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
			var be compute.Backend = &iq2MarkerOnlyBackend{Backend: rec}
			if mode == "marker-only" {
				// Legacy Q2_K gate/up must not hide a missing IQ1_M promise:
				// the third operand, not an earlier IQ2 refusal, must stop staging.
				down := m.kqw[names[2]]
				m, names = q2kExpertModel(t)
				m.kqw[names[2]] = down
			}
			if mode != "marker-only" {
				be = &dtypeAdmissionBackend{expertHALRecordingBackend: rec, marker: true, dtypes: map[compute.Dtype]bool{
					compute.F32: true, compute.IQ2_XS: true, compute.IQ1_M: mode != "explicit-down-refusal",
				}}
			}
			if mode == "unaligned-down-row" {
				m.kqw[names[2]].in = 128
			}
			s := &Session{M: m, Backend: be, halW: map[string]compute.Tensor{}}
			t.Cleanup(s.Close)
			x := q2kExpertInput(256)
			if _, admitted := s.expertSwiGLUHAL(names[0], names[1], names[2], x); admitted {
				t.Fatal("unpromised or unaligned IQ1_M operand admitted")
			}
			if rec.matmuls != 0 || rec.swiglu != 0 || len(rec.uploads) != 0 || len(s.halW) != 0 {
				t.Fatal("failed three-operand admission partially staged or executed")
			}
			if mode != "unaligned-down-row" {
				want := expertSwiGLU(m, 0, 0, x, residentKernel{m})
				got := expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})
				routedExpertParity(t, "unchanged mixed IQ1_M CPU fallback", got, want)
				if rec.matmuls != 0 || rec.swiglu != 0 || len(rec.uploads) != 0 || len(s.halW) != 0 {
					t.Fatal("CPU fallback staged or executed device weights")
				}
			}
		})
	}
}
