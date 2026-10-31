package model

import (
	"encoding/binary"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The interface field deliberately hides optional dtype promises while forwarding
// real CPU operators and the recorder's memory/operation accounting.
type iq2MarkerOnlyBackend struct{ compute.Backend }

func (*iq2MarkerOnlyBackend) SupportsRoutedExpertKQuant() bool { return true }

func iq2ResidentExpertModel(t *testing.T, kind kQuantKind) (*Model, [3]string) {
	t.Helper()
	m, names := q2kExpertModel(t)
	for i, name := range names {
		raw := make([]byte, 256*kind.blockBytes())
		state := uint64(0x71301 + i)
		for j := range raw {
			state ^= state << 13
			state ^= state >> 7
			state ^= state << 17
			raw[j] = byte(state)
		}
		for off := 0; off < len(raw); off += kind.blockBytes() {
			binary.LittleEndian.PutUint16(raw[off:], 0x3000)
		}
		m.kqw[name] = &kQuantTensor{out: 256, in: 256, nblk: 1, kind: kind, raw: raw}
	}
	return m, names
}

// fak-test:runtime fast est=100ms lane=default
func TestIQ2ResidentExpertBackendRoute(t *testing.T) {
	for _, kind := range []kQuantKind{kindIQ2XXS, kindIQ2XS} {
		t.Run(kind.String(), func(t *testing.T) {
			if SupportsHALKQuant(kind) {
				t.Fatal("backend-specific admission changed the generic registry")
			}
			m, names := iq2ResidentExpertModel(t, kind)
			dt, _ := rawIQDenseHALDtype(kind)
			x := q2kExpertInput(256)
			want := expertSwiGLU(m, 0, 0, x, residentKernel{m})
			rec := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
			be := &dtypeAdmissionBackend{expertHALRecordingBackend: rec, dtypes: map[compute.Dtype]bool{compute.F32: true, dt: true}}
			budget := 3 * kQuantResidentBytes(m.kqw[names[0]])
			s := &Session{M: m, Backend: be, halW: map[string]compute.Tensor{}, ExpertRingBytes: budget}
			t.Cleanup(s.Close)
			for token := 0; token < 2; token++ {
				got := expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})
				routedExpertParity(t, "CPU-recorded resident IQ2 route", got, want)
			}
			if rec.matmuls != 6 || rec.swiglu != 2 || rec.uploads[dt] != 3 {
				t.Fatalf("operators/reuse: matmul=%d swiglu=%d packed uploads=%d", rec.matmuls, rec.swiglu, rec.uploads[dt])
			}
			st := s.ExpertRing()
			if !st.Enabled || st.PageIns != 3 || st.ResidentBytes != budget || st.PeakBytes > budget || st.Hits < 3 {
				t.Fatalf("native packed ring accounting: %+v budget=%d", st, budget)
			}
		})
	}
	for _, mode := range []string{"marker-only", "explicit-refusal", "mixed-unsupported", "other-native-iq", "unaligned-row"} {
		t.Run(mode, func(t *testing.T) {
			m, names := iq2ResidentExpertModel(t, kindIQ2XXS)
			rec := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
			var be compute.Backend = &iq2MarkerOnlyBackend{Backend: rec}
			if mode != "marker-only" {
				allowed := map[compute.Dtype]bool{compute.F32: true}
				if mode == "other-native-iq" {
					m, names = iq2ResidentExpertModel(t, kindIQ3S)
					allowed[compute.IQ3_S] = true
				}
				if mode == "unaligned-row" {
					allowed[compute.IQ2_XXS] = true
					m.kqw[names[2]].in = 128
				}
				if mode == "mixed-unsupported" {
					allowed[compute.IQ2_XXS] = true
					other, otherNames := iq2ResidentExpertModel(t, kindIQ2XS)
					m.kqw[names[2]] = other.kqw[otherNames[2]]
				}
				be = &dtypeAdmissionBackend{expertHALRecordingBackend: rec, marker: true, dtypes: allowed}
			}
			s := &Session{M: m, Backend: be, halW: map[string]compute.Tensor{}}
			t.Cleanup(s.Close)
			if _, admitted := s.expertSwiGLUHAL(names[0], names[1], names[2], q2kExpertInput(256)); admitted {
				t.Fatal("unpromised operand admitted")
			}
			if rec.matmuls != 0 || len(rec.uploads) != 0 || len(s.halW) != 0 {
				t.Fatal("failed per-operand admission staged or executed weights")
			}
		})
	}
	t.Run("legacy-Q2-marker", func(t *testing.T) {
		m, _ := q2kExpertModel(t)
		rec := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
		s := &Session{M: m, Backend: &iq2MarkerOnlyBackend{Backend: rec}, halW: map[string]compute.Tensor{}}
		t.Cleanup(s.Close)
		expertSwiGLU(m, 0, 0, q2kExpertInput(256), sessionQ4KKernel{s: s})
		if rec.matmuls != 3 {
			t.Fatal("pre-existing Q2_K marker-only route changed")
		}
	})
}
