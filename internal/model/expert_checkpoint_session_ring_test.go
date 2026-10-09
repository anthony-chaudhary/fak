package model

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

type checkpointSessionReader struct {
	*bytes.Reader
	calls int
}

func (r *checkpointSessionReader) ReadAt(p []byte, offset int64) (int, error) {
	r.calls++
	return r.Reader.ReadAt(p, offset)
}

type checkpointSessionFixture struct {
	m       *Model
	tier    *ExpertCheckpointTier
	be      *expertHALRecordingBackend
	readers []*checkpointSessionReader
	raw     map[string][]byte
	budget  int64
}

func checkpointSessionNames(expert int) []string {
	return []string{
		fmt.Sprintf("model.layers.0.ffn.experts.%d.w1.weight", expert),
		fmt.Sprintf("model.layers.0.ffn.experts.%d.w3.weight", expert),
		fmt.Sprintf("model.layers.0.ffn.experts.%d.w2.weight", expert),
	}
}

func newCheckpointSessionFixture(t *testing.T, bounded bool) *checkpointSessionFixture {
	t.Helper()
	cfg := expertHALTestConfig(256)
	cfg.ModelType, cfg.NumExperts = "deepseek41", 2
	f := &checkpointSessionFixture{
		m: &Model{Cfg: cfg}, tier: NewExpertCheckpointTier(0),
		be:  &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}},
		raw: map[string][]byte{}, budget: 256 * (84 + 84 + 110),
	}
	for projection, proj := range []string{"gate_proj", "up_proj", "down_proj"} {
		blockBytes, quant := 84, ExpertCheckpointQ2K
		if projection == 2 {
			blockBytes, quant = 110, ExpertCheckpointQ3K
		}
		var slab []byte
		for expert := 0; expert < 2; expert++ {
			raw := make([]byte, 256*blockBytes)
			for row := 0; row < 256; row++ {
				block := raw[row*blockBytes : (row+1)*blockBytes]
				for i := range block {
					block[i] = byte(i*7 + expert*31 + row*3 + projection*13)
				}
				if projection < 2 {
					for i := 0; i < 16; i++ {
						block[i] = byte(1 + expert)
					}
					binary.LittleEndian.PutUint16(block[80:], 0x2400)
					binary.LittleEndian.PutUint16(block[82:], 0)
				} else {
					binary.LittleEndian.PutUint16(block[108:], 0x2400)
				}
			}
			f.raw[checkpointSessionNames(expert)[projection]] = raw
			slab = append(slab, raw...)
		}
		reader := &checkpointSessionReader{Reader: bytes.NewReader(slab)}
		f.readers = append(f.readers, reader)
		err := f.tier.AddShard(reader, int64(len(slab)), []FusedExpertTensor{{Name: proj, Layer: 0, Proj: proj, Arch: "deepseek41", Quant: quant, Experts: 2, Rows: 256, Cols: 256}})
		if err != nil {
			t.Fatal(err)
		}
	}
	if bounded {
		if err := f.tier.SetDeviceRingBudget(f.budget); err != nil {
			t.Fatal(err)
		}
	}
	f.m.SetExpertCheckpoint(f.tier)
	return f
}

func (f *checkpointSessionFixture) reads() int {
	var total int
	for _, r := range f.readers {
		total += r.calls
	}
	return total
}

func (f *checkpointSessionFixture) uploads() int {
	var total int
	for _, count := range f.be.uploads {
		total += count
	}
	return total
}

func (f *checkpointSessionFixture) session(t *testing.T) *Session {
	t.Helper()
	s, err := f.m.NewBackendSessionChecked(f.be)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Close)
	return s
}

func checkpointSessionInput() []float32 {
	x := make([]float32, 256)
	for i := range x {
		x[i] = 1 + float32(i%5)/8
	}
	return x
}

func (f *checkpointSessionFixture) reference(expert int, x []float32) []float32 {
	be := compute.Default()
	names := checkpointSessionNames(expert)
	gate := be.MatMul(compute.NewQ2K(be, []int{256, 256}, f.raw[names[0]]), compute.NewF32(be, []int{256}, x))
	up := be.MatMul(compute.NewQ2K(be, []int{256, 256}, f.raw[names[1]]), compute.NewF32(be, []int{256}, x))
	act := be.SwiGLU(gate, up)
	return be.Read(be.MatMul(compute.NewQ3K(be, []int{256, 256}, f.raw[names[2]]), act))
}

func checkpointSessionForward(t *testing.T, s *Session, expert int) []float32 {
	t.Helper()
	names := checkpointSessionNames(expert)
	out, ok := s.expertSwiGLUHAL(names[0], names[1], names[2], checkpointSessionInput())
	if !ok {
		t.Fatal("native mixed Q2/Q3 expert HAL route declined")
	}
	return out
}

func checkpointSessionNoPermanent(t *testing.T, s *Session) {
	t.Helper()
	for key := range s.halW {
		if isRoutedExpertWeight(key) {
			t.Fatalf("routed expert escaped the bounded ring: %s", key)
		}
	}
}

// fak-test:runtime fast est=1s
func TestCheckpointSessionConstructorInheritsDeviceBudget(t *testing.T) {
	t.Parallel()
	for _, bounded := range []bool{false, true} {
		t.Run(fmt.Sprintf("bounded_%t", bounded), func(t *testing.T) {
			t.Parallel()
			f := newCheckpointSessionFixture(t, bounded)
			s := f.session(t)
			want := int64(0)
			if bounded {
				want = f.budget
			}
			if s.ExpertRingBytes != want || s.executionPolicy != ExecutionPolicyPortable {
				t.Fatalf("constructor ring=%d policy=%d, want ring=%d portable", s.ExpertRingBytes, s.executionPolicy, want)
			}
			if f.reads() != 0 || f.uploads() != 0 || f.tier.Stats().BudgetBytes != 0 {
				t.Fatal("constructor read/uploaded checkpoint data or changed host budget")
			}
			if !bounded {
				checkpointSessionForward(t, s, 0)
				if s.ExpertRing().Enabled || len(s.halW) != 3 || f.reads() != 3 {
					t.Fatal("legacy zero tier no longer uses permanent expert residency")
				}
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestCheckpointSessionNativeHALReuseRotationParity(t *testing.T) {
	t.Parallel()
	f := newCheckpointSessionFixture(t, true)
	s := f.session(t)
	if s.ExpertRingBytes != f.budget {
		t.Fatalf("constructor failed to inherit positive checkpoint budget: %d", s.ExpertRingBytes)
	}
	for step, expert := range []int{0, 0, 1, 0} {
		beforeReads, beforeQ2, beforeQ3 := f.reads(), f.be.uploads[compute.Q2_K], f.be.uploads[compute.Q3_K]
		got, want := checkpointSessionForward(t, s, expert), f.reference(expert, checkpointSessionInput())
		if len(got) != len(want) || len(got) != f.m.Cfg.HiddenSize {
			t.Fatal("expert output shape mismatch")
		}
		nonzero := false
		for i := range want {
			if got[i] != want[i] || math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
				t.Fatalf("step=%d expert=%d output=%d got=%g want=%g", step, expert, i, got[i], want[i])
			}
			nonzero = nonzero || want[i] != 0
		}
		if !nonzero {
			t.Fatal("numerical parity fixture is vacuously zero")
		}
		wantReads, wantQ2, wantQ3 := 3, 2, 1
		if step == 1 {
			wantReads, wantQ2, wantQ3 = 0, 0, 0
		}
		if f.reads()-beforeReads != wantReads || f.be.uploads[compute.Q2_K]-beforeQ2 != wantQ2 || f.be.uploads[compute.Q3_K]-beforeQ3 != wantQ3 {
			t.Fatalf("step=%d checkpoint read/quant upload reuse failed", step)
		}
		st := s.ExpertRing()
		if !st.Enabled || st.BudgetBytes != f.budget || st.ResidentBytes != f.budget || st.PeakBytes > f.budget || st.ResidentCount != 3 {
			t.Fatalf("step=%d ring budget=%d resident=%d peak=%d count=%d", step, st.BudgetBytes, st.ResidentBytes, st.PeakBytes, st.ResidentCount)
		}
		checkpointSessionNoPermanent(t, s)
	}
	st := s.ExpertRing()
	if st.Hits != 3 || st.PageIns != 9 || st.Evictions != 6 || f.be.matmuls != 12 || f.be.swiglu != 4 {
		t.Fatalf("ring/real HAL totals hits=%d pageins=%d evictions=%d matmuls=%d swiglu=%d", st.Hits, st.PageIns, st.Evictions, f.be.matmuls, f.be.swiglu)
	}
	if host := f.tier.Stats(); host.BudgetBytes != 0 || host.ResidentBytes != 0 || host.ResidentCount != 0 {
		t.Fatal("bounded device ring retained checkpoint bytes on host")
	}
}

func checkpointSessionMismatch(t *testing.T, f *checkpointSessionFixture, s *Session, observed int64) {
	t.Helper()
	beforeReads, beforeUploads := f.reads(), f.uploads()
	payload := deviceOnlyExpertRingPanic(func() { checkpointSessionForward(t, s, 0) })
	err, ok := payload.(error)
	var operation *BackendForwardOperationError
	var refusal *ExpertCheckpointDeviceBudgetError
	if !ok || !errors.As(err, &operation) || !errors.Is(err, ErrExpertCheckpointDeviceBudget) || !errors.As(err, &refusal) {
		t.Fatalf("mismatch payload=%v, want typed checkpoint operation refusal", payload)
	}
	if refusal.Tensor != checkpointSessionNames(0)[0] || refusal.Bytes != f.budget || refusal.Budget != observed {
		t.Fatalf("mismatch operands tensor=%s bytes=%d budget=%d", refusal.Tensor, refusal.Bytes, refusal.Budget)
	}
	if !s.BackendSessionClosed() || f.reads() != beforeReads || f.uploads() != beforeUploads {
		t.Fatal("mismatch failed to close before checkpoint IO/upload")
	}
	checkpointSessionNoPermanent(t, s)
}

// fak-test:runtime fast est=1s
func TestCheckpointSessionRejectsDeclaredBudgetOverrides(t *testing.T) {
	t.Parallel()
	for _, delta := range []int64{-1, 0, 1} {
		t.Run(fmt.Sprintf("override_%d", delta), func(t *testing.T) {
			t.Parallel()
			f := newCheckpointSessionFixture(t, true)
			s := f.session(t)
			observed := f.budget + delta
			if delta == 0 {
				observed = 0
			}
			s.ExpertRingBytes = observed
			checkpointSessionMismatch(t, f, s, observed)
		})
	}
}

// fak-test:runtime fast est=1s
func TestCheckpointSessionRejectsActualSharedRingMismatch(t *testing.T) {
	t.Parallel()
	f := newCheckpointSessionFixture(t, true)
	s := f.session(t)
	observed := f.budget * 2
	shared, err := NewSharedExpertRing(SharedExpertRingConfig{Model: f.m, Backend: f.be, BudgetBytes: observed})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = shared.Close() })
	if err := shared.Attach(s, "checkpoint-agent"); err != nil {
		t.Fatal(err)
	}
	s.ExpertRingBytes = f.budget
	checkpointSessionMismatch(t, f, s, observed)
}

// fak-test:runtime fast est=1s
func TestCheckpointSessionPortableCapacityRefusalCannotEscapeRing(t *testing.T) {
	t.Parallel()
	f := newCheckpointSessionFixture(t, true)
	s := f.session(t)
	s.ExpertRingBytes = f.budget
	checkpointSessionForward(t, s, 0)
	if s.executionPolicy != ExecutionPolicyPortable || s.expertRing == nil {
		t.Fatal("capacity fixture is not a portable bounded session")
	}
	for _, name := range checkpointSessionNames(0) {
		s.expertRing.hold("kquant-raw:" + name)
	}
	beforeReads, beforeUploads := f.reads(), f.uploads()
	payload := deviceOnlyExpertRingPanic(func() { checkpointSessionForward(t, s, 1) })
	err, ok := payload.(error)
	var operation *BackendForwardOperationError
	if !ok || !errors.As(err, &operation) || !errors.Is(err, polymodel.ErrPinnedNoRoom) {
		t.Fatalf("portable pinned-capacity payload=%v, want typed original pool refusal", payload)
	}
	if !s.BackendSessionClosed() || f.reads() != beforeReads || f.uploads() != beforeUploads {
		t.Fatal("capacity refusal escaped to checkpoint IO/upload or left session open")
	}
	checkpointSessionNoPermanent(t, s)
}
