package model

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// This backend executes through cpu-ref. Its recorded native-seam dispatches
// establish ownership, selection and lifetime wiring, never GPU qualification.
type v41EPOwnerTestBackend struct {
	*v41RoutedSwiGLUTestBackend
	missingQ3      bool
	uploads        map[compute.Buffer]bool
	frees          map[compute.Buffer]int
	expertBuffers  map[compute.Buffer]bool
	session        *Session
	expertCalls    map[string]int
	ownershipError error
}

type v41EPOwnerTestDescriptor struct {
	QuantDescriptor
	dtype                    compute.Dtype
	blockWeights, blockBytes int
}

func (d v41EPOwnerTestDescriptor) Dtype() compute.Dtype { return d.dtype }
func (d v41EPOwnerTestDescriptor) BlockWeights() int    { return d.blockWeights }
func (d v41EPOwnerTestDescriptor) BlockBytes() int      { return d.blockBytes }

func newV41EPOwnerTestBackend() *v41EPOwnerTestBackend {
	return &v41EPOwnerTestBackend{v41RoutedSwiGLUTestBackend: newV41RoutedSwiGLUTestBackend(), uploads: map[compute.Buffer]bool{}, frees: map[compute.Buffer]int{}, expertBuffers: map[compute.Buffer]bool{}, expertCalls: map[string]int{}}
}

func (b *v41EPOwnerTestBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return (!b.missingQ3 || dt != compute.Q3_K) && b.v41HalSeamBackend.SupportsDeviceWeightDtype(dt)
}

func (b *v41EPOwnerTestBackend) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	out := b.Backend.Upload(x, dt)
	b.uploads[out.Buf()] = true
	return out
}

func (b *v41EPOwnerTestBackend) Free(x compute.Tensor) {
	delete(b.uploads, x.Buf())
	b.frees[x.Buf()]++
	b.Backend.Free(x)
}

func (b *v41EPOwnerTestBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	if b.session != nil {
		for name, staged := range b.session.halW {
			if staged.Buf() != w.Buf() || !strings.Contains(name, ".ffn.experts.") {
				continue
			}
			canonical := strings.TrimPrefix(name, "kquant-raw:")
			qt := b.session.M.kqw[canonical]
			want := compute.Q2_K
			if strings.HasSuffix(name, ".w2.weight") {
				want = compute.Q3_K
			}
			if qt == nil || w.Dtype != want {
				b.ownershipError = fmt.Errorf("foreign or wrong-dtype projection %s", name)
			}
			b.expertCalls[canonical]++
			b.expertBuffers[w.Buf()] = true
		}
	}
	return b.v41HalSeamBackend.MatMul(w, x)
}

func v41EPOwnerPackedFixture(rank int) (*Session, TPPlan, *v41EPOwnerTestBackend) {
	cfg := Config{ModelType: "deepseek41", NumLayers: 1, NumExperts: 7, HiddenSize: 256, MoEIntermediateSize: 256, HeadDim: v41KVLoraRank}
	m := &Model{Cfg: cfg, kqw: map[string]*kQuantTensor{}, epRank: rank}
	_ = v41EPConfigure(m, &DistComm{rank: rank, size: 3}, v41EPFixtureBinding(m))
	owners, _ := ExpertParallelPlan(cfg.NumExperts, 3)
	for expert := owners.Shards[rank].Lo; expert < owners.Shards[rank].Hi; expert++ {
		stem := "ffn.experts." + itoa(expert)
		m.kqw[layerName(0, stem+".w1.weight")] = q2kFixtureTensor(256, 256, uint64(100+expert))
		m.kqw[layerName(0, stem+".w3.weight")] = q2kFixtureTensor(256, 256, uint64(200+expert))
		m.kqw[layerName(0, stem+".w2.weight")] = q3kFixtureTensor(256, 256)
	}
	b := newV41EPOwnerTestBackend()
	return &Session{M: m, Backend: b, halW: map[string]compute.Tensor{}}, owners, b
}

func v41EPOwnerTestBind(s *Session, routed v41RoutedExpertFunc) (v41RoutedExpertFunc, error) {
	st := &v41ForwardState{routedExpert: routed, callbackOwner: s}
	s.v41Forward = st
	err := panicAsError(func() { s.bindV41ExpertParallel(st) })
	if err != nil {
		if st.callbackOwner != nil {
			return nil, fmt.Errorf("refused binding retained callback owner: %w", err)
		}
		return nil, err
	}
	return st.routedExpert, nil
}

// fak-test:runtime medium est=5s lane=default
func TestV41ExpertParallelDeviceOwner(t *testing.T) {
	t.Run("admission", func(t *testing.T) {
		for _, mode := range []string{"valid", "foreign-packed", "foreign-f32", "missing", "shape", "dtype", "bytes", "blocks", "bias", "missing-q3", "missing-operation", "offload", "descriptor-dtype", "descriptor-geometry"} {
			t.Run(mode, func(t *testing.T) {
				s, _, b := v41EPOwnerPackedFixture(1)
				defer s.Close()
				name := layerName(0, "ffn.experts.3.w2.weight")
				switch mode {
				case "foreign-packed":
					s.M.kqw[layerName(0, "ffn.experts.0.w2.weight")] = q3kFixtureTensor(256, 256)
				case "foreign-f32":
					s.M.manifest = map[string]tensorMeta{layerName(0, "ffn.experts.0.w2.weight"): {Shape: []int{256, 256}}}
				case "missing":
					delete(s.M.kqw, name)
				case "shape":
					s.M.kqw[name].out++
				case "dtype":
					s.M.kqw[name].kind = kindQ2K
				case "bytes":
					s.M.kqw[name].raw = s.M.kqw[name].raw[:len(s.M.kqw[name].raw)-1]
				case "blocks":
					s.M.kqw[name].nblk++
				case "bias":
					s.M.manifest = map[string]tensorMeta{strings.TrimSuffix(name, "weight") + "bias": {Shape: []int{256}}}
				case "missing-q3":
					b.missingQ3 = true
				case "missing-operation":
					b.available = false
				case "offload":
					s.CPUOffloadExperts = true
				case "descriptor-dtype", "descriptor-geometry":
					original, _ := LookupQuantDescriptor(kindQ3K)
					t.Cleanup(func() { RegisterQuantDescriptor(original) })
					desc := v41EPOwnerTestDescriptor{QuantDescriptor: original, dtype: compute.Q3_K, blockWeights: qkK, blockBytes: q3kBlockBytes}
					if mode == "descriptor-dtype" {
						desc.dtype = compute.Q4_K
					} else {
						// Keep total bytes self-consistent with the overwritten
						// descriptor. Native Q3_K geometry must still reject it.
						desc.blockWeights, desc.blockBytes = qkK/2, q3kBlockBytes/2
						for _, qt := range s.M.kqw {
							if qt.kind == kindQ3K {
								qt.nblk *= 2
							}
						}
					}
					RegisterQuantDescriptor(desc)
				}
				bound, err := v41EPOwnerTestBind(s, s.v41RoutedExpertFunc())
				if mode == "valid" {
					if err != nil || bound == nil {
						t.Fatalf("valid owner: %v", err)
					}
				} else if !errors.Is(err, ErrV41ExpertParallel) || bound != nil {
					t.Fatalf("invalid owner accepted: %v", err)
				}
				if len(b.uploads) != 0 || b.matmuls != 0 || b.reads != 0 {
					t.Fatal("binding admission performed device work")
				}
			})
		}
	})
	t.Run("decline-is-terminal", func(t *testing.T) {
		s, _, b := v41EPOwnerPackedFixture(1)
		defer s.Close()
		routed, err := v41EPOwnerTestBind(s, s.v41RoutedExpertFunc())
		if err != nil {
			t.Fatal(err)
		}
		// This is the production callback's real admission decline, after the
		// session was bound. No gate/up, down or whole-host retry is permitted.
		s.M.Cfg.ActGeluTanh = true
		hostCalls := 0
		_, err = s.M.v41FullRoutedExpertWithDevice(0, "ffn.experts.3", make([]float32, 256), .5, s.M.Cfg, routed,
			func(int, string, []float32) ([]float32, v41ExpertGateUpOutcome, error) {
				hostCalls++
				return nil, v41GateUpDeclined, nil
			},
			nil, func() ([]float32, []float32, []float32, error) { hostCalls++; return nil, nil, nil, nil }, nil, false)
		if !errors.Is(err, ErrV41ExpertParallel) || hostCalls != 0 || b.matmuls != 0 {
			t.Fatalf("decline err=%v host=%d matmuls=%d", err, hostCalls, b.matmuls)
		}
		for _, be := range []compute.Backend{nil, compute.Default()} {
			s.Backend = be
			decline := func(int, string, []float32, float32) ([]float32, v41RoutedExpertOutcome, error) {
				return nil, v41RoutedExpertDeclined, nil
			}
			unchanged, err := v41EPOwnerTestBind(s, decline)
			if err != nil {
				t.Fatal(err)
			}
			if _, outcome, err := unchanged(0, "", nil, 1); outcome != v41RoutedExpertDeclined || err != nil {
				t.Fatal("host transport policy changed")
			}
		}
	})
	t.Run("selected-failure", func(t *testing.T) {
		s, _, b := v41EPOwnerPackedFixture(1)
		defer s.Close()
		routed, err := v41EPOwnerTestBind(s, s.v41RoutedExpertFunc())
		if err != nil {
			t.Fatal(err)
		}
		failure := &compute.BackendError{Backend: "test", Site: "V41RoutedSwiGLU", Err: errors.New("selected owner operation failed")}
		b.failure = failure
		_, outcome, err := routed(0, "ffn.experts.3", make([]float32, 256), .5)
		if outcome != v41RoutedExpertError || !errors.Is(err, failure) || !s.BackendSessionClosed() || len(b.uploads) != 0 || b.matmuls != 2 {
			t.Fatalf("selected failure outcome=%v err=%v closed=%t live=%d matmuls=%d", outcome, err, s.BackendSessionClosed(), len(b.uploads), b.matmuls)
		}
	})
	t.Run("session-tcp", func(t *testing.T) {
		full := v41IncrementalExpertFixture(t, true, false)
		bias := make([]float32, full.Cfg.NumExperts)
		selectedExperts := []int{1, 2, 129, 130, 257, 258}
		for i, expert := range selectedExperts {
			bias[expert] = float32(100 + i)
		}
		v41WriteTensorF32(t, full, layerName(0, "ffn.gate.e_score_correction_bias"), bias)
		binding := v41EPFixtureBinding(full)
		owners, _ := ExpertParallelPlan(full.Cfg.NumExperts, 3)
		models := make([]*Model, 3)
		for rank, band := range owners.Shards {
			m := v41EPFixtureShard(full, band)
			m.kqw = map[string]*kQuantTensor{}
			for expert := band.Lo; expert < band.Hi; expert++ {
				for _, leaf := range []string{"w1", "w3", "w2"} {
					name := layerName(0, "ffn.experts."+itoa(expert)+"."+leaf+".weight")
					m.kqw[name] = full.kqw[name]
				}
			}
			models[rank] = m
		}
		ref := full.NewBackendSession(newV41EPOwnerTestBackend())
		defer full.CloseWeights()
		want := [][]float32{ref.Prefill([]int{1, 2}), ref.Step(3)}
		ref.Close()
		errByRank := v41EPTestGroup(t, func(g *DistComm) error {
			rank, m := g.Rank(), models[g.Rank()]
			if err := v41EPConfigure(m, g, binding); err != nil {
				return err
			}
			b := newV41EPOwnerTestBackend()
			var session *Session
			newSession := func() *Session {
				session = m.NewBackendSession(b)
				b.session = session
				return session
			}
			defer func() {
				if session != nil {
					session.Close()
				}
				_ = m.CloseWeights()
			}()
			ownedRouted := make(map[int]bool)
			band := owners.Shards[rank]
			for _, expert := range selectedExperts {
				if expert >= band.Lo && expert < band.Hi {
					ownedRouted[expert] = true
				}
			}
			check := func() error {
				if b.ownershipError != nil {
					return b.ownershipError
				}
				if len(b.expertCalls) != 3*len(ownedRouted) || b.calls != 6 {
					return fmt.Errorf("rank %d projection identities=%d routed calls=%d, want 6/6", rank, len(b.expertCalls), b.calls)
				}
				for name, count := range b.expertCalls {
					if count != 3 {
						return fmt.Errorf("rank %d %s dispatches=%d want 3", rank, name, count)
					}
				}
				// Session.Close retires private Q2/Q3 expert weights. Dense
				// F32 weights belong to the model and intentionally survive.
				for buffer := range b.expertBuffers {
					if b.uploads[buffer] || b.frees[buffer] != 1 {
						return fmt.Errorf("rank %d expert buffer live=%t frees=%d", rank, b.uploads[buffer], b.frees[buffer])
					}
				}
				remaining := make(map[compute.Buffer]bool, len(b.uploads))
				for buffer := range b.uploads {
					remaining[buffer] = true
				}
				modelOwned := 0
				if !m.VisitImmutableDeviceWeights(b, func(weight compute.Tensor) bool {
					if weight.Dtype != compute.F32 || !b.uploads[weight.Buf()] || b.frees[weight.Buf()] != 0 {
						return false
					}
					delete(remaining, weight.Buf())
					modelOwned++
					return true
				}) || modelOwned == 0 || len(remaining) != 0 {
					return fmt.Errorf("rank %d retained uploads=%d model-owned F32=%d", rank, len(b.uploads), modelOwned)
				}
				for buffer := range b.uploads {
					remaining[buffer] = true
				}
				if err := m.CloseWeights(); err != nil {
					return err
				}
				if len(b.uploads) != 0 {
					return fmt.Errorf("rank %d model close retained %d uploads", rank, len(b.uploads))
				}
				for buffer := range remaining {
					if b.frees[buffer] != 1 {
						return fmt.Errorf("rank %d model buffer frees=%d", rank, b.frees[buffer])
					}
				}
				return nil
			}
			if rank != 0 {
				if err := RunEPFollower(g, newSession); err != nil {
					return err
				}
				return check()
			}
			coord, err := NewEPDecodeCoordinator(g)
			if err != nil {
				return err
			}
			m.SetEPDecodeCoordinator(coord)
			s := newSession()
			got := [][]float32{s.Prefill([]int{1, 2}), s.Step(3)}
			for i := range got {
				if err := v41EPCheckBits(got[i], want[i]); err != nil {
					return err
				}
			}
			s.Close()
			if err := check(); err != nil {
				return err
			}
			return coord.Shutdown()
		})
		for rank, err := range errByRank {
			if err != nil {
				t.Fatalf("rank %d: %v", rank, err)
			}
		}
	})
}
