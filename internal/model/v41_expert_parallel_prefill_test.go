package model

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The packed tiles below execute the CPU reference. These counters witness the
// reachable Session scheduler and real TCP protocol, never device throughput.
type v41EPPrefillTestBackend struct {
	*v41EPOwnerPanelBackend
	scalarMatMuls int
	expertWork    bool
}

// This expert-only recorder does not implement the unrelated ordinary BF16 or
// grouped-output backend ownership contracts. Keep those full-forward seams
// on their existing preselection host paths; advertise expert capabilities
// only under the actual bound routed callback's Session execution ownership.
func (b *v41EPPrefillTestBackend) SupportsV41BF16Publish(n int) bool {
	return b.expertWork && n > 0
}
func (b *v41EPPrefillTestBackend) SupportsV41DeferredStatus(n int) bool {
	return b.expertWork && b.enabled && n > 0
}
func (b *v41EPPrefillTestBackend) V41RowLogAvailable() bool { return b.expertWork }

func (b *v41EPPrefillTestBackend) expertScope() func() {
	previous := b.expertWork
	b.expertWork = true
	return func() { b.expertWork = previous }
}

func newV41EPPrefillTestBackend() *v41EPPrefillTestBackend {
	return &v41EPPrefillTestBackend{v41EPOwnerPanelBackend: &v41EPOwnerPanelBackend{v41ExpertDownTileWitness: &v41ExpertDownTileWitness{
		v41ExpertTileWitness: &v41ExpertTileWitness{
			v41ScalarCohortWitness: &v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: true},
			tiles:                  true, ownedTiles: map[compute.Buffer]bool{},
		}, downTiles: true,
	}}}
}

func (b *v41EPPrefillTestBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	if w.Dtype == compute.Q2_K || w.Dtype == compute.Q3_K {
		b.scalarMatMuls++
	}
	return b.v41EPOwnerPanelBackend.MatMul(w, x)
}

type v41EPPrefillWire struct {
	net.Conn
	bytes []byte
}

func (w *v41EPPrefillWire) Write(p []byte) (int, error) {
	n, err := w.Conn.Write(p)
	w.bytes = append(w.bytes, p[:n]...)
	return n, err
}

func (w *v41EPPrefillWire) checkDigests(want [][32]byte) error {
	round := 0
	for data := w.bytes; len(data) != 0; {
		if len(data) < 4 {
			return errors.New("truncated recorded frame")
		}
		n := int(binary.LittleEndian.Uint32(data[:4]))
		if n <= 0 || n > len(data)-4 {
			return errors.New("invalid recorded frame span")
		}
		payload := data[4 : 4+n]
		data = data[4+n:]
		if distOp(payload[0]) != opV41EPReduce {
			continue
		}
		part, _, err := decodeF32(payload[1:])
		if err != nil || len(part) < v41EPHeaderWords || round >= len(want) {
			return errors.New("invalid recorded reduction")
		}
		var got [32]byte
		for i := 0; i < 8; i++ {
			binary.LittleEndian.PutUint32(got[4*i:], math.Float32bits(part[i+2]))
		}
		if got != want[round] {
			return fmt.Errorf("round %d lost shared ordinal, position, history or domain", round)
		}
		round++
	}
	if round != len(want) {
		return fmt.Errorf("recorded %d reductions, want %d", round, len(want))
	}
	return nil
}

func v41EPPrefillShard(full *Model, band TPShard) *Model {
	m := v41EPFixtureShard(full, band)
	m.kqw = map[string]*kQuantTensor{}
	for expert := band.Lo; expert < band.Hi; expert++ {
		for _, leaf := range []string{"w1", "w3", "w2"} {
			name := layerName(0, "ffn.experts."+itoa(expert)+"."+leaf+".weight")
			m.kqw[name] = full.kqw[name]
		}
	}
	return m
}

// One witness exercises actual Session.Prefill, PrefillNoLogits, suffix Prefill
// and Step against the previous scalar-owner schedule on a real TCP group.
// Duplicate slots carry different coefficients; 35 tokens cross the 32-token
// bound with a short tail. Skew leaves the root and another owner empty.
// fak-test:runtime medium est=10s lane=default
func TestV41ExpertParallelPrefillPanelTCP(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41PrefillExpertPanelEnv, "0")
	full := v41IncrementalExpertFixture(t, true, false)
	defer full.CloseWeights()
	full.Cfg.DeepSeekV41.CompressRatios = []int{0}
	full.Cfg.DeepSeekV41.IndexSourceLayerIDs = nil
	owners, _ := ExpertParallelPlan(full.Cfg.NumExperts, 3)
	binding := v41EPFixtureBinding(full)
	prompt := make([]int, 35)
	for i := range prompt {
		prompt[i] = 1 + i%7
	}
	type result struct {
		logits [][]float32
		state  []*v41ForwardSnapshot
	}
	for _, mode := range []string{"balanced", "skew_empty"} {
		t.Run(mode, func(t *testing.T) {
			experts := []int{1, 1, 129, 129, 257, 257}
			if mode == "skew_empty" {
				experts = []int{257, 257, 258, 258, 259, 259}
			}
			picks := make([]routePick, len(experts))
			for i, expert := range experts {
				picks[i] = routePick{expert: expert, weight: float32(i+1) / 32}
			}
			var reference []result
			for _, scalar := range []bool{true, false} {
				results := make([]result, 3)
				backends := make([]*v41EPPrefillTestBackend, 3)
				scalarCalls := make([]int, 3)
				models := make([]*Model, 3)
				for rank, band := range owners.Shards {
					models[rank] = v41EPPrefillShard(full, band)
				}
				errs := v41EPTestGroup(t, func(g *DistComm) error {
					rank, m := g.Rank(), models[g.Rank()]
					if err := v41EPConfigure(m, g, binding); err != nil {
						return err
					}
					b := newV41EPPrefillTestBackend()
					backends[rank] = b
					var wire *v41EPPrefillWire
					if rank != 0 {
						wire = &v41EPPrefillWire{Conn: g.coord}
						g.coord = wire
					}
					var digests [][32]byte
					var sessions []*Session
					defer func() {
						for _, s := range sessions {
							s.Close()
						}
						_ = m.CloseWeights()
					}()
					newSession := func() *Session {
						s := m.NewBackendSession(b)
						sessions = append(sessions, s)
						st := s.v41State()
						if b.expertWork || b.V41RowLogAvailable() || st.gatherOps.bf16PublishFunc() != nil || st.bf16Boundary != nil || st.routedExpert == nil {
							panic("expert-only recorder bound unrelated capability or omitted packed owner binding")
						}
						st.routerTopK = func(int, []float32, []float32, v41RouterConfig, bool) ([]routePick, error) {
							return append([]routePick(nil), picks...), nil
						}
						routed := st.routedExpert
						st.routedExpert = func(l int, stem string, x []float32, weight float32) ([]float32, v41RoutedExpertOutcome, error) {
							scalarCalls[rank]++
							return routed(l, stem, x, weight)
						}
						config, _ := json.Marshal(m.Cfg)
						configHash := sha256.Sum256(config)
						c := m.epColl.(*v41ExpertParallelCollective)
						var ordinal uint64
						note := func(l, position int, prefill bool, p []routePick, x []float32) {
							ordinal++
							d := v41EPIdentity(c.binding, configHash, ordinal, l, st.history, p, x)
							digests = append(digests, v41EPPositionIdentity(d, position, prefill))
						}
						step, prefill := st.routedLayer, st.routedPrefill
						st.routedLayer = func(l int, p []routePick, x []float32) ([]float32, v41RoutedExpertOutcome, error) {
							defer b.expertScope()()
							note(l, len(st.history), false, p, x)
							return step(l, p, x)
						}
						st.routedPrefill = func(l int, p [][]routePick, x [][]float32, scratch *v41ProjScratch) ([][]float32, error) {
							defer b.expertScope()()
							if !scalar {
								for token := range p {
									note(l, token, true, p[token], x[token])
								}
								return prefill(l, p, x, scratch)
							}
							// The old full-forward owner loop is the independent oracle.
							rows := make([][]float32, len(p))
							for token := range p {
								row, outcome, err := st.routedLayer(l, p[token], x[token])
								if err != nil || outcome != v41RoutedExpertHandled {
									return nil, fmt.Errorf("scalar oracle: %v", err)
								}
								rows[token] = row
							}
							return rows, nil
						}
						return s
					}
					if rank == 0 {
						coord, err := NewEPDecodeCoordinator(g)
						if err != nil {
							return err
						}
						m.SetEPDecodeCoordinator(coord)
						s := newSession()
						results[rank].logits = append(results[rank].logits, s.Prefill(prompt), s.Step(4), s.Prefill([]int{5, 6}))
						s.Close()
						s = newSession()
						results[rank].logits = append(results[rank].logits, s.Prefill([]int{2}), s.Step(3))
						s.Close()
						s = newSession()
						s.PrefillNoLogits([]int{2, 1, 3})
						results[rank].logits = append(results[rank].logits, s.Step(4))
						s.Close()
						if err := coord.Shutdown(); err != nil {
							return err
						}
					} else {
						if err := RunEPFollower(g, newSession); err != nil {
							return err
						}
						if err := wire.checkDigests(digests); err != nil {
							return err
						}
					}
					for _, s := range sessions {
						results[rank].state = append(results[rank].state, captureV41ForwardSnapshot(s.v41Forward))
						if s.modelWeightsHeld || len(s.halW) != 0 {
							return errors.New("closed Session retained owner weights")
						}
					}
					if m.V41ExpertParallelTransportStats().CompletedGathers != 44 || b.expertWork || len(b.pending) != 0 || len(b.ownedTiles) != 0 {
						return errors.New("missing exact rounds or retained cohort temporaries")
					}
					return nil
				})
				for rank, err := range errs {
					if err != nil {
						t.Fatalf("scalar=%t rank=%d: %v", scalar, rank, err)
					}
					owned := 0
					for _, pick := range picks {
						if pick.expert >= owners.Shards[rank].Lo && pick.expert < owners.Shards[rank].Hi {
							owned++
						}
					}
					b, wantScalar := backends[rank], 44*owned
					if !scalar {
						wantScalar = 6 * owned
					}
					if scalarCalls[rank] != wantScalar || b.scalarMatMuls != 3*wantScalar || b.calls != 44*owned {
						t.Fatalf("scalar=%t rank=%d scalar rows=%d/%d scalar GEMVs=%d activations=%d", scalar, rank, scalarCalls[rank], wantScalar, b.scalarMatMuls, b.calls)
					}
					wantCohorts, wantTiles := 0, 0
					if !scalar {
						wantCohorts, wantTiles = 2*owned, 12*owned
					}
					if len(b.cohortRows) != wantCohorts || len(b.tileRows) != wantTiles || len(b.downRows)*2 != wantTiles {
						t.Fatalf("scalar=%t rank=%d cohorts=%v Q2 tiles=%v Q3 tiles=%v", scalar, rank, b.cohortRows, b.tileRows, b.downRows)
					}
					for _, rows := range b.cohortRows {
						if rows > 32 || rows < 2 {
							t.Fatal("unbounded cohort")
						}
					}
					for _, rows := range append(append([]int(nil), b.tileRows...), b.downRows...) {
						if rows != 2 && rows != 4 && rows != 8 {
							t.Fatal("unbounded packed tile")
						}
					}
				}
				if scalar {
					reference = results
				} else {
					for i, row := range results[0].logits {
						if err := v41EPCheckBits(row, reference[0].logits[i]); err != nil {
							t.Fatalf("operation %d: %v", i, err)
						}
					}
					if !reflect.DeepEqual(results, reference) {
						t.Fatal("per-owner committed continuation differs from scalar owner schedule")
					}
				}
			}
		})
	}
	for _, test := range []struct {
		name   string
		tile   int
		rounds uint64
	}{
		{"failure_before_first_reduce", 1, 0},
		{"failure_after_first_window", 49, 32},
	} {
		t.Run(test.name, func(t *testing.T) {
			failure := &compute.BackendError{Backend: "test", Site: "prefill Q2 tile", Err: errors.New("selected tile failed")}
			models := make([]*Model, 3)
			backends := make([]*v41EPPrefillTestBackend, 3)
			sessions := make([]*Session, 3)
			scalarCalls := make([]int, 3)
			for rank, band := range owners.Shards {
				models[rank] = v41EPPrefillShard(full, band)
			}
			errs := v41EPTestGroup(t, func(g *DistComm) error {
				rank, m := g.Rank(), models[g.Rank()]
				if err := v41EPConfigure(m, g, binding); err != nil {
					return err
				}
				b := newV41EPPrefillTestBackend()
				backends[rank] = b
				if rank == 2 {
					b.failTileAt, b.tileFailure = test.tile, failure
				}
				defer func() {
					if s := sessions[rank]; s != nil {
						s.Close()
					}
					_ = m.CloseWeights()
				}()
				newSession := func() *Session {
					s := m.NewBackendSession(b)
					sessions[rank] = s
					st := s.v41State()
					if b.expertWork || b.V41RowLogAvailable() || st.gatherOps.bf16PublishFunc() != nil || st.bf16Boundary != nil || st.routedExpert == nil {
						panic("expert-only recorder bound unrelated capability or omitted packed owner binding")
					}
					prefill, step := st.routedPrefill, st.routedLayer
					st.routedPrefill = func(l int, p [][]routePick, x [][]float32, scratch *v41ProjScratch) ([][]float32, error) {
						defer b.expertScope()()
						return prefill(l, p, x, scratch)
					}
					st.routedLayer = func(l int, p []routePick, x []float32) ([]float32, v41RoutedExpertOutcome, error) {
						defer b.expertScope()()
						return step(l, p, x)
					}
					st.routerTopK = func(int, []float32, []float32, v41RouterConfig, bool) ([]routePick, error) {
						return []routePick{{257, .125}, {257, .25}, {258, .125}, {258, .25}, {259, .125}, {259, .25}}, nil
					}
					routed := st.routedExpert
					st.routedExpert = func(l int, stem string, x []float32, weight float32) ([]float32, v41RoutedExpertOutcome, error) {
						scalarCalls[rank]++
						return routed(l, stem, x, weight)
					}
					return s
				}
				var err error
				if rank == 0 {
					coord, cause := NewEPDecodeCoordinator(g)
					if cause != nil {
						return cause
					}
					m.SetEPDecodeCoordinator(coord)
					s := newSession()
					var got []float32
					err = panicAsError(func() { got = s.Prefill(prompt) })
					if got != nil {
						return errors.New("failed Prefill returned partial logits")
					}
				} else {
					err = RunEPFollower(g, newSession)
				}
				s := sessions[rank]
				if s == nil || len(s.v41Forward.history) != 0 || len(s.v41Forward.layers) != 0 || s.v41Forward.forwardCoverage != nil {
					return fmt.Errorf("owner %d published failed continuation", rank)
				}
				c := m.epColl.(*v41ExpertParallelCollective)
				calls := b.calls
				out, retry := s.v41Forward.routedPrefill(0, nil, nil, &v41ProjScratch{})
				if c.failure == nil || retry != c.failure || out != nil || b.calls != calls {
					return errors.New("failed panel was reusable or replayed")
				}
				return err
			})
			for rank, err := range errs {
				b, s := backends[rank], sessions[rank]
				if !errors.Is(err, ErrV41ExpertParallel) || rank == 2 && !errors.Is(err, failure) {
					t.Fatalf("rank %d lost selected/terminal error: %v", rank, err)
				}
				// The failed owner finished round 32 before selecting its next
				// tile. Its socket abort may race a peer's deadline-clear/count
				// step for that same round; remote accounting is not atomic.
				minimum := test.rounds
				if rank != 2 && minimum > 0 {
					minimum--
				}
				rounds := models[rank].V41ExpertParallelTransportStats().CompletedGathers
				if rounds < minimum || rounds > test.rounds || b.expertWork || scalarCalls[rank] != 0 || len(b.pending) != 0 || len(b.ownedTiles) != 0 || len(s.halW) != 0 || s.modelWeightsHeld {
					t.Fatalf("rank %d partial success, replay or retained ownership: rounds=%d scalar=%d", rank, rounds, scalarCalls[rank])
				}
			}
		})
	}
	t.Run("snapshot_omits_and_rebinds_callback", func(t *testing.T) {
		m := v41EPPrefillShard(full, owners.Shards[0])
		defer m.CloseWeights()
		if err := v41EPConfigure(m, &DistComm{rank: 0, size: 3}, binding); err != nil {
			t.Fatal(err)
		}
		s := m.NewSession()
		defer s.Close()
		st := s.v41State()
		st.history = []int{1}
		restored := captureV41ForwardSnapshot(st).clone().restore()
		if restored.routedPrefill != nil || restored.routedLayer != nil || restored.callbackOwner != nil {
			t.Fatal("snapshot captured Session callbacks")
		}
		other := m.NewSession()
		defer other.Close()
		other.v41Forward = restored
		bound := other.v41State()
		if bound.routedPrefill == nil || bound.routedLayer == nil || bound.callbackOwner != other {
			t.Fatal("restored state did not bind the destination Session")
		}
		if _, err := m.forwardV41([]int{1}, nil); !errors.Is(err, ErrV41ExpertParallel) {
			t.Fatal("Model.Forward bypassed the Session EP binding")
		}
	})
}
