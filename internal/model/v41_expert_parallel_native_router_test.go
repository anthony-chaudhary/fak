package model

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The outer recorder owns ordinary F32 dense/router tensors. Inside the
// qualified expert scope, the inner recorder retains its packed-weight and
// cohort ownership topology. All math remains CPU reference math.
type v41EPNativeRouterBackend struct {
	*v41EPPrefillTestBackend
	owner                   sync.Mutex
	allocated               map[compute.Buffer]bool
	live                    map[compute.Buffer]string
	frees                   map[compute.Buffer]int
	gateSites               map[string]int
	gateWeights             map[compute.Buffer]int
	gateUploads             map[int]int
	gateOutputs             map[compute.Buffer]uint64
	experts, width          int
	gateReads, logitUploads uint64
	native, router, sums    int
	fault                   error
}

func newV41EPNativeRouterBackend(cfg Config) *v41EPNativeRouterBackend {
	b := &v41EPNativeRouterBackend{
		v41EPPrefillTestBackend: newV41EPPrefillTestBackend(),
		allocated:               map[compute.Buffer]bool{}, live: map[compute.Buffer]string{}, frees: map[compute.Buffer]int{},
		gateSites: map[string]int{}, gateWeights: map[compute.Buffer]int{}, gateUploads: map[int]int{}, gateOutputs: map[compute.Buffer]uint64{},
		experts: cfg.NumExperts, width: cfg.HiddenSize,
	}
	for layer := 0; layer < cfg.NumLayers; layer++ {
		b.gateSites["hal-weight "+layerName(layer, "ffn.gate.weight")] = layer
	}
	return b
}

func (b *v41EPNativeRouterBackend) ForwardExecutionOwner() *sync.Mutex { return &b.owner }

func (b *v41EPNativeRouterBackend) own(shape []int, values []float32, site string) compute.Tensor {
	out := compute.NewF32(b, append([]int(nil), shape...), append([]float32(nil), values...))
	if b.allocated[out.Buf()] {
		panic("native recorder reused an allocation identity")
	}
	b.allocated[out.Buf()], b.live[out.Buf()] = true, site
	return out
}

func (b *v41EPNativeRouterBackend) Upload(x compute.Tensor, dtype compute.Dtype) compute.Tensor {
	return b.UploadClass(x, dtype, compute.MemoryActivation, "unclassified")
}

func (b *v41EPNativeRouterBackend) UploadClass(x compute.Tensor, dtype compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if site == "V4.1 routed-sum accumulator" {
		b.sums++
	}
	if b.expertWork || dtype != compute.F32 {
		return b.v41EPPrefillTestBackend.Upload(x, dtype)
	}
	out := b.own(x.Shape, b.v41EPPrefillTestBackend.Read(x), site)
	if layer, gate := b.gateSites[site]; gate {
		if class != compute.MemoryWeights || !reflect.DeepEqual(x.Shape, []int{b.experts, b.width}) || b.gateUploads[layer] != 0 {
			panic("canonical router gate upload lost its exact identity")
		}
		b.gateWeights[out.Buf()], b.gateUploads[layer] = layer, 1
	}
	if site == "V4.1 router logits" {
		b.logitUploads += uint64(out.Numel()) * 4
	}
	return out
}

func (b *v41EPNativeRouterBackend) denseOutput(weight, temporary compute.Tensor) compute.Tensor {
	defer b.v41EPPrefillTestBackend.Free(temporary)
	out := b.own(temporary.Shape, b.v41EPPrefillTestBackend.Read(temporary), "dense output")
	if _, gate := b.gateWeights[weight.Buf()]; gate {
		b.gateOutputs[out.Buf()] = uint64(out.Numel()) * 4
	}
	return out
}

func (b *v41EPNativeRouterBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	if b.expertWork || w.Dtype != compute.F32 {
		return b.v41EPPrefillTestBackend.MatMul(w, x)
	}
	return b.denseOutput(w, b.v41EPPrefillTestBackend.MatMul(w, x))
}

func (b *v41EPNativeRouterBackend) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	if b.expertWork || w.Dtype != compute.F32 {
		return b.v41EPPrefillTestBackend.BatchedMatMul(w, x, rows)
	}
	return b.denseOutput(w, b.v41EPPrefillTestBackend.BatchedMatMul(w, x, rows))
}

func (b *v41EPNativeRouterBackend) requireOuter(x compute.Tensor) {
	if _, live := b.live[x.Buf()]; !b.allocated[x.Buf()] || !live || x.Backend() != b {
		panic("native recorder rejected a foreign or retired buffer")
	}
}

func (b *v41EPNativeRouterBackend) Read(x compute.Tensor) []float32 {
	if b.allocated[x.Buf()] {
		b.requireOuter(x)
		b.gateReads += b.gateOutputs[x.Buf()]
	} else if x.Backend() != b.v41EPPrefillTestBackend.Backend {
		panic("native recorder rejected an unknown Read owner")
	}
	return b.v41EPPrefillTestBackend.Read(x)
}

func (b *v41EPNativeRouterBackend) Free(x compute.Tensor) {
	if b.allocated[x.Buf()] {
		b.requireOuter(x)
		delete(b.live, x.Buf())
		b.frees[x.Buf()]++
	} else if x.Backend() != b.v41EPPrefillTestBackend.Backend {
		panic("native recorder rejected an unknown Free owner")
	}
	// Preserve inner pending/tile bookkeeping even for permitted CPU outputs.
	b.v41EPPrefillTestBackend.Free(x)
}

func (*v41EPNativeRouterBackend) V41RouterTopKAvailable() bool { return true }

func (b *v41EPNativeRouterBackend) V41RouterTopK(logits, bias compute.Tensor, rows, experts, topK int, scale float32, full, hasBias bool) (compute.V41RouterPicks, error) {
	if b.expertWork || rows != 1 || logits.Dtype != compute.F32 || !reflect.DeepEqual(logits.Shape, []int{1, experts}) {
		return compute.V41RouterPicks{}, errors.New("canonical router entered the wrong ownership scope")
	}
	b.requireOuter(logits)
	b.router++
	if b.gateOutputs[logits.Buf()] != 0 {
		b.native++
		if b.fault != nil {
			return compute.V41RouterPicks{}, b.fault
		}
	}
	var correction []float32
	if hasBias {
		b.requireOuter(bias)
		correction = b.Backend.Read(bias)
	}
	// Internal CPU reads implement the recorder's math, not outer seam traffic.
	picks, err := v41RouteForGeometry(b.Backend.Read(logits), correction, v41RouterConfig{Experts: experts, TopK: topK, SharedCount: 1, RouteScale: scale}, full)
	if err != nil {
		return compute.V41RouterPicks{}, err
	}
	out := compute.V41RouterPicks{}
	for _, pick := range picks {
		out.Experts = append(out.Experts, int32(pick.expert))
		out.Weights = append(out.Weights, pick.weight)
	}
	return out, nil
}

func (b *v41EPNativeRouterBackend) retired() error {
	if b.expertWork || len(b.live) != 0 || len(b.pending) != 0 || len(b.ownedTiles) != 0 {
		return fmt.Errorf("retained fixture state: scope=%v outer=%d pending=%d tiles=%d", b.expertWork, len(b.live), len(b.pending), len(b.ownedTiles))
	}
	for buffer := range b.allocated {
		if b.frees[buffer] != 1 {
			return fmt.Errorf("outer allocation retired %d times", b.frees[buffer])
		}
	}
	return nil
}

type v41EPNativeRouterResult struct {
	logits  [][]float32
	states  []*v41ForwardSnapshot
	digests [][32]byte
	routes  [][]routePick
	failure error
}

// Actual canonical native-router selection inside distributed EP Step. These
// are outer API boundary counts; the recorder still copies/reads CPU storage.
// Existing 35-token controls cover panel boundaries separately.
// fak-test:runtime medium est=10s lane=default
func TestV41ExpertParallelNativeRouterTCP(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41PrefillExpertPanelEnv, "0")
	actions := [][]int{{1, 2, 3}, {4}, {5}}
	logicalRows := 0
	for _, ids := range actions {
		logicalRows += len(ids)
	}
	for _, mode := range []string{"plain", "role", "router_fault"} {
		t.Run(mode, func(t *testing.T) {
			full := v41IncrementalExpertFixture(t, true, false)
			defer full.CloseWeights()
			if mode != "role" {
				full.Cfg.DeepSeekV41.CompressRatios = []int{0}
				full.Cfg.DeepSeekV41.IndexSourceLayerIDs = nil
				full.Cfg.DeepSeekV41.KVSourceLayerIDs = nil
			}
			selectedExperts := []int{1, 2, 129, 130, 257, 258}
			bias := make([]float32, full.Cfg.NumExperts)
			for i, expert := range selectedExperts {
				bias[expert] = float32(100 + i)
			}
			v41WriteTensorF32(t, full, layerName(0, "ffn.gate.e_score_correction_bias"), bias)
			owners, _ := ExpertParallelPlan(full.Cfg.NumExperts, 3)
			binding := v41EPFixtureBinding(full)
			selected := errors.New("selected root native router fault")
			var reference []v41EPNativeRouterResult
			for _, ordinary := range []bool{true, false} {
				if mode == "router_fault" && ordinary {
					continue
				}
				results := make([]v41EPNativeRouterResult, 3)
				backends := make([]*v41EPNativeRouterBackend, 3)
				models := make([]*Model, 3)
				closeErrors := make([]error, 3)
				for rank, band := range owners.Shards {
					models[rank] = v41EPPrefillShard(full, band)
					backends[rank] = newV41EPNativeRouterBackend(full.Cfg)
				}
				var prefixReady sync.WaitGroup
				prefixReady.Add(3)
				errs := v41EPTestGroup(t, func(g *DistComm) error {
					rank, m, b := g.Rank(), models[g.Rank()], backends[g.Rank()]
					arrived := false
					defer func() {
						if !arrived {
							prefixReady.Done()
						}
					}()
					defer func() { closeErrors[rank] = m.CloseWeights() }()
					if err := v41EPConfigure(m, g, binding); err != nil {
						return err
					}
					var coord *EPDecodeCoordinator
					var wire *v41EPPrefillWire
					if rank == 0 {
						var err error
						coord, err = NewEPDecodeCoordinator(g)
						if err != nil {
							return err
						}
						m.SetEPDecodeCoordinator(coord)
					} else {
						wire = &v41EPPrefillWire{Conn: g.coord}
						g.coord = wire
					}
					s := m.NewBackendSession(b)
					defer s.Close()
					st := s.v41State()
					bound := st.projectionRoute
					if bound == nil || bound.nativeProjection.owner != s || bound.nativeRouter.owner != s || bound.nativeProjection.backend != b || bound.nativeRouter.backend != b || bound.nativeProjection.model != m || bound.nativeRouter.model != m || st.routedExpert == nil {
						return errors.New("fixture did not bind the canonical native pair and packed owner")
					}
					if ordinary {
						st.setDenseProjection(st.denseProjection())
						bound = st.projectionRoute
					}
					if b.expertWork || st.bf16Boundary != nil || st.gatherOps.bf16PublishFunc() != nil {
						return errors.New("fixture admitted unrelated full-forward capability")
					}
					config, _ := json.Marshal(m.Cfg)
					configHash := sha256.Sum256(config)
					c := m.epColl.(*v41ExpertParallelCollective)
					var ordinal uint64
					note := func(layer, position int, prefill bool, picks []routePick, input []float32) {
						ordinal++
						d := v41EPIdentity(c.binding, configHash, ordinal, layer, st.history, picks, input)
						results[rank].digests = append(results[rank].digests, v41EPPositionIdentity(d, position, prefill))
						results[rank].routes = append(results[rank].routes, append([]routePick(nil), picks...))
					}
					step, prefill := st.routedLayer, st.routedPrefill
					st.routedLayer = func(layer int, picks []routePick, input []float32) ([]float32, v41RoutedExpertOutcome, error) {
						defer b.expertScope()()
						note(layer, len(st.history), false, picks, input)
						return step(layer, picks, input)
					}
					st.routedPrefill = func(layer int, picks [][]routePick, inputs [][]float32, scratch *v41ProjScratch) ([][]float32, error) {
						defer b.expertScope()()
						for token := range picks {
							note(layer, token, true, picks[token], inputs[token])
						}
						return prefill(layer, picks, inputs, scratch)
					}
					for action, ids := range actions {
						if rank != 0 {
							payload, err := g.BroadcastFromRoot(nil)
							if err != nil {
								results[rank].failure = err
								return nil
							}
							op, pos, gotIDs, err := decodeEPDecodeFrame(payload)
							wantOp := epOpStep
							if action == 0 {
								wantOp = epOpPrefill
							}
							if err != nil || op != wantOp || !reflect.DeepEqual(gotIDs, ids) {
								return errors.New("follower received unexpected decode action")
							}
							if err := epCheckMirror(g, s, op, pos); err != nil {
								return err
							}
						}
						beforeNative, beforeRead, beforeUpload, beforeScalar := b.native, b.gateReads, b.logitUploads, b.scalarMatMuls
						beforeRouter, beforeSums := b.router, b.sums
						beforeTiles, beforeDown, beforeCohorts := len(b.tileRows), len(b.downRows), len(b.cohortRows)
						if mode == "router_fault" && action == 1 && rank == 0 {
							b.fault = selected
						}
						var logits []float32
						err := panicAsError(func() {
							if action == 0 {
								logits = s.Prefill(ids)
							} else {
								logits = s.Step(ids[0])
							}
						})
						if b.expertWork || st.projectionRoute != bound {
							return errors.New("execution changed the binding or retained expert scope")
						}
						if err != nil {
							results[rank].failure = err
							if logits != nil || !reflect.DeepEqual(st.history, []int{1, 2, 3}) {
								return errors.New("failed Step committed output or history")
							}
							if rank == 0 {
								if b.native-beforeNative != 1 || b.router-beforeRouter != 1 || b.gateReads != beforeRead || b.logitUploads != beforeUpload || b.sums != beforeSums || b.scalarMatMuls != beforeScalar || len(b.tileRows) != beforeTiles || len(b.downRows) != beforeDown || len(b.cohortRows) != beforeCohorts {
									return errors.New("selected router fault read/replayed the route or reached root expert/sum work")
								}
								if len(results[rank].states) != 1 || !reflect.DeepEqual(captureV41ForwardSnapshot(st), results[rank].states[0]) {
									return errors.New("selected router fault changed the committed continuation")
								}
								var operation *BackendForwardOperationError
								if !errors.Is(err, selected) || !errors.As(err, &operation) || !s.BackendSessionClosed() {
									return fmt.Errorf("native router primary changed: %v", err)
								}
								calls, sums := b.router, b.sums
								if retry := panicAsError(func() { _ = s.Step(4) }); retry == nil || b.router != calls || b.sums != sums {
									return errors.New("failed owner replayed native/router sum work")
								}
							}
							return nil
						}
						results[rank].logits = append(results[rank].logits, append([]float32(nil), logits...))
						results[rank].states = append(results[rank].states, captureV41ForwardSnapshot(st))
						if action == 0 {
							if b.native != 0 || b.gateReads == 0 || b.logitUploads == 0 || len(b.tileRows) == 0 || len(b.downRows) == 0 || len(b.cohortRows) == 0 {
								return errors.New("Prefill missed ordinary routing or actual packed owner cohorts")
							}
							arrived = true
							prefixReady.Done()
							prefixReady.Wait()
						} else {
							wantNative, wantBytes := m.Cfg.NumLayers, uint64(0)
							if ordinary {
								wantNative, wantBytes = 0, uint64(4*m.Cfg.NumExperts*m.Cfg.NumLayers)
							}
							if b.native-beforeNative != wantNative || b.gateReads-beforeRead != wantBytes || b.logitUploads-beforeUpload != wantBytes || b.scalarMatMuls-beforeScalar != 6*m.Cfg.NumLayers {
								return fmt.Errorf("rank %d Step seam counts: native=%d read=%d upload=%d scalar=%d", rank, b.native-beforeNative, b.gateReads-beforeRead, b.logitUploads-beforeUpload, b.scalarMatMuls-beforeScalar)
							}
						}
					}
					if rank == 0 {
						if err := coord.Shutdown(); err != nil {
							return err
						}
					} else {
						payload, err := g.BroadcastFromRoot(nil)
						if err != nil {
							return err
						}
						op, _, _, err := decodeEPDecodeFrame(payload)
						if err != nil || op != epOpShutdown {
							return errors.New("follower missed shutdown")
						}
						if err := wire.checkDigests(results[rank].digests, results[rank].routes[0], owners, rank, m.Cfg.HiddenSize); err != nil {
							return err
						}
					}
					stats := m.V41ExpertParallelTransportStats()
					if stats.CompletedGathers != uint64(len(actions)) || stats.CompletedRows != uint64(logicalRows) {
						return fmt.Errorf("transport counts: %+v", stats)
					}
					return nil
				})
				for rank, err := range errs {
					b := backends[rank]
					if err != nil || closeErrors[rank] != nil || b.retired() != nil {
						t.Fatalf("rank %d operation=%v close=%v retirement=%v", rank, err, closeErrors[rank], b.retired())
					}
					wantSums := 0
					if rank == 0 {
						wantSums = logicalRows
						if mode == "router_fault" {
							wantSums = len(actions[0])
						}
					}
					if b.sums != wantSums || len(b.gateUploads) != full.Cfg.NumLayers {
						t.Fatalf("rank %d root sums/gate identity: %d / %v", rank, b.sums, b.gateUploads)
					}
					if mode == "router_fault" {
						if results[rank].failure == nil {
							t.Fatalf("rank %d accepted faulted Step", rank)
						}
						continue
					}
					if results[rank].failure != nil || len(results[rank].logits) != len(actions) || len(results[rank].digests) != logicalRows {
						t.Fatalf("rank %d missing completed actions: %+v", rank, results[rank])
					}
					for _, picks := range results[rank].routes {
						counts := [3]int{}
						for _, pick := range picks {
							for owner, band := range owners.Shards {
								if pick.expert >= band.Lo && pick.expert < band.Hi {
									counts[owner]++
								}
							}
						}
						if counts != [3]int{2, 2, 2} {
							t.Fatalf("canonical bias did not preserve rank symmetry: %v", counts)
						}
					}
					if rank != 0 && (!reflect.DeepEqual(results[rank].states, results[0].states) || !reflect.DeepEqual(results[rank].digests, results[0].digests)) {
						t.Fatalf("rank %d continuation or identity differs", rank)
					}
					for i, logits := range results[rank].logits {
						if err := v41EPCheckBits(logits, results[0].logits[i]); err != nil {
							t.Fatal(err)
						}
					}
					if !ordinary {
						if !reflect.DeepEqual(results[rank].states, reference[rank].states) || !reflect.DeepEqual(results[rank].digests, reference[rank].digests) {
							t.Fatalf("rank %d native continuation/route identity differs from ordinary EP", rank)
						}
						for i, logits := range results[rank].logits {
							if err := v41EPCheckBits(logits, reference[rank].logits[i]); err != nil {
								t.Fatal(err)
							}
						}
					}
				}
				if ordinary {
					reference = results
				}
			}
		})
	}
}
