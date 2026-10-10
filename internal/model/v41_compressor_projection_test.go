package model

import (
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

var v41CompressorTestLeaves = []string{"attn.compressor.wkv.weight", "attn.compressor.wgate.weight"}
var v41CompressorTestFields = []string{"compressor_projection_device_calls", "compressor_projection_host_calls", "compressor_projection_device_rows", "compressor_projection_host_rows", "compressor_projection_activation_upload_bytes", "compressor_projection_readback_bytes", "compressor_projection_nanos"}

func v41CompressorTestFiniteParity(t *testing.T, got, want []float32, context string) {
	t.Helper()
	for _, values := range [][]float32{got, want} {
		for _, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("%s contains a non-finite comparison value", context)
			}
		}
	}
	if len(got) == len(want) {
		maximum, bad := float64(0), false
		for i := range got {
			difference := math.Abs(float64(got[i] - want[i]))
			maximum = math.Max(maximum, difference)
			bad = bad || difference > 1e-4*math.Max(1, math.Abs(float64(want[i])))
		}
		if bad {
			t.Logf("%s maximum absolute difference=%g", context, maximum)
		}
	}
	assertV41LogitsClose(t, got, want, context)
}

func v41CompressorTestFixture(t *testing.T) *Model {
	t.Helper()
	return v41CompressorTestFixtureIndex(t, 512)
}

func v41CompressorTestFixtureIndex(t *testing.T, indexDim int) *Model {
	t.Helper()
	m := v41IncrementalPlainModel(t, 2)
	c := &m.Cfg
	c.HeadDim, c.NumHeads, c.NumKVHeads, c.OGroups = 512, 1, 1, 1
	c.QKNopeHeadDim, c.QKRopeHeadDim = 496, 16
	c.DeepSeekV41.Attention = DeepSeekV41AttentionGeometry{}
	c.DeepSeekV41.CompressRatios = []int{2, 2}
	c.DeepSeekV41.KVSourceLayerIDs, c.DeepSeekV41.IndexSourceLayerIDs = []int{0}, []int{0}
	c.IndexNHeads, c.IndexHeadDim, c.IndexTopK = 1, indexDim, 2
	var extra []synthTensor
	for layer := 0; layer < 2; layer++ {
		for _, spec := range []struct {
			leaf  string
			shape []int
		}{
			{"mhc.mixes.weight", []int{4 * c.HiddenSize, v41MHCMixWidth}},
			{"mhc.ffn_mixes.weight", []int{4 * c.HiddenSize, v41MHCMixWidth}},
			{"mhc.ffn_base", []int{v41MHCMixWidth}},
			{"mhc.ffn_scale", []int{3}},
			{"attn.wq_a_norm.weight", []int{c.QLoraRank}},
			{"attn.wq_b.weight", []int{512, c.QLoraRank}},
			{"attn.wkv.weight", []int{512, c.HiddenSize}},
			{"attn.kv_norm.weight", []int{512}},
			{"attn.wo_a.weight", []int{c.OLoraRank, 512}},
			{"attn.wo_b.weight", []int{c.HiddenSize, c.OLoraRank}},
			{"attn.sink", []int{1}},
			{"attn.compressor.wkv.weight", []int{512, c.HiddenSize}},
			{"attn.compressor.wgate.weight", []int{512, c.HiddenSize}},
			{"attn.compressor.norm.weight", []int{512}},
			{"indexer.wq_b.weight", []int{indexDim, c.QLoraRank}},
			{"indexer.wk.weight", []int{indexDim, 512}},
			{"indexer.k_norm.weight", []int{indexDim}},
			{"indexer.weights_proj.weight", []int{1, c.HiddenSize}},
		} {
			extra = append(extra, synthTensor{layerName(layer, spec.leaf), spec.shape})
		}
	}
	manifest, raw := synthBuildRaw(extra, func(name string, next func() float32) float32 {
		if strings.HasSuffix(name, "mhc.ffn_scale") {
			return .75
		}
		if strings.HasSuffix(name, "norm.weight") {
			return 0.8 + 0.2*next()
		}
		return synthMatmulFill(name, next)
	})
	for name, meta := range manifest {
		meta.Offset += len(m.raw)
		m.manifest[name] = meta
	}
	m.raw = append(m.raw, raw...)
	return m
}

// v41CompressorProducerTestFixture declares the two producers whose projection,
// retention, cache, and late-failure paths the compressor-specific tests drive.
// The base fixture keeps layer 1 a reader for cross-layer ownership witnesses.
func v41CompressorProducerTestFixture(t *testing.T) *Model {
	t.Helper()
	m := v41CompressorTestFixture(t)
	m.Cfg.DeepSeekV41.KVSourceLayerIDs = []int{0, 1}
	return m
}

type v41CompressorTestBackend struct {
	*v41DenseTestBackend
	apiCalls       int
	operationCalls int
	stages         map[compute.Buffer]int
	frees          map[compute.Buffer]int
	nonfinite      bool
	activationBufs []compute.Buffer
}

func newV41CompressorTestBackend() *v41CompressorTestBackend {
	return &v41CompressorTestBackend{v41DenseTestBackend: newV41DenseTestBackend(), stages: map[compute.Buffer]int{}, frees: map[compute.Buffer]int{}}
}
func (b *v41CompressorTestBackend) Caps() compute.Caps {
	b.apiCalls++
	return b.v41DenseTestBackend.Caps()
}
func (b *v41CompressorTestBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	b.apiCalls++
	return b.v41DenseTestBackend.SupportsDeviceWeightDtype(dt)
}
func (b *v41CompressorTestBackend) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	b.apiCalls++
	b.operationCalls++
	out := b.v41DenseTestBackend.Upload(x, dt)
	if len(x.Shape) == 1 {
		b.activationBufs = append(b.activationBufs, out.Buf())
	}
	if len(x.Shape) == 2 {
		b.stages[out.Buf()]++
	}
	return out
}
func (b *v41CompressorTestBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.apiCalls++
	b.operationCalls++
	return b.v41DenseTestBackend.MatMul(w, x)
}
func (b *v41CompressorTestBackend) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	b.apiCalls++
	b.operationCalls++
	return b.v41DenseTestBackend.BatchedMatMul(w, x, rows)
}
func (b *v41CompressorTestBackend) Read(x compute.Tensor) []float32 {
	b.apiCalls++
	b.operationCalls++
	out := b.v41DenseTestBackend.Read(x)
	if b.nonfinite && b.faultOutputs[x.Buf()] && len(out) > 0 {
		out[0] = float32(math.NaN())
	}
	return out
}
func (b *v41CompressorTestBackend) Free(x compute.Tensor) {
	b.apiCalls++
	b.frees[x.Buf()]++
	b.v41DenseTestBackend.Free(x)
}

func v41CompressorTestPhase(t *testing.T, m *Model, phase string) map[string]float64 {
	t.Helper()
	raw, err := json.Marshal(m.V41ExpertFaultAttribution())
	if err != nil {
		t.Fatal(err)
	}
	var phases map[string]map[string]json.RawMessage
	if err = json.Unmarshal(raw, &phases); err != nil {
		t.Fatal(err)
	}
	out := map[string]float64{}
	for _, key := range v41CompressorTestFields {
		var value *float64
		if err = json.Unmarshal(phases[phase][key], &value); err != nil || value == nil {
			t.Fatalf("default numeric compressor phase missing %s.%s", phase, key)
		}
		out[key] = *value
	}
	return out
}

func v41CompressorTestWeight(t *testing.T, s *Session, layer int, leaf string) compute.Buffer {
	t.Helper()
	var found compute.Buffer
	for key, w := range s.halW {
		if strings.HasSuffix(key, layerName(layer, leaf)) {
			if found != nil {
				t.Fatal("ambiguous compressor cached weight")
			}
			found = w.Buf()
		}
	}
	if found == nil {
		t.Fatalf("actual session did not stage layer %d %s", layer, leaf)
	}
	return found
}

func v41CompressorTestOps(t *testing.T, s *Session, b *v41CompressorTestBackend, from int) (map[string][]v41DenseTestOp, int, int) {
	t.Helper()
	weights := map[compute.Buffer]string{}
	for layer := 0; layer < 2; layer++ {
		for _, leaf := range v41CompressorTestLeaves {
			name := layerName(layer, leaf)
			weights[v41CompressorTestWeight(t, s, layer, leaf)] = name
		}
	}
	named := map[string][]v41DenseTestOp{}
	upload, read := 0, 0
	for _, op := range b.ops[from:] {
		if name, ok := weights[op.weight]; ok {
			named[name] = append(named[name], op)
			upload += b.uploads[op.input]
			read += b.reads[op.output]
		}
	}
	return named, upload, read
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=4s lane=default
func TestV41CompressorProjectionActualSessionRoutes(t *testing.T) {
	t.Parallel()
	for _, deny := range []bool{false, true} {
		t.Run(itoa(boolToIntV41Expert(deny)), func(t *testing.T) {
			m, oracle := v41CompressorProducerTestFixture(t), v41CompressorProducerTestFixture(t)
			b := newV41CompressorTestBackend()
			b.deny = deny
			s := v41DenseTestSession(t, m, b)
			host := oracle.NewSession()
			t.Cleanup(host.Close)
			var history []int
			cache := map[string]compute.Buffer{}
			for index, ids := range [][]int{{1, 2, 3}, {4}, {5, 6}} {
				phase := "prefill"
				if index == 1 {
					phase = "decode"
				}
				before := v41CompressorTestPhase(t, m, phase)
				denseBefore := v41DenseTestPhase(t, m, phase)
				hostDenseBefore := v41DenseTestPhase(t, oracle, phase)
				from := len(b.ops)
				var got, want []float32
				if index == 1 {
					got = s.Step(ids[0])
					want = host.Step(ids[0])
				} else {
					got = s.Prefill(ids)
					want = host.Prefill(ids)
				}
				history = append(history, ids...)
				v41CompressorTestFiniteParity(t, got, want, "actual compressor Session vs separate host Session")
				cold := v41CompressorProducerTestFixture(t).NewSession()
				coldLogits := cold.Prefill(history)
				v41CompressorTestFiniteParity(t, got, coldLogits, "actual compressor continuation vs fresh full-history cold Session")
				actualKeys, _ := s.v41Forward.attn.IndexKeys(0)
				coldKeys, _ := cold.v41Forward.attn.IndexKeys(0)
				if !reflect.DeepEqual(actualKeys, coldKeys) {
					t.Fatal("continuation index publication differs from fresh full-history cold Session")
				}
				cold.Close()
				if !reflect.DeepEqual(s.v41Forward.history, history) {
					t.Fatal("compressor route did not commit exact history")
				}
				for layer := 0; layer < 2; layer++ {
					state, reference := s.v41Forward.layerState(layer), host.v41Forward.layerState(layer)
					if !reflect.DeepEqual(state.partialPositions, reference.partialPositions) || len(state.partialInputs) != len(reference.partialInputs) {
						t.Fatal("host compressor carrier retention differs")
					}
					for row := range state.partialInputs {
						v41CompressorTestFiniteParity(t, state.partialInputs[row], reference.partialInputs[row], "retained original compressor carrier")
					}
					if state.nextWindowPos != len(history) {
						t.Fatal("source/reader continuation cursor did not advance")
					}
					rows, ok := state.KVSourceRows(0)
					refRows, refOK := reference.KVSourceRows(0)
					if ok != refOK || !reflect.DeepEqual(rows, refRows) {
						t.Fatal("source publication differs from host pooling/norm oracle")
					}
				}
				delta := v41DenseTestDelta(v41CompressorTestPhase(t, m, phase), before)
				if delta["compressor_projection_nanos"] <= 0 {
					t.Fatal("actual compressor work absent from elapsed counter")
				}
				if deny {
					if delta["compressor_projection_device_calls"] != 0 || delta["compressor_projection_device_rows"] != 0 || delta["compressor_projection_host_calls"] <= 0 || delta["compressor_projection_host_rows"] <= 0 {
						t.Fatalf("unsupported compressor route accounting %v", delta)
					}
				} else {
					named, upload, read := v41CompressorTestOps(t, s, b, from)
					calls, rows := 0, 0
					for layer := 0; layer < 2; layer++ {
						for _, leaf := range v41CompressorTestLeaves {
							name := layerName(layer, leaf)
							if len(named[name]) == 0 {
								t.Fatalf("actual producer compressor product missing %s", name)
							}
							for _, op := range named[name] {
								if op.in != 64 || op.out != 512 {
									t.Fatal("compressor projection orientation/width mismatch")
								}
								calls++
								rows += op.rows
							}
							w := v41CompressorTestWeight(t, s, layer, leaf)
							if old, ok := cache[name]; ok && old != w {
								t.Fatal("resident compressor cache restaged across continuation")
							}
							cache[name] = w
						}
					}
					if delta["compressor_projection_device_calls"] != float64(calls) || delta["compressor_projection_device_rows"] != float64(rows) || delta["compressor_projection_host_calls"] != 0 || delta["compressor_projection_host_rows"] != 0 || delta["compressor_projection_activation_upload_bytes"] != float64(upload) || delta["compressor_projection_readback_bytes"] != float64(read) {
						t.Fatalf("compressor component ledger %v actual calls=%d rows=%d upload=%d read=%d", delta, calls, rows, upload, read)
					}
				}
				dense := v41DenseTestDelta(v41DenseTestPhase(t, m, phase), denseBefore)
				hostDense := v41DenseTestDelta(v41DenseTestPhase(t, oracle, phase), hostDenseBefore)
				if dense["dense_projection_device_rows"]+dense["dense_projection_host_rows"] != hostDense["dense_projection_host_rows"] {
					t.Fatalf("compressor changed existing dense/indexer row accounting: %v host=%v", dense, hostDense)
				}
				if !deny {
					for layer := 0; layer < 2; layer++ {
						for _, leaf := range v41DenseTestLeaves {
							weight := v41CompressorTestWeight(t, s, layer, leaf)
							rows := 0
							for _, op := range b.ops[from:] {
								if op.weight == weight {
									rows += op.rows
								}
							}
							if rows != len(ids) {
								t.Fatalf("ordinary dense leaf %d/%s rows=%d want%d", layer, leaf, rows, len(ids))
							}
						}
					}
					if dense["dense_projection_device_rows"] != float64(14*len(ids)) {
						t.Fatalf("compressor contaminated seven device dense leaves: %v", dense)
					}
				}
			}
		})
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=2s lane=default
func TestV41CompressorProjectionRestoreOwnerAndStrictRefusal(t *testing.T) {
	t.Parallel()
	m := v41CompressorProducerTestFixture(t)
	b := newV41CompressorTestBackend()
	source, target := v41DenseTestSession(t, m, b), v41DenseTestSession(t, m, b)
	source.Prefill([]int{1, 2, 3})
	snap, err := source.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer snap.Close()
	if err = snap.Restore(target); err != nil {
		t.Fatal(err)
	}
	source.Close()
	from := len(b.ops)
	got := target.Step(4)
	named, _, _ := v41CompressorTestOps(t, target, b, from)
	for layer := 0; layer < 2; layer++ {
		for _, leaf := range v41CompressorTestLeaves {
			if len(named[layerName(layer, leaf)]) == 0 {
				t.Fatal("restored compressor bridge did not bind target owner")
			}
		}
	}
	oracle := v41CompressorProducerTestFixture(t).NewSession()
	t.Cleanup(oracle.Close)
	oracle.Prefill([]int{1, 2, 3})
	v41CompressorTestFiniteParity(t, got, oracle.Step(4), "restored compressor owner")
	target.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
	calls := b.operationCalls
	var closed *BackendForwardOperationError
	if err = panicAsError(func() { target.Step(5) }); !errors.As(err, &closed) || closed.Path != "device-only" || closed.Layer != -1 || closed != target.halFailure || !target.BackendSessionClosed() || b.operationCalls != calls {
		t.Fatal("strict Flash DeviceOnly refusal changed")
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=20ms lane=default
func TestV41CompressorProjectionAbsoluteQueryCausality(t *testing.T) {
	t.Parallel()
	rows := [][]float32{{0.4, -0.3}, {-0.6, 0.8}, {9, 7}}
	queries := [][]float32{{0.1, 0.2}, {0.3, -0.4}, {-0.2, 0.6}, {0.7, 0.5}}
	sink := []float32{0.15}
	scale := float32(0.5)
	options := V41AttentionSharedKVOptions{Layer: 1, Ratio: 2, Groups: 3, HeadDim: 2, Heads: 1, Softmax: scale, Sink: sink}
	setOffset := func(opts *V41AttentionSharedKVOptions, offset int) {
		t.Helper()
		field := reflect.ValueOf(opts).Elem().FieldByName("QueryOffset")
		if !field.IsValid() || !field.CanSet() || field.Kind() != reflect.Int {
			t.Fatal("compressed attention lacks an absolute query-position contract")
		}
		field.SetInt(int64(offset))
	}
	full, err := V41AttentionCompressedForward(flatten(queries), rows, options)
	if err != nil {
		t.Fatal(err)
	}
	independent := flatten(v41OracleCompressedForward(t, 2, queries, rows, sink, scale))
	v41CompressorTestFiniteParity(t, full, independent, "independent full-panel compressed group visibility")
	if loraMaxAbsDiff(full[6:], []float32{0, 0}) < 1e-4 {
		t.Fatal("absolute-query oracle is vacuous")
	}
	setOffset(&options, 3)
	single, err := V41AttentionCompressedForward(queries[3], rows, options)
	if err != nil {
		t.Fatal(err)
	}
	v41CompressorTestFiniteParity(t, single, independent[6:], "absolute last query equals independent full-panel tail")
	options.IndexTopK, options.Idx = 3, []int32{1, -1, 2}
	selected, err := V41AttentionCompressedForward(queries[3], rows, options)
	if err != nil {
		t.Fatal(err)
	}
	dot := float64(scale) * (float64(queries[3][0])*float64(rows[1][0]) + float64(queries[3][1])*float64(rows[1][1]))
	probability := math.Exp(dot) / (math.Exp(dot) + math.Exp(float64(sink[0])))
	want := []float32{float32(probability * float64(rows[1][0])), float32(probability * float64(rows[1][1]))}
	v41CompressorTestFiniteParity(t, selected, want, "selected row with masked and causally future groups")
	options.Idx = []int32{-1, -1, 2}
	empty, err := V41AttentionCompressedForward(queries[3], rows, options)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(empty, []float32{0, 0}) {
		t.Fatal("masked/future-only selected groups must emit zero")
	}
	for _, offset := range []int{-1, int(^uint(0) >> 1)} {
		invalid := options
		invalid.IndexTopK, invalid.Idx = 0, nil
		setOffset(&invalid, offset)
		_, err := V41AttentionCompressedForward(flatten(queries[:2]), rows, invalid)
		if !errors.Is(err, ErrV41ForwardStage) {
			t.Fatal("negative or overflowing absolute query range must refuse with typed stage sentinel")
		}
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=1s lane=default
func TestV41CompressorProjectionDirectStepAdmittedSourceReader(t *testing.T) {
	t.Parallel()
	m := v41CompressorProducerTestFixture(t)
	b := newV41CompressorTestBackend()
	s := v41DenseTestSession(t, m, b)
	s.Prefill([]int{1, 2, 3})
	if !s.v41IncrementalEligible() {
		t.Fatal("seeded source/reader Session must be incrementally eligible")
	}
	from := len(b.ops)
	got, stats, err := m.forwardV41Step(4, s.v41State(), &v41ProjScratch{})
	if err != nil {
		t.Fatalf("admitted source/reader direct Step refused: %T %v", err, err)
	}
	if !stats.Committed || stats.LayerCalls != 2 {
		t.Fatal("source/reader direct Step did not commit exactly one position per layer")
	}
	oracle := v41CompressorProducerTestFixture(t).NewSession()
	t.Cleanup(oracle.Close)
	v41CompressorTestFiniteParity(t, got, oracle.Prefill([]int{1, 2, 3, 4}), "admitted direct source/reader Step vs fresh full-history cold Session")
	actualKeys, _ := s.v41Forward.attn.IndexKeys(0)
	coldKeys, _ := oracle.v41Forward.attn.IndexKeys(0)
	if !reflect.DeepEqual(actualKeys, coldKeys) {
		t.Fatal("direct Step index publication differs from fresh full-history cold Session")
	}
	for layer := 0; layer < 2; layer++ {
		for _, leaf := range v41DenseTestLeaves {
			weight := v41CompressorTestWeight(t, s, layer, leaf)
			rows := 0
			for _, op := range b.ops[from:] {
				if op.weight == weight {
					rows += op.rows
				}
			}
			if rows != 1 {
				t.Fatalf("direct Step dense leaf %d/%s projected%drows", layer, leaf, rows)
			}
		}
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=2s lane=default
func TestV41CompressorProjectionShortPrefixSourceReader(t *testing.T) {
	t.Parallel()
	defer func() {
		if value := recover(); value != nil {
			t.Fatalf("short-prefix source/reader unexpectedly panicked: %T %v", value, value)
		}
	}()
	m, oracle := v41CompressorTestFixture(t), v41CompressorTestFixture(t)
	if m.Cfg.IndexHeadDim <= 0 || m.Cfg.IndexHeadDim == m.Cfg.QLoraRank {
		t.Fatal("source index fixture must distinguish index width from query rank")
	}
	b := newV41CompressorTestBackend()
	s, host := v41DenseTestSession(t, m, b), oracle.NewSession()
	t.Cleanup(host.Close)
	v41CompressorTestFiniteParity(t, s.Prefill([]int{1}), host.Prefill([]int{1}), "single token source/reader")
	for layer := 0; layer < 2; layer++ {
		state := s.v41Forward.layerState(layer)
		if layer == 0 {
			if !reflect.DeepEqual(state.partialPositions, []int{0}) || len(state.partialInputs) != 1 {
				t.Fatal("short prefix must retain the source compressor carrier")
			}
		} else if len(state.partialPositions) != 0 || len(state.partialInputs) != 0 || len(state.partialKV) != 0 {
			t.Fatal("reader retained a private compressor group")
		}
	}
	if rows, _ := s.v41Forward.attn.KVSourceRows(0); len(rows) != 0 {
		t.Fatal("incomplete first group published a KV placeholder")
	}
	if keys, _ := s.v41Forward.attn.IndexKeys(0); len(keys) != 0 {
		t.Fatal("incomplete first group published a query-latent index placeholder")
	}
	selection, ratio, ok := s.v41Forward.attn.TopK()
	if !ok || ratio != 2 || len(selection) != 1 || len(selection[0]) != 2 {
		t.Fatal("short-prefix source did not publish the fixed-width masked selection")
	}
	for _, row := range selection {
		for _, id := range row {
			if id != -1 {
				t.Fatal("empty source selection must mask every slot")
			}
		}
	}
	roles := m.v41AttentionRolesCached()
	plan, err := v41AttentionPlanFor(m.Cfg, 1, roles)
	if err != nil || plan.Role != V41AttentionRoleReader || plan.KVSourceLayer != 0 || plan.IndexSourceLayer != 0 {
		t.Fatal("short-prefix logits did not traverse a configured source reader")
	}
	for index, ids := range [][]int{{2}, {3, 4}} {
		var got, want []float32
		if index == 0 {
			got, want = s.Step(ids[0]), host.Step(ids[0])
		} else {
			got, want = s.Prefill(ids), host.Prefill(ids)
		}
		v41CompressorTestFiniteParity(t, got, want, "first complete compressed group and suffix")
		history := []int{1, 2}
		if index == 1 {
			history = []int{1, 2, 3, 4}
		}
		cold := v41CompressorTestFixture(t).NewSession()
		v41CompressorTestFiniteParity(t, got, cold.Prefill(history), "short-prefix continuation vs fresh full-history cold Session")
		coldKeys, _ := cold.v41Forward.attn.IndexKeys(0)
		actualKeys, _ := s.v41Forward.attn.IndexKeys(0)
		if !reflect.DeepEqual(actualKeys, coldKeys) {
			t.Fatal("short-prefix index publication differs from fresh cold full-history")
		}
		cold.Close()
		groups := index + 1
		rows, rowOK := s.v41Forward.attn.KVSourceRows(0)
		keys, keyOK := s.v41Forward.attn.IndexKeys(0)
		refRows, _ := host.v41Forward.attn.KVSourceRows(0)
		refKeys, _ := host.v41Forward.attn.IndexKeys(0)
		if !rowOK || !keyOK || len(rows) != groups || len(keys) != groups || !reflect.DeepEqual(rows, refRows) || !reflect.DeepEqual(keys, refKeys) {
			t.Fatal("completed group KV/index publication differs from separate host Session")
		}
		for _, key := range keys {
			if len(key) != m.Cfg.IndexHeadDim {
				t.Fatal("source index key used query rank instead of configured index width")
			}
		}
		for layer := 0; layer < 2; layer++ {
			if len(s.v41Forward.layerState(layer).partialInputs) != 0 {
				t.Fatal("complete group retained stale source/reader carriers")
			}
		}
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=3s lane=default
func TestV41CompressorProjectionPrismLoRAIndependentScalar(t *testing.T) {
	t.Parallel()
	m := v41CompressorProducerTestFixture(t)
	inputs := make([][]float32, 2)
	for row := range inputs {
		inputs[row] = make([]float32, 64)
		for col := range inputs[row] {
			inputs[row][col] = float32(math.Sin(float64(row)*0.7 + float64(col)*0.19))
		}
	}
	original := make([][]float32, 2)
	for row := range inputs {
		original[row] = append([]float32(nil), inputs[row]...)
	}
	signs := make([]int, 64)
	for i := range signs {
		signs[i] = 1
		if i%3 == 1 {
			signs[i] = -1
		}
	}
	names := []string{layerName(0, v41CompressorTestLeaves[0]), layerName(0, v41CompressorTestLeaves[1])}
	if err := m.SetPrismHadamard(PrismHadamardSpec{BlockSize: 4, SignWidths: []int{64}, SignValues: signs, WeightNames: names}); err != nil {
		t.Fatal(err)
	}
	set := NewLoRASet()
	projected := make([][][]float64, 2)
	for component, name := range names {
		adapter := newTestAdapter("compressor", name, 512, 64, 2, 8, uint64(13668+component))
		if err := set.Add(adapter); err != nil {
			t.Fatal(err)
		}
		weight := append([]float32(nil), m.tensor(name)...)
		projected[component] = make([][]float64, 2)
		for row, input := range inputs {
			rotated := make([]float64, 64)
			for start := 0; start < 64; start += 4 {
				a, b, c, d := float64(input[start])*float64(signs[start]), float64(input[start+1])*float64(signs[start+1]), float64(input[start+2])*float64(signs[start+2]), float64(input[start+3])*float64(signs[start+3])
				rotated[start], rotated[start+1], rotated[start+2], rotated[start+3] = (a+b+c+d)/2, (a-b+c-d)/2, (a+b-c-d)/2, (a-b-c+d)/2
			}
			latent := make([]float64, 2)
			for rank := 0; rank < 2; rank++ {
				for col := 0; col < 64; col++ {
					latent[rank] += float64(adapter.A[rank*64+col]) * float64(input[col])
				}
			}
			out := make([]float64, 512)
			for dst := range out {
				for col := 0; col < 64; col++ {
					out[dst] += float64(weight[dst*64+col]) * rotated[col]
				}
				for rank := 0; rank < 2; rank++ {
					out[dst] += 4 * float64(adapter.B[dst*2+rank]) * latent[rank]
				}
			}
			projected[component][row] = out
		}
	}
	set.Activate("compressor")
	m.SetLoRA(set)
	norm := append([]float32(nil), m.tensor(layerName(0, "attn.compressor.norm.weight"))...)
	want := v41SRRefPool(2, 512, projected[0], projected[1], norm, m.Cfg.RMSNormEps)
	got, err := m.v41CompressedRows(0, 2, inputs, inputs)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || len(want) != 1 {
		t.Fatal("adapted compressor failed to emit one group")
	}
	v41CompressorTestFiniteParity(t, got[0], want[0], "independent signed H4 plus LoRA on original input and BF16/norm tail")
	if !reflect.DeepEqual(inputs, original) {
		t.Fatal("adapted compressor mutated original source carriers")
	}
	plain := v41CompressorProducerTestFixture(t)
	base, err := plain.v41CompressedRows(0, 2, inputs, inputs)
	if err != nil {
		t.Fatal(err)
	}
	if loraMaxAbsDiff(got[0], base[0]) < 1e-4 {
		t.Fatal("independent adapted compressor fixture is vacuous")
	}
	deviceOracle := v41CompressorProducerTestFixture(t)
	if err := deviceOracle.SetPrismHadamard(PrismHadamardSpec{BlockSize: 4, SignWidths: []int{64}, SignValues: signs, WeightNames: names}); err != nil {
		t.Fatal(err)
	}
	deviceOracle.SetLoRA(set)
	b := newV41CompressorTestBackend()
	s := v41DenseTestSession(t, m, b)
	host := deviceOracle.NewSession()
	t.Cleanup(host.Close)
	v41CompressorTestFiniteParity(t, s.Prefill([]int{1, 2, 3}), host.Prefill([]int{1, 2, 3}), "actual adapted compressor cold Session")
	v41CompressorTestFiniteParity(t, s.Step(4), host.Step(4), "actual adapted compressor Step")
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=2s lane=default
func TestV41CompressorProjectionSharedWeightLifetime(t *testing.T) {
	t.Parallel()
	m := v41CompressorProducerTestFixture(t)
	b := newV41CompressorTestBackend()
	source := v41DenseTestSession(t, m, b)
	source.Prefill([]int{1, 2, 3})
	weights := map[string]compute.Buffer{}
	for layer := 0; layer < 2; layer++ {
		for _, leaf := range v41CompressorTestLeaves {
			weights[layerName(layer, leaf)] = v41CompressorTestWeight(t, source, layer, leaf)
		}
	}
	snapshot, err := source.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	clone, err := snapshot.Clone()
	if err != nil {
		t.Fatal(err)
	}
	left, right := v41DenseTestSession(t, m, b), v41DenseTestSession(t, m, b)
	if err = snapshot.Restore(left); err != nil {
		t.Fatal(err)
	}
	if err = clone.Restore(right); err != nil {
		t.Fatal(err)
	}
	snapshot.Close()
	clone.Close()
	source.Close()
	for index, branch := range []*Session{left, right} {
		branch.Step(4 + index)
		for layer := 0; layer < 2; layer++ {
			for _, leaf := range v41CompressorTestLeaves {
				name := layerName(layer, leaf)
				weight := v41CompressorTestWeight(t, branch, layer, leaf)
				if weight != weights[name] || b.stages[weight] != 1 || b.frees[weight] != 0 {
					t.Fatal("compressor sibling did not share a single live model-owned staged weight")
				}
			}
		}
	}
	left.Close()
	for _, weight := range weights {
		if b.frees[weight] != 0 {
			t.Fatal("closing compressor sibling freed shared resident")
		}
	}
	right.Close()
	if err = m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	for _, weight := range weights {
		if b.frees[weight] != 1 {
			t.Fatal("final model owner did not free compressor resident exactly once")
		}
	}
}

func v41CompressorTestFreshCold(t *testing.T, s *Session, history []int, indexDim int, fixture func() *Model, got []float32) {
	t.Helper()
	cold := fixture().NewSession()
	defer cold.Close()
	v41CompressorTestFiniteParity(t, got, cold.Prefill(history), "compressed continuation vs independent fresh cold Session")
	keys, keyOK := s.v41Forward.attn.IndexKeys(0)
	want, wantOK := cold.v41Forward.attn.IndexKeys(0)
	if keyOK != wantOK || len(keys) != len(history)/2 || len(keys) != len(want) {
		t.Fatal("compressed continuation index source history differs from fresh cold Session")
	}
	for row, key := range keys {
		if len(key) != indexDim {
			t.Fatalf("published index key width=%d want%d", len(key), indexDim)
		}
		v41CompressorTestFiniteParity(t, key, want[row], "index publication vs independent fresh cold Session")
	}
	if !reflect.DeepEqual(s.v41Forward.history, history) {
		t.Fatal("compressed continuation history differs")
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=4s lane=default
func TestV41CompressorProjectionDeclaredIndexWidthProducers(t *testing.T) {
	t.Parallel()
	defer func() {
		if value := recover(); value != nil {
			t.Fatalf("declared index-width Session unexpectedly panicked: %T %v", value, value)
		}
	}()
	fixture := func() *Model {
		m := v41CompressorTestFixtureIndex(t, 128)
		m.Cfg.DeepSeekV41.KVSourceLayerIDs = []int{0, 1}
		return m
	}
	m := fixture()
	b := newV41CompressorTestBackend()
	s := v41DenseTestSession(t, m, b)
	history := []int{1, 2, 3}
	v41CompressorTestFreshCold(t, s, history, 128, fixture, s.Prefill(history))
	weights := map[string]compute.Buffer{}
	for layer := 0; layer < 2; layer++ {
		for _, leaf := range v41CompressorTestLeaves {
			weights[layerName(layer, leaf)] = v41CompressorTestWeight(t, s, layer, leaf)
		}
	}
	got, stats, err := m.forwardV41Step(4, s.v41State(), &v41ProjScratch{})
	if err != nil || !stats.Committed || stats.LayerCalls != 2 {
		t.Fatalf("declared index-width direct Step refused: %T %v", err, err)
	}
	history = append(history, 4)
	v41CompressorTestFreshCold(t, s, history, 128, fixture, got)
	got = s.Prefill([]int{5, 6})
	history = append(history, 5, 6)
	v41CompressorTestFreshCold(t, s, history, 128, fixture, got)
	snapshot, err := s.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer snapshot.Close()
	clone, err := snapshot.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer clone.Close()
	left, right := v41DenseTestSession(t, m, b), v41DenseTestSession(t, m, b)
	if err = snapshot.Restore(left); err != nil {
		t.Fatal(err)
	}
	if err = clone.Restore(right); err != nil {
		t.Fatal(err)
	}
	s.Close()
	for branchIndex, branch := range []*Session{left, right} {
		token := 7 - branchIndex
		from := len(b.ops)
		got = branch.Step(token)
		branchHistory := append(append([]int(nil), history...), token)
		v41CompressorTestFreshCold(t, branch, branchHistory, 128, fixture, got)
		for _, op := range b.ops[from:] {
			for _, weight := range weights {
				if op.weight == weight {
					t.Fatal("incomplete restored group invoked compressor projection before group closure")
				}
			}
		}
		from = len(b.ops)
		got = branch.Step(branchIndex)
		branchHistory = append(branchHistory, branchIndex)
		v41CompressorTestFreshCold(t, branch, branchHistory, 128, fixture, got)
		named, _, _ := v41CompressorTestOps(t, branch, b, from)
		for layer := 0; layer < 2; layer++ {
			for _, leaf := range v41CompressorTestLeaves {
				name := layerName(layer, leaf)
				if len(named[name]) == 0 {
					t.Fatal("completed restored group did not invoke its target compressor product")
				}
				buffer := v41CompressorTestWeight(t, branch, layer, leaf)
				if buffer != weights[name] || b.stages[buffer] != 1 || b.frees[buffer] != 0 {
					t.Fatal("declared-width restored owner failed to retain shared compressor cache")
				}
			}
		}
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=3s lane=default
func TestV41CompressorProjectionPerLayerIndexSourceOwnHistory(t *testing.T) {
	t.Parallel()
	defer func() {
		if value := recover(); value != nil {
			t.Fatalf("per-layer compressed source unexpectedly panicked: %T %v", value, value)
		}
	}()
	fixture := func() *Model {
		m := v41CompressorProducerTestFixture(t)
		m.Cfg.NumLayers = 1
		m.Cfg.DeepSeekV41.CompressRatios = []int{2}
		m.Cfg.DeepSeekV41.KVSourceLayerIDs = nil
		m.Cfg.DeepSeekV41.IndexSourceLayerIDs = []int{0}
		return m
	}
	m := fixture()
	b := newV41CompressorTestBackend()
	s := v41DenseTestSession(t, m, b)
	history := []int{1, 2, 3}
	v41CompressorTestFreshCold(t, s, history, 512, fixture, s.Prefill(history))
	for _, token := range []int{4, 5, 6} {
		before, _ := s.v41Forward.attn.IndexKeys(0)
		got, stats, err := m.forwardV41Step(token, s.v41State(), &v41ProjScratch{})
		if err != nil || !stats.Committed || stats.LayerCalls != 1 {
			t.Fatalf("per-layer source direct Step refused: %T %v", err, err)
		}
		history = append(history, token)
		v41CompressorTestFreshCold(t, s, history, 512, fixture, got)
		after, _ := s.v41Forward.attn.IndexKeys(0)
		if len(before) > 0 && !reflect.DeepEqual(before, after[:len(before)]) {
			t.Fatal("per-layer source replaced historical index rows")
		}
		state := s.v41Forward.layerState(0)
		if len(state.partialInputs) != len(history)%2 || len(state.partialPositions) != len(history)%2 {
			t.Fatal("per-layer source retained incorrect complete/incomplete group carrier")
		}
	}
}
