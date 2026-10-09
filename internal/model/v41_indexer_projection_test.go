package model

import (
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

var v41IndexerTestLeaves = []string{"indexer.wq_b.weight", "indexer.wk.weight", "indexer.weights_proj.weight"}
var v41IndexerTestFields = []string{"indexer_projection_device_calls", "indexer_projection_host_calls", "indexer_projection_device_rows", "indexer_projection_host_rows", "indexer_projection_activation_upload_bytes", "indexer_projection_readback_bytes", "indexer_projection_nanos", "indexer_scoring_calls", "indexer_scoring_nanos"}

var v41IndexerTestExportedFields = []string{"IndexerProjectionDeviceCalls", "IndexerProjectionHostCalls", "IndexerProjectionDeviceRows", "IndexerProjectionHostRows", "IndexerProjectionActivationUploadBytes", "IndexerProjectionReadbackBytes", "IndexerProjectionNanos", "IndexerScoringCalls", "IndexerScoringNanos"}

func v41IndexerTestFixture(t *testing.T) *Model {
	t.Helper()
	return v41CompressorTestFixtureIndex(t, 128)
}

type v41IndexerDiagnosticProduct struct{ input, output []float32 }
type v41IndexerDiagnosticBackend struct {
	*v41CompressorTestBackend
	products map[compute.Buffer][]v41IndexerDiagnosticProduct
}

func (b *v41IndexerDiagnosticBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	var input []float32
	if len(w.Shape) == 2 && w.Shape[0] == 128 && w.Shape[1] == 512 {
		input = append([]float32(nil), b.v41DenseTestBackend.Backend.Read(x)...)
	}
	out := b.v41CompressorTestBackend.MatMul(w, x)
	if input != nil {
		b.products[w.Buf()] = append(b.products[w.Buf()], v41IndexerDiagnosticProduct{input, append([]float32(nil), b.v41DenseTestBackend.Backend.Read(out)...)})
	}
	return out
}
func (b *v41IndexerDiagnosticBackend) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	var input []float32
	if len(w.Shape) == 2 && w.Shape[0] == 128 && w.Shape[1] == 512 {
		input = append([]float32(nil), b.v41DenseTestBackend.Backend.Read(x)...)
	}
	out := b.v41CompressorTestBackend.BatchedMatMul(w, x, rows)
	if input != nil {
		b.products[w.Buf()] = append(b.products[w.Buf()], v41IndexerDiagnosticProduct{input, append([]float32(nil), b.v41DenseTestBackend.Backend.Read(out)...)})
	}
	return out
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=3s lane=optin
func TestV41IndexerProjectionBorrowedKVFirstDivergenceDiagnostic(t *testing.T) {
	fixture := func() *Model {
		m := v41IndexerTestFixture(t)
		m.Cfg.DeepSeekV41.IndexSourceLayerIDs = []int{0, 1}
		return m
	}
	m := fixture()
	b := &v41IndexerDiagnosticBackend{newV41CompressorTestBackend(), map[compute.Buffer][]v41IndexerDiagnosticProduct{}}
	s := v41DenseTestSession(t, m, b)
	var history []int

	maxDiff := func(a, b []float32) float64 {
		t.Helper()
		if len(a) != len(b) {
			t.Fatal("diagnostic operands have unequal widths")
		}
		value := float64(0)
		for i := range a {
			value = math.Max(value, math.Abs(float64(a[i])-float64(b[i])))
		}
		return value
	}
	for segment, ids := range [][]int{{1, 2, 3}, {4}, {5, 6}} {
		var got []float32
		if len(ids) == 1 {
			got = s.Step(ids[0])
		} else {
			got = s.Prefill(ids)
		}
		history = append(history, ids...)
		coldModel := fixture()
		cold := coldModel.NewSession()
		wantLogits := cold.Prefill(history)
		logitDiff := maxDiff(got, wantLogits)
		probeModel := fixture()
		probe := probeModel.NewSession()
		state := probe.v41State()
		var hostInput, hostRaw []float32
		callback := reflect.ValueOf(&state.denseProjection).Elem()
		callback.Set(reflect.MakeFunc(callback.Type(), func(args []reflect.Value) []reflect.Value {
			if int(args[0].Int()) == 1 && args[1].String() == "indexer.wk.weight" {
				input := args[2].Interface().([]float32)
				out, in, rows := int(args[3].Int()), int(args[4].Int()), int(args[5].Int())
				hostInput = append(hostInput, input...)
				weight := probeModel.tensor(layerName(1, "indexer.wk.weight"))
				for row := 0; row < rows; row++ {
					for dst := 0; dst < out; dst++ {
						sum := float64(0)
						for col := 0; col < in; col++ {
							sum += float64(weight[dst*in+col]) * float64(input[row*in+col])
						}
						hostRaw = append(hostRaw, float32(sum))
					}
				}
			}
			return []reflect.Value{reflect.Zero(callback.Type().Out(0)), reflect.ValueOf(v41ProjectionDeclined), reflect.Zero(callback.Type().Out(2))}
		}))
		if _, err := probeModel.forwardV41(history, state); err != nil {
			t.Fatal(err)
		}
		weight := v41CompressorTestWeight(t, s, 1, "indexer.wk.weight")
		var selectedInput, selectedRaw []float32
		for _, product := range b.products[weight] {
			selectedInput = append(selectedInput, product.input...)
			selectedRaw = append(selectedRaw, product.output...)
		}
		if len(hostInput) == 0 {
			t.Fatal("host forward diagnostic did not observe own wk input")
		}
		inputDiff := maxDiff(selectedInput, hostInput)
		rawDiff := maxDiff(selectedRaw, hostRaw)
		keys, _ := s.v41Forward.attn.IndexKeys(1)
		wantKeys, _ := cold.v41Forward.attn.IndexKeys(1)
		keyDiff := maxDiff(flatten(keys), flatten(wantKeys))
		maxRow, maxCol := 0, 0
		for row := range keys {
			for col := range keys[row] {
				if math.Abs(float64(keys[row][col])-float64(wantKeys[row][col])) == keyDiff {
					maxRow, maxCol = row, col
				}
			}
		}
		t.Logf("diagnostic control segment=%d positions=%d own-input-maxdiff=%g raw-wk-maxdiff=%g final-key-maxdiff=%g key-row=%d key-col=%d scalar-cold-Session-logits-maxdiff=%g", segment, len(history), inputDiff, rawDiff, keyDiff, maxRow, maxCol, logitDiff)
		if m.Cfg.RMSNormEps != coldModel.Cfg.RMSNormEps || !reflect.DeepEqual(m.tensor(layerName(1, "indexer.k_norm.weight")), coldModel.tensor(layerName(1, "indexer.k_norm.weight"))) {
			t.Fatal("fixture indexer normalization weights or EPS disagree")
		}
		cold.Close()
		probe.Close()
	}
	t.Logf("fixture EPS=%g; fixture construction sets no Prism or LoRA adapters", m.Cfg.RMSNormEps)
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=3s lane=default
func TestV41IndexerProjectionBorrowedKVSameBackendKeyParity(t *testing.T) {
	t.Parallel()
	fixture := func() *Model {
		m := v41CompressorTestFixtureIndex(t, 128)
		m.Cfg.DeepSeekV41.IndexSourceLayerIDs = []int{0, 1}
		return m
	}
	m := fixture()
	b := newV41CompressorTestBackend()
	s := v41DenseTestSession(t, m, b)
	var history []int
	for segment, ids := range [][]int{{1, 2, 3}, {4}, {5, 6}} {
		var got []float32
		if len(ids) == 1 {
			got = s.Step(ids[0])
		} else {
			got = s.Prefill(ids)
		}
		history = append(history, ids...)
		cold := fixture().NewSession()
		want := cold.Prefill(history)
		v41CompressorTestFiniteParity(t, got, want, "borrowed KV vs independent CPU cold scalar logits")
		coldHALModel := fixture()
		t.Cleanup(func() {
			if err := coldHALModel.CloseWeights(); err != nil {
				t.Error(err)
			}
		})
		coldHAL := v41DenseTestSession(t, coldHALModel, newV41CompressorTestBackend())
		coldHAL.Prefill(history)
		keys, ok := s.v41Forward.attn.IndexKeys(1)
		wantKeys, wantOK := coldHAL.v41Forward.attn.IndexKeys(1)
		if !ok || !wantOK || len(keys) != len(history)/2 || len(wantKeys) != len(keys) {
			t.Fatalf("same-backend own key history absent at segment %d", segment)
		}
		for row := range keys {
			t.Run(itoa(segment)+"/"+itoa(row), func(t *testing.T) {
				v41CompressorTestFiniteParity(t, keys[row], wantKeys[row], "borrowed KV retained vs independent cold same recording-HAL own key")
			})
		}
		v41IndexerTestColdSelection(t, s, history, fixture)
		cold.Close()
		coldHAL.Close()
	}
}

func v41IndexerTestColdSelection(t *testing.T, s *Session, history []int, fixture func() *Model) {
	t.Helper()
	cold := fixture().NewSession()
	defer cold.Close()
	cold.Prefill(history)
	got, ratio, ok := s.v41Forward.attn.TopK()
	want, wantRatio, wantOK := cold.v41Forward.attn.TopK()
	if !ok || !wantOK || ratio != 2 || wantRatio != 2 || len(got) == 0 || len(want) == 0 || !reflect.DeepEqual(got[len(got)-1], want[len(want)-1]) {
		t.Fatal("retained Session last query selected IDs differ from independent cold history")
	}
	v41IndexerTestSelectionCount(t, got[len(got)-1], len(history)/ratio, s.M.Cfg.IndexTopK)
	v41IndexerTestSelectionCount(t, want[len(want)-1], len(history)/wantRatio, cold.M.Cfg.IndexTopK)
}

func v41IndexerTestSelectionCount(t *testing.T, ids []int32, groups, topK int) {
	t.Helper()
	if len(ids) != topK {
		t.Fatal("Session index selection lost its fixed topK width")
	}
	count := 0
	seen := map[int32]bool{}
	for _, id := range ids {
		if id == -1 {
			continue
		}
		if id < 0 || int(id) >= groups || seen[id] {
			t.Fatal("Session index selection contains invalid, future, or duplicate row")
		}
		seen[id] = true
		count++
	}
	want := groups
	if want > topK {
		want = topK
	}
	if count != want {
		t.Fatalf("Session index selection contains %d valid rows, want %d", count, want)
	}
}

func v41IndexerTestPhase(t *testing.T, m *Model, phase string) map[string]float64 {
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
	for _, key := range v41IndexerTestFields {
		var value *float64
		if err = json.Unmarshal(phases[phase][key], &value); err != nil || value == nil {
			t.Fatalf("numeric indexer phase missing %s.%s", phase, key)
		}
		out[key] = *value
	}
	return out
}

func v41IndexerTestOps(t *testing.T, s *Session, b *v41CompressorTestBackend, from int) (map[string][]v41DenseTestOp, int, int) {
	t.Helper()
	weights := map[compute.Buffer]string{}
	for _, leaf := range v41IndexerTestLeaves {
		weights[v41CompressorTestWeight(t, s, 0, leaf)] = leaf
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
// fak-test:runtime medium est=3s lane=default
func TestV41IndexerProjectionActualSessionOncePerCompletedKey(t *testing.T) {
	t.Parallel()
	m := v41IndexerTestFixture(t)
	if m.Cfg.IndexHeadDim <= 0 || m.Cfg.IndexHeadDim == v41KVLoraRank {
		t.Fatal("indexer fixture must distinguish index width from the full KV latent width")
	}
	b := newV41CompressorTestBackend()
	s := v41DenseTestSession(t, m, b)
	var history []int
	cache := map[string]compute.Buffer{}
	for index, ids := range [][]int{{1, 2, 3}, {4}, {5}, {6, 7, 0}} {
		from, oldLength := len(b.ops), len(history)
		var got []float32
		if len(ids) == 1 {
			got = s.Step(ids[0])
		} else {
			got = s.Prefill(ids)
		}
		history = append(history, ids...)
		named, _, _ := v41IndexerTestOps(t, s, b, from)
		for _, leaf := range v41IndexerTestLeaves {
			rows := 0
			for _, op := range named[leaf] {
				rows += op.rows
				wantIn, wantOut := 32, 128
				if leaf == "indexer.wk.weight" {
					wantIn = 512
				}
				if leaf == "indexer.weights_proj.weight" {
					wantIn, wantOut = 64, 1
				}
				if op.in != wantIn || op.out != wantOut {
					t.Fatalf("indexer orientation %s got %dx%d", leaf, op.out, op.in)
				}
			}
			want := len(ids)
			if leaf == "indexer.wk.weight" {
				want = len(history)/2 - oldLength/2
			}
			if rows != want {
				t.Fatalf("actual Session indexer %s projected %d rows, want %d newly completed/query rows at segment %d", leaf, rows, want, index)
			}
			weight := v41CompressorTestWeight(t, s, 0, leaf)
			if old, ok := cache[leaf]; ok && old != weight {
				t.Fatal("indexer restaged cached weight")
			}
			cache[leaf] = weight
		}
		v41CompressorTestFreshCold(t, s, history, 128, func() *Model { return v41IndexerTestFixture(t) }, got)
		v41IndexerTestColdSelection(t, s, history, func() *Model { return v41IndexerTestFixture(t) })
		keys, _ := s.v41Forward.attn.IndexKeys(0)
		kv, _ := s.v41Forward.attn.KVSourceRows(0)
		if len(keys) != len(history)/2 || len(kv) != len(keys) {
			t.Fatal("index and KV streams lost completed group identity")
		}
		for row := range keys {
			if len(keys[row]) != m.Cfg.IndexHeadDim || len(kv[row]) != v41KVLoraRank {
				t.Fatal("index width confused with KV width")
			}
		}
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=2s lane=default
func TestV41IndexerProjectionRestoreForkAndStrictRefusal(t *testing.T) {
	t.Parallel()
	m := v41IndexerTestFixture(t)
	b := newV41CompressorTestBackend()
	s := v41DenseTestSession(t, m, b)
	s.Prefill([]int{1, 2, 3})
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
	for i, branch := range []*Session{left, right} {
		from := len(b.ops)
		got := branch.Step(4 + i)
		named, _, _ := v41IndexerTestOps(t, branch, b, from)
		rows := 0
		for _, op := range named["indexer.wk.weight"] {
			rows += op.rows
		}
		if rows != 1 {
			t.Fatalf("restored owner replayed %d key rows, want one", rows)
		}
		v41CompressorTestFreshCold(t, branch, []int{1, 2, 3, 4 + i}, 128, func() *Model { return v41IndexerTestFixture(t) }, got)
		v41IndexerTestColdSelection(t, branch, []int{1, 2, 3, 4 + i}, func() *Model { return v41IndexerTestFixture(t) })
	}
	left.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
	calls := b.operationCalls
	var closed *BackendForwardOperationError
	err = panicAsError(func() { left.Step(6) })
	if !errors.As(err, &closed) || closed.Path != "device-only" || closed.Layer != -1 || !left.BackendSessionClosed() || b.operationCalls != calls {
		t.Fatal("strict DeviceOnly refusal changed")
	}
	if reflect.DeepEqual(left.v41Forward.history, right.v41Forward.history) {
		t.Fatal("fork fixture did not create independent histories")
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=3s lane=default
func TestV41IndexerProjectionComponentAccountingAndHostDecline(t *testing.T) {
	t.Parallel()
	for _, deny := range []bool{false, true} {
		t.Run(itoa(boolToIntV41Expert(deny)), func(t *testing.T) {
			m := v41IndexerTestFixture(t)
			b := newV41CompressorTestBackend()
			b.deny = deny
			s := v41DenseTestSession(t, m, b)
			var history []int
			for _, ids := range [][]int{{1, 2, 3}, {4}, {5, 6}} {
				phase := "prefill"
				if len(ids) == 1 {
					phase = "decode"
				}
				before := v41IndexerTestPhase(t, m, phase)
				from, oldLength := len(b.ops), len(history)
				if len(ids) == 1 {
					s.Step(ids[0])
				} else {
					s.Prefill(ids)
				}
				history = append(history, ids...)
				delta := v41DenseTestDelta(v41IndexerTestPhase(t, m, phase), before)
				if delta["indexer_projection_nanos"] <= 0 || delta["indexer_scoring_calls"] <= 0 || delta["indexer_scoring_nanos"] <= 0 {
					t.Fatal("actual indexer work lacks elapsed/scoring attribution")
				}
				if deny {
					wantRows := 2*len(ids) + len(history)/2 - oldLength/2
					if delta["indexer_projection_host_rows"] != float64(wantRows) || delta["indexer_projection_host_calls"] <= 0 || delta["indexer_projection_device_calls"] != 0 || delta["indexer_projection_device_rows"] != 0 || delta["indexer_projection_activation_upload_bytes"] != 0 || delta["indexer_projection_readback_bytes"] != 0 {
						t.Fatalf("unsupported indexer component accounting %v want rows %d", delta, wantRows)
					}
				} else {
					named, upload, read := v41IndexerTestOps(t, s, b, from)
					calls, rows := 0, 0
					for _, ops := range named {
						for _, op := range ops {
							calls++
							rows += op.rows
						}
					}
					if delta["indexer_projection_device_calls"] != float64(calls) || delta["indexer_projection_device_rows"] != float64(rows) || delta["indexer_projection_activation_upload_bytes"] != float64(upload) || delta["indexer_projection_readback_bytes"] != float64(read) || delta["indexer_projection_host_calls"] != 0 || delta["indexer_projection_host_rows"] != 0 {
						t.Fatalf("selected indexer accounting %v recorder calls=%d rows=%d upload=%d read=%d", delta, calls, rows, upload, read)
					}
				}
			}
			value := reflect.ValueOf(m.V41ExpertFaultAttribution().Prefill)
			wire := v41IndexerTestPhase(t, m, "prefill")
			for i, name := range v41IndexerTestExportedFields {
				field := value.FieldByName(name)
				if !field.IsValid() || !field.CanInt() || float64(field.Int()) != wire[v41IndexerTestFields[i]] {
					t.Fatalf("exported model phase field %s absent or differs from wire source", name)
				}
			}
		})
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=2s lane=default
func TestV41IndexerProjectionOwnHistoryWithoutKVPublication(t *testing.T) {
	t.Parallel()
	fixture := func() *Model {
		m := v41IndexerTestFixture(t)
		m.Cfg.NumLayers = 1
		m.Cfg.DeepSeekV41.CompressRatios = []int{2}
		m.Cfg.DeepSeekV41.KVSourceLayerIDs = nil
		return m
	}
	m := fixture()
	b := newV41CompressorTestBackend()
	s := v41DenseTestSession(t, m, b)
	history := []int{1, 2, 3}
	v41CompressorTestFreshCold(t, s, history, 128, fixture, s.Prefill(history))
	for _, token := range []int{4, 5, 6} {
		previous, _ := s.v41Forward.attn.IndexKeys(0)
		from := len(b.ops)
		got := s.Step(token)
		history = append(history, token)
		v41CompressorTestFreshCold(t, s, history, 128, fixture, got)
		v41IndexerTestColdSelection(t, s, history, fixture)
		current, _ := s.v41Forward.attn.IndexKeys(0)
		if !reflect.DeepEqual(previous, current[:len(previous)]) {
			t.Fatal("per-layer own index history was replaced")
		}
		named, _, _ := v41IndexerTestOps(t, s, b, from)
		rows := 0
		for _, op := range named["indexer.wk.weight"] {
			rows += op.rows
		}
		if rows != len(current)-len(previous) {
			t.Fatal("per-layer key projection replayed historical keys")
		}
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=3s lane=default
func TestV41IndexerProjectionOwnIndexSourceWithBorrowedKV(t *testing.T) {
	t.Parallel()
	fixture := func() *Model {
		m := v41IndexerTestFixture(t)
		m.Cfg.DeepSeekV41.IndexSourceLayerIDs = []int{0, 1}
		return m
	}
	m := fixture()
	b := newV41CompressorTestBackend()
	s := v41DenseTestSession(t, m, b)
	var history []int
	for _, ids := range [][]int{{1, 2, 3}, {4}, {5, 6}} {
		from, oldLength := len(b.ops), len(history)
		var got []float32
		if len(ids) == 1 {
			got = s.Step(ids[0])
		} else {
			got = s.Prefill(ids)
		}
		history = append(history, ids...)
		cold := fixture().NewSession()
		v41CompressorTestFiniteParity(t, got, cold.Prefill(history), "own index source with borrowed KV vs independent cold history")
		selection, ratio, ok := s.v41Forward.attn.TopK()
		wantSelection, wantRatio, wantOK := cold.v41Forward.attn.TopK()
		if !ok || !wantOK || ratio != 2 || wantRatio != 2 || len(selection) == 0 || len(wantSelection) == 0 || !reflect.DeepEqual(selection[len(selection)-1], wantSelection[len(wantSelection)-1]) {
			t.Fatal("borrowed KV own index source selected IDs differ from independent cold last query")
		}
		v41IndexerTestSelectionCount(t, selection[len(selection)-1], len(history)/ratio, m.Cfg.IndexTopK)
		v41IndexerTestSelectionCount(t, wantSelection[len(wantSelection)-1], len(history)/wantRatio, cold.M.Cfg.IndexTopK)
		coldHALModel := fixture()
		t.Cleanup(func() {
			if err := coldHALModel.CloseWeights(); err != nil {
				t.Error(err)
			}
		})
		coldHAL := v41DenseTestSession(t, coldHALModel, newV41CompressorTestBackend())
		coldHAL.Prefill(history)
		for layer := 0; layer < 2; layer++ {
			keys, ok := s.v41Forward.attn.IndexKeys(layer)
			want, wantOK := coldHAL.v41Forward.attn.IndexKeys(layer)
			if !ok || !wantOK || len(keys) != len(history)/2 || len(want) != len(keys) {
				t.Fatal("own layer index history absent or incomplete")
			}
			for row := range keys {
				v41CompressorTestFiniteParity(t, keys[row], want[row], "own layer independent cold same recording-HAL normalized index key")
			}
			weight := v41CompressorTestWeight(t, s, layer, "indexer.wk.weight")
			rows := 0
			for _, op := range b.ops[from:] {
				if op.weight == weight {
					rows += op.rows
				}
			}
			if rows != len(history)/2-oldLength/2 {
				t.Fatalf("own index layer%d replayed historical key rows", layer)
			}
		}
		left, _ := s.v41Forward.attn.IndexKeys(0)
		right, _ := s.v41Forward.attn.IndexKeys(1)
		if reflect.DeepEqual(left, right) {
			t.Fatal("own index source fixture aliases nearest KV source history")
		}
		cold.Close()
		coldHAL.Close()
	}
}
