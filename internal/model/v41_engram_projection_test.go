package model

import (
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

var v41EngProjFields = []string{"engram_projection_device_calls", "engram_projection_host_calls", "engram_projection_device_rows", "engram_projection_host_rows", "engram_projection_matmul_calls", "engram_projection_activation_upload_bytes", "engram_projection_readback_bytes", "engram_projection_nanos", "engram_projection_host_weight_f32_bytes"}

func v41EngProjPhase(t *testing.T, m *Model, phase string) map[string]float64 {
	t.Helper()
	raw, err := json.Marshal(m.V41ExpertFaultAttribution())
	if err != nil {
		t.Fatal(err)
	}
	var objects map[string]map[string]json.RawMessage
	if err := json.Unmarshal(raw, &objects); err != nil {
		t.Fatal(err)
	}
	out := map[string]float64{}
	for _, key := range v41EngProjFields {
		var n float64
		if err := json.Unmarshal(objects[phase][key], &n); err != nil {
			t.Errorf("default %s.%s unavailable: %v", phase, key, err)
			continue
		}
		out[key] = n
	}
	return out
}
func v41EngProjFixture(t *testing.T, dtype string, compressed bool) *Model {
	t.Helper()
	m, _ := v41IncrementalEngramFixture(t, compressed)
	stage := m.v41EngramStageFor()
	out, in := (stage.hc+1)*m.Cfg.HiddenSize, stage.columns*stage.headDim
	for index, layer := range m.Cfg.DeepSeekV41.EngramLayerIDs {
		name := layerName(layer, "engram_kv.weight")
		w, ok := m.residentF32Mat(name)
		if !ok {
			t.Fatal("Engram projection fixture missing raw oracle")
		}
		switch dtype {
		case "Q2_K":
			if m.kqw == nil {
				m.kqw = map[string]*kQuantTensor{}
			}
			m.kqw[name] = q2kFixtureTensor(out, in, uint64(13668080+index))
		case "Q4_K":
			if m.q4kw == nil {
				m.q4kw = map[string]*q4kTensor{}
			}
			m.q4kw[name] = quantizeQ4KFromRaw(v41DenseTestQ4Raw(out, in), out, in)
		case "Q8_0":
			if m.q8w == nil {
				m.q8w = map[string]*q8Tensor{}
			}
			m.q8w[name] = quantizeQ8(w, out, in)
		}
		if dtype != "F32" {
			delete(m.manifest, name)
		}
	}
	return m
}

type v41EngProjBackend struct {
	*v41DenseTestBackend
	out, in          int
	weightBuffers    map[compute.Buffer]bool
	outputBuffers    map[compute.Buffer]int
	operations       []v41EngProjOperation
	attempts, stages int
	weightFrees      map[compute.Buffer]int
	decline          compute.Dtype
	site             string
	cause            any
	failed           bool
	decodeFault      bool
	decodeAttempts   int
}
type v41EngProjOperation struct{ rows, upload, read int }

func newV41EngProjBackend(m *Model) *v41EngProjBackend {
	stage := m.v41EngramStageFor()
	return &v41EngProjBackend{v41DenseTestBackend: newV41DenseTestBackend(), out: (stage.hc + 1) * m.Cfg.HiddenSize, in: stage.columns * stage.headDim, weightBuffers: map[compute.Buffer]bool{}, outputBuffers: map[compute.Buffer]int{}, weightFrees: map[compute.Buffer]int{}, decline: compute.Dtype(255)}
}
func (b *v41EngProjBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return dt != b.decline && b.v41DenseTestBackend.SupportsDeviceWeightDtype(dt)
}
func (b *v41EngProjBackend) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	out := b.v41DenseTestBackend.Upload(x, dt)
	if len(x.Shape) == 2 && x.Shape[0] == b.out && x.Shape[1] == b.in {
		b.weightBuffers[out.Buf()] = true
		b.stages++
	}
	return out
}
func (b *v41EngProjBackend) multiply(w, x compute.Tensor, rows int, batch bool) compute.Tensor {
	engram := b.weightBuffers[w.Buf()]
	if engram {
		b.attempts++
		if b.decodeFault {
			b.decodeAttempts++
		}
		if b.decodeFault && b.decodeAttempts == 2 && b.site == "matmul" {
			b.failed = true
			panic(b.cause)
		}
	}
	var out compute.Tensor
	if batch {
		out = b.v41DenseTestBackend.BatchedMatMul(w, x, rows)
	} else {
		out = b.v41DenseTestBackend.MatMul(w, x)
	}
	if engram {
		b.operations = append(b.operations, v41EngProjOperation{rows: rows, upload: b.uploads[x.Buf()]})
		b.outputBuffers[out.Buf()] = len(b.operations) - 1
	}
	return out
}
func (b *v41EngProjBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	return b.multiply(w, x, 1, false)
}
func (b *v41EngProjBackend) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	return b.multiply(w, x, rows, true)
}
func (b *v41EngProjBackend) Read(x compute.Tensor) []float32 {
	op, engram := b.outputBuffers[x.Buf()]
	late := engram && b.decodeFault && b.decodeAttempts == 2
	if late && b.site == "read" {
		b.failed = true
		panic(b.cause)
	}
	values := b.v41DenseTestBackend.Read(x)
	if late && b.site == "length" {
		b.failed = true
		values = append(values, 0)
	}
	if late && b.site == "finite" {
		b.failed = true
		values[0] = float32(math.NaN())
	}
	if engram {
		b.operations[op].read += 4 * len(values)
	}
	return values
}
func (b *v41EngProjBackend) Free(x compute.Tensor) {
	if b.weightBuffers[x.Buf()] {
		b.weightFrees[x.Buf()]++
	}
	b.v41DenseTestBackend.Free(x)
}
func v41EngProjSession(t *testing.T, m *Model, b compute.Backend) *Session {
	t.Helper()
	s, err := m.NewBackendSessionChecked(b)
	if err != nil {
		t.Fatal(err)
	}
	s.Q4K, s.Quant = m.q4kw != nil, m.q8w != nil
	t.Cleanup(s.Close)
	return s
}

// fak-test:justify why=integration when=changed:internal/model/**
// fak-test:runtime medium est=12s lane=default
func TestV41EngramProjectionActualDefaultSession(t *testing.T) {
	t.Parallel()
	for _, scenario := range []struct {
		name, dtype         string
		compressed, decline bool
	}{
		{"f32", "F32", false, false}, {"q2", "Q2_K", false, false}, {"q4", "Q4_K", false, false}, {"q8", "Q8_0", false, false}, {"compressed-mhc", "F32", true, false}, {"declined-q2", "Q2_K", false, true},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			t.Parallel()
			m, oracle := v41EngProjFixture(t, scenario.dtype, scenario.compressed), v41EngProjFixture(t, scenario.dtype, scenario.compressed)
			b := newV41EngProjBackend(m)
			if scenario.decline {
				b.decline = compute.Q2_K
			}
			s := v41EngProjSession(t, m, b)
			host := oracle.NewSession()
			host.Q4K, host.Quant = oracle.q4kw != nil, oracle.q8w != nil
			t.Cleanup(host.Close)
			layers := len(m.Cfg.DeepSeekV41.EngramLayerIDs)
			for index, ids := range [][]int{{1, 2, 3}, {4}, {5, 6}} {
				phase := "prefill"
				if index == 1 {
					phase = "decode"
				}
				before := v41EngProjPhase(t, m, phase)
				from, attempts := len(b.operations), b.attempts
				var got, want []float32
				if index == 1 {
					got, want = s.Step(ids[0]), host.Step(ids[0])
				} else {
					got, want = s.Prefill(ids), host.Prefill(ids)
				}
				tolerance := 1e-4
				if scenario.dtype == "Q8_0" {
					tolerance = 0.01
				}
				v41GroupedParity(t, got, want, tolerance)
				delta := v41DenseTestDelta(v41EngProjPhase(t, m, phase), before)
				rows, upload, read := 0, 0, 0
				for _, op := range b.operations[from:] {
					rows += op.rows
					upload += op.upload
					read += op.read
				}
				wantRows := layers * len(ids)
				if scenario.decline {
					materializations := layers
					if index == 2 {
						materializations *= len(ids)
					}
					if rows != 0 || delta["engram_projection_device_calls"] != 0 || delta["engram_projection_device_rows"] != 0 || delta["engram_projection_matmul_calls"] != 0 || delta["engram_projection_host_rows"] != float64(wantRows) || delta["engram_projection_host_weight_f32_bytes"] != float64(4*b.out*b.in*materializations) {
						t.Errorf("declined whole Engram operation actual accounting=%v rows=%d", delta, rows)
					}
				} else {
					if rows != wantRows || delta["engram_projection_device_rows"] != float64(wantRows) || delta["engram_projection_device_calls"] <= 0 || delta["engram_projection_host_calls"] != 0 || delta["engram_projection_host_rows"] != 0 || delta["engram_projection_host_weight_f32_bytes"] != 0 {
						t.Errorf("actual default Session Engram route=%v recorded_rows=%d want=%d", delta, rows, wantRows)
					}
					if delta["engram_projection_matmul_calls"] != float64(b.attempts-attempts) || delta["engram_projection_activation_upload_bytes"] != float64(upload) || delta["engram_projection_readback_bytes"] != float64(read) {
						t.Errorf("Engram actual API/transfer accounting=%v matmuls=%d upload=%d read=%d", delta, b.attempts-attempts, upload, read)
					}
				}
				if delta["engram_projection_nanos"] <= 0 {
					t.Error("Engram projection has no elapsed observation")
				}
			}
			if !scenario.decline && b.stages != layers {
				t.Errorf("Engram immutable KV staged %d times want %d", b.stages, layers)
			}
		})
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=1ms lane=default
func TestV41EngramProjectionDefaultExportedLedger(t *testing.T) {
	t.Parallel()
	typeOfPhase := reflect.TypeOf(V41ExpertFaultAttribution{}.Prefill)
	for i, name := range []string{"EngramProjectionDeviceCalls", "EngramProjectionHostCalls", "EngramProjectionDeviceRows", "EngramProjectionHostRows", "EngramProjectionMatMulCalls", "EngramProjectionActivationUploadBytes", "EngramProjectionReadbackBytes", "EngramProjectionNanos", "EngramProjectionHostWeightF32Bytes"} {
		field, ok := typeOfPhase.FieldByName(name)
		if !ok || field.PkgPath != "" {
			t.Errorf("missing exported phase field %s", name)
			continue
		}
		if field.Tag.Get("json") != v41EngProjFields[i] {
			t.Errorf("phase field %s lacks default JSON contract %s", name, v41EngProjFields[i])
		}
	}
	for _, phase := range []string{"prefill", "decode"} {
		for key, n := range v41EngProjPhase(t, &Model{}, phase) {
			if n != 0 {
				t.Errorf("inert %s=%g", key, n)
			}
		}
	}
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=3s lane=default
func TestV41EngramProjectionLateSelectedFailure(t *testing.T) {
	t.Parallel()
	for _, site := range []string{"matmul", "read", "length", "finite", "unknown"} {
		t.Run(site, func(t *testing.T) {
			t.Parallel()
			m := v41EngProjFixture(t, "F32", false)
			b := newV41EngProjBackend(m)
			s := v41EngProjSession(t, m, b)
			s.Prefill([]int{1, 2, 3})
			before := captureV41ForwardSnapshot(s.v41Forward)
			ledgerBefore := v41EngProjPhase(t, m, "decode")
			b.live = map[compute.Buffer]bool{}
			b.decodeFault, b.site = true, site
			b.cause = &compute.BackendError{Backend: "cpu-ref-recording", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
			unknown := struct{ Number int }{1366808}
			if site == "unknown" {
				b.site, b.cause = "matmul", unknown
			}
			var recovered any
			func() { defer func() { recovered = recover() }(); s.Step(4) }()
			if !b.failed {
				t.Fatal("late second Engram projection injection was never reached")
			}
			if site == "unknown" {
				if recovered != unknown {
					t.Error("late selected Engram operation changed unknown panic identity")
				}
			} else {
				err, ok := recovered.(error)
				var selected *BackendForwardOperationError
				if !ok || !errors.As(err, &selected) || !errors.Is(err, ErrV41ForwardStage) {
					t.Errorf("Engram selected failure lacks typed forward error/cause: %T %v", recovered, recovered)
				}
				if site == "matmul" || site == "read" {
					if !errors.Is(err, b.cause.(error)) {
						t.Error("Engram selected failure lost backend cause identity")
					}
				}
			}
			if !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
				t.Error("late Engram failure changed retained state")
			}
			if len(b.live) != 0 {
				t.Errorf("late Engram failure leaked %d transient buffers", len(b.live))
			}
			delta := v41DenseTestDelta(v41EngProjPhase(t, m, "decode"), ledgerBefore)
			if delta["engram_projection_device_calls"] != 2 || delta["engram_projection_device_rows"] != 1 || delta["engram_projection_host_calls"] != 0 || delta["engram_projection_host_rows"] != 0 || delta["engram_projection_host_weight_f32_bytes"] != 0 || delta["engram_projection_matmul_calls"] != 2 {
				t.Errorf("late Engram failure attempted/completed accounting=%v", delta)
			}
			b.decodeFault, b.site, b.cause, b.failed = false, "", nil, false
			oracle := v41EngProjFixture(t, "F32", false).NewSession()
			t.Cleanup(oracle.Close)
			oracle.Prefill([]int{1, 2, 3})
			v41GroupedParity(t, s.Step(4), oracle.Step(4), 1e-4)
		})
	}
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=1s lane=default
func TestV41EngramProjectionRestoreForkOwnership(t *testing.T) {
	t.Parallel()
	m := v41EngProjFixture(t, "F32", false)
	b := newV41EngProjBackend(m)
	source := v41EngProjSession(t, m, b)
	source.Prefill([]int{1, 2, 3})
	snap, err := source.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer snap.Close()
	clone, err := snap.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer clone.Close()
	left, right := v41EngProjSession(t, m, b), v41EngProjSession(t, m, b)
	if err := snap.Restore(left); err != nil {
		t.Fatal(err)
	}
	if err := clone.Restore(right); err != nil {
		t.Fatal(err)
	}
	source.Close()
	for _, frees := range b.weightFrees {
		if frees != 0 {
			t.Error("source close freed model-owned Engram KV")
		}
	}
	for index, branch := range []*Session{left, right} {
		oracle := v41EngProjFixture(t, "F32", false).NewSession()
		t.Cleanup(oracle.Close)
		oracle.Prefill([]int{1, 2, 3})
		before := b.attempts
		v41GroupedParity(t, branch.Step(4+index), oracle.Step(4+index), 1e-4)
		if b.attempts-before != 2 {
			t.Error("restored target did not invoke its two Engram projections")
		}
	}
	left.Close()
	for _, frees := range b.weightFrees {
		if frees != 0 {
			t.Error("fork close freed sibling's shared Engram KV")
		}
	}
	if b.stages != 2 {
		t.Errorf("restored Engram cache staged %d weights want two shared residents", b.stages)
	}
	right.Close()
	if err := m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	for weight := range b.weightBuffers {
		if b.weightFrees[weight] != 1 {
			t.Errorf("model owner freed Engram KV %d times want once", b.weightFrees[weight])
		}
	}
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=3s lane=default
func TestV41EngramProjectionRectangularRawHCAndOrientation(t *testing.T) {
	t.Parallel()
	for _, standard := range []bool{false, true} {
		t.Run(map[bool]string{false: "legacy-input-output", true: "standard-output-input"}[standard], func(t *testing.T) {
			t.Parallel()
			m, layout := v41RectangularEngramModel(t, 32)
			oracle, _ := v41RectangularEngramModel(t, 32)
			name := layerName(0, "engram_kv.weight")
			if standard {
				meta := m.manifest[name]
				meta.Shape = []int{320, 768}
				m.manifest[name] = meta
			}
			ids := []int{1, 3, 5}
			x, streams := v41FullEngramAsymmetricStreams(ids, m.Cfg.HiddenSize)
			before := make([][][]float32, len(streams))
			for i := range streams {
				before[i] = make([][]float32, 4)
				for h := range streams[i] {
					before[i][h] = append([]float32(nil), streams[i][h]...)
				}
			}
			wantStreams := v41OracleRectangularEngramUpdates(t, oracle, layout, 0, before, ids, float32(m.Cfg.RMSNormEps))
			if err := m.v41EngramInject(0, x, streams, true, ids, float32(m.Cfg.RMSNormEps)); err != nil {
				t.Fatal(err)
			}
			for i := range streams {
				for h := range streams[i] {
					moved := false
					for j, value := range streams[i][h] {
						if math.Abs(float64(value-wantStreams[i][h][j])) > cpuOracleTol {
							t.Fatalf("independent BF16/per-HC oracle mismatch position=%d stream=%d cell=%d", i, h, j)
						}
						moved = moved || value != before[i][h][j]
					}
					if !moved {
						t.Fatalf("raw projection vacuous stream=%d", h)
					}
				}
			}
			for h := 1; h < 4; h++ {
				identical := true
				for j := range streams[0][h] {
					identical = identical && streams[0][h][j]-before[0][h][j] == streams[0][0][j]-before[0][0][j]
				}
				if identical {
					t.Error("four independent HC gates collapsed")
				}
			}
			b := newV41EngProjBackend(m)
			s := v41EngProjSession(t, m, b)
			host := oracle.NewSession()
			t.Cleanup(host.Close)
			for index, prompt := range [][]int{ids, {7}, {2, 4}} {
				var got, want []float32
				from := b.attempts
				if index == 1 {
					got, want = s.Step(prompt[0]), host.Step(prompt[0])
				} else {
					got, want = s.Prefill(prompt), host.Prefill(prompt)
				}
				v41GroupedParity(t, got, want, 1e-4)
				if b.attempts-from != len(prompt) {
					t.Errorf("rectangular actual Session device projections=%d want %d", b.attempts-from, len(prompt))
				}
			}
		})
	}
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime fast est=200ms lane=default
func TestV41EngramProjectionRejectsOtherSameCountShape(t *testing.T) {
	t.Parallel()
	m := v41EngProjFixture(t, "F32", false)
	b := newV41EngProjBackend(m)
	name := layerName(0, "engram_kv.weight")
	meta := m.manifest[name]
	meta.Shape = []int{meta.Shape[0] * 2, meta.Shape[1] / 2}
	m.manifest[name] = meta
	s, err := m.NewBackendSessionChecked(b)
	if err == nil {
		defer s.Close()
		err = panicAsError(func() { s.Prefill([]int{1, 2, 3}) })
	}
	if !errors.Is(err, ErrV41NativeUnsupported) {
		t.Errorf("non-contract shape must refuse with native sentinel: %v", err)
	}
	if b.stages != 0 || b.attempts != 0 {
		t.Error("malformed projection geometry reached grouped device APIs")
	}
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=1s lane=default
func TestV41EngramProjectionHostPartialMaterializationFailure(t *testing.T) {
	t.Parallel()
	m := v41EngProjFixture(t, "Q4_K", false)
	b := newV41EngProjBackend(m)
	b.decline = compute.Q4_K
	s := v41EngProjSession(t, m, b)
	s.Prefill([]int{1, 2, 3})
	before := captureV41ForwardSnapshot(s.v41Forward)
	ledger := v41EngProjPhase(t, m, "decode")
	first, ok := m.residentF32Mat(layerName(0, "engram_kv.weight"))
	if !ok {
		t.Fatal("first host returned payload missing")
	}
	weight := m.q4kw[layerName(2, "engram_kv.weight")]
	reader := &v41GroupedFailingReader{}
	weight.lazy = &LazyQ4KRange{Reader: reader, Bytes: len(weight.raw)}
	weight.raw = nil
	err := panicAsError(func() { s.Step(4) })
	if !errors.Is(err, ErrV41ForwardStage) {
		t.Errorf("host retrieval failure lost closed stage identity: %v", err)
	}
	if reader.calls == 0 {
		t.Fatal("lazy host failure fixture was not invoked")
	}
	delta := v41DenseTestDelta(v41EngProjPhase(t, m, "decode"), ledger)
	if delta["engram_projection_host_weight_f32_bytes"] != float64(4*len(first)*reader.calls) || delta["engram_projection_device_calls"] != 0 || delta["engram_projection_matmul_calls"] != 0 || delta["engram_projection_host_rows"] < float64(reader.calls) || delta["engram_projection_nanos"] <= 0 {
		t.Errorf("actual partial returned host payload accounting=%v reader_attempts=%d", delta, reader.calls)
	}
	if !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
		t.Error("late host materialization failure changed retained state")
	}
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime fast est=100ms lane=default
func TestV41EngramProjectionSourceNamesAndCollision(t *testing.T) {
	t.Parallel()
	cfg := v41EngProjFixture(t, "F32", false).Cfg
	cfg.ModelType, cfg.DeepSeekV41 = "deepseek41", nil
	for _, leaf := range []string{"engram_kv.weight", "engram_q_norm.weight", "engram_k_norm.weight"} {
		source, native := "model.engram.0."+leaf, layerName(0, leaf)
		if got, keep := quantSourceTensorNameWithRetention(cfg, source, true); !keep || got != native {
			t.Errorf("source %s resolves to %s retained=%t", source, got, keep)
		}
		shape := []int{4}
		if leaf == "engram_kv.weight" {
			shape = []int{2, 256}
		}
		n := 1
		for _, width := range shape {
			n *= width
		}
		first, second := make([]float32, n), make([]float32, n)
		for i := range first {
			first[i] = float32((i%13)-6) / 32
			second[i] = float32((i%7)+1) / 16
		}
		for _, order := range [][]string{{source, native}, {native, source}} {
			for _, name := range order {
				one, err := NewFromF32Tensors(cfg, []NamedTensorF32{{Name: name, Shape: shape, Data: first}})
				if err != nil {
					t.Fatalf("individually valid plain source refused: %v", err)
				}
				if got, ok := one.residentF32Mat(native); !ok || !reflect.DeepEqual(got, first) {
					t.Error("plain source mapping changed flat payload")
				}
			}
			collision, err := NewFromF32Tensors(cfg, []NamedTensorF32{{Name: order[0], Shape: shape, Data: first}, {Name: order[1], Shape: shape, Data: second}})
			if err == nil || collision != nil {
				t.Error("plain source/native collision admitted duplicate writer")
			}
			b := NewQuantBuilder(cfg, false)
			if err := b.AddF32Tensor("lm_head.weight", []int{1, 32}, make([]float32, 32)); err != nil {
				t.Fatal(err)
			}
			if err := b.AddF32Tensor(order[0], shape, first); err != nil {
				t.Fatal(err)
			}
			if err := b.AddF32Tensor(order[1], shape, second); err == nil {
				t.Error("quantized F32 source/native collision admitted duplicate writer")
			}
			built, err := b.Build()
			if err != nil {
				t.Fatal(err)
			}
			if leaf == "engram_kv.weight" {
				want := quantizeQ8(first, shape[0], shape[1])
				got := built.q8w[native]
				if got == nil || got.out != want.out || got.in != want.in || !reflect.DeepEqual(got.q, want.q) || !reflect.DeepEqual(got.d, want.d) {
					t.Error("refused writer replaced first quantized KV payload")
				}
			} else if got, ok := built.residentF32Mat(native); !ok || !reflect.DeepEqual(got, first) {
				t.Error("refused writer replaced first norm payload")
			}
			if leaf == "engram_kv.weight" {
				b := NewQuantBuilder(cfg, false)
				raw := q2kFixtureTensor(shape[0], shape[1], 136681).raw
				other := q2kFixtureTensor(shape[0], shape[1], 136682).raw
				if err := b.AddResidentQ2K(order[0], shape, raw); err != nil {
					t.Fatal(err)
				}
				if b.m.kqw[native] == nil {
					t.Fatal("individually valid packed KV was not retained")
				}
				if err := b.AddResidentQ2K(order[1], shape, other); err == nil {
					t.Error("packed source/native collision admitted duplicate writer")
				}
				built, err := b.Build()
				if err != nil {
					t.Fatal(err)
				}
				if got := built.kqw[native]; got == nil || !reflect.DeepEqual(got.raw, raw) {
					t.Error("refused packed writer replaced first resident payload")
				}
			}
		}
	}
	for _, name := range []string{"model.engram.0.engram_embd.weight", "model.engram.0.engram_kv.weight.extra", "model.engram.0.other.weight", "model.engram.x.engram_kv.weight"} {
		if got, keep := quantSourceTensorNameWithRetention(cfg, name, true); !keep || got != name {
			t.Errorf("strict source adapter altered unrelated name %s -> %s", name, got)
		}
	}
	other := cfg
	other.ModelType = "llama"
	other.DeepSeekV41 = nil
	name := "model.engram.0.engram_kv.weight"
	if got, keep := quantSourceTensorNameWithRetention(other, name, true); !keep || got != name {
		t.Error("V4.1 adapter affected another architecture")
	}
}
