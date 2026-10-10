package model

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"sort"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

var v41HeadProjFields = []string{"head_projection_device_calls", "head_projection_host_calls", "head_projection_device_rows", "head_projection_host_rows", "head_projection_activation_upload_bytes", "head_projection_readback_bytes", "head_projection_nanos"}

func v41HeadProjPhase(t *testing.T, m *Model, phase string) map[string]float64 {
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
	for _, key := range v41HeadProjFields {
		var n float64
		if err := json.Unmarshal(objects[phase][key], &n); err != nil {
			t.Errorf("default %s.%s unavailable: %v", phase, key, err)
			continue
		}
		out[key] = n
	}
	return out
}

func v41HeadProjFixture(t *testing.T) *Model {
	t.Helper()
	cfg := v41ResidentHeadConfig(t)
	cfg.OGroups = 1
	cfg.LogitScale = .0625
	m := v41BuildFullModel(cfg, v41KVLoraRank)
	meta := m.manifest[layerName(0, "mhc.mixes.weight")]
	meta.Shape = []int{24, 4 * cfg.HiddenSize}
	m.manifest[layerName(0, "mhc.mixes.weight")] = meta
	names := make([]string, 0, len(m.manifest))
	for name := range m.manifest {
		names = append(names, name)
	}
	sort.Strings(names)
	tensors := make([]synthTensor, 0, len(names))
	for _, name := range names {
		tensors = append(tensors, synthTensor{name, m.manifest[name].Shape})
	}
	m.manifest, m.raw = synthBuildRaw(tensors, func(name string, next func() float32) float32 {
		switch {
		case name == "model.norm.weight" || hasSuffix(name, "attn_norm.weight") || hasSuffix(name, "ffn_norm.weight") || hasSuffix(name, "attn.wq_a_norm.weight") || hasSuffix(name, "attn.kv_norm.weight"):
			return 1
		case hasSuffix(name, "mhc.ffn_scale"):
			return .75
		case hasSuffix(name, "mhc.ffn_base"):
			return .125 * next()
		case hasSuffix(name, "mhc.scale"):
			return 1
		case hasSuffix(name, "mhc.base"):
			return 0
		case hasSuffix(name, "attn.sink"):
			return .25 * next()
		default:
			return synthMatmulFill(name, next)
		}
	})
	m.kqw = map[string]*kQuantTensor{"lm_head.weight": q2kFixtureTensor(cfg.VocabSize, cfg.HiddenSize, 1366813001)}
	delete(m.manifest, "lm_head.weight")
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	})
	return m
}

type v41HeadProjOp struct {
	rows, upload, read int
	weight             compute.Buffer
	activation, result []float32
}
type v41HeadProjBackend struct {
	*v41DenseTestBackend
	raw                              []byte
	dtype                            compute.Dtype
	decline                          bool
	weights                          map[compute.Buffer]bool
	outputs                          map[compute.Buffer]int
	payloads                         map[compute.Buffer][]float32
	operations                       []v41HeadProjOp
	stages                           int
	fault                            bool
	site                             string
	cause                            any
	faultAttempts                    int
	apiUploads, apiMatMuls, apiReads int
	weightFrees                      map[compute.Buffer]int
}

func newV41HeadProjBackend(raw []byte, decline bool) *v41HeadProjBackend {
	return &v41HeadProjBackend{v41DenseTestBackend: newV41DenseTestBackend(), raw: raw, dtype: compute.Q2_K, decline: decline, weightFrees: map[compute.Buffer]int{}, weights: map[compute.Buffer]bool{}, outputs: map[compute.Buffer]int{}, payloads: map[compute.Buffer][]float32{}}
}
func (b *v41HeadProjBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return !(b.decline && dt == b.dtype) && b.v41DenseTestBackend.SupportsDeviceWeightDtype(dt)
}
func v41HeadProjCodesEqual(codes []int8, raw []byte) bool {
	if len(codes) != len(raw) {
		return false
	}
	for i, code := range codes {
		if byte(code) != raw[i] {
			return false
		}
	}
	return true
}
func (b *v41HeadProjBackend) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	b.apiUploads++
	y := b.v41DenseTestBackend.Upload(x, dt)
	if host, ok := x.Buf().(compute.HostBuffer); ok {
		if dt == b.dtype && v41HeadProjCodesEqual(host.I8(), b.raw) {
			b.weights[y.Buf()] = true
			b.stages++
		}
		if dt == compute.F32 {
			b.payloads[y.Buf()] = append([]float32(nil), host.F32()...)
		}
	}
	return y
}
func (b *v41HeadProjBackend) multiply(w, x compute.Tensor, rows int, batched bool) compute.Tensor {
	b.apiMatMuls++
	if b.weights[w.Buf()] && b.fault {
		b.faultAttempts++
		if b.site == "matmul" {
			panic(b.cause)
		}
	}

	var y compute.Tensor
	if batched {
		y = b.v41DenseTestBackend.BatchedMatMul(w, x, rows)
	} else {
		y = b.v41DenseTestBackend.MatMul(w, x)
	}
	if b.weights[w.Buf()] {
		b.operations = append(b.operations, v41HeadProjOp{rows: rows, upload: b.uploads[x.Buf()], weight: w.Buf(), activation: append([]float32(nil), b.payloads[x.Buf()]...)})
		b.outputs[y.Buf()] = len(b.operations) - 1
	}
	return y
}
func (b *v41HeadProjBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	return b.multiply(w, x, 1, false)
}
func (b *v41HeadProjBackend) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	return b.multiply(w, x, rows, true)
}
func (b *v41HeadProjBackend) Read(x compute.Tensor) []float32 {
	b.apiReads++
	_, head := b.outputs[x.Buf()]
	if head && b.fault && b.site == "read" {
		panic(b.cause)
	}

	y := b.v41DenseTestBackend.Read(x)
	if head && b.fault {
		if b.site == "length" {
			y = append(y, 0)
		}
		if b.site == "finite" {
			y[0] = float32(math.NaN())
		}
	}
	if index, ok := b.outputs[x.Buf()]; ok {
		b.operations[index].read += 4 * len(y)
		b.operations[index].result = append([]float32(nil), y...)
	}
	return y
}
func (b *v41HeadProjBackend) Free(x compute.Tensor) {
	if b.weights[x.Buf()] {
		b.weightFrees[x.Buf()]++
	}
	delete(b.outputs, x.Buf())
	delete(b.payloads, x.Buf())
	b.v41DenseTestBackend.Free(x)
}

// fak-test:justify why=integration when=changed:internal/model/**
// fak-test:runtime medium est=5s lane=default
func TestV41HeadProjectionActualDefaultSession(t *testing.T) {
	t.Parallel()
	// The bounded full-HC fixture shares about 76 MiB immutable source bytes; each Session owns its runtime state.
	m := v41HeadProjFixture(t)
	control := v41HeadProjClone(t, m)
	b, hb := newV41HeadProjBackend(m.kqw["lm_head.weight"].raw, false), newV41HeadProjBackend(m.kqw["lm_head.weight"].raw, true)
	s, host := v41EngProjSession(t, m, b), v41EngProjSession(t, control, hb)
	for index, ids := range [][]int{{1, 2, 3}, {4}, {5, 6}} {
		phase := "prefill"
		if index == 1 {
			phase = "decode"
		}
		before, hbefore := v41HeadProjPhase(t, m, phase), v41HeadProjPhase(t, control, phase)
		db, gb, mb := v41DenseTestPhase(t, m, phase), v41GroupedPhase(t, m, phase), v41MHCProjPhase(t, m, phase)
		from := len(b.operations)
		var got, want []float32
		if index == 1 {
			got, want = s.Step(ids[0]), host.Step(ids[0])
		} else {
			got, want = s.Prefill(ids), host.Prefill(ids)
		}
		v41GroupedParity(t, got, want, 1e-4)
		delta, hd := v41DenseTestDelta(v41HeadProjPhase(t, m, phase), before), v41DenseTestDelta(v41HeadProjPhase(t, control, phase), hbefore)
		rows, upload, read := 0, 0, 0
		for _, op := range b.operations[from:] {
			rows += op.rows
			upload += op.upload
			read += op.read
		}
		if rows != len(ids) || delta["head_projection_device_rows"] != float64(len(ids)) || delta["head_projection_device_calls"] <= 0 || delta["head_projection_host_calls"] != 0 || delta["head_projection_host_rows"] != 0 {
			t.Errorf("default head route rows=%d want=%d ledger=%v", rows, len(ids), delta)
		}
		if upload != 4*m.Cfg.HiddenSize*len(ids) || read != 4*m.Cfg.VocabSize*len(ids) || delta["head_projection_activation_upload_bytes"] != float64(upload) || delta["head_projection_readback_bytes"] != float64(read) {
			t.Errorf("actual head transfers upload=%d read=%d ledger=%v", upload, read, delta)
		}
		if hd["head_projection_device_calls"] != 0 || hd["head_projection_device_rows"] != 0 || hd["head_projection_host_rows"] != float64(len(ids)) || len(hb.operations) != 0 {
			t.Errorf("natural head dtype decline=%v operations=%d", hd, len(hb.operations))
		}
		if delta["head_projection_nanos"] <= 0 || hd["head_projection_nanos"] <= 0 {
			t.Error("head timing missing on selected or declined arm")
		}
		dense, group, mhc := v41DenseTestDelta(v41DenseTestPhase(t, m, phase), db), v41DenseTestDelta(v41GroupedPhase(t, m, phase), gb), v41DenseTestDelta(v41MHCProjPhase(t, m, phase), mb)
		if dense["dense_projection_device_rows"] != float64(7*len(ids)) || group["grouped_output_device_rows"] != float64(len(ids)) || mhc["mhc_projection_device_rows"] != float64(2*len(ids)) {
			t.Errorf("ordinary default composition dense=%v grouped=%v mhc=%v", dense, group, mhc)
		}
	}
	if b.stages != 1 {
		t.Errorf("immutable packed head uploads=%d want1", b.stages)
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=1ms lane=default
func TestV41HeadProjectionDefaultExportedLedger(t *testing.T) {
	t.Parallel()
	typ := reflect.TypeOf(V41ExpertFaultAttribution{}.Prefill)
	for i, name := range []string{"HeadProjectionDeviceCalls", "HeadProjectionHostCalls", "HeadProjectionDeviceRows", "HeadProjectionHostRows", "HeadProjectionActivationUploadBytes", "HeadProjectionReadbackBytes", "HeadProjectionNanos"} {
		field, ok := typ.FieldByName(name)
		if !ok || field.PkgPath != "" {
			t.Errorf("missing exported phase field %s", name)
			continue
		}
		if field.Tag.Get("json") != v41HeadProjFields[i] {
			t.Errorf("phase field %s lacks default JSON contract", name)
		}
	}
	for _, phase := range []string{"prefill", "decode"} {
		for key, n := range v41HeadProjPhase(t, &Model{}, phase) {
			if n != 0 {
				t.Errorf("inert %s=%g", key, n)
			}
		}
	}
}

func v41HeadProjClone(t *testing.T, m *Model) *Model {
	t.Helper()
	clone := &Model{Cfg: m.Cfg, manifest: m.manifest, raw: m.raw, kqw: m.kqw}
	t.Cleanup(func() {
		if err := clone.CloseWeights(); err != nil {
			t.Error(err)
		}
	})
	return clone
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=6s lane=default
func TestV41HeadProjectionSelectedFailureRollback(t *testing.T) {
	t.Parallel()
	fixture := v41HeadProjFixture(t)
	for _, site := range []string{"matmul", "read", "length", "finite", "unknown"} {
		t.Run(site, func(t *testing.T) {
			m := v41HeadProjClone(t, fixture)
			b := newV41HeadProjBackend(m.kqw["lm_head.weight"].raw, false)
			s := v41EngProjSession(t, m, b)
			s.Prefill([]int{1, 2, 3})
			before := captureV41ForwardSnapshot(s.v41Forward)
			ledger := v41HeadProjPhase(t, m, "decode")
			b.live = map[compute.Buffer]bool{}
			b.fault = true
			b.site = site
			b.cause = &compute.BackendError{Backend: "cpu-ref-recording", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
			unknown := &struct{ Number int }{1366813}
			if site == "unknown" {
				b.site = "matmul"
				b.cause = unknown
			}
			var recovered any
			var firstClosed *BackendForwardOperationError
			func() { defer func() { recovered = recover() }(); s.Step(4) }()
			if b.faultAttempts != 1 {
				t.Errorf("selected head attempts=%d want1", b.faultAttempts)
			}
			if site == "unknown" {
				if recovered != unknown {
					t.Error("unknown head panic identity changed")
				}
			} else {
				err, ok := recovered.(error)
				var selected *V41ProjectionOperationError
				if !ok || !errors.As(err, &selected) || !errors.Is(err, ErrV41ForwardStage) {
					t.Errorf("selected head failure lacks closed typed cause: %T", recovered)
				}
				if selected != nil && (selected.Stage != string(v41StageHead) || selected.Layer != -1) {
					t.Error("selected failure lost global head stage identity")
				}
				if !errors.As(err, &firstClosed) || !s.BackendSessionClosed() {
					t.Error("selected head failure lost closed Session contract")
				}
				if (site == "matmul" || site == "read") && !errors.Is(err, b.cause.(error)) {
					t.Error("selected head backend failure changed cause identity")
				}
			}
			if !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
				t.Error("selected head failure changed retained state")
			}
			if len(b.live) != 0 {
				t.Errorf("head failure leaked %d transient buffers", len(b.live))
			}
			delta := v41DenseTestDelta(v41HeadProjPhase(t, m, "decode"), ledger)
			if delta["head_projection_device_calls"] != 1 || delta["head_projection_device_rows"] != 0 || delta["head_projection_host_calls"] != 0 || delta["head_projection_host_rows"] != 0 {
				t.Errorf("selected head completion/no-host-retry ledger=%v", delta)
			}
			if site != "unknown" {
				apiBefore := [3]int{b.apiUploads, b.apiMatMuls, b.apiReads}
				opsBefore := append([]v41DenseTestOp(nil), b.ops...)
				uploadBefore, readBefore := map[compute.Buffer]int{}, map[compute.Buffer]int{}
				liveBefore := map[compute.Buffer]bool{}
				for key, value := range b.uploads {
					uploadBefore[key] = value
				}
				for key, value := range b.reads {
					readBefore[key] = value
				}
				for key, value := range b.live {
					liveBefore[key] = value
				}
				attempts := b.faultAttempts
				var retry any
				func() { defer func() { retry = recover() }(); s.Step(4) }()
				var closed *BackendForwardOperationError
				err, ok := retry.(error)
				if !ok || !errors.As(err, &closed) || closed != firstClosed || !errors.Is(err, ErrV41ForwardStage) || b.faultAttempts != attempts {
					t.Error("closed head Session retried execution or changed cause")
				}
				if apiBefore != [3]int{b.apiUploads, b.apiMatMuls, b.apiReads} || !reflect.DeepEqual(opsBefore, b.ops) || !reflect.DeepEqual(uploadBefore, b.uploads) || !reflect.DeepEqual(readBefore, b.reads) || !reflect.DeepEqual(liveBefore, b.live) {
					t.Error("closed Session invoked backend or changed retained backend resources")
				}
				return
			}
			b.fault = false
			b.cause = nil
			control := v41HeadProjClone(t, fixture)
			host := v41EngProjSession(t, control, newV41HeadProjBackend(control.kqw["lm_head.weight"].raw, true))
			host.Prefill([]int{1, 2, 3})
			v41GroupedParity(t, s.Step(4), host.Step(4), 1e-4)
		})
	}
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=4s lane=default
func TestV41HeadProjectionRestoreForkOwnership(t *testing.T) {
	t.Parallel()
	m := v41HeadProjFixture(t)
	b := newV41HeadProjBackend(m.kqw["lm_head.weight"].raw, false)
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
	for index, branch := range []*Session{left, right} {
		control := v41HeadProjClone(t, m)
		host := v41EngProjSession(t, control, newV41HeadProjBackend(m.kqw["lm_head.weight"].raw, true))
		host.Prefill([]int{1, 2, 3})
		from := len(b.operations)
		v41GroupedParity(t, branch.Step(4+index), host.Step(4+index), 1e-4)
		if len(b.operations)-from != 1 {
			t.Error("restored branch did not invoke head once")
		}
	}
	left.Close()
	for _, frees := range b.weightFrees {
		if frees != 0 {
			t.Error("Session close freed model-owned head")
		}
	}
	if b.stages != 1 {
		t.Errorf("restored immutable head stages=%d want1", b.stages)
	}
	right.Close()
	if err := m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	if len(b.weights) != 1 {
		t.Errorf("owned head buffers=%d want1", len(b.weights))
	}
	for weight := range b.weights {
		if b.weightFrees[weight] != 1 {
			t.Errorf("owned head freed=%d want1", b.weightFrees[weight])
		}
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=4s lane=default
func TestV41HeadProjectionHostNormAndScaleSemantics(t *testing.T) {
	t.Parallel()
	fixture := v41HeadProjFixture(t)
	for _, layerNorm := range []bool{false, true} {
		t.Run(map[bool]string{false: "gain1p-rms", true: "layernorm-bias"}[layerNorm], func(t *testing.T) {
			m := v41HeadProjClone(t, fixture)
			m.Cfg.LayerNorm = layerNorm
			m.Cfg.NormGain1p = !layerNorm
			if layerNorm {
				m.manifest = make(map[string]tensorMeta, len(fixture.manifest)+1)
				for name, meta := range fixture.manifest {
					m.manifest[name] = meta
				}
				offset := len(m.raw)
				m.raw = append([]byte(nil), m.raw...)
				for cell := 0; cell < m.Cfg.HiddenSize; cell++ {
					var payload [4]byte
					binary.LittleEndian.PutUint32(payload[:], math.Float32bits(float32(cell%11-5)*.03125))
					m.raw = append(m.raw, payload[:]...)
				}
				m.manifest["model.norm.bias"] = tensorMeta{Dtype: "F32", Shape: []int{m.Cfg.HiddenSize}, Offset: offset, Nbytes: 4 * m.Cfg.HiddenSize}
			}
			oracle := v41HeadProjClone(t, m)
			b := newV41HeadProjBackend(m.kqw["lm_head.weight"].raw, false)
			s := v41EngProjSession(t, m, b)
			ids := []int{1, 2, 3}
			got := s.Prefill(ids)
			want := lastLogits(oracle.Forward(ids))
			v41GroupedParity(t, got, want, 1e-4)
			rows := 0
			for _, op := range b.operations {
				rows += op.rows
			}
			if rows != len(ids) || len(b.operations) == 0 {
				t.Fatal("host norm contract did not reach actual head GEMM")
			}
			last := b.operations[len(b.operations)-1]
			if len(last.result) < m.Cfg.VocabSize {
				t.Fatal("head readback missing")
			}
			start := len(last.result) - m.Cfg.VocabSize
			for cell, value := range got {
				expected := last.result[start+cell] * float32(m.Cfg.LogitScale)
				if math.Abs(float64(value-expected)) > 1e-6*math.Max(1, math.Abs(float64(expected))) {
					t.Error("host logit scale was omitted or applied more than once")
				}
			}
		})
	}
}

// fak-test:justify why=integration when=changed:internal/model/**
// fak-test:runtime medium est=5s lane=default
func TestV41HeadProjectionDedicatedPackedTiedStores(t *testing.T) {
	t.Parallel()
	fixture := v41HeadProjFixture(t)
	for _, dtype := range []compute.Dtype{compute.Q2_K, compute.Q4_K} {
		t.Run(map[compute.Dtype]string{compute.Q2_K: "q2", compute.Q4_K: "q4"}[dtype], func(t *testing.T) {
			m := v41HeadProjClone(t, fixture)
			m.manifest = make(map[string]tensorMeta, len(fixture.manifest))
			for name, meta := range fixture.manifest {
				m.manifest[name] = meta
			}
			delete(m.manifest, "model.embed_tokens.weight")
			m.kqw = nil
			raw := fixture.kqw["lm_head.weight"].raw
			var err error
			if dtype == compute.Q2_K {
				m.Q2KEmbedding, err = NewQ2KEmbedding(raw, m.Cfg.VocabSize, m.Cfg.HiddenSize)
				if err != nil {
					t.Fatal(err)
				}
			} else {
				raw = v41DenseTestQ4Raw(m.Cfg.VocabSize, m.Cfg.HiddenSize)
				m.Q2KEmbedding, err = NewQ4KEmbedding(raw, m.Cfg.VocabSize, m.Cfg.HiddenSize)
				if err != nil {
					t.Fatal(err)
				}
			}
			if m.headName() != "model.embed_tokens.weight" {
				t.Fatal("dedicated packed store did not resolve actual tied head")
			}
			control := v41HeadProjClone(t, m)
			control.Q2KEmbedding = m.Q2KEmbedding
			b, hb := newV41HeadProjBackend(raw, false), newV41HeadProjBackend(raw, true)
			b.dtype, hb.dtype = dtype, dtype
			s, host := v41EngProjSession(t, m, b), v41EngProjSession(t, control, hb)
			var weight compute.Buffer
			for index, ids := range [][]int{{1, 2, 3}, {4}, {5, 6}} {
				phase := "prefill"
				if index == 1 {
					phase = "decode"
				}
				before, hbefore := v41HeadProjPhase(t, m, phase), v41HeadProjPhase(t, control, phase)
				denseBefore, groupBefore, mhcBefore := v41DenseTestPhase(t, m, phase), v41GroupedPhase(t, m, phase), v41MHCProjPhase(t, m, phase)
				from := len(b.operations)
				var got, want []float32
				if index == 1 {
					got, want = s.Step(ids[0]), host.Step(ids[0])
				} else {
					got, want = s.Prefill(ids), host.Prefill(ids)
				}
				v41GroupedParity(t, got, want, 1e-4)
				delta, hd := v41DenseTestDelta(v41HeadProjPhase(t, m, phase), before), v41DenseTestDelta(v41HeadProjPhase(t, control, phase), hbefore)
				rows, upload, read := 0, 0, 0
				for _, op := range b.operations[from:] {
					rows += op.rows
					upload += op.upload
					read += op.read
					if weight == nil {
						weight = op.weight
					} else if weight != op.weight {
						t.Error("tied head immutable cache identity changed")
					}
				}
				if rows != len(ids) || delta["head_projection_device_rows"] != float64(len(ids)) || delta["head_projection_device_calls"] <= 0 || delta["head_projection_host_rows"] != 0 || delta["head_projection_host_calls"] != 0 || delta["head_projection_activation_upload_bytes"] != float64(upload) || delta["head_projection_readback_bytes"] != float64(read) || upload != 4*m.Cfg.HiddenSize*len(ids) || read != 4*m.Cfg.VocabSize*len(ids) {
					t.Errorf("dedicated tied head actual route/traffic=%v rows=%d", delta, rows)
				}
				if hd["head_projection_device_rows"] != 0 || hd["head_projection_host_rows"] != float64(len(ids)) || hd["head_projection_host_calls"] <= 0 || len(hb.operations) != 0 {
					t.Errorf("dedicated tied natural dtype decline=%v", hd)
				}
				dense, group, mhc := v41DenseTestDelta(v41DenseTestPhase(t, m, phase), denseBefore), v41DenseTestDelta(v41GroupedPhase(t, m, phase), groupBefore), v41DenseTestDelta(v41MHCProjPhase(t, m, phase), mhcBefore)
				if dense["dense_projection_device_rows"] != float64(7*len(ids)) || group["grouped_output_device_rows"] != float64(len(ids)) || mhc["mhc_projection_device_rows"] != float64(2*len(ids)) {
					t.Error("dedicated tied head suppressed ordinary default composition")
				}
			}
			if b.stages != 1 {
				t.Errorf("dedicated packed tied head uploads=%d want1", b.stages)
			}
		})
	}
}
