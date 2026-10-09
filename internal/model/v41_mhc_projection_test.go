package model

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

var v41MHCProjFields = []string{"mhc_projection_device_calls", "mhc_projection_host_calls", "mhc_projection_device_rows", "mhc_projection_host_rows", "mhc_projection_matmul_calls", "mhc_projection_activation_upload_bytes", "mhc_projection_readback_bytes", "mhc_projection_nanos", "mhc_projection_host_weight_f32_bytes"}

func v41MHCProjPhase(t *testing.T, m *Model, phase string) map[string]float64 {
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
	for _, key := range v41MHCProjFields {
		var n float64
		if err := json.Unmarshal(objects[phase][key], &n); err != nil {
			t.Errorf("default %s.%s unavailable: %v", phase, key, err)
			continue
		}
		out[key] = n
	}
	return out
}

func v41MHCProjFixture(t *testing.T) *Model {
	t.Helper()
	m := v41RawFullFlattenedMHC(t)
	name := layerName(0, "mhc.mixes.weight")
	m.kqw = map[string]*kQuantTensor{name: q2kFixtureTensor(24, 4*m.Cfg.HiddenSize, 136681001)}
	delete(m.manifest, name)
	return m
}

type v41MHCProjBackend struct {
	*v41DenseTestBackend
	in               int
	decline          bool
	weights          map[compute.Buffer]bool
	outputs          map[compute.Buffer]int
	operations       []v41MHCProjOperation
	stages, attempts int
	payloads         map[compute.Buffer][]float32
	weightFrees      map[compute.Buffer]int
	fault            bool
	faultAttempts    int
	site             string
	cause            any
	failed           bool
}

type v41MHCProjOperation struct {
	rows, upload, read int
	weight             compute.Buffer
	activation, result []float32
}

func newV41MHCProjBackend(in int, decline bool) *v41MHCProjBackend {
	return &v41MHCProjBackend{v41DenseTestBackend: newV41DenseTestBackend(), in: in, decline: decline, weights: map[compute.Buffer]bool{}, outputs: map[compute.Buffer]int{}, payloads: map[compute.Buffer][]float32{}, weightFrees: map[compute.Buffer]int{}}
}
func (b *v41MHCProjBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return !(b.decline && dt == compute.Q2_K) && b.v41DenseTestBackend.SupportsDeviceWeightDtype(dt)
}
func (b *v41MHCProjBackend) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	y := b.v41DenseTestBackend.Upload(x, dt)
	if dt == compute.F32 && len(x.Shape) == 1 && x.Shape[0] == b.in {
		if host, ok := x.Buf().(compute.HostBuffer); ok {
			b.payloads[y.Buf()] = append([]float32(nil), host.F32()...)
		}
	}
	if len(x.Shape) == 2 && x.Shape[0] == 24 && x.Shape[1] == b.in {
		b.weights[y.Buf()] = true
		b.stages++
	}
	return y
}
func (b *v41MHCProjBackend) multiply(w, x compute.Tensor, rows int, batched bool) compute.Tensor {
	mhc := b.weights[w.Buf()]
	if mhc {
		b.attempts++
		if b.fault {
			b.faultAttempts++
			if b.faultAttempts == 2 && b.site == "matmul" {
				b.failed = true
				panic(b.cause)
			}
		}
	}
	var y compute.Tensor
	if batched {
		y = b.v41DenseTestBackend.BatchedMatMul(w, x, rows)
	} else {
		y = b.v41DenseTestBackend.MatMul(w, x)
	}
	if mhc {
		b.operations = append(b.operations, v41MHCProjOperation{rows: rows, upload: b.uploads[x.Buf()], weight: w.Buf(), activation: append([]float32(nil), b.payloads[x.Buf()]...)})
		b.outputs[y.Buf()] = len(b.operations) - 1
	}
	return y
}
func (b *v41MHCProjBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	return b.multiply(w, x, 1, false)
}
func (b *v41MHCProjBackend) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	return b.multiply(w, x, rows, true)
}
func (b *v41MHCProjBackend) Read(x compute.Tensor) []float32 {
	index, mhc := b.outputs[x.Buf()]
	late := mhc && b.fault && b.faultAttempts == 2
	if late && b.site == "read" {
		b.failed = true
		panic(b.cause)
	}
	y := b.v41DenseTestBackend.Read(x)
	if late && b.site == "length" {
		b.failed = true
		y = append(y, 0)
	}
	if late && b.site == "finite" {
		b.failed = true
		y[0] = float32(math.NaN())
	}
	if mhc {
		b.operations[index].read += 4 * len(y)
		b.operations[index].result = append([]float32(nil), y...)
	}
	return y
}

func (b *v41MHCProjBackend) Free(x compute.Tensor) {
	if b.weights[x.Buf()] {
		b.weightFrees[x.Buf()]++
	}
	delete(b.payloads, x.Buf())
	delete(b.outputs, x.Buf())
	b.v41DenseTestBackend.Free(x)
}

// fak-test:justify why=integration when=changed:internal/model/**
// fak-test:runtime medium est=4s lane=default
func TestV41MHCProjectionActualDefaultSession(t *testing.T) {
	t.Parallel()
	m, control := v41MHCProjFixture(t), v41MHCProjFixture(t)
	b, hostBackend := newV41MHCProjBackend(4*m.Cfg.HiddenSize, false), newV41MHCProjBackend(4*control.Cfg.HiddenSize, true)
	s, host := v41EngProjSession(t, m, b), v41EngProjSession(t, control, hostBackend)
	for index, ids := range [][]int{{1, 2, 3}, {4}, {5, 6}} {
		phase := "prefill"
		if index == 1 {
			phase = "decode"
		}
		before, hostBefore := v41MHCProjPhase(t, m, phase), v41MHCProjPhase(t, control, phase)
		denseBefore, groupBefore := v41DenseTestPhase(t, m, phase), v41GroupedPhase(t, m, phase)
		from, attempts := len(b.operations), b.attempts
		var got, want []float32
		if index == 1 {
			got, want = s.Step(ids[0]), host.Step(ids[0])
		} else {
			got, want = s.Prefill(ids), host.Prefill(ids)
		}
		v41GroupedParity(t, got, want, 1e-4)
		delta, hostDelta := v41DenseTestDelta(v41MHCProjPhase(t, m, phase), before), v41DenseTestDelta(v41MHCProjPhase(t, control, phase), hostBefore)
		rows, upload, read := 0, 0, 0
		for _, op := range b.operations[from:] {
			rows += op.rows
			upload += op.upload
			read += op.read
		}
		if rows != len(ids) || delta["mhc_projection_device_rows"] != float64(len(ids)) || delta["mhc_projection_device_calls"] <= 0 || delta["mhc_projection_host_calls"] != 0 || delta["mhc_projection_host_rows"] != 0 || delta["mhc_projection_host_weight_f32_bytes"] != 0 {
			t.Errorf("actual default Session mHC route=%v recorded_rows=%d want=%d", delta, rows, len(ids))
		}
		if delta["mhc_projection_matmul_calls"] != float64(b.attempts-attempts) || delta["mhc_projection_activation_upload_bytes"] != float64(upload) || delta["mhc_projection_readback_bytes"] != float64(read) || upload != 4*b.in*len(ids) || read != 4*24*len(ids) {
			t.Errorf("actual mHC API/transfer ledger=%v matmuls=%d upload=%d read=%d", delta, b.attempts-attempts, upload, read)
		}
		materializations := 1
		if index == 2 {
			materializations = len(ids)
		}
		if hostDelta["mhc_projection_device_calls"] != 0 || hostDelta["mhc_projection_device_rows"] != 0 || hostDelta["mhc_projection_matmul_calls"] != 0 || hostDelta["mhc_projection_host_rows"] != float64(len(ids)) || hostDelta["mhc_projection_host_weight_f32_bytes"] != float64(4*24*b.in*materializations) {
			t.Errorf("whole mHC dtype decline host ledger=%v", hostDelta)
		}
		if delta["mhc_projection_nanos"] <= 0 || hostDelta["mhc_projection_nanos"] <= 0 {
			t.Error("mHC timing missing on selected or declined operation")
		}
		dense := v41DenseTestDelta(v41DenseTestPhase(t, m, phase), denseBefore)
		group := v41DenseTestDelta(v41GroupedPhase(t, m, phase), groupBefore)
		if dense["dense_projection_device_rows"] != float64(7*len(ids)) || group["grouped_output_device_rows"] != float64(len(ids)) {
			t.Errorf("ordinary default composition dense=%v grouped=%v", dense, group)
		}
	}
	if b.stages != 1 || hostBackend.stages != 0 || len(hostBackend.operations) != 0 {
		t.Errorf("immutable mHC staging device=%d declined=%d declined_ops=%d", b.stages, hostBackend.stages, len(hostBackend.operations))
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=1ms lane=default
func TestV41MHCProjectionDefaultExportedLedger(t *testing.T) {
	t.Parallel()
	phaseType := reflect.TypeOf(V41ExpertFaultAttribution{}.Prefill)
	for i, name := range []string{"MHCProjectionDeviceCalls", "MHCProjectionHostCalls", "MHCProjectionDeviceRows", "MHCProjectionHostRows", "MHCProjectionMatMulCalls", "MHCProjectionActivationUploadBytes", "MHCProjectionReadbackBytes", "MHCProjectionNanos", "MHCProjectionHostWeightF32Bytes"} {
		field, ok := phaseType.FieldByName(name)
		if !ok || field.PkgPath != "" {
			t.Errorf("missing exported phase field %s", name)
			continue
		}
		if field.Tag.Get("json") != v41MHCProjFields[i] {
			t.Errorf("phase field %s lacks default JSON contract %s", name, v41MHCProjFields[i])
		}
	}
	for _, phase := range []string{"prefill", "decode"} {
		for key, n := range v41MHCProjPhase(t, &Model{}, phase) {
			if n != 0 {
				t.Errorf("inert %s=%g", key, n)
			}
		}
	}
}

func v41MHCProjVariant(t *testing.T, dtype string, transposed, reduced bool, layers int) *Model {
	t.Helper()
	m := v41RawFullFlattenedMHC(t)
	if reduced {
		m = v41ReducedModelLayers(t, layers)
		if m.Cfg.DeepSeekV41 != nil {
			cfg := *m.Cfg.DeepSeekV41
			cfg.EngramLayerIDs = nil
			m.Cfg.DeepSeekV41 = &cfg
		}
	} else if layers == 2 {
		m.Cfg.NumLayers = layers
		cfg := *m.Cfg.DeepSeekV41
		cfg.CompressRatios = []int{0, 0}
		cfg.EngramLayerIDs = nil
		m.Cfg.DeepSeekV41 = &cfg
		originalNames := make([]string, 0, len(m.manifest))
		for name := range m.manifest {
			originalNames = append(originalNames, name)
		}
		prefix := layerName(0, "")
		for _, name := range originalNames {
			if strings.HasPrefix(name, prefix) {
				m.manifest[layerName(1, strings.TrimPrefix(name, prefix))] = m.manifest[name]
			}
		}
	}
	in := 4 * m.Cfg.HiddenSize
	if reduced {
		in = m.Cfg.HiddenSize
	}
	for layer := 0; layer < layers; layer++ {
		name := layerName(layer, "mhc.mixes.weight")
		meta := m.manifest[name]
		if dtype == "Q2_K" {
			if m.kqw == nil {
				m.kqw = map[string]*kQuantTensor{}
			}
			m.kqw[name] = q2kFixtureTensor(24, in, uint64(136681020+layer))
			delete(m.manifest, name)
			continue
		}
		logical := make([]float32, 24*in)
		for i := range logical {
			logical[i] = float32((i*(layer+3)+7)%29-14) * .025
		}
		stored := logical
		meta.Shape = []int{24, in}
		if transposed {
			stored = v41TransposeMixBlock(logical, in)
			meta.Shape = []int{in, 24}
		}
		meta.Offset = len(m.raw)
		for _, v := range stored {
			var bytes [4]byte
			binary.LittleEndian.PutUint32(bytes[:], math.Float32bits(v))
			m.raw = append(m.raw, bytes[:]...)
		}
		m.manifest[name] = meta
	}
	return m
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=6s lane=default
func TestV41MHCProjectionRawFullAndReduced(t *testing.T) {
	t.Parallel()
	for _, scenario := range []struct {
		name, dtype string
		reduced     bool
	}{{"full-f32", "F32", false}, {"full-q2", "Q2_K", false}, {"reduced-f32", "F32", true}} {
		t.Run(scenario.name, func(t *testing.T) {
			t.Parallel()
			m := v41MHCProjVariant(t, scenario.dtype, false, scenario.reduced, 2)
			control := v41MHCProjVariant(t, scenario.dtype, scenario.dtype == "F32" && !scenario.reduced, scenario.reduced, 2)
			if m.Cfg.DeepSeekV41 == nil || m.Cfg.DeepSeekV41.HCMult != 4 {
				t.Fatal("raw mHC fixture lost HC4 geometry")
			}
			hc := m.Cfg.DeepSeekV41.HCMult
			// Each stream has a pre/post coefficient and a residual coefficient for every stream pair.
			coefficientWidth := 2*hc + hc*hc
			in := 4 * m.Cfg.HiddenSize
			if scenario.reduced {
				in = m.Cfg.HiddenSize
			}
			b := newV41MHCProjBackend(in, false)
			s := v41EngProjSession(t, m, b)
			var host *Session
			if scenario.reduced {
				host = control.NewSession()
				t.Cleanup(host.Close)
			} else {
				host = v41EngProjSession(t, control, newV41MHCProjBackend(in, scenario.dtype == "Q2_K"))
			}
			weights := make([][]float32, 2)
			for layer := range weights {
				var ok bool
				weights[layer], ok = m.residentF32Mat(layerName(layer, "mhc.mixes.weight"))
				if !ok {
					t.Fatal("independent raw weights unavailable")
				}
			}
			for index, ids := range [][]int{{1, 2, 3}, {4}, {5, 6}} {
				phase := "prefill"
				if index == 1 {
					phase = "decode"
				}
				before := v41MHCProjPhase(t, m, phase)
				from := len(b.operations)
				var got, want []float32
				if index == 1 {
					got, want = s.Step(ids[0]), host.Step(ids[0])
				} else {
					got, want = s.Prefill(ids), host.Prefill(ids)
				}
				v41GroupedParity(t, got, want, 1e-4)
				delta := v41DenseTestDelta(v41MHCProjPhase(t, m, phase), before)
				if delta["mhc_projection_device_rows"] != float64(2*len(ids)) || delta["mhc_projection_host_rows"] != 0 || delta["mhc_projection_host_weight_f32_bytes"] != 0 {
					t.Errorf("raw route ledger=%v", delta)
				}
				if len(b.operations)-from != 2*len(ids) {
					t.Fatalf("raw operations=%d want %d", len(b.operations)-from, 2*len(ids))
				}
				for j, op := range b.operations[from:] {
					layer := j / len(ids)
					if index > 0 {
						layer = j % 2
					}
					if len(op.activation) != in || len(op.result) != coefficientWidth {
						t.Fatalf("observed raw payload width=%d/%d want %d/%d", len(op.activation), len(op.result), in, coefficientWidth)
					}
					var ss, signal float64
					for _, v := range op.activation {
						ss += float64(v) * float64(v)
					}
					for out := 0; out < 24; out++ {
						var dot float64
						for cell, v := range op.activation {
							dot += float64(weights[layer][out*in+cell]) * float64(v)
						}
						if math.Abs(float64(op.result[out])-dot) > 1e-4*math.Max(1, math.Abs(dot)) {
							t.Fatalf("raw-before-RMS output=%d got=%g want=%g", out, op.result[out], dot)
						}
						signal = math.Max(signal, math.Abs(dot))
					}
					if signal == 0 {
						t.Fatal("raw scalar oracle vacuous")
					}
					if scenario.reduced {
						if math.Abs(ss/float64(in)-1) > 1e-3 {
							t.Errorf("reduced input must be normalized H row, RMS squared=%g", ss/float64(in))
						}
					} else if index == 0 && layer == 0 {
						embed := cpuOracleTensor(t, m, "model.embed_tokens.weight")
						for cell := 0; cell < m.Cfg.HiddenSize; cell++ {
							if op.activation[cell] != embed[ids[j%len(ids)]*m.Cfg.HiddenSize+cell] {
								t.Fatal("full input normalized before raw projection")
							}
						}
						if math.Abs(ss/float64(in)-1) < .01 {
							t.Fatal("wrong prenormalized projection control is vacuous")
						}
					} else if layer == 1 {
						for h := 1; h < 4; h++ {
							same := true
							nonzero := false
							for cell := 0; cell < m.Cfg.HiddenSize; cell++ {
								v := op.activation[h*m.Cfg.HiddenSize+cell]
								same = same && v == op.activation[cell]
								nonzero = nonzero || v != 0
							}
							if same || !nonzero {
								t.Fatalf("layer two lost independent nonzero stream %d", h)
							}
						}
					}
				}
			}
		})
	}
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=6s lane=default
func TestV41MHCProjectionLateSelectedFailure(t *testing.T) {
	t.Parallel()
	for _, site := range []string{"matmul", "read", "length", "finite", "unknown"} {
		t.Run(site, func(t *testing.T) {
			t.Parallel()
			m := v41MHCProjVariant(t, "Q2_K", false, false, 2)
			b := newV41MHCProjBackend(4*m.Cfg.HiddenSize, false)
			api := &v41MHCClosedRecorder{v41MHCProjBackend: b}
			s := v41EngProjSession(t, m, api)
			s.Prefill([]int{1, 2, 3})
			before := captureV41ForwardSnapshot(s.v41Forward)
			ledgerBefore := v41MHCProjPhase(t, m, "decode")
			b.live = map[compute.Buffer]bool{}
			b.fault, b.site = true, site
			b.cause = &compute.BackendError{Backend: "cpu-ref-recording", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
			unknown := struct{ Number int }{1366810}
			if site == "unknown" {
				b.site, b.cause = "matmul", unknown
			}
			var closedFailure *BackendForwardOperationError
			var recovered any
			func() { defer func() { recovered = recover() }(); s.Step(4) }()
			if !b.failed {
				t.Fatal("late layer two selected mHC fault not reached")
			}
			if site == "unknown" {
				if recovered != unknown {
					t.Error("selected unknown panic identity changed")
				}
			} else {
				err, ok := recovered.(error)
				if !ok || !errors.As(err, &closedFailure) || !s.BackendSessionClosed() {
					t.Error("selected typed failure did not retain a closed Session/error identity")
				}

				var selected *BackendForwardOperationError
				if !ok || !errors.As(err, &selected) || !errors.Is(err, ErrV41ForwardStage) {
					t.Errorf("selected mHC failure lacks typed identity/cause: %T", recovered)
				}
				if (site == "matmul" || site == "read") && !errors.Is(err, b.cause.(error)) {
					t.Error("selected backend cause identity changed")
				}
			}
			if !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
				t.Error("late selected failure changed retained state")
			}
			if len(b.live) != 0 {
				t.Errorf("selected fault leaked %d transient buffers", len(b.live))
			}
			delta := v41DenseTestDelta(v41MHCProjPhase(t, m, "decode"), ledgerBefore)
			if b.faultAttempts != 2 || delta["mhc_projection_device_calls"] != 2 || delta["mhc_projection_device_rows"] != 1 || delta["mhc_projection_host_calls"] != 0 || delta["mhc_projection_host_rows"] != 0 || delta["mhc_projection_host_weight_f32_bytes"] != 0 || delta["mhc_projection_matmul_calls"] != 2 {
				t.Errorf("selected attempt/completion/no-retry ledger=%v attempts=%d", delta, b.faultAttempts)
			}

			if site != "unknown" {
				callsBefore := api.calls
				stateBefore := captureV41ForwardSnapshot(s.v41Forward)
				dataBefore := v41MHCProjPhase(t, m, "decode")
				var repeated any
				func() { defer func() { repeated = recover() }(); s.Step(4) }()
				var closed *BackendForwardOperationError
				err, ok := repeated.(error)
				if !ok || !errors.As(err, &closed) || closed != closedFailure || !errors.Is(err, ErrV41ForwardStage) {
					t.Error("closed public entry changed selected failure identity/cause")
				}
				if api.calls != callsBefore {
					t.Error("closed public entry invoked backend API")
				}
				if !reflect.DeepEqual(stateBefore, captureV41ForwardSnapshot(s.v41Forward)) || !reflect.DeepEqual(dataBefore, v41MHCProjPhase(t, m, "decode")) {
					t.Error("closed public entry changed retained state or projection attribution")
				}
				if len(b.live) != 0 {
					t.Error("closed public entry changed transient buffer cleanup")
				}
				return
			}
			b.fault, b.site, b.cause = false, "", nil
			control := v41MHCProjVariant(t, "Q2_K", false, false, 2)
			host := v41EngProjSession(t, control, newV41MHCProjBackend(4*control.Cfg.HiddenSize, true))
			host.Prefill([]int{1, 2, 3})
			v41GroupedParity(t, s.Step(4), host.Step(4), 1e-4)
		})
	}
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=3s lane=default
func TestV41MHCProjectionRestoreForkOwnership(t *testing.T) {
	t.Parallel()
	m := v41MHCProjVariant(t, "Q2_K", false, false, 2)
	b := newV41MHCProjBackend(4*m.Cfg.HiddenSize, false)
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
			t.Error("source close freed immutable mHC weight")
		}
	}
	for index, branch := range []*Session{left, right} {
		control := v41MHCProjVariant(t, "Q2_K", false, false, 2)
		host := v41EngProjSession(t, control, newV41MHCProjBackend(4*control.Cfg.HiddenSize, true))
		host.Prefill([]int{1, 2, 3})
		before := b.attempts
		v41GroupedParity(t, branch.Step(4+index), host.Step(4+index), 1e-4)
		if b.attempts-before != 2 {
			t.Error("restored target did not select both mHC layers")
		}
	}
	left.Close()
	for _, frees := range b.weightFrees {
		if frees != 0 {
			t.Error("fork close freed sibling immutable mHC weight")
		}
	}
	if b.stages != 2 {
		t.Errorf("shared immutable mHC staging=%d want 2", b.stages)
	}
	right.Close()
	if err := m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	if len(b.weights) != 2 {
		t.Errorf("model-owned mHC residents=%d want 2", len(b.weights))
	}
	for weight := range b.weights {
		if b.weightFrees[weight] != 1 {
			t.Errorf("immutable mHC freed %d times want 1", b.weightFrees[weight])
		}
	}
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=2s lane=default
func TestV41MHCProjectionInvalidSourceBeforeAnyMHCAPI(t *testing.T) {
	t.Parallel()
	for _, scenario := range []string{"same-count-wrong-shape", "q2-inf", "q2-nan"} {
		t.Run(scenario, func(t *testing.T) {
			t.Parallel()
			m := v41MHCProjFixture(t)
			b := newV41MHCProjBackend(4*m.Cfg.HiddenSize, false)
			s := v41EngProjSession(t, m, b)
			name := layerName(0, "mhc.mixes.weight")
			if scenario == "same-count-wrong-shape" {
				m.kqw[name] = q2kFixtureTensor(12, 8*m.Cfg.HiddenSize, 136681099)
			} else {
				bits := uint16(0x7c00)
				if scenario == "q2-nan" {
					bits = 0x7e00
				}
				binary.LittleEndian.PutUint16(m.kqw[name].raw[q2kBlockBytes-4:], bits)
				values, ok := m.residentF32Mat(name)
				if !ok || len(values) != 24*b.in {
					t.Fatal("finite corruption changed structural payload length")
				}
				invalid := false
				for _, v := range values {
					invalid = invalid || math.IsNaN(float64(v)) || math.IsInf(float64(v), 0)
				}
				if !invalid {
					t.Fatal("packed finite rejection control vacuous")
				}
			}
			err := panicAsError(func() { s.Prefill([]int{1, 2, 3}) })
			if !errors.Is(err, ErrV41ForwardStage) {
				t.Errorf("invalid mHC source lacks closed forward refusal: %T", err)
			}
			if b.stages != 0 || b.attempts != 0 {
				t.Errorf("invalid source reached mHC device API stages=%d matmuls=%d", b.stages, b.attempts)
			}
			phase := v41MHCProjPhase(t, m, "prefill")
			if phase["mhc_projection_host_calls"] != 0 || phase["mhc_projection_host_weight_f32_bytes"] != 0 || phase["mhc_projection_host_rows"] != 0 || phase["mhc_projection_device_rows"] != 0 {
				t.Errorf("invalid selected source materialized/retried/completed: %v", phase)
			}
			if s.v41Forward != nil && len(s.v41Forward.history) != 0 {
				t.Error("invalid mHC source committed prefix history")
			}
		})
	}
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=2s lane=default
func TestV41MHCProjectionHostReturnedWeightSpanOnFailure(t *testing.T) {
	t.Parallel()
	m := v41MHCProjVariant(t, "F32", true, false, 2)
	b := newV41MHCProjBackend(4*m.Cfg.HiddenSize, false)
	s := v41EngProjSession(t, m, b)
	meta := m.manifest[layerName(1, "mhc.mixes.weight")]
	binary.LittleEndian.PutUint32(m.raw[meta.Offset:], math.Float32bits(float32(math.NaN())))
	weights, ok := m.residentF32Mat(layerName(1, "mhc.mixes.weight"))
	if !ok || len(weights) != 24*b.in {
		t.Fatal("host failure source changed returned whole-weight span")
	}
	streams := make([][]float32, 4)
	for h := range streams {
		streams[h] = make([]float32, m.Cfg.HiddenSize)
		for cell := range streams[h] {
			streams[h][cell] = float32((h+1)*(cell%7+1)) / 9
		}
	}
	coefficients, projectErr := v41MHCProjectFull(weights, streams, m.Cfg.HiddenSize, float32(m.Cfg.RMSNormEps), true)
	// The input streams require pre/post vectors and a square residual-mixing matrix.
	if projectErr != nil || len(coefficients) != 2*len(streams)+len(streams)*len(streams) {
		t.Fatal("parent host raw projection did not complete its coefficient vector")
	}
	nonfinite := false
	for _, value := range coefficients {
		nonfinite = nonfinite || math.IsNaN(float64(value)) || math.IsInf(float64(value), 0)
	}
	if !nonfinite {
		t.Fatal("host downstream nonfinite rejection control is vacuous")
	}
	if _, splitErr := v41MHCSplit(coefficients, cpuOracleTensor(t, m, layerName(1, "mhc.scale")), cpuOracleTensor(t, m, layerName(1, "mhc.base")), 4, m.Cfg.DeepSeekV41.HCSinkhornIters, float32(m.Cfg.DeepSeekV41.HCEps)); splitErr == nil {
		t.Fatal("parent mHC split admitted nonfinite raw coefficients")
	}
	err := panicAsError(func() { s.Prefill([]int{1, 2, 3}) })
	var stage *V41ForwardError
	if !errors.As(err, &stage) || stage.Stage != v41StageMHC || stage.Layer != 1 || stage.Err == nil {
		t.Errorf("host downstream refusal lacks existing typed MHC layer/cause contract: %T", err)
	}
	phase := v41MHCProjPhase(t, m, "prefill")
	if phase["mhc_projection_host_weight_f32_bytes"] != float64(2*4*24*b.in) {
		t.Errorf("actual returned host span, including failing layer,=%g want %d", phase["mhc_projection_host_weight_f32_bytes"], 2*4*24*b.in)
	}
	if phase["mhc_projection_host_calls"] != 4 || phase["mhc_projection_host_rows"] != 4 || phase["mhc_projection_device_rows"] != 0 || b.stages != 0 || b.attempts != 0 {
		t.Errorf("transposed host completion/device isolation=%v stages=%d attempts=%d", phase, b.stages, b.attempts)
	}
	if s.v41Forward != nil && len(s.v41Forward.history) != 0 {
		t.Error("host projection failure committed prefix history")
	}
}

type v41MHCClosedRecorder struct {
	*v41MHCProjBackend
	calls [4]int
}

func (b *v41MHCClosedRecorder) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	b.calls[0]++
	return b.v41MHCProjBackend.Upload(x, dt)
}
func (b *v41MHCClosedRecorder) MatMul(w, x compute.Tensor) compute.Tensor {
	b.calls[1]++
	return b.v41MHCProjBackend.MatMul(w, x)
}
func (b *v41MHCClosedRecorder) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	b.calls[1]++
	return b.v41MHCProjBackend.BatchedMatMul(w, x, rows)
}
func (b *v41MHCClosedRecorder) Read(x compute.Tensor) []float32 {
	b.calls[2]++
	return b.v41MHCProjBackend.Read(x)
}
func (b *v41MHCClosedRecorder) Free(x compute.Tensor) { b.calls[3]++; b.v41MHCProjBackend.Free(x) }
