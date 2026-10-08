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

var v41DenseTestLeaves = []string{"attn.wq_a.weight", "attn.wq_b.weight", "attn.wkv.weight", "ffn.gate.weight", "ffn.shared_experts.w1.weight", "ffn.shared_experts.w3.weight", "ffn.shared_experts.w2.weight"}
var v41DenseTestFields = []string{"dense_projection_device_calls", "dense_projection_host_calls", "dense_projection_device_rows", "dense_projection_host_rows", "dense_projection_activation_upload_bytes", "dense_projection_readback_bytes", "dense_projection_nanos"}

type v41DenseTestOp struct {
	weight, input, output compute.Buffer
	rows, in, out         int
	batch                 bool
}

// The cpu-ref recorder witnesses software dispatch through the actual session,
// without claiming a physical device. Dtype promises govern admission explicitly.
type v41DenseTestBackend struct {
	compute.Backend
	ops            []v41DenseTestOp
	uploads, reads map[compute.Buffer]int
	live           map[compute.Buffer]bool
	deny           bool
	fail           any
	failSite       string
	beforeFailure  func()
	malformed      bool
}

func newV41DenseTestBackend() *v41DenseTestBackend {
	return &v41DenseTestBackend{Backend: compute.Default(), uploads: map[compute.Buffer]int{}, reads: map[compute.Buffer]int{}, live: map[compute.Buffer]bool{}}
}
func (b *v41DenseTestBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory = true
	c.UploadDtype = true
	return c
}
func (b *v41DenseTestBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return !b.deny && compute.BackendSupportsDeviceWeightDtype(b.Backend, dt)
}
func (b *v41DenseTestBackend) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	if b.fail != nil && b.failSite == "upload" && len(x.Shape) == 1 {
		panic(b.fail)
	}
	out := b.Backend.Upload(x, dt)
	if dt == compute.F32 && len(out.Shape) == 1 {
		b.live[out.Buf()] = true
	}
	if dt == compute.F32 {
		n := 1
		for _, width := range out.Shape {
			n *= width
		}
		b.uploads[out.Buf()] = 4 * n
	}
	return out
}
func (b *v41DenseTestBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	if b.fail != nil && b.failSite == "matmul" {
		if b.beforeFailure != nil {
			b.beforeFailure()
		}
		panic(b.fail)
	}
	out := b.Backend.MatMul(w, x)
	b.ops = append(b.ops, v41DenseTestOp{w.Buf(), x.Buf(), out.Buf(), 1, w.Shape[1], w.Shape[0], false})
	b.live[x.Buf()], b.live[out.Buf()] = true, true
	return out
}
func (b *v41DenseTestBackend) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	if b.fail != nil && b.failSite == "matmul" {
		if b.beforeFailure != nil {
			b.beforeFailure()
		}
		panic(b.fail)
	}
	out := b.Backend.BatchedMatMul(w, x, rows)
	b.ops = append(b.ops, v41DenseTestOp{w.Buf(), x.Buf(), out.Buf(), rows, w.Shape[1], w.Shape[0], true})
	b.live[x.Buf()], b.live[out.Buf()] = true, true
	return out
}
func (b *v41DenseTestBackend) Read(x compute.Tensor) []float32 {
	if b.fail != nil && b.failSite == "read" {
		panic(b.fail)
	}
	values := b.Backend.Read(x)
	if b.malformed {
		values = append(append([]float32(nil), values...), 0)
	}
	b.reads[x.Buf()] = 4 * len(values)
	return values
}
func (b *v41DenseTestBackend) Free(x compute.Tensor) { delete(b.live, x.Buf()); b.Backend.Free(x) }

func v41DenseTestPhase(t *testing.T, m *Model, phase string) map[string]float64 {
	t.Helper()
	raw, err := json.Marshal(m.V41ExpertFaultAttribution())
	if err != nil {
		t.Fatal(err)
	}
	var phases map[string]map[string]json.RawMessage
	if err := json.Unmarshal(raw, &phases); err != nil {
		t.Fatal(err)
	}
	out := map[string]float64{}
	for _, key := range v41DenseTestFields {
		value, ok := phases[phase][key]
		if !ok {
			t.Errorf("default %s missing %s", phase, key)
			continue
		}
		var n float64
		if err := json.Unmarshal(value, &n); err != nil {
			t.Fatal(err)
		}
		out[key] = n
	}
	return out
}
func v41DenseTestDelta(after, before map[string]float64) map[string]float64 {
	out := map[string]float64{}
	for key, value := range after {
		out[key] = value - before[key]
	}
	return out
}
func v41DenseTestSession(t *testing.T, m *Model, b compute.Backend) *Session {
	t.Helper()
	s, err := m.NewBackendSessionChecked(b)
	if err != nil {
		t.Fatal(err)
	}
	s.Q4K = m.q4kw != nil
	s.Quant = m.q8w != nil
	t.Cleanup(s.Close)
	return s
}
func v41DenseTestQ4Raw(out, in int) []byte {
	raw := makeTestQ4KRaw(out, in)
	for offset := 0; offset < len(raw); offset += q4kBlockBytes {
		raw[offset], raw[offset+1] = 0, 0x10
	}
	return raw
}
func v41DenseTestFixture(t *testing.T, role, packed bool) *Model {
	t.Helper()
	var m *Model
	if role || packed {
		m = v41IncrementalExpertFixture(t, role, false)
	} else {
		m = v41IncrementalPlainModel(t, 1)
	}
	if !packed {
		return m
	}
	if m.q8w == nil {
		m.q8w = map[string]*q8Tensor{}
	}
	if m.q4kw == nil {
		m.q4kw = map[string]*q4kTensor{}
	}
	for _, leaf := range v41DenseTestLeaves {
		name := layerName(0, leaf)
		shape := v41ProjectionShape(t, m, leaf)
		out, in := shape[0], shape[1]
		if leaf == "attn.wq_b.weight" {
			continue
		}
		switch leaf {
		case "attn.wq_a.weight", "ffn.shared_experts.w2.weight":
			m.kqw[name] = q2kFixtureTensor(out, in, 0x61366)
		case "ffn.gate.weight", "ffn.shared_experts.w3.weight":
			m.q4kw[name] = quantizeQ4KFromRaw(v41DenseTestQ4Raw(out, in), out, in)
		default:
			w, ok := m.residentF32Mat(name)
			if !ok {
				t.Fatal("missing Q8 oracle weight")
			}
			m.q8w[name] = quantizeQ8(w, out, in)
		}
		delete(m.manifest, name)
	}
	return m
}

// cpu-ref Q8 MatMul quantizes activations like the host Q8 oracle. Its Q4/K
// branches consume F32 activations, so dequantize only those oracle weights.
func v41DenseTestPackedBackendOracle(t *testing.T, m *Model) {
	t.Helper()
	for _, leaf := range v41DenseTestLeaves {
		name := layerName(0, leaf)
		shape := v41ProjectionShape(t, m, leaf)
		if m.has(name) || m.q8w[name] != nil {
			continue
		}
		weights, ok := m.residentF32Mat(name)
		if !ok {
			t.Fatal("missing packed oracle projection")
		}
		cursor := 0
		manifest, raw := synthBuildRaw([]synthTensor{{name, shape}}, func(_ string, _ func() float32) float32 { value := weights[cursor]; cursor++; return value })
		meta := manifest[name]
		meta.Offset += len(m.raw)
		m.raw = append(m.raw, raw...)
		m.manifest[name] = meta
	}
}

func v41DenseTestOps(s *Session, b *v41DenseTestBackend, from int) (map[string][]v41DenseTestOp, int, int) {
	named := map[string][]v41DenseTestOp{}
	upload, read := 0, 0
	for _, op := range b.ops[from:] {
		for _, leaf := range v41DenseTestLeaves {
			name := layerName(0, leaf)
			for key, weight := range s.halW {
				if strings.HasSuffix(key, name) && weight.Buf() == op.weight {
					named[leaf] = append(named[leaf], op)
					upload += b.uploads[op.input]
					read += b.reads[op.output]
				}
			}
		}
	}
	return named, upload, read
}
func v41DenseTestParity(t *testing.T, got, want []float32) {
	t.Helper()
	assertV41LogitsClose(t, got, want, "dense device bridge vs host oracle")
	var largest, diff float64
	for i, value := range got {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatal("dense bridge logits nonfinite")
		}
		largest = max(largest, math.Abs(float64(value)))
		diff = max(diff, math.Abs(float64(value-want[i])))
	}
	if largest == 0 {
		t.Fatal("dense oracle is vacuous")
	}
	t.Logf("dense_logits width=%d maximum_abs=%g maximum_diff=%g", len(got), largest, diff)
}

// fak-test:runtime slow est=30s lane=default
func TestV41DenseProjectionActualSessionRoutes(t *testing.T) {
	t.Parallel()
	for _, scenario := range []struct {
		name               string
		role, packed, deny bool
	}{{"plain", false, false, false}, {"role", true, false, false}, {"packed", false, true, false}, {"unsupported", false, false, true}} {
		t.Run(scenario.name, func(t *testing.T) {
			t.Parallel()
			m := v41DenseTestFixture(t, scenario.role, scenario.packed)
			oracle := v41DenseTestFixture(t, scenario.role, scenario.packed)
			if scenario.packed {
				v41DenseTestPackedBackendOracle(t, oracle)
			}
			b := newV41DenseTestBackend()
			b.deny = scenario.deny
			s := v41DenseTestSession(t, m, b)
			history := []int{1, 2, 3}
			for index, ids := range [][]int{{1, 2, 3}, {4}, {5, 6}} {
				phase := "prefill"
				if index == 1 {
					phase = "decode"
				}
				from := len(b.ops)
				before := v41DenseTestPhase(t, m, phase)
				var got []float32
				if index == 1 {
					got = s.Step(ids[0])
				} else {
					got = s.Prefill(ids)
				}
				if index > 0 {
					history = append(history, ids...)
				}
				v41DenseTestParity(t, got, lastLogits(oracle.Forward(history)))
				named, upload, read := v41DenseTestOps(s, b, from)
				delta := v41DenseTestDelta(v41DenseTestPhase(t, m, phase), before)
				rows, calls := 0, 0
				for _, leaf := range v41DenseTestLeaves {
					leafRows := 0
					for _, op := range named[leaf] {
						leafRows += op.rows
						calls++
					}
					rows += leafRows
					want := len(ids)
					if scenario.deny {
						want = 0
					}
					if leafRows != want {
						t.Errorf("actual session leaf %s device rows=%d want %d", leaf, leafRows, want)
					}
					if index == 0 && !scenario.deny && (leaf == "attn.wq_a.weight" || leaf == "attn.wq_b.weight") {
						if len(named[leaf]) != 1 || !named[leaf][0].batch {
							t.Errorf("cold multirow query %s did not use one bounded device panel", leaf)
						}
					}
				}
				wantHostRows, wantHostCalls := 0, 0
				if scenario.deny {
					wantHostRows = 7 * len(ids)
					wantHostCalls = 7 * len(ids)
					if index == 0 {
						wantHostCalls = 2 + 5*len(ids)
					}
				}
				if delta["dense_projection_device_calls"] != float64(calls) || delta["dense_projection_device_rows"] != float64(rows) || delta["dense_projection_host_rows"] != float64(wantHostRows) || delta["dense_projection_host_calls"] != float64(wantHostCalls) {
					t.Errorf("dense selected ledger=%v ops=%d rows=%d host=%d", delta, calls, rows, wantHostRows)
				}
				if delta["dense_projection_activation_upload_bytes"] != float64(upload) || delta["dense_projection_readback_bytes"] != float64(read) || delta["dense_projection_nanos"] <= 0 {
					t.Errorf("dense transfer/time ledger=%v recorded upload=%d read=%d", delta, upload, read)
				}
				raw, err := json.Marshal(delta)
				if err != nil {
					t.Fatal(err)
				}
				t.Logf("dense_phase=%s tokens=%d scenario=%s delta=%s", phase, len(ids), scenario.name, raw)
			}
			if scenario.packed {
				for _, leaf := range v41DenseTestLeaves {
					if leaf != "attn.wq_b.weight" && m.has(layerName(0, leaf)) {
						t.Error("compressed dense projection expanded into F32 manifest")
					}
				}
			}
		})
	}
}

// fak-test:runtime fast est=500ms lane=default
func TestV41DenseProjectionSelectedFailureRollbackAndRetry(t *testing.T) {
	t.Parallel()
	for _, site := range []string{"upload", "matmul", "read", "malformed"} {
		t.Run(site, func(t *testing.T) {
			t.Parallel()
			m := v41DenseTestFixture(t, false, false)
			b := newV41DenseTestBackend()
			s := v41DenseTestSession(t, m, b)
			s.Prefill([]int{1, 2, 3})
			b.live = map[compute.Buffer]bool{}
			before := captureV41ForwardSnapshot(s.v41Forward)
			phaseBefore := v41DenseTestPhase(t, m, "decode")
			if site == "malformed" {
				b.malformed = true
			} else {
				b.failSite = site
				b.fail = &compute.BackendError{Backend: "test-device", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
			}
			err := panicAsError(func() { s.Step(4) })
			var selected interface {
				SelectedProjectionOperation() bool
				Unwrap() error
			}
			if !errors.As(err, &selected) || !selected.SelectedProjectionOperation() || !errors.Is(err, ErrV41ForwardStage) {
				t.Errorf("selected dense failure lost closed marker/cause: %v", err)
			}
			if !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
				t.Error("failed dense projection changed retained state")
			}
			if s.BackendSessionClosed() {
				t.Error("retryable raw compute failure closed session")
			}
			if len(b.live) != 0 {
				t.Errorf("failed dense projection leaked %d transient buffers", len(b.live))
			}
			delta := v41DenseTestDelta(v41DenseTestPhase(t, m, "decode"), phaseBefore)
			if delta["dense_projection_device_calls"] != 1 || delta["dense_projection_device_rows"] != 0 || delta["dense_projection_host_calls"] != 0 || delta["dense_projection_host_rows"] != 0 || delta["dense_projection_nanos"] <= 0 {
				t.Errorf("failed projection attempt accounting=%v", delta)
			}
			expectedUpload := 4 * m.Cfg.HiddenSize
			if site == "upload" {
				expectedUpload = 0
			}
			if delta["dense_projection_activation_upload_bytes"] != float64(expectedUpload) {
				t.Error("failure lost actual successful upload bytes")
			}
			if site != "malformed" && delta["dense_projection_readback_bytes"] != 0 {
				t.Error("failed Read counted nonexistent returned bytes")
			}
			if site == "malformed" {
				shape := v41ProjectionShape(t, m, "attn.wq_a.weight")
				if delta["dense_projection_readback_bytes"] != float64(4*(shape[0]+1)) {
					t.Error("malformed successful Read bytes lost")
				}
			}
			b.fail, b.failSite, b.malformed = nil, "", false
			got := s.Step(4)
			oracle := v41DenseTestFixture(t, false, false)
			v41DenseTestParity(t, got, lastLogits(oracle.Forward([]int{1, 2, 3, 4})))
		})
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestV41DenseProjectionRestoreTargetOwnership(t *testing.T) {
	t.Parallel()
	m := v41DenseTestFixture(t, false, false)
	shared := newV41DenseTestBackend()
	source, target := v41DenseTestSession(t, m, shared), v41DenseTestSession(t, m, shared)
	source.Prefill([]int{1, 2, 3})
	snap, err := source.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer snap.Close()
	if err := snap.Restore(target); err != nil {
		t.Fatal(err)
	}
	source.Close()
	before := len(shared.ops)
	got := target.Step(4)
	named, _, _ := v41DenseTestOps(target, shared, before)
	if len(shared.ops)-before != 7 {
		t.Error("restored projection callback did not execute target session")
	}
	for _, leaf := range v41DenseTestLeaves {
		if len(named[leaf]) != 1 {
			t.Errorf("restored target cache did not register projection %s", leaf)
		}
	}
	oracle := v41DenseTestFixture(t, false, false)
	v41DenseTestParity(t, got, lastLogits(oracle.Forward([]int{1, 2, 3, 4})))
	target.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
	calls := len(shared.ops)
	err = panicAsError(func() { target.Step(5) })
	var closed *BackendForwardOperationError
	if !errors.As(err, &closed) || len(shared.ops) != calls {
		t.Error("strict DeviceOnly V4.1 guard changed")
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestV41DenseProjectionPrismLoRAOriginalInput(t *testing.T) {
	t.Parallel()
	fixture := func() *Model {
		m := v41DenseTestFixture(t, false, false)
		name := layerName(0, "attn.wq_a.weight")
		shape := v41ProjectionShape(t, m, "attn.wq_a.weight")
		signs := make([]int, shape[1])
		for i := range signs {
			signs[i] = 1
			if i%3 == 1 {
				signs[i] = -1
			}
		}
		if err := m.SetPrismHadamard(PrismHadamardSpec{BlockSize: 4, SignWidths: []int{shape[1]}, SignValues: signs, WeightNames: []string{name}}); err != nil {
			t.Fatal(err)
		}
		set := NewLoRASet()
		if err := set.Add(newTestAdapter("dense", name, shape[0], shape[1], 1, 8, 0x61368)); err != nil {
			t.Fatal(err)
		}
		set.Activate("dense")
		m.SetLoRA(set)
		return m
	}
	m, oracle := fixture(), fixture()
	b := newV41DenseTestBackend()
	s := v41DenseTestSession(t, m, b)
	got := s.Prefill([]int{1, 2, 3})
	want := lastLogits(oracle.Forward([]int{1, 2, 3}))
	v41DenseTestParity(t, got, want)
	base := v41DenseTestFixture(t, false, false).Forward([]int{1, 2, 3})
	if loraMaxAbsDiff(want, lastLogits(base)) < 1e-6 {
		t.Fatal("Prism/LoRA witness vacuous")
	}
	got = s.Step(4)
	v41DenseTestParity(t, got, lastLogits(oracle.Forward([]int{1, 2, 3, 4})))
}

// fak-test:runtime fast est=200ms lane=default
func TestV41DenseProjectionClosedAndUnknownPanicPriority(t *testing.T) {
	t.Parallel()
	for _, closed := range []bool{false, true} {
		t.Run(itoa(boolToIntV41Expert(closed)), func(t *testing.T) {
			t.Parallel()
			m := v41DenseTestFixture(t, false, false)
			b := newV41DenseTestBackend()
			s := v41DenseTestSession(t, m, b)
			s.Prefill([]int{1, 2, 3})
			unknown := struct{ value int }{17}
			var expected any = unknown
			if closed {
				expected = &BackendForwardOperationError{Backend: "test-device", Cause: ErrV41ForwardStage}
			}
			if closed {
				b.beforeFailure = func() { s.halFailure = expected.(*BackendForwardOperationError); s.Close() }
			}
			b.failSite, b.fail = "matmul", expected
			var recovered any
			func() { defer func() { recovered = recover() }(); s.Step(4) }()
			if recovered != expected {
				t.Errorf("selected projection changed closed/unknown panic identity: %T", recovered)
			}
			if closed && !s.BackendSessionClosed() {
				t.Error("closed operation failure did not retain teardown priority")
			}
		})
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestV41DenseProjectionHostPrismPanelScalarOracle(t *testing.T) {
	t.Parallel()
	m := v41DenseTestFixture(t, false, false)
	const leaf = "attn.wq_a.weight"
	name := layerName(0, leaf)
	shape := v41ProjectionShape(t, m, leaf)
	out, in := shape[0], shape[1]
	signs := make([]int, in)
	for i := range signs {
		signs[i] = 1
		if i%3 == 1 {
			signs[i] = -1
		}
	}
	if err := m.SetPrismHadamard(PrismHadamardSpec{BlockSize: 4, SignWidths: []int{in}, SignValues: signs, WeightNames: []string{name}}); err != nil {
		t.Fatal(err)
	}
	const rows = 3
	panel := v41SyntheticPanel(rows, in)
	original := append([]float32(nil), panel...)
	want := make([]float32, rows*out)
	for row := 0; row < rows; row++ {
		copy(want[row*out:(row+1)*out], m.residentMatRows(name, panel[row*in:(row+1)*in], out, in))
	}
	got, err := m.v41ProjPanel(0, leaf, panel, out, in, rows)
	if err != nil {
		t.Fatal(err)
	}
	assertV41LogitsClose(t, got, want, "host Prism panel vs independent scalar transformed rows")
	unrotated := v41DenseTestFixture(t, false, false)
	raw, err := unrotated.v41ProjPanel(0, leaf, panel, out, in, rows)
	if err != nil {
		t.Fatal(err)
	}
	if loraMaxAbsDiff(want, raw) < 1e-4 {
		t.Fatal("Prism-only panel saturation fixture vacuous")
	}
	if !reflect.DeepEqual(panel, original) {
		t.Error("host Prism panel mutated original caller input")
	}
}
