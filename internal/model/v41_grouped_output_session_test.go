package model

import (
	"encoding/binary"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func v41GroupedCheckPhase(t *testing.T, m *Model, b *v41GroupedBackend, from, attempts, rows int, phase string, before map[string]float64, host bool) {
	t.Helper()
	delta := v41DenseTestDelta(v41GroupedPhase(t, m, phase), before)
	completed, upload, read := 0, 0, 0
	for _, op := range b.grouped[from:] {
		if op.leaf == "b" {
			completed += op.rows
		}
		upload += op.upload
		read += op.read
	}
	if host {
		a, _ := m.residentF32Mat(layerName(0, "attn.wo_a.weight"))
		bb, _ := m.residentF32Mat(layerName(0, "attn.wo_b.weight"))
		if len(b.grouped) != from || delta["grouped_output_device_calls"] != 0 || delta["grouped_output_matmul_calls"] != 0 || delta["grouped_output_device_rows"] != 0 {
			t.Errorf("declined whole operation touched device: %v", delta)
		}
		materializations := 1
		if rows == 2 {
			materializations = rows
		}
		wantBytes := 4 * (len(a) + len(bb)) * materializations
		if delta["grouped_output_host_rows"] != float64(rows) || delta["grouped_output_host_calls"] != float64(rows) || delta["grouped_output_host_weight_f32_bytes"] != float64(wantBytes) {
			t.Errorf("host grouped materialization accounting=%v want bytes=%d rows=%d", delta, wantBytes, rows)
		}
	} else {
		if completed != rows {
			t.Errorf("actual default Session grouped output completed %d rows want %d", completed, rows)
		}
		if delta["grouped_output_device_rows"] != float64(rows) || delta["grouped_output_device_calls"] != float64(rows) || delta["grouped_output_host_calls"] != 0 || delta["grouped_output_host_rows"] != 0 || delta["grouped_output_host_weight_f32_bytes"] != 0 {
			t.Errorf("grouped selected ledger=%v", delta)
		}
		if b.matmulAttempts-attempts != rows*(m.Cfg.OGroups+1) {
			t.Error("grouped output did not contract each isolated group then the joined output")
		}
		if delta["grouped_output_matmul_calls"] != float64(b.matmulAttempts-attempts) || delta["grouped_output_activation_upload_bytes"] != float64(upload) || delta["grouped_output_readback_bytes"] != float64(read) {
			t.Errorf("grouped actual API/transfer accounting=%v actual matmul=%d upload=%d read=%d", delta, b.matmulAttempts-attempts, upload, read)
		}
	}
	if delta["grouped_output_nanos"] <= 0 {
		t.Error("grouped selected/host operation has no elapsed observation")
	}
}

// fak-test:justify why=integration when=changed:internal/model/**
// fak-test:runtime slow est=40s lane=default
func TestV41GroupedOutputActualSessionRoutes(t *testing.T) {
	t.Parallel()
	for _, scenario := range []struct {
		name, dtype      string
		role, flat, deny bool
	}{
		{"grouped-f32", "F32", false, false, false},
		{"legacy-flat-f32", "F32", false, true, false},
		{"q8", "Q8_0", false, false, false},
		{"q4", "Q4_K", false, false, false},
		{"q2", "Q2_K", false, false, false},
		{"role-mhc", "F32", true, false, false},
		{"declined", "F32", false, false, true},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			t.Parallel()
			m := v41GroupedFixture(t, scenario.dtype, scenario.role, scenario.flat)
			oracle := v41GroupedFixture(t, scenario.dtype, scenario.role, scenario.flat)
			b := newV41GroupedBackend(t, m)
			b.deny = scenario.deny
			s := v41DenseTestSession(t, m, b)
			history := []int{1, 2, 3}
			batches := [][]int{{1, 2, 3}, {4}, {5, 6}}
			if scenario.role {
				batches = [][]int{{1, 2, 3}, {4}, {5}, {6, 7}}
			}
			for index, ids := range batches {
				decode := index > 0 && len(ids) == 1
				phase := "prefill"
				if decode {
					phase = "decode"
				}
				before := v41GroupedPhase(t, m, phase)
				denseBefore := v41DenseTestPhase(t, m, phase)
				from, attempts := len(b.grouped), b.matmulAttempts
				var got []float32
				if decode {
					got = s.Step(ids[0])
				} else {
					got = s.Prefill(ids)
				}
				if index > 0 {
					history = append(history, ids...)
				}
				tolerance := 1e-5
				if scenario.dtype == "Q8_0" {
					tolerance = 0.01
				}
				v41GroupedParity(t, got, lastLogits(oracle.Forward(history)), tolerance)
				projectedRows := len(ids)
				if scenario.role && index > 0 {
					ratio := m.Cfg.DeepSeekV41.CompressRatios[0]
					priorTokens := len(history) - len(ids)
					projectedRows = 0
					for row := range ids {
						if priorTokens+row >= ratio-1 {
							projectedRows++
						}
					}
				}
				v41GroupedCheckPhase(t, m, b, from, attempts, projectedRows, phase, before, scenario.deny)
				named, _, _ := v41DenseTestOps(s, b.v41DenseTestBackend, 0)
				denseDelta := v41DenseTestDelta(v41DenseTestPhase(t, m, phase), denseBefore)
				wantDenseRows := 7 * len(ids)
				key := "dense_projection_device_rows"
				if scenario.deny {
					key = "dense_projection_host_rows"
				}
				if denseDelta[key] != float64(wantDenseRows) {
					t.Errorf("grouped bridge changed existing dense ledger: %v named_leaves=%d", denseDelta, len(named))
				}
			}
			if scenario.dtype != "F32" {
				for _, leaf := range []string{"attn.wo_a.weight", "attn.wo_b.weight"} {
					if m.has(layerName(0, leaf)) {
						t.Error("selected grouped weight expanded into F32 manifest")
					}
				}
			}
		})
	}
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime fast est=500ms lane=default
func TestV41GroupedOutputUnsupportedPairKeepsDenseDevice(t *testing.T) {
	t.Parallel()
	m := v41GroupedFixture(t, "Q8_0", false, false)
	b := newV41GroupedBackend(t, m)
	b.decline = compute.Q8_0
	s := v41DenseTestSession(t, m, b)
	for index, ids := range [][]int{{1, 2, 3}, {4}, {5, 6}} {
		phase := "prefill"
		if index == 1 {
			phase = "decode"
		}
		before := v41GroupedPhase(t, m, phase)
		denseBefore := v41DenseTestPhase(t, m, phase)
		from, attempts := len(b.grouped), b.matmulAttempts
		var got []float32
		if index == 1 {
			got = s.Step(ids[0])
		} else {
			got = s.Prefill(ids)
		}
		if len(got) != m.Cfg.VocabSize {
			t.Error("declined grouped operation did not produce logits")
		}
		v41GroupedCheckPhase(t, m, b, from, attempts, len(ids), phase, before, true)
		dense := v41DenseTestDelta(v41DenseTestPhase(t, m, phase), denseBefore)
		if dense["dense_projection_device_rows"] != float64(7*len(ids)) || dense["dense_projection_host_rows"] != 0 {
			t.Errorf("unsupported grouped dtype changed independently supported dense path: %v", dense)
		}
	}
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime fast est=500ms lane=default
func TestV41GroupedOutputInvalidGeometryBeforeDevice(t *testing.T) {
	t.Parallel()
	for _, malformed := range []string{"groups", "weight-shape"} {
		t.Run(malformed, func(t *testing.T) {
			t.Parallel()
			m := v41GroupedFixture(t, "F32", false, false)
			b := newV41GroupedBackend(t, m)
			s := v41DenseTestSession(t, m, b)
			s.Prefill([]int{1, 2, 3})
			before := captureV41ForwardSnapshot(s.v41Forward)
			attempts := b.matmulAttempts
			if malformed == "groups" {
				m.Cfg.OGroups = 3
			} else {
				name := layerName(0, "attn.wo_a.weight")
				meta := m.manifest[name]
				meta.Shape = []int{meta.Shape[0] + 1, meta.Shape[1]}
				m.manifest[name] = meta
			}
			err := panicAsError(func() { s.Step(4) })
			if !errors.Is(err, ErrV41ForwardStage) {
				t.Errorf("invalid grouped geometry lost closed forward refusal: %v", err)
			}
			if b.matmulAttempts != attempts {
				t.Error("invalid grouped geometry reached a device grouped contraction")
			}
			if !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
				t.Error("invalid grouped geometry advanced retained session state")
			}
		})
	}
}

type v41GroupedFailingReader struct{ calls int }

func (r *v41GroupedFailingReader) ReadAt(_ []byte, _ int64) (int, error) {
	r.calls++
	return 0, ErrV41ForwardStage
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime fast est=500ms lane=default
func TestV41GroupedOutputHostPartialMaterializationFailure(t *testing.T) {
	t.Parallel()
	m := v41GroupedFixture(t, "Q4_K", false, false)
	b := newV41GroupedBackend(t, m)
	b.deny = true
	s := v41DenseTestSession(t, m, b)
	s.Prefill([]int{1, 2, 3})
	before := captureV41ForwardSnapshot(s.v41Forward)
	ledgerBefore := v41GroupedPhase(t, m, "decode")
	a, ok := m.residentF32Mat(layerName(0, "attn.wo_a.weight"))
	if !ok {
		t.Fatal("positive host A materialization witness absent")
	}
	name := layerName(0, "attn.wo_b.weight")
	weight := m.q4kw[name]
	reader := &v41GroupedFailingReader{}
	weight.lazy = &LazyQ4KRange{Reader: reader, Bytes: len(weight.raw)}
	weight.raw = nil
	err := panicAsError(func() { s.Step(4) })
	if !errors.Is(err, ErrV41ForwardStage) {
		t.Errorf("missing host B payload lost closed stage refusal: %v", err)
	}
	delta := v41DenseTestDelta(v41GroupedPhase(t, m, "decode"), ledgerBefore)
	if reader.calls != 2 {
		t.Errorf("parent incremental-stage plus full-history fallback retrieval attempts=%d want 2", reader.calls)
	}
	if delta["grouped_output_host_weight_f32_bytes"] != float64(4*len(a)*reader.calls) || delta["grouped_output_host_rows"] != 0 || delta["grouped_output_device_calls"] != 0 || delta["grouped_output_host_calls"] != float64(reader.calls) || delta["grouped_output_nanos"] <= 0 {
		t.Errorf("partial host materialization accounting=%v want returned A bytes=%d", delta, 4*len(a)*reader.calls)
	}
	if !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
		t.Error("partial host grouped failure advanced retained state")
	}
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=2s lane=default
func TestV41GroupedOutputLateFailureAtomic(t *testing.T) {
	t.Parallel()
	for _, site := range []string{"matmul", "read", "length", "finite", "unknown"} {
		t.Run(site, func(t *testing.T) {
			t.Parallel()
			m := v41GroupedFixture(t, "F32", false, false)
			b := newV41GroupedBackend(t, m)
			api := &v41GroupedClosedRecorder{v41GroupedBackend: b}
			s := v41DenseTestSession(t, m, api)
			s.Prefill([]int{1, 2, 3})
			b.live = map[compute.Buffer]bool{}
			before := captureV41ForwardSnapshot(s.v41Forward)
			ledgerBefore := v41GroupedPhase(t, m, "decode")
			from, attempts := len(b.grouped), b.matmulAttempts
			b.fault = site
			b.cause = &compute.BackendError{Backend: "cpu-ref-recording", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
			unknown := struct{ Number int }{13668}
			if site == "unknown" {
				b.fault, b.cause = "matmul", unknown
			}
			var closedFailure *BackendForwardOperationError
			var recovered any
			func() { defer func() { recovered = recover() }(); s.Step(4) }()
			if !b.failing {
				t.Fatal("late grouped B failure injection was never reached")
			}
			if site == "unknown" {
				if recovered != unknown {
					t.Error("selected grouped operation changed unknown panic identity")
				}
			} else {
				err, ok := recovered.(error)
				if !ok || !errors.As(err, &closedFailure) || !s.BackendSessionClosed() {
					t.Error("selected typed failure did not retain a closed Session/error identity")
				}

				var typed *BackendForwardOperationError
				if !ok || !errors.As(err, &typed) || !errors.Is(err, ErrV41ForwardStage) {
					t.Errorf("grouped failure lacks typed operation error/cause: %T %v", recovered, recovered)
				}
				if site == "matmul" || site == "read" {
					if !errors.Is(err, b.cause.(error)) {
						t.Error("selected grouped failure lost backend cause identity")
					}
				}
			}
			if !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
				t.Error("late grouped failure committed continuation state")
			}
			if len(b.live) != 0 {
				t.Errorf("grouped failure leaked %d transient buffers", len(b.live))
			}
			delta := v41DenseTestDelta(v41GroupedPhase(t, m, "decode"), ledgerBefore)
			if delta["grouped_output_device_calls"] != 1 || delta["grouped_output_device_rows"] != 0 || delta["grouped_output_host_calls"] != 0 || delta["grouped_output_host_rows"] != 0 || delta["grouped_output_host_weight_f32_bytes"] != 0 {
				t.Errorf("failed whole-operation accounting=%v", delta)
			}
			if delta["grouped_output_matmul_calls"] != float64(b.matmulAttempts-attempts) {
				t.Error("late failure lost attempted MatMul call")
			}
			upload, read := 0, 0
			for _, op := range b.grouped[from:] {
				upload += op.upload
				read += op.read
			}
			// The failing MatMul receives an already successfully uploaded joined activation.
			if site == "matmul" || site == "unknown" {
				upload += 4 * m.Cfg.OGroups * m.Cfg.OLoraRank
			}
			if delta["grouped_output_activation_upload_bytes"] != float64(upload) || delta["grouped_output_readback_bytes"] != float64(read) {
				t.Errorf("late failure transfer accounting=%v actual upload=%d read=%d", delta, upload, read)
			}

			if site != "unknown" {
				callsBefore := api.calls
				stateBefore := captureV41ForwardSnapshot(s.v41Forward)
				dataBefore := v41GroupedPhase(t, m, "decode")
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
				if !reflect.DeepEqual(stateBefore, captureV41ForwardSnapshot(s.v41Forward)) || !reflect.DeepEqual(dataBefore, v41GroupedPhase(t, m, "decode")) {
					t.Error("closed public entry changed retained state or projection attribution")
				}
				if len(b.live) != 0 {
					t.Error("closed public entry changed transient buffer cleanup")
				}
				return
			}
			b.fault, b.cause, b.failing = "", nil, false
			v41GroupedParity(t, s.Step(4), lastLogits(v41GroupedFixture(t, "F32", false, false).Forward([]int{1, 2, 3, 4})), 1e-5)
		})
	}
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime fast est=500ms lane=default
func TestV41GroupedOutputRestoreForkTargetOwnsBridge(t *testing.T) {
	t.Parallel()
	m := v41GroupedFixture(t, "F32", false, false)
	b := newV41GroupedBackend(t, m)
	source := v41DenseTestSession(t, m, b)
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
	left, right := v41DenseTestSession(t, m, b), v41DenseTestSession(t, m, b)
	if err := snap.Restore(left); err != nil {
		t.Fatal(err)
	}
	if err := clone.Restore(right); err != nil {
		t.Fatal(err)
	}
	source.Close()
	for _, frees := range b.weightFrees {
		if frees != 0 {
			t.Error("source close freed shared immutable grouped weight")
		}
	}
	for index, branch := range []*Session{left, right} {
		from := len(b.grouped)
		token := 4 + index
		v41GroupedParity(t, branch.Step(token), lastLogits(v41GroupedFixture(t, "F32", false, false).Forward([]int{1, 2, 3, token})), 1e-5)
		rows := 0
		for _, op := range b.grouped[from:] {
			if op.leaf == "b" {
				rows += op.rows
			}
		}
		if rows != 1 {
			t.Error("restored target bridge did not execute exactly one grouped row")
		}
	}
	left.Close()
	for _, frees := range b.weightFrees {
		if frees != 0 {
			t.Error("fork close freed sibling's shared grouped weight")
		}
	}
	v41GroupedParity(t, right.Step(6), lastLogits(v41GroupedFixture(t, "F32", false, false).Forward([]int{1, 2, 3, 5, 6})), 1e-5)
	right.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
	before := len(b.ops)
	err = panicAsError(func() { right.Step(7) })
	var strict *BackendForwardOperationError
	if !errors.As(err, &strict) || len(b.ops) != before {
		t.Error("strict V4.1 device-only whole-model guard changed")
	}
	right.Close()
	if err := m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	if len(b.weights) != m.Cfg.OGroups+1 {
		t.Errorf("shared grouped cache contains %d weights want %d", len(b.weights), m.Cfg.OGroups+1)
	}
	for weight := range b.weights {
		if b.weightFrees[weight] != 1 {
			t.Errorf("model owner freed grouped weight %d times want once", b.weightFrees[weight])
		}
	}
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime fast est=500ms lane=default
func TestV41GroupedOutputPreservesRawAdapterSemantics(t *testing.T) {
	t.Parallel()
	m := v41GroupedFixture(t, "F32", false, true)
	spec := PrismHadamardSpec{BlockSize: 4}
	widths := map[int]bool{}
	for _, leaf := range []string{"attn.wo_a.weight", "attn.wo_b.weight"} {
		name := layerName(0, leaf)
		shape := m.manifest[name].Shape
		signs := make([]int, shape[1])
		for i := range signs {
			signs[i] = 1
			if i%3 == 1 {
				signs[i] = -1
			}
		}
		spec.WeightNames = append(spec.WeightNames, name)
		if !widths[shape[1]] {
			widths[shape[1]] = true
			spec.SignWidths = append(spec.SignWidths, shape[1])
			spec.SignValues = append(spec.SignValues, signs...)
		}
	}
	if err := m.SetPrismHadamard(spec); err != nil {
		t.Fatal(err)
	}
	set := NewLoRASet()
	for _, leaf := range []string{"attn.wo_a.weight", "attn.wo_b.weight"} {
		name := layerName(0, leaf)
		shape := m.manifest[name].Shape
		if err := set.Add(newTestAdapter("grouped", name, shape[0], shape[1], 1, 8, 13668)); err != nil {
			t.Fatal(err)
		}
	}
	set.Activate("grouped")
	m.SetLoRA(set)
	b := newV41GroupedBackend(t, m)
	s := v41DenseTestSession(t, m, b)
	for index, ids := range [][]int{{1, 2, 3}, {4}} {
		history := []int{1, 2, 3}
		var got []float32
		if index == 0 {
			got = s.Prefill(ids)
		} else {
			got = s.Step(ids[0])
			history = append(history, ids...)
		}
		want := lastLogits(v41GroupedFixture(t, "F32", false, true).Forward(history))
		v41GroupedParity(t, lastLogits(m.Forward(history)), want, 1e-6)
		v41GroupedParity(t, got, want, 1e-5)
	}
}

type v41GroupedFiniteBackend struct {
	*v41GroupedBackend
	codes                          [][]int8
	groupedUploads, groupedMatmuls int
}

func (b *v41GroupedFiniteBackend) isGrouped(x compute.Tensor) bool {
	if !x.Dtype.Quantized() {
		return false
	}
	host, ok := x.Buf().(compute.HostBuffer)
	if !ok {
		return false
	}
	for _, codes := range b.codes {
		if reflect.DeepEqual(codes, host.I8()) {
			return true
		}
	}
	return false
}
func (b *v41GroupedFiniteBackend) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	if b.isGrouped(x) {
		b.groupedUploads++
	}
	return b.v41GroupedBackend.Upload(x, dt)
}
func (b *v41GroupedFiniteBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	if b.isGrouped(w) {
		b.groupedMatmuls++
	}
	return b.v41GroupedBackend.MatMul(w, x)
}
func (b *v41GroupedFiniteBackend) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	if b.isGrouped(w) {
		b.groupedMatmuls++
	}
	return b.v41GroupedBackend.BatchedMatMul(w, x, rows)
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=2s lane=default
func TestV41GroupedOutputPackedNonfiniteBeforeAnyGroupedAPI(t *testing.T) {
	t.Parallel()
	for _, scenario := range []string{"q2-inf", "q2-nan", "q8-overflow"} {
		for _, leaf := range []string{"attn.wo_a.weight", "attn.wo_b.weight"} {
			t.Run(scenario+"/"+leaf, func(t *testing.T) {
				t.Parallel()
				dtype := "Q2_K"
				if scenario == "q8-overflow" {
					dtype = "Q8_0"
				}
				m := v41GroupedFixture(t, dtype, false, false)
				base := newV41GroupedBackend(t, m)
				name := layerName(0, leaf)
				if dtype == "Q8_0" {
					q := m.q8w[name]
					found := false
					for i, code := range q.q {
						if code > 1 || code < -1 {
							q.d[i/qBlk] = math.MaxFloat32
							found = true
							break
						}
					}
					if !found {
						t.Fatal("Q8 overflow witness has no nonzero magnitude>1 code")
					}
				} else {
					q := m.kqw[name]
					d := uint16(0x7c00)
					if scenario == "q2-nan" {
						d = 0x7e00
					}
					binary.LittleEndian.PutUint16(q.raw[q2kBlockBytes-4:], d)
				}
				b := &v41GroupedFiniteBackend{v41GroupedBackend: base}
				for _, weightLeaf := range []string{"attn.wo_a.weight", "attn.wo_b.weight"} {
					weightName := layerName(0, weightLeaf)
					var codes []int8
					if dtype == "Q8_0" {
						codes = append([]int8(nil), m.q8w[weightName].q...)
					} else {
						raw := m.kqw[weightName].raw
						codes = make([]int8, len(raw))
						for i, value := range raw {
							codes[i] = int8(value)
						}
					}
					b.codes = append(b.codes, codes)
					if weightLeaf == "attn.wo_a.weight" {
						for group := 0; group < m.Cfg.OGroups; group++ {
							b.codes = append(b.codes, codes[group*len(codes)/m.Cfg.OGroups:(group+1)*len(codes)/m.Cfg.OGroups])
						}
					}
				}
				// Prove the source corruption remains valid-length but decodes nonfinite before invoking the session.
				values, ok := m.residentF32Mat(name)
				if !ok {
					t.Fatal("invalid-finite payload is structurally absent")
				}
				wantElements := len(base.b)
				if leaf == "attn.wo_a.weight" {
					wantElements = len(base.a) * len(base.a[0])
				}
				if len(values) != wantElements {
					t.Fatalf("packed finite-validation source length=%d want %d", len(values), wantElements)
				}
				nonfinite := false
				for _, value := range values {
					if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
						nonfinite = true
						break
					}
				}
				if !nonfinite {
					t.Fatal("packed finite-validation negative control is vacuous")
				}
				s := v41DenseTestSession(t, m, b)
				err := panicAsError(func() { s.Prefill([]int{1, 2, 3}) })
				if !errors.Is(err, ErrV41ForwardStage) {
					t.Errorf("packed nonfinite payload lost closed forward refusal: %v", err)
				}
				if b.groupedUploads != 0 || b.groupedMatmuls != 0 {
					t.Errorf("nonfinite packed source reached grouped API uploads=%d matmuls=%d", b.groupedUploads, b.groupedMatmuls)
				}
				phase := v41GroupedPhase(t, m, "prefill")
				if phase["grouped_output_host_calls"] != 0 || phase["grouped_output_host_weight_f32_bytes"] != 0 || phase["grouped_output_host_rows"] != 0 || phase["grouped_output_device_rows"] != 0 {
					t.Errorf("invalid selected packed payload retried/materialized on host or completed rows: %v", phase)
				}
				if s.v41Forward != nil && len(s.v41Forward.history) != 0 {
					t.Error("nonfinite packed source committed prefix history")
				}
			})
		}
	}
}

type v41GroupedClosedRecorder struct {
	*v41GroupedBackend
	calls [4]int
}

func (b *v41GroupedClosedRecorder) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	b.calls[0]++
	return b.v41GroupedBackend.Upload(x, dt)
}
func (b *v41GroupedClosedRecorder) MatMul(w, x compute.Tensor) compute.Tensor {
	b.calls[1]++
	return b.v41GroupedBackend.MatMul(w, x)
}
func (b *v41GroupedClosedRecorder) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	b.calls[1]++
	return b.v41GroupedBackend.BatchedMatMul(w, x, rows)
}
func (b *v41GroupedClosedRecorder) Read(x compute.Tensor) []float32 {
	b.calls[2]++
	return b.v41GroupedBackend.Read(x)
}
func (b *v41GroupedClosedRecorder) Free(x compute.Tensor) { b.calls[3]++; b.v41GroupedBackend.Free(x) }
