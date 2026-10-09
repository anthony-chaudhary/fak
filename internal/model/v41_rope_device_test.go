package model

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41TailRoPECall struct {
	q, kv, table              []float32
	qOut, kvOut               []float32
	heads, headDim, rotaryDim int
}

// This recorder executes an independent scalar formula over cpu-ref storage.
// Its DeviceMemory promise tests routing only, never physical qualification.
type v41TailRoPETestBackend struct {
	*v41DenseTestBackend
	supported                bool
	calls                    []v41TailRoPECall
	transients               map[compute.Buffer]bool
	outputs                  map[compute.Buffer]string
	uploadCount, uploadBytes int
	readCount, readBytes     int
	readScratch              []float32
	fault                    string
	failCall                 int
	cause                    error
	panicValue               any
}

func newV41TailRoPETestBackend() *v41TailRoPETestBackend {
	return &v41TailRoPETestBackend{
		v41DenseTestBackend: newV41DenseTestBackend(), supported: true,
		transients: map[compute.Buffer]bool{}, outputs: map[compute.Buffer]string{},
	}
}

func (b *v41TailRoPETestBackend) SupportsV41TailRoPE() bool { return b.supported }

func (b *v41TailRoPETestBackend) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if !strings.HasPrefix(site, "V4.1 tail-RoPE ") {
		return b.v41DenseTestBackend.Upload(x, dt)
	}
	if class != compute.MemoryActivation || dt != compute.F32 {
		panic("tail-RoPE upload is not a classed F32 activation")
	}
	if b.fault == "upload" && b.uploadCount+1 == b.failCall {
		panic(b.cause)
	}
	// Device uploads own storage. The test must not hide input mutation behind
	// cpu-ref's otherwise identity Upload implementation.
	values := append([]float32(nil), b.Backend.Read(x)...)
	y := compute.NewF32(b, append([]int(nil), x.Shape...), values)
	b.transients[y.Buf()] = true
	b.uploadCount++
	b.uploadBytes += 4 * len(values)
	return y
}

func v41TailRoPEExactOracle(input, table []float32, headDim, rotaryDim int) []float32 {
	out := append([]float32(nil), input...)
	for row := 0; row < len(input)/headDim; row++ {
		for j := 0; j < rotaryDim/2; j++ {
			i := row*headDim + headDim - rotaryDim + 2*j
			a, b := float64(input[i]), float64(input[i+1])
			s, c := float64(table[2*j]), float64(table[2*j+1])
			ac, bs := float32(a*c), float32(b*s)
			bc, as := float32(b*c), float32(a*s)
			out[i] = float32(float64(ac) - float64(bs))
			out[i+1] = float32(float64(bc) + float64(as))
		}
	}
	return out
}

func (b *v41TailRoPETestBackend) V41TailRoPEQK(q, kv, table compute.Tensor, heads, headDim, rotaryDim int) (compute.Tensor, compute.Tensor, error) {
	if q.Backend() != b || kv.Backend() != b || table.Backend() != b ||
		q.Dtype != compute.F32 || kv.Dtype != compute.F32 || table.Dtype != compute.F32 ||
		!reflect.DeepEqual(q.Shape, []int{heads, headDim}) || !reflect.DeepEqual(kv.Shape, []int{headDim}) || !reflect.DeepEqual(table.Shape, []int{rotaryDim / 2, 2}) {
		return compute.Tensor{}, compute.Tensor{}, errors.New("invalid tail-RoPE tensor contract")
	}
	call := v41TailRoPECall{
		q: append([]float32(nil), b.Backend.Read(q)...), kv: append([]float32(nil), b.Backend.Read(kv)...),
		table: append([]float32(nil), b.Backend.Read(table)...), heads: heads, headDim: headDim, rotaryDim: rotaryDim,
	}
	call.qOut = v41TailRoPEExactOracle(call.q, call.table, headDim, rotaryDim)
	call.kvOut = v41TailRoPEExactOracle(call.kv, call.table, headDim, rotaryDim)
	b.calls = append(b.calls, call)
	fail := len(b.calls) == b.failCall
	if fail && b.fault == "dispatch" {
		return compute.Tensor{}, compute.Tensor{}, b.cause
	}
	if fail && b.fault == "panic" {
		if b.panicValue != nil {
			panic(b.panicValue)
		}
		panic(b.cause)
	}
	qo := compute.NewF32(b, []int{heads, headDim}, append([]float32(nil), call.qOut...))
	ko := compute.NewF32(b, []int{headDim}, append([]float32(nil), call.kvOut...))
	b.transients[qo.Buf()], b.transients[ko.Buf()] = true, true
	b.outputs[qo.Buf()], b.outputs[ko.Buf()] = "query", "kv"
	if fail && b.fault == "partial-output" {
		return qo, ko, b.cause
	}
	if fail && b.fault == "alias" {
		b.Free(qo)
		return q, ko, nil
	}
	return qo, ko, nil
}

func (b *v41TailRoPETestBackend) Read(x compute.Tensor) []float32 {
	which := b.outputs[x.Buf()]
	if which == "" {
		return b.v41DenseTestBackend.Read(x)
	}
	values := b.Backend.Read(x)
	b.readCount++
	b.readBytes += 4 * len(values)
	if len(b.calls) == b.failCall {
		if b.fault == "read-"+which {
			panic(b.cause)
		}
		if b.fault == "short-"+which {
			return values[:len(values)-1]
		}
		if b.fault == "nonfinite" && which == "kv" {
			values[len(values)-1] = float32(math.NaN())
		}
		if b.fault == "prefix" && which == "query" {
			values[0] = math.Float32frombits(math.Float32bits(values[0]) ^ 1)
		}
	}
	// Reuse a host readback buffer, then poison it on Free. Both kinds of
	// borrowing are invalid: Q must be retained before KV's read and both rows
	// must be retained before output release.
	if cap(b.readScratch) < len(values) {
		b.readScratch = make([]float32, len(values))
	}
	b.readScratch = b.readScratch[:len(values)]
	copy(b.readScratch, values)
	return b.readScratch
}

func (b *v41TailRoPETestBackend) Free(x compute.Tensor) {
	if b.outputs[x.Buf()] != "" {
		values := b.Backend.Read(x)
		for i := range values {
			values[i] = float32(math.NaN())
		}
		for i := range b.readScratch {
			b.readScratch[i] = float32(math.NaN())
		}
		delete(b.outputs, x.Buf())
	}
	delete(b.transients, x.Buf())
	b.v41DenseTestBackend.Free(x)
}

func v41TailRoPECheckBits(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s width=%d want=%d", label, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d]=%08x want=%08x", label, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

func v41TailRoPEOnly(s *Session) {
	st := s.v41State()
	// Isolate this boundary; other selected components have their own witnesses.
	st.expertGateUp, st.expertDown = nil, nil
	st.denseProjection, st.groupedOutput, st.engramProjection = nil, nil, nil
	st.mhcProjection, st.finalNorm, st.queryNorm, st.kvNorm = nil, nil, nil, nil
	st.ffnNorm, st.sharedActivation = nil, nil
}

// Refs #13668. One bounded software witness covers exact operands, all three
// assembly blocks, continuation ownership, transfers and selected failures.
// Shader/NoContraction and cross-device tail tolerance remain physical gates.
// fak-test:runtime medium est=5s lane=default
func TestV41TailRoPEDeviceDispatch(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("tail-RoPE software witness requires cpu-ref")
	}
	newSession := func(t *testing.T, m *Model) (*Session, *v41TailRoPETestBackend) {
		b := newV41TailRoPETestBackend()
		s := v41DenseTestSession(t, m, b)
		v41TailRoPEOnly(s)
		if s.v41State().tailRoPE == nil {
			t.Fatal("available operation was not bound")
		}
		return s, b
	}

	t.Run("exact-current-tables-and-distinct-tails", func(t *testing.T) {
		for _, geometry := range [][3]int{{3, 8, 4}, {2, 512, 64}} {
			heads, hd, rd := geometry[0], geometry[1], geometry[2]
			b := newV41TailRoPETestBackend()
			s := &Session{M: &Model{}, Backend: b}
			t.Cleanup(s.Close)
			cfg := Config{RopeTheta: 10000, QKRopeHeadDim: rd, RopeScaling: "yarn", LongRope: &RopeScaling{Type: "yarn", AttentionFactor: 1.75}}
			cos, sin := v41RopeTableForLayer(cfg, 0, 7)
			table := make([]float32, rd)
			for j := 0; j < rd/2; j++ {
				angle := float64(7) / math.Pow(10000, float64(2*j)/float64(rd))
				table[2*j] = float32(math.Sin(angle)) * float32(1.75)
				table[2*j+1] = float32(math.Cos(angle)) * float32(1.75)
			}
			q, kv := make([]float32, heads*hd), make([]float32, hd)
			for i := range q {
				q[i] = float32((i%17)-8)*0.1234567 + float32(i)*0.007
			}
			for i := range kv {
				kv[i] = float32((i%13)-6)*0.234567 + float32(i)*0.003
			}
			q[0], kv[0] = math.Float32frombits(0x80000000), math.Float32frombits(0x80000000)
			beforeQ, beforeKV := append([]float32(nil), q...), append([]float32(nil), kv...)
			qo, ko, err := s.v41TailRoPEFunc()(0, q, kv, cos, sin, heads, hd, rd)
			if err != nil {
				t.Fatal(err)
			}
			v41TailRoPECheckBits(t, "packed current [sin,cos]", b.calls[0].table, table)
			v41TailRoPECheckBits(t, "query", qo, v41TailRoPEExactOracle(q, table, hd, rd))
			v41TailRoPECheckBits(t, "KV", ko, v41TailRoPEExactOracle(kv, table, hd, rd))
			v41TailRoPECheckBits(t, "unchanged Q input", q, beforeQ)
			v41TailRoPECheckBits(t, "unchanged KV input", kv, beforeKV)
			if b.uploadCount != 3 || b.readCount != 2 || b.uploadBytes != 4*((heads+1)*hd+rd) || b.readBytes != 4*(heads+1)*hd || len(b.transients) != 0 {
				t.Fatalf("transfer/ownership contract failed: upload=%d/%d read=%d/%d live=%d", b.uploadCount, b.uploadBytes, b.readCount, b.readBytes, len(b.transients))
			}
			// Controls must disagree at the seam, before later projections can
			// hide a wrong pairing, missing amplitude, or BF16 substitution.
			prefix := append([]float32(nil), q...)
			for h := 0; h < heads; h++ {
				copy(prefix[h*hd:h*hd+rd], v41TailRoPEExactOracle(q[h*hd:h*hd+rd], table, rd, rd))
			}
			halfSplit := append([]float32(nil), q...)
			for h := 0; h < heads; h++ {
				for j := 0; j < rd/2; j++ {
					i := h*hd + hd - rd + j
					a, b := q[i], q[i+rd/2]
					halfSplit[i] = float32(a*table[2*j+1]) - float32(b*table[2*j])
					halfSplit[i+rd/2] = float32(b*table[2*j+1]) + float32(a*table[2*j])
				}
			}
			unit := append([]float32(nil), table...)
			for i := range unit {
				unit[i] /= 1.75
			}
			bf16 := append([]float32(nil), qo...)
			for i := range bf16 {
				bits := math.Float32bits(bf16[i])
				bf16[i] = math.Float32frombits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
			}
			if reflect.DeepEqual(qo, prefix) || reflect.DeepEqual(qo, halfSplit) || reflect.DeepEqual(qo, v41TailRoPEExactOracle(q, unit, hd, rd)) || reflect.DeepEqual(qo, bf16) {
				t.Fatal("prefix/half-split/scale/BF16 controls do not discriminate")
			}
		}
		// The product (1+2^-23)*(1-2^-23) rounds to 1 before
		// subtraction. A fused multiply-add preserves a nonzero residual.
		b := newV41TailRoPETestBackend()
		s := &Session{M: &Model{}, Backend: b}
		t.Cleanup(s.Close)
		a, c := math.Float32frombits(0x3f800001), math.Float32frombits(0x3f7ffffe)
		q := []float32{5, -7, a, 1}
		qo, _, err := s.v41TailRoPEFunc()(0, q, q, []float32{c}, []float32{1}, 1, 4, 2)
		if err != nil || qo[2] != 0 || float32(math.FMA(float64(a), float64(c), -1)) == qo[2] {
			t.Fatalf("separate-F32 product control failed: q=%v err=%v", qo, err)
		}
	})

	t.Run("partial-upload-cleanup", func(t *testing.T) {
		for failedUpload, stage := range []string{"query upload", "kv upload", "table upload"} {
			b := newV41TailRoPETestBackend()
			s := &Session{M: &Model{}, Backend: b}
			t.Cleanup(s.Close)
			b.fault, b.failCall = "upload", failedUpload+1
			b.cause = &compute.BackendError{Backend: "test-device", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
			q, kv := []float32{5, -7, 3, 11}, []float32{-2, 4, 6, 8}
			beforeQ, beforeKV := append([]float32(nil), q...), append([]float32(nil), kv...)
			err := v41TailRoPEInPlace(0, q, kv, []float32{0.3}, []float32{0.7}, 1, 4, 2, s.v41TailRoPEFunc())
			var selected *V41TailRoPEOperationError
			var closed *BackendForwardOperationError
			if !errors.As(err, &selected) || !errors.As(err, &closed) || closed.Stage != stage || !errors.Is(err, b.cause) || !s.BackendSessionClosed() {
				t.Fatalf("partial upload lost selected failure identity: %v", err)
			}
			if b.uploadCount != failedUpload || len(b.calls) != 0 || len(b.transients) != 0 {
				t.Fatal("partial upload dispatched or leaked a prior allocation")
			}
			v41TailRoPECheckBits(t, "failed upload Q", q, beforeQ)
			v41TailRoPECheckBits(t, "failed upload KV", kv, beforeKV)
		}
	})

	for _, profile := range []string{"reduced", "full", "compressed-role"} {
		t.Run(profile, func(t *testing.T) {
			var m *Model
			switch profile {
			case "reduced":
				m = v41IncrementalPlainModel(t, 2)
			case "full":
				m = v41FullStepPatchedNorms(t)
			default:
				m = v41IncrementalExpertFixture(t, true, false)
			}
			// An explicit test configuration, not an inference about a pinned GGUF.
			m.Cfg.RopeScaling = "yarn"
			m.Cfg.LongRope = &RopeScaling{Type: "yarn", AttentionFactor: 1.75}
			s, b := newSession(t, m)
			v41GroupedParity(t, s.Prefill([]int{1, 2}), lastLogits(m.Forward([]int{1, 2})), 1e-4)
			if !s.v41IncrementalEligible() || len(b.calls) != 2*m.Cfg.NumLayers {
				t.Fatal("cold prefill did not select one operation per layer/position")
			}
			v41GroupedParity(t, s.Step(3), lastLogits(m.Forward([]int{1, 2, 3})), 1e-4)
			v41GroupedParity(t, s.Prefill([]int{4}), lastLogits(m.Forward([]int{1, 2, 3, 4})), 1e-4)
			if len(b.calls) != 4*m.Cfg.NumLayers || b.uploadCount != 3*len(b.calls) || b.readCount != 2*len(b.calls) || len(b.transients) != 0 {
				t.Fatal("Step/suffix replayed a prefix, skipped RoPE, or leaked a transient")
			}
			for i, call := range b.calls {
				layer, pos := i/2, i%2 // cold prefill is layer-major
				if i >= 2*m.Cfg.NumLayers {
					layer, pos = i%m.Cfg.NumLayers, i/m.Cfg.NumLayers
				}
				cos, sin := v41RopeTableForLayer(m.Cfg, layer, pos)
				packed := make([]float32, len(cos)*2)
				for j := range cos {
					packed[2*j], packed[2*j+1] = sin[j], cos[j]
				}
				v41TailRoPECheckBits(t, "absolute-position current table", call.table, packed)
				if call.heads != m.Cfg.NumHeads || call.headDim != m.Cfg.HeadDim || call.rotaryDim != m.Cfg.QKRopeHeadDim {
					t.Fatal("query/KV geometry changed at rotation seam")
				}
			}
			snap, err := s.PrefixSnapshot()
			if err != nil {
				t.Fatal(err)
			}
			defer snap.Close()
			target := v41DenseTestSession(t, m, b)
			if err := snap.Restore(target); err != nil {
				t.Fatal(err)
			}
			v41TailRoPEOnly(target)
			s.Close()
			before := len(b.calls)
			v41GroupedParity(t, target.Prefill([]int{5}), lastLogits(m.Forward([]int{1, 2, 3, 4, 5})), 1e-4)
			if target.v41Forward.callbackOwner != target || target.v41Forward.tailRoPE == nil || len(b.calls) != before+m.Cfg.NumLayers {
				t.Fatal("restored callback did not rebind to the target session")
			}
			before = len(b.calls)
			target.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
			var refused *BackendForwardOperationError
			if err := recoverError(func() { target.Step(6) }); !errors.As(err, &refused) || refused.Stage != "decode: architecture uses host model compute" || len(b.calls) != before {
				t.Fatalf("whole-architecture DeviceOnly admission changed: %v", err)
			}
		})
	}

	t.Run("decline-before-selection", func(t *testing.T) {
		for _, missing := range []bool{false, true} {
			m := v41IncrementalPlainModel(t, 1)
			b := newV41TailRoPETestBackend()
			b.supported = false
			var be compute.Backend = b
			if missing {
				be = b.v41DenseTestBackend // hides the optional operation
			}
			s := v41DenseTestSession(t, m, be)
			v41TailRoPEOnly(s)
			if s.v41State().tailRoPE != nil {
				t.Fatal("unavailable capability bound a selected callback")
			}
			v41GroupedParity(t, s.Prefill([]int{1, 2}), lastLogits(m.Forward([]int{1, 2})), 0)
			v41GroupedParity(t, s.Step(3), lastLogits(m.Forward([]int{1, 2, 3})), 1e-5)
			if len(b.calls) != 0 || b.uploadCount != 0 || b.readCount != 0 {
				t.Fatal("declined operation performed transfers or dispatch")
			}
		}
	})

	t.Run("later-suffix-failure-keeps-prior-committed-token", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 2)
		s, b := newSession(t, m)
		s.Prefill([]int{1, 2})
		// Existing suffix semantics commit each successful token. Independently
		// advance a host snapshot through the first suffix token; failure in the
		// second must restore this boundary, not the whole Prefill call boundary.
		expected := captureV41ForwardSnapshot(s.v41Forward).restore()
		scratch := &v41ProjScratch{tailRoPE: s.v41State().tailRoPE}
		if _, _, err := m.forwardV41Step(3, expected, scratch); err != nil {
			t.Fatal(err)
		}
		if scratch.tailRoPE != nil {
			t.Fatal("reused scratch retained a session's rotary callback")
		}
		b.fault, b.failCall = "dispatch", len(b.calls)+2*m.Cfg.NumLayers
		b.cause = &compute.BackendError{Backend: "test-device", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
		err := recoverError(func() { s.Prefill([]int{3, 4}) })
		var selected *V41TailRoPEOperationError
		if !errors.As(err, &selected) || !errors.Is(err, b.cause) || !s.BackendSessionClosed() || len(b.calls) != b.failCall || len(b.transients) != 0 {
			t.Fatalf("later suffix failure replayed or lost selection: %v", err)
		}
		if !reflect.DeepEqual(s.v41Forward.history, []int{1, 2, 3}) || !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), captureV41ForwardSnapshot(expected)) {
			t.Fatal("later suffix failure did not restore the last committed-token boundary")
		}
		if retry := recoverError(func() { s.Prefill([]int{4}) }); retry != s.halFailure || len(b.calls) != b.failCall {
			t.Fatalf("closed suffix session retried: %v", retry)
		}
	})

	for _, tc := range []struct {
		name  string
		value any
	}{
		{"plain-error", errors.New("unclassified rotary failure")},
		{"non-error", "unclassified rotary panic"},
	} {
		t.Run("unclassified-panic-"+tc.name, func(t *testing.T) {
			m := v41IncrementalPlainModel(t, 2)
			s, b := newSession(t, m)
			s.Prefill([]int{1, 2})
			before := captureV41ForwardSnapshot(s.v41Forward)
			b.fault, b.failCall, b.panicValue = "panic", len(b.calls)+m.Cfg.NumLayers, tc.value
			caught := func() (value any) {
				defer func() { value = recover() }()
				s.Step(3)
				return nil
			}()
			var closed *BackendForwardOperationError
			if caught != tc.value || !errors.As(s.halFailure, &closed) || closed.Path != "v41-tail-rope" || closed.Stage != "tail-rope" || !s.BackendSessionClosed() {
				t.Fatalf("unclassified panic lost identity or failed to close: panic=%v failure=%v", caught, s.halFailure)
			}
			if original, ok := tc.value.(error); ok && !errors.Is(closed, original) {
				t.Fatal("unclassified error cause was not retained")
			}
			if len(b.calls) != b.failCall || len(b.transients) != 0 || !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), before) {
				t.Fatal("unclassified panic replayed, leaked, or advanced continuation state")
			}
			if retry := recoverError(func() { s.Step(3) }); retry != s.halFailure || len(b.calls) != b.failCall {
				t.Fatalf("closed session retried after unclassified panic: %v", retry)
			}
		})
	}

	for _, tc := range []struct {
		fault, entry, stage string
		role                bool
	}{
		{"dispatch", "cold", "tail-rope", false},
		{"dispatch", "step", "tail-rope", false},
		{"dispatch", "suffix", "tail-rope", true},
		{"panic", "step", "tail-rope", false},
		{"partial-output", "step", "tail-rope", false},
		{"read-query", "step", "query readback", false},
		{"read-kv", "suffix", "kv readback", true},
		{"short-query", "step", "query readback", false},
		{"short-kv", "step", "kv readback", false},
		{"nonfinite", "suffix", "kv readback", true},
		{"prefix", "step", "query readback", false},
		{"alias", "step", "tail-rope", false},
	} {
		t.Run(tc.fault+"-"+tc.entry, func(t *testing.T) {
			m := v41IncrementalPlainModel(t, 2)
			if tc.role {
				m = v41IncrementalExpertFixture(t, true, false)
			}
			s, b := newSession(t, m)
			if tc.entry != "cold" {
				s.Prefill([]int{1, 2})
			}
			before := captureV41ForwardSnapshot(s.v41Forward)
			b.fault, b.failCall = tc.fault, len(b.calls)+m.Cfg.NumLayers
			// A structural sentinel hidden inside a selected error is still a
			// selected failure. Step and suffix must not re-fold on the host.
			b.cause = &compute.BackendError{Backend: "test-device", Class: compute.VulkanClassExecutionFailed, Err: fmt.Errorf("selected cause: %w", ErrV41ForwardStage)}
			invoke := func() {
				if tc.entry == "step" {
					s.Step(3)
				} else {
					s.Prefill([]int{3})
				}
			}
			err := recoverError(invoke)
			var selected *V41TailRoPEOperationError
			var closed *BackendForwardOperationError
			if !errors.As(err, &selected) || selected.Layer != m.Cfg.NumLayers-1 || !errors.As(err, &closed) || closed.Path != "v41-tail-rope" || closed.Stage != tc.stage {
				t.Fatalf("selected failure lost operation/layer/stage: %v", err)
			}
			if (tc.fault == "dispatch" || tc.fault == "panic" || tc.fault == "partial-output" || strings.HasPrefix(tc.fault, "read-")) && (!errors.Is(err, b.cause) || !errors.Is(err, ErrV41ForwardStage)) {
				t.Fatalf("selected failure lost nested cause identity: %v", err)
			}
			if !s.BackendSessionClosed() || len(b.calls) != b.failCall || len(b.transients) != 0 || len(b.outputs) != 0 {
				t.Fatalf("selected failure replayed or leaked: calls=%d want=%d live=%d", len(b.calls), b.failCall, len(b.transients))
			}
			if !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), before) {
				t.Fatal("selected failure changed committed history/cache")
			}
			if retry := recoverError(invoke); retry != s.halFailure || len(b.calls) != b.failCall {
				t.Fatalf("closed session retried selected operation: %v", retry)
			}
		})
	}
}
