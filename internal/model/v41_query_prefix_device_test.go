package model

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// These candidate-only tests exercise real Session consumers. The composite
// recorder uses the same cpu-ref RMSNorm/MatMul operations as the uncomposed
// device recorder, with four explicit checked BF16 boundaries. It witnesses
// routing, publication and lifecycle, not native commands or physical arithmetic.
type v41QueryPrefixRecorder struct {
	*v41DenseTestBackend
	owner                                 *Session
	available                             bool
	calls, oldAttention, oldQuery, oldWQA int
	uploadCalls, readCalls                int
	flushCalls, retireCalls               int
	faultAt                               int
	fault, busy, cleanup                  string
	primary                               error
	armed, poison                         bool
	opsAtCall, uploadsAtCall, readsAtCall int
	lastInput                             []float32
	borrowed                              compute.V41QueryPrefixResult
	cleanupHits                           int
	cleanupFreeBuffer                     compute.Buffer
}

func newV41QueryPrefixRecorder() *v41QueryPrefixRecorder {
	return &v41QueryPrefixRecorder{v41DenseTestBackend: newV41DenseTestBackend(), available: true}
}
func (b *v41QueryPrefixRecorder) V41QueryPrefixAvailable() bool { return b.available }
func (b *v41QueryPrefixRecorder) poisonBorrowed() {
	if b.poison {
		for _, row := range [][]float32{b.borrowed.Collapsed, b.borrowed.QLat} {
			for i := range row {
				row[i] = float32(math.NaN())
			}
		}
	}
}
func (b *v41QueryPrefixRecorder) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	b.poisonBorrowed()
	b.uploadCalls++
	if dt != compute.F32 {
		return b.v41DenseTestBackend.Upload(x, dt)
	}
	values := append([]float32(nil), b.Backend.Read(x)...)
	out := compute.NewF32(b, append([]int(nil), x.Shape...), values)
	b.uploads[out.Buf()] = 4 * len(values)
	return out
}
func (b *v41QueryPrefixRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, _ compute.MemoryClass, _ string) compute.Tensor {
	return b.Upload(x, dt)
}
func (b *v41QueryPrefixRecorder) Read(x compute.Tensor) []float32 {
	b.poisonBorrowed()
	b.readCalls++
	return b.v41DenseTestBackend.Read(x)
}
func (b *v41QueryPrefixRecorder) isWeight(x compute.Tensor, leaf string) bool {
	if b.owner == nil {
		return false
	}
	for l := 0; l < b.owner.M.Cfg.NumLayers; l++ {
		w, ok := b.owner.halW[layerName(l, leaf)]
		if ok && w.Buf() == x.Buf() {
			return true
		}
	}
	return false
}
func (b *v41QueryPrefixRecorder) RMSNorm(x, w compute.Tensor, eps float32) compute.Tensor {
	b.poisonBorrowed()
	if b.isWeight(w, "attn_norm.weight") {
		b.oldAttention++
	}
	if b.isWeight(w, "attn.wq_a_norm.weight") {
		b.oldQuery++
	}
	return b.Backend.RMSNorm(x, w, eps)
}
func (b *v41QueryPrefixRecorder) MatMul(w, x compute.Tensor) compute.Tensor {
	b.poisonBorrowed()
	if b.isWeight(w, "attn.wq_a.weight") {
		b.oldWQA++
	}
	return b.v41DenseTestBackend.MatMul(w, x)
}
func (b *v41QueryPrefixRecorder) BeginBatch() {}
func (b *v41QueryPrefixRecorder) FlushBatch() {
	b.flushCalls++
	if b.armed && b.cleanup == "flush" {
		b.cleanupHits++
		panic(errors.New("secondary prefix Close flush"))
	}
}
func (b *v41QueryPrefixRecorder) Recycle() {
	if b.armed {
		b.retireCalls++
		if b.cleanup == "recycle" {
			b.cleanupHits++
			panic(errors.New("secondary prefix Close recycle"))
		}
	}
}
func (b *v41QueryPrefixRecorder) Free(x compute.Tensor) {
	if b.armed && b.cleanup == "free" && x.Buf() == b.cleanupFreeBuffer {
		b.cleanupHits++
		panic(errors.New("secondary prefix Close Free"))
	}
	b.v41DenseTestBackend.Free(x)
}

func v41QueryPrefixCast(values []float32, site uint32) ([]float32, error) {
	out := make([]float32, len(values))
	for i, value := range values {
		u := math.Float32bits(value)
		rounded := (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000
		if u&0x7f800000 == 0x7f800000 || rounded&0x7f800000 == 0x7f800000 {
			return nil, &compute.V41BF16CheckedArithmeticError{Site: site, Index: i}
		}
		out[i] = math.Float32frombits(rounded)
	}
	return out, nil
}
func (b *v41QueryPrefixRecorder) V41QueryPrefix(in compute.V41QueryPrefixInputs) (compute.V41QueryPrefixResult, error) {
	b.calls++
	b.opsAtCall, b.uploadsAtCall, b.readsAtCall = len(b.ops), b.uploadCalls, b.readCalls
	b.lastInput = append([]float32(nil), in.Collapse...)
	if b.owner == nil || in.AttentionGain.Backend() != b || in.WQA.Backend() != b || in.QueryGain.Backend() != b {
		panic("query prefix operands do not belong to the selected owner")
	}
	layer := -1
	for l := 0; l < b.owner.M.Cfg.NumLayers; l++ {
		if w := b.owner.halW[layerName(l, "attn.wq_a.weight")]; w.Buf() == in.WQA.Buf() {
			layer = l
			break
		}
	}
	if layer < 0 {
		panic("query prefix did not borrow the current session memo")
	}
	a := b.owner.M.tensor(layerName(layer, "attn_norm.weight"))
	q := b.owner.M.tensor(layerName(layer, "attn.wq_a_norm.weight"))
	if len(a) == 0 || len(q) == 0 || len(in.AttentionGainHost) != len(a) || len(in.QueryGainHost) != len(q) ||
		&a[0] != &in.AttentionGainHost[0] || &q[0] != &in.QueryGainHost[0] ||
		b.owner.halW[layerName(layer, "attn_norm.weight")].Buf() != in.AttentionGain.Buf() ||
		b.owner.halW[layerName(layer, "attn.wq_a_norm.weight")].Buf() != in.QueryGain.Buf() {
		panic("query prefix canonical gain provenance changed")
	}
	fail := b.faultAt == b.calls
	if fail {
		b.armed = true
		if b.fault == "panic" {
			panic(b.primary)
		}
		if b.fault == "error" {
			return compute.V41QueryPrefixResult{}, b.primary
		}
	}
	if b.busy != "" {
		out := compute.V41QueryPrefixResult{}
		if b.fault == "busy-partial" {
			out.Collapsed = []float32{1}
		}
		busy := &compute.V41QueryPrefixBusyError{State: b.busy}
		if b.fault == "busy-wrapped" {
			return out, fmt.Errorf("wrapped busy is outside the healthy contract: %w", busy)
		}
		if b.fault == "busy-selected" {
			return out, &compute.BackendError{Backend: "prefix-recorder", Class: compute.VulkanClassExecutionFailed, Err: busy}
		}
		return out, busy
	}
	for _, row := range [][]float32{in.AttentionGainHost, in.QueryGainHost} {
		for _, value := range row {
			if !finite32(value) {
				return compute.V41QueryPrefixResult{}, &compute.V41QueryPrefixStageError{Stage: "gain", Cause: errors.New("nonfinite gain")}
			}
		}
	}
	input, err := v41QueryPrefixCast(in.Collapse, 3)
	if err != nil {
		return compute.V41QueryPrefixResult{}, err
	}
	x := compute.NewF32(b.Backend, []int{len(input)}, input)
	n := b.Backend.RMSNorm(x, in.AttentionGain, in.Epsilon)
	collapsed, err := v41QueryPrefixCast(b.Backend.Read(n), 4)
	b.Backend.Free(n)
	b.Backend.Free(x)
	if err != nil {
		return compute.V41QueryPrefixResult{}, err
	}
	x = compute.NewF32(b.Backend, []int{len(collapsed)}, collapsed)
	y := b.Backend.MatMul(in.WQA, x)
	latent, err := v41QueryPrefixCast(b.Backend.Read(y), 5)
	b.Backend.Free(y)
	b.Backend.Free(x)
	if err != nil {
		return compute.V41QueryPrefixResult{}, err
	}
	x = compute.NewF32(b.Backend, []int{len(latent)}, latent)
	n = b.Backend.RMSNorm(x, in.QueryGain, in.Epsilon)
	latent, err = v41QueryPrefixCast(b.Backend.Read(n), 6)
	b.Backend.Free(n)
	b.Backend.Free(x)
	if err != nil {
		return compute.V41QueryPrefixResult{}, err
	}
	out := compute.V41QueryPrefixResult{Collapsed: collapsed, QLat: latent}
	if fail {
		switch b.fault {
		case "partial-error":
			return out, b.primary
		case "collapsed-short":
			out.Collapsed = out.Collapsed[:len(out.Collapsed)-1]
		case "latent-short":
			out.QLat = out.QLat[:len(out.QLat)-1]
		case "collapsed-nan":
			out.Collapsed[0] = float32(math.NaN())
		case "latent-inf":
			out.QLat[0] = float32(math.Inf(1))
		case "collapsed-not-bf16":
			out.Collapsed[0] = math.Float32frombits(0x3f800001)
		case "latent-not-bf16":
			out.QLat[0] = math.Float32frombits(0x3f800001)
		}
	}
	b.borrowed = out
	return out, nil
}

func v41QueryPrefixRequireCPU(t *testing.T) {
	t.Helper()
	if b := compute.Default(); b == nil || b.Name() != "cpu-ref" || b.Caps().DeviceMemory {
		t.Fatal("query-prefix software witness requires cpu-ref")
	}
}
func v41QueryPrefixFixture(t *testing.T, role bool) *Model {
	t.Helper()
	if role {
		return v41IncrementalExpertFixture(t, true, false)
	}
	return v41FullStepPatchedNorms(t)
}
func v41QueryPrefixSession(t *testing.T, m *Model, b *v41QueryPrefixRecorder) *Session {
	t.Helper()
	s := v41DenseTestSession(t, m, b)
	b.owner = s
	t.Cleanup(func() { b.armed = false })
	if s.v41State().queryPrefix == nil {
		t.Fatal("available prefix did not bind")
	}
	return s
}
func v41QueryPrefixWarm(t *testing.T, role bool) (*Session, *v41QueryPrefixRecorder) {
	t.Helper()
	b := newV41QueryPrefixRecorder()
	s := v41QueryPrefixSession(t, v41QueryPrefixFixture(t, role), b)
	s.Prefill([]int{1, 2})
	if b.calls != 0 || !s.v41IncrementalEligible() {
		t.Fatal("cold full prefill unexpectedly selected prefix")
	}
	for _, leaf := range []string{"attn_norm.weight", "attn.wq_a.weight", "attn.wq_a_norm.weight"} {
		if !s.halW[layerName(0, leaf)].Ready() {
			t.Fatalf("real prefill did not warm %s", leaf)
		}
	}
	return s, b
}
func v41QueryPrefixOldCounts(b *v41QueryPrefixRecorder) [3]int {
	return [3]int{b.oldAttention, b.oldWQA, b.oldQuery}
}

func v41QueryPrefixFiniteParity(t *testing.T, got, want []float32) {
	t.Helper()
	for _, row := range [][]float32{got, want} {
		for _, value := range row {
			if !finite32(value) {
				t.Fatal("query-prefix parity contains a nonfinite value")
			}
		}
	}
	v41GroupedParity(t, got, want, 1e-5)
}

// fak-test:runtime medium est=3s lane=default
func TestV41QueryPrefixSessionRouting(t *testing.T) {
	v41QueryPrefixRequireCPU(t)
	for _, role := range []bool{false, true} {
		t.Run(fmt.Sprintf("role=%t", role), func(t *testing.T) {
			s, b := v41QueryPrefixWarm(t, role)
			controlBackend := newV41QueryPrefixRecorder()
			controlBackend.available = false
			control := v41DenseTestSession(t, s.M, controlBackend)
			controlBackend.owner = control
			if control.v41State().queryPrefix != nil {
				t.Fatal("unavailable backend bound query prefix")
			}
			control.Prefill([]int{1, 2})
			before := v41QueryPrefixOldCounts(b)
			if before[0] == 0 || before[2] == 0 {
				t.Fatal("old norm controls were never executed")
			}
			b.poison = true
			v41QueryPrefixFiniteParity(t, s.Step(3), control.Step(3))
			if b.calls != s.M.Cfg.NumLayers || v41QueryPrefixOldCounts(b) != before {
				t.Fatal("warm Step replayed the old prefix or missed composite")
			}
			v41QueryPrefixFiniteParity(t, s.Prefill([]int{4, 5}), control.Prefill([]int{4, 5}))
			if b.calls != 3*s.M.Cfg.NumLayers || v41QueryPrefixOldCounts(b) != before {
				t.Fatal("suffix did not use exactly one composite per layer and token")
			}
			if !reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), captureV41ForwardSnapshot(control.v41Forward)) {
				t.Fatal("prefix publication changed retained continuation")
			}
			if controlBackend.calls != 0 {
				t.Fatal("unavailable control selected the composite")
			}
		})
	}
}

// fak-test:runtime medium est=2s lane=default
func TestV41QueryPrefixRestoredWarmOwnership(t *testing.T) {
	v41QueryPrefixRequireCPU(t)
	for _, role := range []bool{false, true} {
		t.Run(fmt.Sprintf("role=%t", role), func(t *testing.T) {
			source, b := v41QueryPrefixWarm(t, role)
			snap, err := source.PrefixSnapshot()
			if err != nil {
				t.Fatal(err)
			}
			defer snap.Close()
			target := v41QueryPrefixSession(t, source.M, b)
			if err := snap.Restore(target); err != nil {
				t.Fatal(err)
			}
			source.Close()
			if target.v41State().callbackOwner != target {
				t.Fatal("restore retained source callback owner")
			}
			before := v41QueryPrefixOldCounts(b)
			target.Step(3)
			if b.calls != 0 || v41QueryPrefixOldCounts(b) == before {
				t.Fatal("fresh restored memo did not take the legitimate old route")
			}
			before = v41QueryPrefixOldCounts(b)
			target.Prefill([]int{4})
			if b.calls != target.M.Cfg.NumLayers || v41QueryPrefixOldCounts(b) != before || target.BackendSessionClosed() {
				t.Fatal("restored destination did not select after its own warmup")
			}
			primary := fmt.Errorf("restored wrapped primary: %w", &BackendForwardOperationError{Path: "incoming", Layer: 9, Stage: "test", Cause: ErrV41ForwardStage})
			beforeState := captureV41ForwardSnapshot(target.v41Forward)
			b.primary, b.fault, b.faultAt = primary, "panic", b.calls+1
			if got := recoverError(func() { target.Step(5) }); got != primary || target.halFailure != primary || !target.BackendSessionClosed() {
				t.Fatal("restored current owner did not retain the original primary", got)
			}
			if !reflect.DeepEqual(beforeState, captureV41ForwardSnapshot(target.v41Forward)) {
				t.Fatal("restored prefix failure committed continuation")
			}
			if got := recoverError(func() { target.Step(5) }); got != primary || b.calls != b.faultAt {
				t.Fatal("retired restored owner replayed")
			}
		})
	}
}

// fak-test:runtime medium est=2s lane=default
func TestV41QueryPrefixBusyRetry(t *testing.T) {
	v41QueryPrefixRequireCPU(t)
	for _, state := range []string{"batch", "restore"} {
		for _, suffix := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/suffix=%t", state, suffix), func(t *testing.T) {
				s, b := v41QueryPrefixWarm(t, false)
				before := captureV41ForwardSnapshot(s.v41Forward)
				b.busy = state
				flush := b.flushCalls
				call := func() {
					if suffix {
						s.Prefill([]int{3})
					} else {
						s.Step(3)
					}
				}
				err := recoverError(call)
				var busy *compute.V41QueryPrefixBusyError
				if !errors.As(err, &busy) || busy.State != state || !errors.Is(err, ErrV41QueryPrefixOperation) {
					t.Fatal("busy refusal lost no-replay identity", err)
				}
				if s.BackendSessionClosed() || s.halFailure != nil || b.flushCalls != flush || b.busy != state || b.calls != 1 || len(b.ops) != b.opsAtCall || b.uploadCalls != b.uploadsAtCall || b.readCalls != b.readsAtCall {
					t.Fatal("busy prefix replayed, closed, or changed the active batch")
				}
				if !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
					t.Fatal("busy prefix committed state")
				}
				b.busy = ""
				call()
				if b.calls != 2 || s.BackendSessionClosed() || len(s.v41Forward.history) != 3 {
					t.Fatal("healthy busy session could not retry")
				}
			})
		}
	}
}

// fak-test:runtime medium est=4s lane=default
func TestV41QueryPrefixSelectedFailure(t *testing.T) {
	v41QueryPrefixRequireCPU(t)
	cases := []struct {
		name, fault, stage string
		cause              error
	}{
		{"transport", "error", "output read", &compute.V41QueryPrefixStageError{Stage: "output read", Cause: ErrV41ForwardStage}},
		{"partial-error", "partial-error", "composite", ErrV41ForwardStage},
		{"collapsed-short", "collapsed-short", "output", nil},
		{"latent-short", "latent-short", "output", nil},
		{"collapsed-nan", "collapsed-nan", "output", nil},
		{"latent-inf", "latent-inf", "output", nil},
		{"collapsed-not-bf16", "collapsed-not-bf16", "output", nil},
		{"latent-not-bf16", "latent-not-bf16", "output", nil},
		{"malformed-busy-output", "busy-partial", "protocol", errV41QueryPrefixResult},
		{"unknown-busy-state", "busy-state", "protocol", errV41QueryPrefixResult},
		{"wrapped-busy", "busy-wrapped", "composite", nil},
		{"selected-error-wraps-busy", "busy-selected", "composite", nil},
	}
	for site := uint32(3); site <= 6; site++ {
		cases = append(cases, struct {
			name, fault, stage string
			cause              error
		}{fmt.Sprintf("site%d", site), "error", "composite", &compute.V41BF16CheckedArithmeticError{Site: site, Index: 0}})
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s, b := v41QueryPrefixWarm(t, true)
			before := captureV41ForwardSnapshot(s.v41Forward)
			b.fault, b.primary, b.faultAt = tc.fault, tc.cause, 1
			var publishedA, publishedQ []float32
			original := s.v41State().queryPrefix
			s.v41State().queryPrefix = func(layer int, collapse []float32) (a, q []float32, handled bool, err error) {
				a, q, handled, err = original(layer, collapse)
				publishedA, publishedQ = a, q
				return
			}
			if tc.fault == "busy-partial" || tc.fault == "busy-wrapped" || tc.fault == "busy-selected" {
				b.busy = "batch"
			}
			if tc.fault == "busy-state" {
				b.busy = "unknown"
			}
			err := recoverError(func() { s.Prefill([]int{3}) })
			var selected *BackendForwardOperationError
			if !errors.As(err, &selected) || selected.Path != "v41-query-prefix" || selected.Stage != tc.stage || !errors.Is(err, ErrV41QueryPrefixOperation) || !s.BackendSessionClosed() || s.halFailure != err {
				t.Fatal("selected failure lost attribution or retirement", err)
			}
			if publishedA != nil || publishedQ != nil {
				t.Fatal("failing prefix published a partial row")
			}
			if tc.cause != nil && !errors.Is(err, tc.cause) {
				t.Fatal("selected failure lost primary cause")
			}
			if tc.fault == "busy-partial" || tc.fault == "busy-state" {
				var busy *compute.V41QueryPrefixBusyError
				if !errors.As(err, &busy) || busy.State != b.busy {
					t.Fatal("malformed busy tuple lost its original busy cause")
				}
			}
			if b.calls != 1 || len(b.ops) != b.opsAtCall || b.uploadCalls != b.uploadsAtCall || b.readCalls != b.readsAtCall {
				t.Fatal("selected failure replayed or ran downstream projections")
			}
			if !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
				t.Fatal("selected failure published cache, cursor or retained rows")
			}
			if got := recoverError(func() { s.Step(3) }); got != err || b.calls != 1 {
				t.Fatal("retired prefix changed primary on retry")
			}
		})
	}
}

// fak-test:runtime medium est=2s lane=default
func TestV41QueryPrefixWrappedPrimary(t *testing.T) {
	v41QueryPrefixRequireCPU(t)
	for _, cleanup := range []string{"", "flush", "free", "recycle"} {
		t.Run("cleanup="+cleanup, func(t *testing.T) {
			s, b := v41QueryPrefixWarm(t, false)
			peer := v41DenseTestSession(t, s.M, b)
			t.Cleanup(func() { b.armed = false })
			// Warm canonical weights are borrowed; Close intentionally does not
			// Free them. An unrelated, session-owned disposable makes this exact
			// secondary Free injection reachable without fabricating warm memos.
			if cleanup == "free" {
				disposable := b.Upload(compute.NewF32(b.Backend, []int{1}, []float32{1}), compute.F32)
				s.halW["query-prefix-test-disposable"] = disposable
				b.cleanupFreeBuffer = disposable.Buf()
				t.Cleanup(func() { b.armed = false; b.Backend.Free(disposable) })
			}
			before := captureV41ForwardSnapshot(s.v41Forward)
			primary := fmt.Errorf("preserve this wrapper: %w", &BackendForwardOperationError{Path: "different-owner", Layer: 8, Stage: "incoming", Cause: ErrV41ForwardStage})
			b.fault, b.primary, b.faultAt, b.cleanup = "panic", primary, 1, cleanup
			got := recoverError(func() { s.Step(3) })
			if got != primary || s.halFailure != primary || !s.BackendSessionClosed() || peer.BackendSessionClosed() || peer.halFailure != nil {
				t.Fatal("wrapped primary or current-owner retirement changed", got)
			}
			wantHits := 0
			if cleanup != "" {
				wantHits = 1
			}
			if b.cleanupHits != wantHits {
				t.Fatalf("secondary %q hook hits=%d, want%d", cleanup, b.cleanupHits, wantHits)
			}
			if b.calls != 1 || len(b.ops) != b.opsAtCall || !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
				t.Fatal("wrapped failure replayed or committed")
			}
			if retry := recoverError(func() { s.Step(3) }); retry != primary || b.calls != 1 {
				t.Fatal("retry replaced the wrapped primary")
			}
			b.armed = false
		})
	}
}

// fak-test:runtime medium est=2s lane=default
func TestV41QueryPrefixOwnedPairAndDecline(t *testing.T) {
	v41QueryPrefixRequireCPU(t)
	s, b := v41QueryPrefixWarm(t, false)
	fn := s.v41State().queryPrefix
	input := make([]float32, s.M.Cfg.HiddenSize)
	for i := range input {
		input[i] = float32(i%5+1) / 8
	}
	before := append([]float32(nil), input...)
	a, q, handled, err := fn(0, input)
	if !handled || err != nil || len(a) != len(input) || len(q) != s.M.Cfg.QLoraRank {
		t.Fatal("warm direct prefix failed", err)
	}
	wantA, wantQ := append([]float32(nil), a...), append([]float32(nil), q...)
	b.poison = true
	b.poisonBorrowed()
	if !reflect.DeepEqual(a, wantA) || !reflect.DeepEqual(q, wantQ) || !reflect.DeepEqual(input, before) {
		t.Fatal("published pair or input aliases borrowed storage")
	}
	if len(a) > 0 && len(q) > 0 {
		a[0] = 88
		if !reflect.DeepEqual(q, wantQ) {
			t.Fatal("published rows alias one another")
		}
	}
	for _, leaf := range []string{"attn_norm.weight", "attn.wq_a.weight", "attn.wq_a_norm.weight"} {
		t.Run("missing-"+leaf, func(t *testing.T) {
			name := layerName(0, leaf)
			saved := s.halW[name]
			delete(s.halW, name)
			defer func() { s.halW[name] = saved }()
			calls, uploads, reads, memo := b.calls, b.uploadCalls, b.readCalls, len(s.halW)
			a, q, handled, err := fn(0, input)
			if handled || err != nil || a != nil || q != nil || b.calls != calls || b.uploadCalls != uploads || b.readCalls != reads || len(s.halW) != memo {
				t.Fatal("missing memo staged, selected, or published")
			}
		})
	}
}

// fak-test:runtime medium est=1s lane=default
func TestV41QueryPrefixEligibility(t *testing.T) {
	v41QueryPrefixRequireCPU(t)
	s, b := v41QueryPrefixWarm(t, false)
	fn := s.v41State().queryPrefix
	input := make([]float32, s.M.Cfg.HiddenSize)
	declined := func(t *testing.T) {
		t.Helper()
		calls, uploads, reads, memo := b.calls, b.uploadCalls, b.readCalls, len(s.halW)
		a, q, handled, err := fn(0, input)
		if handled || err != nil || a != nil || q != nil || b.calls != calls || b.uploadCalls != uploads || b.readCalls != reads || len(s.halW) != memo || s.BackendSessionClosed() {
			t.Fatal("ineligible prefix did not decline before operation/memo mutation")
		}
	}
	t.Run("layer-norm", func(t *testing.T) {
		saved := s.M.Cfg.LayerNorm
		s.M.Cfg.LayerNorm = true
		defer func() { s.M.Cfg.LayerNorm = saved }()
		declined(t)
	})
	t.Run("gain-one-plus", func(t *testing.T) {
		saved := s.M.Cfg.NormGain1p
		s.M.Cfg.NormGain1p = true
		defer func() { s.M.Cfg.NormGain1p = saved }()
		declined(t)
	})
	t.Run("installed-lora", func(t *testing.T) {
		saved := s.M.lora
		s.M.lora = &LoRASet{}
		defer func() { s.M.lora = saved }()
		declined(t)
	})
	t.Run("active-prism", func(t *testing.T) {
		saved := s.M.prism
		s.M.prism = &prismHadamardState{weightWidth: map[string]int{layerName(0, "attn.wq_a.weight"): s.M.Cfg.HiddenSize}}
		defer func() { s.M.prism = saved }()
		declined(t)
	})
	t.Run("noncanonical-matrix", func(t *testing.T) {
		name := layerName(0, "attn.wq_a.weight")
		saved := s.M.manifest[name]
		changed := saved
		changed.Dtype = "F16"
		s.M.manifest[name] = changed
		defer func() { s.M.manifest[name] = saved }()
		declined(t)
	})
	for _, tc := range []struct {
		name  string
		value float64
	}{
		{"zero", 0}, {"negative", -1}, {"nan", math.NaN()},
		{"positive-infinity", math.Inf(1)}, {"negative-infinity", math.Inf(-1)},
		{"f32-underflow", math.SmallestNonzeroFloat64}, {"f32-overflow", math.MaxFloat64},
	} {
		t.Run("epsilon-"+tc.name, func(t *testing.T) {
			saved := s.M.Cfg.RMSNormEps
			s.M.Cfg.RMSNormEps = tc.value
			defer func() { s.M.Cfg.RMSNormEps = saved }()
			declined(t)
		})
	}
	for _, rotary := range []int{0, 1, s.M.Cfg.HeadDim + 2} {
		t.Run(fmt.Sprintf("invalid-rotary-%d", rotary), func(t *testing.T) {
			saved := s.M.Cfg.QKRopeHeadDim
			s.M.Cfg.QKRopeHeadDim = rotary
			defer func() { s.M.Cfg.QKRopeHeadDim = saved }()
			declined(t)
		})
	}
	t.Run("misaligned-canonical-offset", func(t *testing.T) {
		name := layerName(0, "attn.wq_a.weight")
		saved := s.M.manifest[name]
		changed := saved
		changed.Offset++
		s.M.manifest[name] = changed
		defer func() { s.M.manifest[name] = saved }()
		declined(t)
	})
	t.Run("capability-absent", func(t *testing.T) {
		b.available = false
		defer func() { b.available = true }()
		if s.v41QueryPrefixFunc() != nil {
			t.Fatal("absent capability bound a callback")
		}
	})
}
