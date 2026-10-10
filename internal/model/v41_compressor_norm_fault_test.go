package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func v41CompressorNormAssertFailure(t *testing.T, s *Session, err error, layer int, stage string) *BackendForwardOperationError {
	t.Helper()
	var selected *V41ProjectionOperationError
	var closed *BackendForwardOperationError
	if !errors.As(err, &selected) || selected.Leaf != v41CompressorNormTestLeaf || selected.Layer != layer || selected.Stage != string(v41StageCompress) ||
		!errors.As(err, &closed) || closed.Forward != ForwardPathKind("deepseek41") || closed.Path != "v41-compressor-norm" || closed.Layer != layer || closed.Stage != stage ||
		!s.BackendSessionClosed() || s.halFailure != closed {
		t.Fatalf("selected normalization lost compress/path/phase/close identity: %v", err)
	}
	return closed
}

func v41CompressorNormAssertNoTransient(t *testing.T, b *v41CompressorNormRecorder) {
	t.Helper()
	if len(b.outputs) != 0 {
		t.Fatal("normalization output survived release")
	}
	for buffer := range b.live {
		if !b.weights[buffer] {
			t.Fatal("selected normalization leaked a transient buffer")
		}
	}
}

// Canonical checks use bit identity, including the sign of zero. Both cast
// overflow cases start finite, distinguishing post-cast checks from payload checks.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=300ms lane=default
func TestV41CompressorNormSelectedOperandFailure(t *testing.T) {
	t.Parallel()
	for _, name := range []string{"payload-width", "payload-nan", "gain-width", "gain-value", "gain-signed-zero", "gain-nan", "epsilon-bits", "BF16 input"} {
		t.Run(name, func(t *testing.T) {
			gain := []float32{0, -1, 0.5, 2}
			input := []float32{1, 2, 3, 4}
			eps := float32(1e-5)
			m := v41CompressorNormTinyModel(t, gain, eps)
			s, b := v41CompressorNormTestSession(t, m)
			normalize := s.v41CompressorNormFunc()
			if normalize == nil {
				t.Fatal("test adapter not selected")
			}
			stage := "canonical operands"
			switch name {
			case "payload-width":
				input = input[:3]
				stage = "payload"
			case "payload-nan":
				input[0] = float32(math.NaN())
				stage = "payload"
			case "gain-width":
				gain = gain[:3]
			case "gain-value":
				gain[1] = 1
			case "gain-signed-zero":
				gain[0] = math.Float32frombits(0x80000000)
			case "gain-nan":
				gain[0] = float32(math.NaN())
			case "epsilon-bits":
				eps = math.Float32frombits(math.Float32bits(eps) + 1)
			case "BF16 input":
				input[0] = math.MaxFloat32
				stage = "BF16 input"
			}
			if name == "BF16 input" && (!finite32(input[0]) || finite32(v41CompressorNormRefCast(input[0]))) {
				t.Fatal("input cast-overflow control is vacuous")
			}
			got, err := normalize(0, input, gain, eps)
			closed := v41CompressorNormAssertFailure(t, s, err, 0, stage)
			if got != nil || len(b.records) != 0 || b.readbacks != 0 || b.uploadBytes != 0 {
				t.Fatal("selected operand/cast failure dispatched or returned a row")
			}
			calls := b.io
			if recovered := v41CompressorTestRecover(func() { _, _ = normalize(0, input, gain, eps) }); recovered != closed || b.io != calls {
				t.Fatal("closed operand failure retried or lost its exact latch")
			}
		})
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=300ms lane=default
func TestV41CompressorNormSelectedDeviceFailure(t *testing.T) {
	t.Parallel()
	for _, site := range []string{"weight upload", "activation upload", "rmsnorm", "malformed output", "readback", "short readback", "nonfinite readback", "BF16 output"} {
		t.Run(site, func(t *testing.T) {
			gain, input, eps := []float32{1, -1, 0.5, 2}, []float32{1, 2, 3, 4}, float32(1e-5)
			m := v41CompressorNormTinyModel(t, gain, eps)
			s, b := v41CompressorNormTestSession(t, m)
			cause := &compute.BackendError{Backend: "norm-recorder", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
			b.faultLayer, b.faultSite, b.fault = 0, site, cause
			normalize := s.v41CompressorNormFunc()
			if normalize == nil {
				t.Fatal("test adapter not selected")
			}
			got, err := normalize(0, input, gain, eps)
			stage := site
			switch site {
			case "malformed output":
				stage = "rmsnorm"
			case "short readback", "nonfinite readback":
				stage = "readback"
			}
			closed := v41CompressorNormAssertFailure(t, s, err, 0, stage)
			if got != nil {
				t.Fatal("selected device failure returned a latent")
			}
			if site == "weight upload" || site == "activation upload" || site == "rmsnorm" || site == "readback" {
				if !errors.Is(err, cause) {
					t.Fatal("normalization lost exact backend cause")
				}
			}
			v41CompressorNormAssertNoTransient(t, b.v41CompressorNormRecorder)
			calls, norms := b.io, len(b.records)
			if recovered := v41CompressorTestRecover(func() { _, _ = normalize(0, input, gain, eps) }); recovered != closed || b.io != calls || len(b.records) != norms {
				t.Fatal("closed selected device failure retried or changed latch")
			}
		})
	}
}

// A fault in layer 1 follows layer 0's newly published group. The whole Session
// boundary must roll back both layers and must not replay history on the host.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=3s lane=default
func TestV41CompressorNormSelectedFailure(t *testing.T) {
	t.Parallel()
	for _, site := range []string{"rmsnorm", "short readback", "BF16 output"} {
		t.Run(site, func(t *testing.T) {
			m := v41CompressorProducerTestFixture(t)
			s, b := v41CompressorNormTestSession(t, m)
			s.Prefill([]int{1, 2, 3})
			before := captureV41ForwardSnapshot(s.v41Forward)
			phaseBefore := v41CompressorTestPhase(t, m, "decode")
			beforeNorms := len(b.records)
			b.live = map[compute.Buffer]bool{}
			b.faultLayer, b.faultSite = 1, site
			b.fault = &compute.BackendError{Backend: "norm-recorder", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
			b.beforeFault = func() {
				rows, _ := s.v41Forward.attn.KVSourceRows(0)
				if len(rows) != 2 {
					t.Fatal("late failure did not follow an earlier source publication")
				}
			}
			err := recoverError(func() { s.Step(4) })
			stage := site
			if site == "short readback" {
				stage = "readback"
			}
			closed := v41CompressorNormAssertFailure(t, s, err, 1, stage)
			if len(b.records) != beforeNorms+2 || b.records[beforeNorms].layer != 0 || b.records[beforeNorms+1].layer != 1 {
				t.Fatal("selected late failure skipped or repeated a complete group")
			}
			if !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
				t.Fatal("late normalization failure committed history, retained rows, publications, or selections")
			}
			v41CompressorNormAssertNoTransient(t, b.v41CompressorNormRecorder)
			delta := v41DenseTestDelta(v41CompressorTestPhase(t, m, "decode"), phaseBefore)
			if delta["compressor_projection_host_calls"] != 0 || delta["compressor_projection_host_rows"] != 0 || delta["compressor_projection_device_calls"] <= 0 {
				t.Fatalf("selected normalization failure fell into historical host replay: %v", delta)
			}
			calls, norms, projections := b.io, len(b.records), len(b.ops)
			ledger := v41CompressorTestPhase(t, m, "decode")
			underlying := b.Backend
			b.Backend = nil // Any accidental backend reuse now fails loudly.
			defer func() { b.Backend = underlying }()
			for _, entry := range []func(){func() { s.Step(4) }, func() { s.Prefill([]int{4, 5}) }} {
				if recovered := v41CompressorTestRecover(entry); recovered != closed {
					t.Fatal("closed Session failed to preserve selected operation pointer")
				}
				if b.io != calls || len(b.records) != norms || len(b.ops) != projections || !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) || !reflect.DeepEqual(ledger, v41CompressorTestPhase(t, m, "decode")) {
					t.Fatal("closed Session changed backend work, continuation, or counters")
				}
			}
		})
	}
}

// The existing index helper publishes KV before projection. Verify rollback at
// the actual Session transaction, rather than assuming that helper is atomic.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=1s lane=default
func TestV41CompressorNormDownstreamIndexRollback(t *testing.T) {
	t.Parallel()
	m := v41CompressorProducerTestFixture(t)
	s, b := v41CompressorNormTestSession(t, m)
	s.Prefill([]int{1, 2, 3})
	before, norms := captureV41ForwardSnapshot(s.v41Forward), len(b.records)
	weight := s.halW[layerName(0, "indexer.wk.weight")].Buf()
	if weight == nil {
		t.Fatal("source index weight was not staged")
	}
	cause := &compute.BackendError{Backend: "index-recorder", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
	b.faultWeight, b.failSite, b.fail = weight, "matmul", cause
	b.live = map[compute.Buffer]bool{}
	err := recoverError(func() { s.Step(4) })
	var selected *V41ProjectionOperationError
	if !errors.As(err, &selected) || selected.Leaf != "indexer.wk.weight" || !errors.Is(err, cause) || !s.BackendSessionClosed() || len(b.records) != norms+1 {
		t.Fatalf("index fault did not follow exactly one successful compressor normalization: %v", err)
	}
	if !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
		t.Fatal("downstream index failure retained normalized KV publication")
	}
	v41CompressorNormAssertNoTransient(t, b.v41CompressorNormRecorder)
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=1s lane=default
func TestV41CompressorNormPanicIdentity(t *testing.T) {
	t.Parallel()
	for _, alreadyClosed := range []bool{false, true} {
		t.Run(itoa(boolToIntV41Expert(alreadyClosed)), func(t *testing.T) {
			m := v41CompressorProducerTestFixture(t)
			s, b := v41CompressorNormTestSession(t, m)
			s.Prefill([]int{1, 2, 3})
			before := captureV41ForwardSnapshot(s.v41Forward)
			var cause any = &struct{ Marker string }{"unknown-compressor-norm-panic"}
			if alreadyClosed {
				failure := &BackendForwardOperationError{Backend: "already-closed", Forward: ForwardPathKind("deepseek41"), Path: "v41-compressor-norm", Layer: 1, Stage: "rmsnorm", Cause: ErrV41ForwardStage}
				cause = failure
				b.beforeFault = func() { s.halFailure = failure; s.Close() }
			}
			b.faultLayer, b.faultSite, b.fault = 1, "rmsnorm", cause
			if recovered := v41CompressorTestRecover(func() { s.Step(4) }); recovered != cause {
				t.Fatal("selected normalization changed unknown/already-closed panic identity")
			}
			if !s.BackendSessionClosed() || s.halFailure == nil || !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
				t.Fatal("selected panic did not close/latch and roll back Session")
			}
		})
	}
}
