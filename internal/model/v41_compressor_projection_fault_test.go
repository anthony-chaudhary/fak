package model

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func v41CompressorTestRecover(fn func()) (value any) {
	defer func() { value = recover() }()
	fn()
	return
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=5s lane=default
func TestV41CompressorProjectionSelectedFailureClosesAndRollsBack(t *testing.T) {
	t.Parallel()
	for _, leaf := range v41CompressorTestLeaves {
		for _, site := range []string{"upload", "matmul", "read", "malformed", "nonfinite"} {
			t.Run(leaf+"/"+site, func(t *testing.T) {
				m := v41CompressorProducerTestFixture(t)
				b := newV41CompressorTestBackend()
				s := v41DenseTestSession(t, m, b)
				s.Prefill([]int{1, 2, 3})
				b.faultWeight = v41CompressorTestWeight(t, s, 1, leaf)
				injectedUpload := false
				if site == "upload" {
					probeBackend := newV41CompressorTestBackend()
					probe := v41DenseTestSession(t, v41CompressorProducerTestFixture(t), probeBackend)
					probe.Prefill([]int{1, 2, 3})
					from, firstUpload := len(probeBackend.ops), len(probeBackend.activationBufs)
					probe.Step(4)
					weight := v41CompressorTestWeight(t, probe, 1, leaf)
					var input compute.Buffer
					for _, op := range probeBackend.ops[from:] {
						if op.weight == weight {
							input = op.input
							break
						}
					}
					ordinal := 0
					for index, buffer := range probeBackend.activationBufs[firstUpload:] {
						if buffer == input {
							ordinal = index + 1
							break
						}
					}
					if input == nil || ordinal == 0 {
						t.Fatal("compressor upload lacks an observed actual target input buffer")
					}
					attempts := 0
					b.faultUpload = func() bool { attempts++; injectedUpload = attempts == ordinal; return injectedUpload }
				}
				cause := &compute.BackendError{Backend: "compressor-recorder", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
				if site == "malformed" {
					b.malformed = true
				} else if site == "nonfinite" {
					b.nonfinite = true
				} else {
					b.failSite, b.fail = site, cause
				}
				b.live = map[compute.Buffer]bool{}
				before := captureV41ForwardSnapshot(s.v41Forward)
				phaseBefore := v41CompressorTestPhase(t, m, "decode")
				recovered := v41CompressorTestRecover(func() { s.Step(4) })
				if site == "upload" && !injectedUpload {
					t.Fatal("observed target compressor activation upload was not reached")
				}
				err, ok := recovered.(error)
				var selected *BackendForwardOperationError
				if !ok || !errors.As(err, &selected) || !errors.Is(err, ErrV41ForwardStage) || !s.BackendSessionClosed() {
					t.Fatalf("selected compressor failure must close with typed operation cause: %T", recovered)
				}
				if selected.Layer != 1 || selected.Stage == "" {
					t.Fatalf("selected compressor failure attributed to layer=%d stage=%s", selected.Layer, selected.Stage)
				}
				var architecture *V41ProjectionOperationError
				if !errors.As(err, &architecture) || architecture.Layer != 1 || architecture.Stage != string(v41StageCompress) {
					t.Fatal("selected compressor failure lost architecture stage attribution")
				}
				value := reflect.Indirect(reflect.ValueOf(selected))
				path := value.FieldByName("Path")
				if !path.IsValid() || path.Kind() != reflect.String || path.String() != "v41-compressor-projection" {
					t.Fatal("selected compressor operation path missing")
				}
				if site != "malformed" && site != "nonfinite" && !errors.Is(err, cause) {
					t.Fatal("selected compressor failure lost backend cause identity")
				}
				if !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
					t.Fatal("late producer compressor failure changed whole Session retained state")
				}
				if len(b.live) != 0 {
					t.Fatal("selected compressor failure leaked transient buffers")
				}
				delta := v41DenseTestDelta(v41CompressorTestPhase(t, m, "decode"), phaseBefore)
				if delta["compressor_projection_host_calls"] != 0 || delta["compressor_projection_host_rows"] != 0 || delta["compressor_projection_device_calls"] <= 0 {
					t.Fatalf("selected compressor failure retried on host or lost dispatch: %v", delta)
				}
				calls := b.apiCalls
				state := captureV41ForwardSnapshot(s.v41Forward)
				ledger := v41CompressorTestPhase(t, m, "decode")
				underlying := b.v41DenseTestBackend.Backend
				b.v41DenseTestBackend.Backend = nil
				defer func() { b.v41DenseTestBackend.Backend = underlying }()
				for _, entry := range []func(){func() { s.Step(4) }, func() { s.Prefill([]int{4, 5}) }} {
					if next := v41CompressorTestRecover(entry); next != selected {
						t.Fatal("closed Session changed selected error pointer")
					}
					if b.apiCalls != calls || !reflect.DeepEqual(state, captureV41ForwardSnapshot(s.v41Forward)) || !reflect.DeepEqual(ledger, v41CompressorTestPhase(t, m, "decode")) {
						t.Fatal("closed Session retried backend or mutated state/accounting")
					}
				}
			})
		}
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=2s lane=default
func TestV41CompressorProjectionUnknownAndClosedPanicIdentity(t *testing.T) {
	t.Parallel()
	for _, closed := range []bool{false, true} {
		t.Run(itoa(boolToIntV41Expert(closed)), func(t *testing.T) {
			m := v41CompressorProducerTestFixture(t)
			b := newV41CompressorTestBackend()
			s := v41DenseTestSession(t, m, b)
			s.Prefill([]int{1, 2, 3})
			b.faultWeight = v41CompressorTestWeight(t, s, 1, v41CompressorTestLeaves[1])
			var cause any = &struct{ Marker int }{13668}
			if closed {
				failure := &BackendForwardOperationError{Backend: "already-closed", Cause: ErrV41ForwardStage}
				cause = failure
				b.beforeFailure = func() { s.halFailure = failure; s.Close() }
			}
			b.failSite, b.fail = "matmul", cause
			before := captureV41ForwardSnapshot(s.v41Forward)
			if recovered := v41CompressorTestRecover(func() { s.Step(4) }); recovered != cause {
				t.Fatal("compressor selected boundary changed unknown/already-closed panic identity")
			}
			if !reflect.DeepEqual(before, captureV41ForwardSnapshot(s.v41Forward)) {
				t.Fatal("unknown compressor panic changed retained state")
			}
			if s.BackendSessionClosed() != closed {
				t.Fatal("unknown panic changed Session closure priority")
			}
		})
	}
}

type v41CompressorTestFailReader struct {
	cause error
	calls int
}

type v41CompressorTestCountReader struct {
	*bytes.Reader
	calls int
}

func (r *v41CompressorTestCountReader) ReadAt(dst []byte, offset int64) (int, error) {
	r.calls++
	return r.Reader.ReadAt(dst, offset)
}

type v41CompressorTestIntegratedBackend struct{ *v41CompressorTestBackend }

func (b *v41CompressorTestIntegratedBackend) Tier() string { return "integrated:software-witness" }

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=500ms lane=optin
func TestV41CompressorProjectionQuantHotCacheAndRelease(t *testing.T) {
	t.Setenv("FAK_Q4K_FREE_CPU", "1")
	for _, lazy := range []bool{false, true} {
		t.Run(itoa(boolToIntV41Expert(lazy)), func(t *testing.T) {
			const out, in = 512, 256
			name := layerName(0, v41CompressorTestLeaves[0])
			raw := buildRawQ4K(t, out, in, 1366815)
			for offset := 0; offset < len(raw); offset += q4kBlockBytes {
				binary.LittleEndian.PutUint16(raw[offset:], 0x2000)
				binary.LittleEndian.PutUint16(raw[offset+2:], 0x1800)
			}
			reader := &v41CompressorTestCountReader{Reader: bytes.NewReader(raw)}
			weight := quantizeQ4KFromRaw(raw, out, in)
			if lazy {
				weight = &q4kTensor{out: out, in: in, nblk: 1, lazy: &LazyQ4KRange{Reader: reader, Bytes: len(raw)}}
			}
			input := make([]float32, in)
			for col := range input {
				input[col] = float32(math.Sin(float64(col)*0.17)) * 0.2
			}
			decoded := make([]float32, out*in)
			dequantQ4KRef(decoded, raw)
			want := make([]float64, out)
			sumAbsTerms := make([]float64, out)
			for row := range want {
				for col := range input {
					term := float64(decoded[row*in+col]) * float64(input[col])
					want[row] += term
					sumAbsTerms[row] += math.Abs(term)
				}
			}
			// The two dot orders budget 2*in rounding steps plus scale/min operations for eight Q4 subgroups.
			const unitRoundoff = 1.0 / (1 << 24)
			gamma := float64(2*in+32) * unitRoundoff / (1 - float64(2*in+32)*unitRoundoff)
			m := v41CompressorProducerTestFixture(t)
			m.Cfg.HiddenSize = in
			delete(m.manifest, name)
			m.q4kw = map[string]*q4kTensor{name: weight}
			b := newV41CompressorTestBackend()
			integrated := &v41CompressorTestIntegratedBackend{b}
			s := &Session{M: m, Backend: integrated, halW: map[string]compute.Tensor{}, Q4K: true}
			t.Cleanup(s.Close)
			m.v41SetExpertFaultPhase(V41PhasePrefill)
			callback := s.v41DenseProjectionFunc()
			if callback == nil {
				t.Fatal("valid packed compressor owner lacks callback")
			}
			var cached compute.Buffer
			reads := 0
			before := v41CompressorTestPhase(t, m, "prefill")
			for call := 0; call < 2; call++ {
				got, outcome, err := callback(0, v41CompressorTestLeaves[0], input, out, in, 1)
				if err != nil || outcome != v41ProjectionHandled || s.BackendSessionClosed() {
					t.Fatalf("valid packed compressor projection was not handled: %T", err)
				}
				if len(got) != out {
					t.Fatal("valid packed compressor output width changed")
				}
				for row := range want {
					if math.IsNaN(float64(got[row])) || math.IsInf(float64(got[row]), 0) || math.IsNaN(want[row]) || math.IsInf(want[row], 0) {
						t.Fatal("packed compressor scalar parity requires finite values")
					}
					bound := gamma*sumAbsTerms[row] + unitRoundoff*math.Abs(want[row])
					if difference := math.Abs(float64(got[row]) - want[row]); difference > bound {
						t.Fatalf("packed compressor exceeds float32 dot error bound row=%d got=%g want=%g absdelta=%g sumabs=%g bound=%g", row, got[row], want[row], difference, sumAbsTerms[row], bound)
					}
				}
				buffer := v41CompressorTestWeight(t, s, 0, v41CompressorTestLeaves[0])
				if call == 0 {
					cached, reads = buffer, reader.calls
				} else if cached != buffer || reads != reader.calls {
					t.Fatal("packed compressor hot cache restaged or reread source")
				}
				if b.stages[buffer] != 1 {
					t.Fatal("packed compressor did not stage exactly one resident")
				}
				if lazy && weight.lazy == nil {
					t.Fatal("integrated release discarded lazy source descriptor")
				}
				if !lazy && len(weight.raw) != 0 {
					t.Fatal("integrated compressor staging retained released host packed bytes")
				}
			}
			if lazy && reads != 1 {
				t.Fatal("lazy compressor must read packed source exactly once")
			}
			delta := v41DenseTestDelta(v41CompressorTestPhase(t, m, "prefill"), before)
			if delta["compressor_projection_device_calls"] != 2 || delta["compressor_projection_device_rows"] != 2 || delta["compressor_projection_host_calls"] != 0 || delta["compressor_projection_host_rows"] != 0 {
				t.Fatal("packed hot-cache compressor lost separate selected accounting")
			}
		})
	}
}

func (r *v41CompressorTestFailReader) ReadAt(_ []byte, _ int64) (int, error) {
	r.calls++
	return 0, r.cause
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=500ms lane=default
func TestV41CompressorProjectionSelectedQuantStagingFailure(t *testing.T) {
	t.Parallel()
	for _, kind := range []string{"Q4_K-lazy-read", "Q2_K-truncated", "Q4_K-unsupported"} {
		t.Run(kind, func(t *testing.T) {
			defer func() {
				if value := recover(); value != nil {
					t.Fatalf("selected packed compressor callback unexpectedly panicked: %T %v", value, value)
				}
			}()
			const out, in = 512, 256
			leaf := v41CompressorTestLeaves[0]
			name := layerName(0, leaf)
			m := v41CompressorProducerTestFixture(t)
			m.Cfg.HiddenSize = in
			delete(m.manifest, name)
			cause := errors.New("independent compressor packed reader refusal")
			reader := &v41CompressorTestFailReader{cause: cause}
			if kind == "Q2_K-truncated" {
				weight := q2kFixtureTensor(out, in, 13668)
				weight.raw = weight.raw[:len(weight.raw)-1]
				m.kqw = map[string]*kQuantTensor{name: weight}
			} else {
				m.q4kw = map[string]*q4kTensor{name: {out: out, in: in, nblk: 1, lazy: &LazyQ4KRange{Reader: reader, Bytes: out * q4kBlockBytes}}}
			}
			b := newV41CompressorTestBackend()
			b.deny = kind == "Q4_K-unsupported"
			s := &Session{M: m, Backend: b, halW: map[string]compute.Tensor{}, Q4K: m.q4kw != nil}
			t.Cleanup(s.Close)
			callback := s.v41DenseProjectionFunc()
			if callback == nil {
				t.Fatal("device-capable owner did not install a projection callback")
			}
			_, outcome, err := callback(0, leaf, make([]float32, in), out, in, 1)
			if b.deny {
				if outcome != v41ProjectionDeclined || err != nil || s.BackendSessionClosed() || reader.calls != 0 || b.operationCalls != 0 || len(s.halW) != 0 || len(b.ops) != 0 {
					t.Fatal("unsupported packed compressor did not decline before selection/read")
				}
				return
			}
			var closed *BackendForwardOperationError
			if outcome != v41ProjectionError || !errors.As(err, &closed) || !s.BackendSessionClosed() {
				t.Fatalf("admitted packed compressor staging failure did not close owner: %T", err)
			}
			if closed.Path != "v41-compressor-projection" || closed.Layer != 0 || closed.Stage != "weight upload" {
				t.Fatal("packed compressor failure lost selected weight-upload identity")
			}
			if kind == "Q4_K-lazy-read" && (reader.calls == 0 || !errors.Is(err, cause)) {
				t.Fatal("packed compressor staging lost actual reader/cause identity")
			}
			if len(s.halW) != 0 || len(b.ops) != 0 {
				t.Fatal("failed packed compressor staging published cache or attempted MatMul")
			}
			calls := b.apiCalls
			original := b.v41DenseTestBackend.Backend
			b.v41DenseTestBackend.Backend = nil
			defer func() { b.v41DenseTestBackend.Backend = original }()
			if recovered := v41CompressorTestRecover(func() { s.Step(1) }); recovered != closed || b.apiCalls != calls {
				t.Fatal("closed packed compressor owner retried or changed error identity")
			}
		})
	}
}
