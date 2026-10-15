package model

import (
	"errors"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"math"
	"testing"
)

type v41ResidentBF16Recorder struct {
	*v41BF16PublishRecorder
	uploads, groupReads, rawChecks int
	fault                          string
	cause                          error
	verdict                        error
	attempts                       map[compute.Buffer]int
	born                           []compute.Tensor
	afterPublish                   func()
}

func newV41ResidentBF16Recorder() *v41ResidentBF16Recorder {
	b := &v41ResidentBF16Recorder{v41BF16PublishRecorder: newV41BF16PublishRecorder(), attempts: map[compute.Buffer]int{}, cause: errors.New("injected grouped transfer")}
	b.self = b
	return b
}
func (b *v41ResidentBF16Recorder) UploadClass(x compute.Tensor, dt compute.Dtype, c compute.MemoryClass, site string) compute.Tensor {
	b.uploads++
	return b.v41BF16PublishRecorder.UploadClass(x, dt, c, site)
}
func (b *v41ResidentBF16Recorder) SupportsV41DeferredStatus(n int) bool { return n > 0 }
func (b *v41ResidentBF16Recorder) V41BF16PublishDeferred(x compute.Tensor, site uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
	b.calls++
	values := append([]float32(nil), b.Backend.Read(x)...)
	b.verdict = nil
	for i, v := range values {
		if _, bad := v41BF16PublishFault([]float32{v}); bad && b.verdict == nil {
			b.verdict = &compute.V41BF16CheckedArithmeticError{Site: site, Index: i}
		}
		values[i] = math.Float32frombits(v41BF16BoundaryOracle(math.Float32bits(v)))
	}
	out := compute.NewF32(b, []int{len(values)}, values)
	record := compute.NewF32(b, []int{4}, make([]float32, 4))
	b.born = append(b.born, out, record)
	if b.fault == "output-input-alias" {
		b.born = b.born[:len(b.born)-2]
		b.Backend.Free(out)
		out = x
		b.born = append(b.born, record)
	}
	if b.fault == "status-output-alias" {
		b.born = b.born[:len(b.born)-1]
		b.Backend.Free(record)
		record = out
	}
	if b.fault == "output-shape" {
		out.Shape = []int{len(values) + 1}
	}
	if b.afterPublish != nil {
		b.afterPublish()
	}
	status := compute.V41DeferredStatus{Kind: compute.V41DeferredBF16Publish, Count: len(values), Site: site, Record: record}
	if b.fault == "publish-error" {
		return out, status, b.cause
	}
	return out, status, nil
}
func (b *v41ResidentBF16Recorder) V41ReadDeferred(st []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	b.groupReads++
	for _, s := range st {
		defer b.Free(s.Record)
	}
	if b.fault == "read" {
		return nil, nil, b.cause
	}
	out := make([][]float32, len(rows))
	for i, x := range rows {
		out[i] = b.Backend.Read(x)
		if len(rows) == 1 {
			out[i] = append([]float32(nil), out[i]...)
		}
	}
	return []error{b.verdict}, out, nil
}
func (b *v41ResidentBF16Recorder) Free(x compute.Tensor) {
	b.attempts[x.Buf()]++
	if b.fault == "free" && b.attempts[x.Buf()] == 1 {
		panic(b.cause)
	}
	for i := range b.Backend.Read(x) {
		b.Backend.Read(x)[i] = 91
	}
	b.v41DenseTestBackend.Free(x)
}
func (b *v41ResidentBF16Recorder) FlushBatch() {
	if b.fault == "flush" {
		panic(b.cause)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41BF16ResidentBinding(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	for _, mode := range []string{"success", "publish-error", "overflow", "overflow-before-nan", "read", "output-input-alias", "status-output-alias", "output-shape", "scope-change", "free", "flush"} {
		t.Run(mode, func(t *testing.T) {
			b := newV41ResidentBF16Recorder()
			b.fault = mode
			m := &Model{Cfg: Config{HiddenSize: 4, NumLayers: 1}}
			s := &Session{M: m, Backend: b}
			binding := s.v41BindBF16Publication()
			if binding == nil || binding.core == nil {
				t.Fatal("missing typed binding")
			}
			seed := []float32{1.00390625, math.Float32frombits(0x80000000), 1.0078125, 2}
			if mode == "overflow" || mode == "overflow-before-nan" {
				seed[0] = math.MaxFloat32
			}
			if mode == "overflow-before-nan" || mode == "flush" {
				seed[3] = float32(math.NaN())
			}
			input := compute.NewF32(b, []int{4}, append([]float32(nil), seed...))
			valid := true
			loan := &v41ProjectionLoan{backend: b, valid: func() bool { return valid }}
			if !loan.own(input) {
				t.Fatal("loan refused input")
			}
			primary := &BackendForwardOperationError{Backend: b.Name(), Path: "v41-indexer-projection", Layer: 0, Stage: "readback", Cause: errors.New("raw nonfinite")}
			resident := &v41BF16ResidentInput{tensor: input, loan: loan, valid: func() bool { return valid }, rawCheck: func(raw []float32) error {
				b.rawChecks++
				for _, v := range raw {
					if !finite32(v) {
						return primary
					}
				}
				return nil
			}}
			if mode == "scope-change" {
				b.afterPublish = func() { valid = false }
			}
			var got []float32
			var err error
			var panicValue any
			func() {
				defer func() { panicValue = recover() }()
				got, err = binding.core(0, "indexer.wq_b.weight projection", nil, resident)
			}()
			if mode == "success" {
				if err != nil || panicValue != nil || len(got) != len(seed) {
					t.Fatalf("success %v %v %v", got, err, panicValue)
				}
				for i, v := range seed {
					if math.Float32bits(got[i]) != v41BF16BoundaryOracle(math.Float32bits(v)) {
						t.Fatalf("output[%d] changed", i)
					}
				}
				if s.BackendSessionClosed() {
					t.Fatal("success closed owner")
				}
			} else {
				if len(got) != 0 || !s.BackendSessionClosed() {
					t.Fatalf("partial publication or live failed owner: %v %v %v", got, err, panicValue)
				}
				switch mode {
				case "overflow-before-nan", "flush":
					if err != primary {
						t.Fatalf("raw precedence lost: %v", err)
					}
				case "overflow":
					var arithmetic *compute.V41BF16CheckedArithmeticError
					var boundary *V41BF16BoundaryOperationError
					if !errors.As(err, &arithmetic) || arithmetic.Index != 0 || !errors.As(err, &boundary) {
						t.Fatalf("BF16 context lost: %v", err)
					}
				case "publish-error":
					if !errors.Is(err, b.cause) {
						t.Fatalf("publication cause lost: %v", err)
					}
				case "read":
					var outer *BackendForwardOperationError
					if !errors.As(err, &outer) || outer.Stage != "grouped projection/publication readback" {
						t.Fatalf("grouped stage lost: %v", err)
					}
					var grouped *V41BF16GroupedReadError
					if !errors.As(err, &grouped) || grouped.Layer != 0 || grouped.Site != "indexer.wq_b.weight projection" || !errors.Is(err, b.cause) {
						t.Fatalf("grouped cause lost: %v", err)
					}
				case "free":
					if panicValue != b.cause {
						t.Fatalf("cleanup panic changed: %v", panicValue)
					}
				}
			}
			if b.uploads != 0 || b.calls != 1 || b.groupReads > 1 {
				t.Fatalf("unexpected replay/upload: %d %d %d", b.uploads, b.calls, b.groupReads)
			}
			if mode == "success" && b.groupReads != 1 {
				t.Fatal("success needs exactly one grouped fence")
			}
			if len(loan.owned) != 0 || b.attempts[input.Buf()] != 1 {
				t.Fatal("loan not retired exactly once")
			}
			for _, x := range b.born {
				if b.attempts[x.Buf()] != 1 {
					t.Fatalf("temporary free attempts=%d", b.attempts[x.Buf()])
				}
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41BF16ResidentBindingReplacement(t *testing.T) {
	b := newV41ResidentBF16Recorder()
	s := &Session{M: &Model{Cfg: Config{HiddenSize: 4, NumLayers: 1}}, Backend: b}
	old := s.v41BindBF16Publication()
	marker := errors.New("opaque wrapper")
	replacement := old.withPublication(func(int, string, []float32) ([]float32, error) { return nil, marker })
	if replacement.core != nil || replacement.owner != nil || old.core == nil {
		t.Fatal("ordinary replacement inherited or mutated native authority")
	}
	if _, err := replacement.publicationFunc()(0, "arbitrary", nil); err != marker {
		t.Fatal("wrapper refusal lost")
	}
	// Raw ordinary callers still execute their ordinary upload/publication path.
	got, err := old.publicationFunc()(0, "indexer.wq_b.weight projection", []float32{1.1, 2.2, 3.3, 4.4})
	if err != nil || !v41BF16Widened(got) || b.uploads != 1 || b.groupReads != 1 {
		t.Fatalf("ordinary adapter changed: %v %v uploads=%d reads=%d", got, err, b.uploads, b.groupReads)
	}
}
