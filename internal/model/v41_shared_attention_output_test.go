package model

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

// Exercise the production composer with synthetic typed-core transport.
// This does not qualify physical readback or the full session constructor.
func v41SharedAttentionOutputFixture(values []float32, failure error) (*v41SharedAttentionContinuation, v41SharedAttentionFunc, *[]float32) {
	var last []float32
	core := &v41SharedAttentionCore{}
	core.validated = func(_ int, _ *v41SharedAttentionRequest, _ *v41SharedAttentionValidation) (v41SharedAttentionResult, error) {
		last = append([]float32(nil), values...)
		return v41SharedAttentionResult{values: last}, failure
	}
	ordinary := func(_ int, _ v41SharedAttentionRequest) ([]float32, error) {
		return append([]float32(nil), values...), failure
	}
	b := &v41SharedAttentionBinding{ordinary: ordinary, native: v41BoundSharedAttention{core: core}}
	c := &v41SharedAttentionContinuation{binding: b, layer: 1, pos: 0, sourceRatio: 1, heads: 1, headDim: 2, scale: 1, live: func() bool { return true }}
	return c, ordinary, &last
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=1s lane=default
func TestV41SharedAttentionOutputOwnership(t *testing.T) {
	c, ordinary, last := v41SharedAttentionOutputFixture([]float32{math.Float32frombits(0x80000000), 2}, nil)
	invoke := func(c *v41SharedAttentionContinuation, callback v41SharedAttentionFunc) ([]float32, error) {
		return v41CombinedAttentionWithContinuation(1, 0, 1, 1, 2, []float32{1, 2}, nil, [][]float32{{3, 4}}, nil, nil, 1, &v41ResidentKVView{}, callback, c)
	}
	out, err := invoke(c, ordinary)
	if err != nil || len(*last) != 2 || &out[0] != &(*last)[0] || math.Float32bits(out[0]) != 0x80000000 {
		t.Fatalf("owned output not transferred: %v %v", out, err)
	}
	borrowed := []float32{1, 2}
	got, err := invoke(nil, func(int, v41SharedAttentionRequest) ([]float32, error) { return borrowed, nil })
	if err != nil || &got[0] == &borrowed[0] {
		t.Fatalf("ordinary output was borrowed: %v", err)
	}
	borrowed[0] = 99
	if got[0] != 1 {
		t.Fatal("retained callback output changed result")
	}
	got[1] = 88
	if borrowed[1] != 2 {
		t.Fatal("returned result mutated callback storage")
	}
	c.live = func() bool { return false }
	before := &(*last)[0]
	got, err = invoke(c, ordinary)
	if err != nil || &(*last)[0] != before || math.Float32bits(got[0]) != 0x80000000 {
		t.Fatalf("revoked continuation selected owned core: %v", err)
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=1s lane=default
func TestV41SharedAttentionOutputValidation(t *testing.T) {
	sentinel := errors.New("selected transport fault")
	for _, tc := range []struct {
		name   string
		values []float32
		cause  error
	}{
		{"width", []float32{1}, nil}, {"nan", []float32{1, float32(math.NaN())}, nil},
		{"infinity", []float32{float32(math.Inf(1)), 2}, nil}, {"partial-error", []float32{1, 2}, sentinel},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, ordinary, _ := v41SharedAttentionOutputFixture(tc.values, tc.cause)
			call := func(c *v41SharedAttentionContinuation) ([]float32, error) {
				return v41CombinedAttentionWithContinuation(1, 0, 1, 1, 2, []float32{1, 2}, nil, [][]float32{{3, 4}}, nil, nil, 1, &v41ResidentKVView{}, ordinary, c)
			}
			got, ge := call(c)
			want, we := call(nil)
			if got != nil || want != nil || ge == nil || we == nil || ge.Error() != we.Error() || reflect.TypeOf(ge) != reflect.TypeOf(we) || errors.Is(ge, sentinel) != errors.Is(we, sentinel) {
				t.Fatalf("got %v/%v want %v/%v", got, ge, want, we)
			}
		})
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=1s lane=default
func TestV41SharedAttentionOutputAllocations(t *testing.T) {
	c, ordinary, _ := v41SharedAttentionOutputFixture([]float32{1, 2}, nil)
	q := []float32{1, 2}
	window := [][]float32{{3, 4}}
	view := &v41ResidentKVView{}
	var output []float32
	call := func() {
		output, _ = v41CombinedAttentionWithContinuation(1, 0, 1, 1, 2, q, nil, window, nil, nil, 1, view, ordinary, c)
	}
	// Add precisely the former final clone to the identical canonical path.
	before := testing.AllocsPerRun(100, func() { call(); output = append([]float32(nil), output...) })
	after := testing.AllocsPerRun(100, call)
	if len(output) != 2 || before-after != 1 {
		t.Fatalf("output allocations former-clone=%g owned=%g", before, after)
	}
	t.Logf("composer allocations with former final clone=%g owned=%g", before, after)
}
