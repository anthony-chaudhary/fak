package compute

import (
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestV41SharedAttentionGeometryAndStatus(t *testing.T) {
	t.Parallel()
	tensor := func(shape ...int) Tensor { return Tensor{Dtype: F32, Layout: RowMajor, Shape: shape} }
	q, kv, sink := tensor(2, 3), tensor(4, 3), tensor(2)
	if qb, kb, sb, fb, err := validateV41SharedAttention(q, kv, sink, 4, 2, 3, 0.25, V41SharedAttentionPlain, true); err != nil || qb != 24 || kb != 48 || sb != 8 || fb != 32 {
		t.Fatalf("valid geometry bytes=%d/%d/%d/%d err=%v", qb, kb, sb, fb, err)
	}
	if _, _, sb, _, err := validateV41SharedAttention(q, kv, tensor(1), 4, 2, 3, -0.25, V41SharedAttentionCompressed, false); err != nil || sb != 4 {
		t.Fatalf("absent-sink live dummy: bytes=%d err=%v", sb, err)
	}
	for _, dims := range [][3]int{{0, 2, 3}, {-1, 2, 3}, {4, 0, 3}, {4, 2, 0}, {1 << 30, 2, 3}, {1, 1 << 30, 1}, {1, 2, 1<<31 - 1}} {
		if _, _, _, _, err := validateV41SharedAttention(q, kv, sink, dims[0], dims[1], dims[2], 1, V41SharedAttentionPlain, true); err == nil {
			t.Errorf("accepted invalid rows/heads/dim %v", dims)
		}
	}
	for _, scale := range []float32{0, float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		if _, _, _, _, err := validateV41SharedAttention(q, kv, sink, 4, 2, 3, scale, V41SharedAttentionPlain, true); err == nil {
			t.Errorf("accepted invalid scale %g", scale)
		}
	}
	if _, _, _, _, err := validateV41SharedAttention(q, kv, sink, 4, 2, 3, 1, V41SharedAttentionMode(2), true); err == nil {
		t.Error("accepted unknown mode")
	}
	for operand := range 3 {
		for _, mutation := range []func(*Tensor){
			func(x *Tensor) { x.Shape = []int{12} },
			func(x *Tensor) { x.Dtype = BF16 },
			func(x *Tensor) { x.Layout = ColMajor },
			func(x *Tensor) { x.Quant = &QuantSpec{} },
		} {
			inputs := [3]Tensor{q, kv, sink}
			mutation(&inputs[operand])
			if _, _, _, _, err := validateV41SharedAttention(inputs[0], inputs[1], inputs[2], 4, 2, 3, 1, V41SharedAttentionPlain, true); err == nil {
				t.Errorf("accepted invalid operand %d", operand)
			}
		}
	}
	if _, _, _, _, err := validateV41SharedAttention(q, kv, sink, 4, 2, 3, 1, V41SharedAttentionPlain, false); err == nil {
		t.Error("absent sink accepted [heads] instead of a live [1] dummy")
	}
	if v41SharedAttentionPushConstantBytes != 24 || v41SharedAttentionStatusBytes != 16 || v41SharedAttentionStatusWords != 4 {
		t.Fatal("native wire size changed")
	}
	if err := decodeV41SharedAttentionStatus(make([]uint32, 8), 2, 4, 3); err != nil {
		t.Fatalf("zero success status: %v", err)
	}
	// Lower head wins even when its producer stage is later; NaN bits survive.
	words := []uint32{5, 3, 2, 0x7fc00123, 1, 1, 0, 0xff800000}
	err := fmt.Errorf("selected backend: %w", decodeV41SharedAttentionStatus(words, 2, 4, 3))
	var arithmetic *V41SharedAttentionArithmeticError
	if !errors.As(err, &arithmetic) || arithmetic.Stage != V41SharedAttentionStageValue || arithmetic.Head != 0 || arithmetic.SelectedSlot != 2 || arithmetic.Element != 1 || arithmetic.ValueBits != 0x7fc00123 {
		t.Fatalf("fault attribution lost: %v", err)
	}
	// Validate all records before returning an earlier arithmetic failure.
	words[4] = 7
	var protocol *V41SharedAttentionProtocolError
	if err := decodeV41SharedAttentionStatus(words, 2, 4, 3); !errors.As(err, &protocol) || protocol.Head != 1 {
		t.Fatalf("later malformed status was hidden: %v", err)
	}
	for _, record := range [][4]uint32{
		{1, 1, 0, 0x7f800000}, {2, 0, 0, 0x7f800000}, {2, 4, 0, 0x7f800000},
		{3, 0, 0, 0x7f800000}, {4, 2, 0, 0x7f800000}, {5, 4, 3, 0x7f800000}, {6, 0, 3, 0x7f800000},
	} {
		if err := decodeV41SharedAttentionStatus(record[:], 1, 4, 3); !errors.As(err, &arithmetic) || arithmetic.Stage != V41SharedAttentionStage(record[0]) {
			t.Errorf("valid stage record %v: %v", record, err)
		}
	}
	for _, record := range [][4]uint32{
		{0, 0, 0, 1}, {0, 1, 0, 0}, {7, 0, 0, 0x7f800000}, {1, 5, 0, 0x7f800000},
		{1, 0, 0, 0x7f800000}, {1, 1, 1, 0x7f800000}, {2, 1, 1, 0x7f800000},
		{3, 1, 0, 0x7f800000}, {4, 0, 0, 0x7f800000}, {5, 1, 0, 0x7f800000},
		{5, 1, 4, 0x7f800000}, {6, 1, 1, 0x7f800000}, {6, 0, 0, 0x7f800000}, {1, 1, 0, 0x3f800000},
	} {
		if err := decodeV41SharedAttentionStatus(record[:], 1, 4, 3); !errors.As(err, &protocol) {
			t.Errorf("malformed record %v accepted: %v", record, err)
		}
	}
	for _, n := range []int{0, 3, 5} {
		if err := decodeV41SharedAttentionStatus(make([]uint32, n), 1, 4, 3); !errors.As(err, &protocol) {
			t.Errorf("malformed status length %d accepted: %v", n, err)
		}
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestV41SharedAttentionPortableCorpus(t *testing.T) {
	t.Parallel()
	// These expected values are algebraic, independent of either scalar oracle.
	for _, tc := range []struct {
		name string
		kv   []float32
		sink []float32
		want []float32
	}{
		{"nil sink", []float32{2, -4, 6, 10}, nil, []float32{4, 3, 4, 3}},
		{"zero sink", []float32{2, -4, 6, 10}, []float32{0, 0}, []float32{8.0 / 3, 2, 8.0 / 3, 2}},
		{"unequal sinks", []float32{2, -4, 6, 10}, []float32{0, float32(math.Ln2)}, []float32{8.0 / 3, 2, 2, 1.5}},
		// The model gathers ordered slots [1,1,0,-1], excluding only padding.
		{"duplicates", []float32{6, 10, 6, 10, 2, -4}, []float32{0, 0}, []float32{3.5, 4, 3.5, 4}},
		// TopKLength=2 masks the final two original slots, retaining both B rows.
		{"length mask", []float32{6, 10, 6, 10}, []float32{0, 0}, []float32{4, 20.0 / 3, 4, 20.0 / 3}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, mode := range []V41SharedAttentionMode{V41SharedAttentionPlain, V41SharedAttentionCompressed} {
				got, err := v41SharedAttentionF32Control(make([]float32, 4), tc.kv, tc.sink, 2, 2, 0.75, mode)
				if err != nil {
					t.Fatal(err)
				}
				v41SharedAttentionNear(t, got, tc.want, 1e-6, 1e-6)
			}
		})
	}
	for _, dim := range []int{3, 512} {
		q, kv := make([]float32, 2*dim), make([]float32, 3*dim)
		for i := range q {
			q[i] = float32(i%13-6) / 37 // Deliberately not BF16-rounded.
		}
		for i := range kv {
			kv[i] = float32(i%9-4) / 16 // Independent exact BF16-widened literals.
		}
		for _, scale := range []float32{0.25, -0.375} {
			sink := []float32{-0.3125, 0.625}
			want := v41SharedAttentionFloat64Oracle(q, kv, sink, 2, dim, scale)
			for _, mode := range []V41SharedAttentionMode{V41SharedAttentionPlain, V41SharedAttentionCompressed} {
				got, err := v41SharedAttentionF32Control(q, kv, sink, 2, dim, scale, mode)
				if err != nil {
					t.Fatal(err)
				}
				// Frozen for this bounded corpus before any device measurement.
				v41SharedAttentionNear(t, got, want, 2e-5, 2e-5)
			}
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestV41SharedAttentionFiniteExtremeControls(t *testing.T) {
	t.Parallel()
	for _, mode := range []V41SharedAttentionMode{V41SharedAttentionPlain, V41SharedAttentionCompressed} {
		for _, sink := range [][]float32{nil, {-math.MaxFloat32}} {
			got, err := v41SharedAttentionF32Control([]float32{-math.MaxFloat32}, []float32{1}, sink, 1, 1, 1, mode)
			want := float32(1)
			if sink != nil {
				want = 0.5
			}
			if mode == V41SharedAttentionCompressed {
				want = 0
			}
			if err != nil || len(got) != 1 || got[0] != want {
				t.Fatalf("sentinel mode=%d sink=%v got=%v want=%g err=%v", mode, sink, got, want, err)
			}
		}
		for _, sink := range [][]float32{nil, {0}} {
			got, err := v41SharedAttentionF32Control([]float32{math.MaxFloat32}, []float32{1}, sink, 1, 1, 1, mode)
			if err != nil || len(got) != 1 || got[0] != 1 {
				t.Fatalf("finite huge score mode=%d sink=%v got=%v err=%v", mode, sink, got, err)
			}
		}
		for _, sign := range []float32{-1, 1} {
			got, err := v41SharedAttentionF32Control([]float32{sign * math.MaxFloat32}, []float32{2}, nil, 1, 1, 1, mode)
			var arithmetic *V41SharedAttentionArithmeticError
			if got != nil || !errors.As(err, &arithmetic) || arithmetic.Stage != V41SharedAttentionStageScore || arithmetic.Head != 0 || arithmetic.SelectedSlot != 0 || arithmetic.Element != -1 || arithmetic.ValueBits != math.Float32bits(float32(math.Inf(int(sign)))) {
				t.Fatalf("overflow must fail selected execution mode=%d sign=%g got=%v err=%v", mode, sign, got, err)
			}
		}
		// Both scores are finite; subtraction of -MaxFloat32 and +MaxFloat32
		// becomes -Inf. Rejecting the subtraction would incorrectly reject this.
		got, err := v41SharedAttentionF32Control([]float32{math.MaxFloat32}, []float32{-1, 1}, nil, 1, 1, 1, mode)
		if err != nil || len(got) != 1 || got[0] != 1 {
			t.Fatalf("valid exp(-Inf) mode=%d got=%v err=%v", mode, got, err)
		}
	}
	// Cancellation identifies an otherwise invisible FMA change in the score.
	a, b := math.Float32frombits(0x3f800001), math.Float32frombits(0x3f7ffffe)
	separate := v41SharedAttentionControlScore([]float32{a, 1}, []float32{b, -1}, 1)
	fused := float32(float64(a)*float64(b) - 1)
	if separate != 0 || fused == separate {
		t.Fatalf("FMA control separate=%g fused=%g", separate, fused)
	}
}

// fak-test:runtime fast est=1ms lane=default
// This is a source guard, not compilation, a SPIR-V receipt, or a device witness.
func TestV41SharedAttentionShaderSourceContract(t *testing.T) {
	t.Parallel()
	raw, err := os.ReadFile("shaders/v41_shared_attention.comp")
	if err != nil {
		t.Fatal(err)
	}
	s := string(raw)
	for _, token := range []string{
		"layout(local_size_x = 1) in;",
		"binding = 0) readonly buffer QBuf { float Q[]; }", "binding = 1) readonly buffer KVBuf { float KV[]; }",
		"binding = 2) readonly buffer SinkBuf { float Sink[]; }", "binding = 3) buffer OutBuf { float Out[]; }",
		"binding = 4) writeonly buffer StatusBuf { uint Status[]; }",
		"int heads;\n    int headDim;\n    int selectedRows;\n    int mode;\n    int hasSink;\n    float scale;",
		"uintBitsToFloat(0xff800000u)", "uintBitsToFloat(0xff7fffffu)",
		"precise float product = Q[qBase + uint(d)] * KV[kvBase + uint(d)];", "precise float nextDot = dot + product;",
		"precise float score = dot * pc.scale;", "precise float shift = score - maximum;",
		"precise float nextDenominator = denominator + term;", "precise float weight = term / denominator;",
		"precise float product = weight * KV[kvBase + uint(d)];", "precise float nextValue = Out[outBase + uint(d)] + product;",
		"fault(head, 1u, slot, -1, score)", "fault(head, 2u, -1, -1, term)", "fault(head, 2u, slot, -1, term)",
		"fault(head, 3u, -1, -1, denominator)", "fault(head, 4u, slot, -1, weight)",
		"fault(head, 5u, slot, d, nextValue)", "fault(head, 6u, -1, d, value)",
		"Status[statusBase] = 0u;", "Status[statusBase + 1u] = 0u;", "Status[statusBase + 2u] = 0u;", "Status[statusBase + 3u] = 0u;",
	} {
		if !strings.Contains(s, token) {
			t.Errorf("missing shader contract %q", token)
		}
	}
	for _, forbidden := range []string{"fma(", "float16_t", "atomic", "subgroup", "shared ", "finiteValue(shift)"} {
		if strings.Contains(s, forbidden) {
			t.Errorf("unreviewed shader behavior %q", forbidden)
		}
	}
	if strings.Count(s, "binding = ") != 5 || strings.Count(s, "for (int slot = 0;") != 3 {
		t.Fatal("five bindings or three ordered row passes changed")
	}
	if check, skip := strings.Index(s, "if (!finiteValue(score))"), strings.Index(s, "if (maximum == seed) return;"); check < 0 || skip <= check {
		t.Fatal("compressed sentinel skipped score validation")
	}
}

// This mathematical oracle evaluates logits/normalization in binary64 with
// stored probabilities, independently of the production three-pass code. It is
// used only away from the explicitly separate compressed sentinel behavior.
func v41SharedAttentionFloat64Oracle(q, kv, sink []float32, heads, dim int, scale float32) []float32 {
	rows := len(kv) / dim
	out := make([]float32, heads*dim)
	for h := range heads {
		logits := make([]float64, rows)
		maximum := math.Inf(-1)
		if sink != nil {
			maximum = float64(sink[h])
		}
		for r := range rows {
			for d := range dim {
				logits[r] += float64(q[h*dim+d]) * float64(kv[r*dim+d])
			}
			logits[r] *= float64(scale)
			maximum = math.Max(maximum, logits[r])
		}
		var denominator float64
		if sink != nil {
			denominator = math.Exp(float64(sink[h]) - maximum)
		}
		for r := range logits {
			logits[r] = math.Exp(logits[r] - maximum)
			denominator += logits[r]
		}
		for d := range dim {
			var value float64
			for r, probability := range logits {
				value += probability / denominator * float64(kv[r*dim+d])
			}
			out[h*dim+d] = float32(value)
		}
	}
	return out
}

func v41SharedAttentionControlScore(q, kv []float32, scale float32) float32 {
	var dot float32
	for d := range q {
		product := float32(q[d] * kv[d])
		dot = float32(dot + product)
	}
	return float32(dot * scale)
}

// This separate rounded control pins algorithmic order and safety tightening.
// It never calls model attention, selection, BF16, RoPE, or production helpers;
// it cannot be selected by a backend as an implementation or fallback.
func v41SharedAttentionF32Control(q, kv, sink []float32, heads, dim int, scale float32, mode V41SharedAttentionMode) ([]float32, error) {
	rows := len(kv) / dim
	out := make([]float32, heads*dim)
	finite := func(v float32) bool { return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) }
	exp := func(v float32) float32 { return float32(math.Exp(float64(v))) }
	for h := range heads {
		fault := func(stage V41SharedAttentionStage, slot, element int, value float32) error {
			return &V41SharedAttentionArithmeticError{Stage: stage, Head: h, SelectedSlot: slot, Element: element, ValueBits: math.Float32bits(value)}
		}
		score := func(r int) float32 {
			return v41SharedAttentionControlScore(q[h*dim:(h+1)*dim], kv[r*dim:(r+1)*dim], scale)
		}
		seed := float32(math.Inf(-1))
		if mode == V41SharedAttentionCompressed {
			seed = -math.MaxFloat32
		}
		maximum := seed
		if sink != nil {
			maximum = sink[h]
		}
		for r := range rows {
			v := score(r)
			if !finite(v) {
				return nil, fault(V41SharedAttentionStageScore, r, -1, v)
			}
			if v > maximum {
				maximum = v
			}
		}
		if maximum == seed {
			continue
		}
		var denominator float32
		if sink != nil {
			denominator = exp(float32(sink[h] - maximum))
			if !finite(denominator) {
				return nil, fault(V41SharedAttentionStageExp, -1, -1, denominator)
			}
		}
		for r := range rows {
			term := exp(float32(score(r) - maximum))
			if !finite(term) {
				return nil, fault(V41SharedAttentionStageExp, r, -1, term)
			}
			denominator = float32(denominator + term)
		}
		if !finite(denominator) {
			return nil, fault(V41SharedAttentionStageDenominator, -1, -1, denominator)
		}
		if denominator == 0 {
			continue
		}
		for r := range rows {
			term := exp(float32(score(r) - maximum))
			if !finite(term) {
				return nil, fault(V41SharedAttentionStageExp, r, -1, term)
			}
			weight := float32(term / denominator)
			if !finite(weight) {
				return nil, fault(V41SharedAttentionStageWeight, r, -1, weight)
			}
			for d := range dim {
				product := float32(weight * kv[r*dim+d])
				value := float32(out[h*dim+d] + product)
				if !finite(value) {
					return nil, fault(V41SharedAttentionStageValue, r, d, value)
				}
				out[h*dim+d] = value
			}
		}
	}
	return out, nil
}

func v41SharedAttentionNear(t *testing.T, got, want []float32, absTol, relTol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length=%d want=%d", len(got), len(want))
	}
	var maxAbs, maxRel float64
	for i, value := range got {
		a, b := float64(value), float64(want[i])
		if math.IsNaN(a) || math.IsInf(a, 0) || math.IsNaN(b) || math.IsInf(b, 0) {
			t.Fatalf("nonfinite output/oracle at %d: %g/%g", i, a, b)
		}
		delta := math.Abs(a - b)
		maxAbs = math.Max(maxAbs, delta)
		maxRel = math.Max(maxRel, delta/math.Max(math.Abs(b), 1e-30))
		if delta > absTol+relTol*math.Abs(b) {
			t.Errorf("output[%d]=%g want=%g abs=%g tolerance=%g", i, a, b, delta, absTol+relTol*math.Abs(b))
		}
	}
	t.Logf("scalar control maxAbs=%g maxRel=%g", maxAbs, maxRel)
}
