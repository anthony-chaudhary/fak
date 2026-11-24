package ggufload

import (
	"encoding/binary"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// randomInkling2Raw fills a [rows, cols] payload of typ with random codes and
// small finite f16 scales at each block's scale offsets.
func randomInkling2Raw(t *testing.T, r *rand.Rand, typ TensorType, rows, cols int) []byte {
	t.Helper()
	if typ == TensorNVFP4 {
		return randomInkling2NVFP4(r, rows, cols)
	}
	if typ == TensorF32 {
		raw := make([]byte, rows*cols*4)
		for i := 0; i < rows*cols; i++ {
			binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(float32(r.NormFloat64())))
		}
		return raw
	}
	rb, err := inkling2RowBytes(typ, cols)
	if err != nil {
		t.Fatal(err)
	}
	raw := make([]byte, rows*rb)
	r.Read(raw)
	var blk int
	var scaleOffs []int
	switch typ {
	case TensorQ8_0:
		blk, scaleOffs = 34, []int{0}
	case TensorIQ4_NL:
		blk, scaleOffs = 18, []int{0}
	case TensorQ4_K:
		blk, scaleOffs = 144, []int{0, 2}
	case TensorQ6_K:
		blk, scaleOffs = 210, []int{208}
	default:
		t.Fatalf("no generator for %s", typ)
	}
	for b := 0; b < len(raw); b += blk {
		for _, o := range scaleOffs {
			binary.LittleEndian.PutUint16(raw[b+o:], 0x2000+uint16(r.Intn(0x0C00))) // ~[0.008, 0.06]
		}
	}
	return raw
}

func TestInkling2DeviceWeightMatchesHostRows(t *testing.T) {
	be := compute.Default()
	r := rand.New(rand.NewSource(11))
	cases := []struct {
		typ        TensorType
		rows, cols int
		padK       int
	}{
		{TensorNVFP4, 7, 1152, 1152},
		{TensorNVFP4, 5, 2304, 2304},
		{TensorQ8_0, 6, 2304, 2304},
		{TensorIQ4_NL, 4, 1152, 1280},
		{TensorIQ4_NL, 3, 2304, 2304},
		{TensorQ4_K, 5, 512, 512},
		{TensorQ6_K, 5, 512, 512},
		{TensorF32, 9, 64, 64},
	}
	for _, c := range cases {
		raw := randomInkling2Raw(t, r, c.typ, c.rows, c.cols)
		rb, err := inkling2RowBytes(c.typ, c.cols)
		if err != nil {
			t.Fatal(err)
		}
		x := make([]float32, c.cols)
		for i := range x {
			x[i] = float32(r.NormFloat64())
		}
		want := make([]float32, c.rows)
		inkling2RowsMatVec(c.typ, raw, c.rows, c.cols, rb, want, x)
		w, padK, err := inkling2DeviceWeight(be, c.typ, raw, c.rows, c.cols)
		if err != nil {
			t.Fatalf("%s: %v", c.typ, err)
		}
		if padK != c.padK {
			t.Fatalf("%s [%d %d]: padK %d, want %d", c.typ, c.rows, c.cols, padK, c.padK)
		}
		got := make([]float32, c.rows)
		inkling2DeviceMatVec(be, w, c.cols, padK, got, x)
		row := make([]float32, c.cols)
		for o := range want {
			if err := inkling2DequantRow(c.typ, raw[o*rb:(o+1)*rb], row); err != nil {
				t.Fatal(err)
			}
			var mag float64
			for i, v := range row {
				mag += math.Abs(float64(v * x[i]))
			}
			// NVFP4 W4A16 rounds activations to BF16 (2^-8 relative).
			if d := math.Abs(float64(got[o] - want[o])); d > 1e-2*mag+1e-5 {
				t.Fatalf("%s [%d %d] row %d: device %v, host %v (|w||x| %v)", c.typ, c.rows, c.cols, o, got[o], want[o], mag)
			}
		}
	}
}

type inkling2FreeCounter struct {
	compute.Backend
	freed int
}

func (b *inkling2FreeCounter) Free(t compute.Tensor) { b.freed++; b.Backend.Free(t) }

type inkling2CloseCounter struct{ closed int }

func (c *inkling2CloseCounter) Close() error { c.closed++; return nil }

func TestInkling2BindDeviceSwapsPackedWeightsAndFreesOnClose(t *testing.T) {
	be := &inkling2FreeCounter{Backend: compute.Default()}
	r := rand.New(rand.NewSource(3))
	quant := func(typ TensorType, rows, cols int) *inkling2QuantMatrix {
		rb, _ := inkling2RowBytes(typ, cols)
		return &inkling2QuantMatrix{name: typ.String(), typ: typ, raw: randomInkling2Raw(t, r, typ, rows, cols), rows: rows, cols: cols, rowBytes: rb}
	}
	const experts, rows, cols = 3, 4, 128
	rb, _ := inkling2RowBytes(TensorNVFP4, cols)
	exps := &inkling2QuantExperts{typ: TensorNVFP4, raw: randomInkling2NVFP4(r, experts*rows, cols), experts: experts, rows: rows, cols: cols, rowLen: rb}
	hostExps := *exps
	router := quant(TensorF32, 8, 16)
	closer := &inkling2CloseCounter{}
	m := &Inkling2LoopModel{closer: closer, Weights: &model.Inkling2LoopWeights{
		Output: quant(TensorQ8_0, 5, 64),
		Layers: []model.Inkling2LoopLayerWeights{{Wq: quant(TensorQ8_0, 3, 64), Router: router, GateUp: exps}},
	}}
	if err := bindInkling2Device(m, be); err != nil {
		t.Fatal(err)
	}
	if _, ok := m.Weights.Output.(*inkling2DeviceMatrix); !ok {
		t.Fatalf("output stayed %T", m.Weights.Output)
	}
	lw := m.Weights.Layers[0]
	if lw.Router != model.Inkling2LoopMatrix(router) {
		t.Fatalf("small F32 router moved to %T", lw.Router)
	}
	dev, ok := lw.GateUp.(*inkling2DeviceExperts)
	if !ok || dev.Experts() != experts || dev.Rows() != rows || dev.Cols() != cols {
		t.Fatalf("experts = %T %+v", lw.GateUp, lw.GateUp)
	}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	for e := 0; e < experts; e++ {
		got, want := make([]float32, rows), make([]float32, rows)
		dev.ExpertMatVec(e, got, x)
		hostExps.ExpertMatVec(e, want, x)
		row := make([]float32, cols)
		for i := range got {
			if err := inkling2DequantRow(TensorNVFP4, hostExps.raw[(e*rows+i)*rb:(e*rows+i+1)*rb], row); err != nil {
				t.Fatal(err)
			}
			var mag float64
			for k, v := range row {
				mag += math.Abs(float64(v * x[k]))
			}
			if math.Abs(float64(got[i]-want[i])) > 1e-2*mag+1e-5 {
				t.Fatalf("expert %d row %d: device %v, host %v (|w||x| %v)", e, i, got[i], want[i], mag)
			}
		}
	}
	be.freed = 0 // per-call activation/output frees are not model tensors
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	// Two matrices plus ONE expert stack: the per-expert row views own nothing.
	if be.freed != 3 || closer.closed != 1 {
		t.Fatalf("Close freed %d tensors (want 3) and closed the mapping %d times", be.freed, closer.closed)
	}
}

func TestInkling2DeviceExpertsShareOneStack(t *testing.T) {
	be := compute.Default()
	r := rand.New(rand.NewSource(17))
	const experts, rows = 3, 4
	for _, c := range []struct {
		typ        TensorType
		cols, padK int
	}{
		{TensorNVFP4, 1152, 1152},
		{TensorIQ4_NL, 1152, 1280},
		{TensorQ8_0, 256, 256},
		{TensorQ4_K, 512, 512},
		{TensorQ6_K, 512, 512},
	} {
		rb, err := inkling2RowBytes(c.typ, c.cols)
		if err != nil {
			t.Fatal(err)
		}
		host := &inkling2QuantExperts{typ: c.typ, raw: randomInkling2Raw(t, r, c.typ, experts*rows, c.cols),
			experts: experts, rows: rows, cols: c.cols, rowLen: rb}
		s := &inkling2DeviceSet{be: be}
		got, err := s.experts(host)
		if err != nil {
			t.Fatalf("%s: %v", c.typ, err)
		}
		dev := got.(*inkling2DeviceExperts)
		if len(s.tensors) != 1 || dev.padK != c.padK || dev.Experts() != experts {
			t.Fatalf("%s: %d owned tensors (want 1), padK %d (want %d), %d experts", c.typ, len(s.tensors), dev.padK, c.padK, dev.Experts())
		}
		x := make([]float32, c.cols)
		for i := range x {
			x[i] = float32(r.NormFloat64())
		}
		row := make([]float32, c.cols)
		for e := 0; e < experts; e++ {
			gotY, want := make([]float32, rows), make([]float32, rows)
			dev.ExpertMatVec(e, gotY, x)
			host.ExpertMatVec(e, want, x)
			for i := range want {
				if err := inkling2DequantRow(c.typ, host.raw[(e*rows+i)*rb:(e*rows+i+1)*rb], row); err != nil {
					t.Fatal(err)
				}
				var mag float64
				for k, v := range row {
					mag += math.Abs(float64(v * x[k]))
				}
				if math.Abs(float64(gotY[i]-want[i])) > 1e-2*mag+1e-5 {
					t.Fatalf("%s expert %d row %d: device %v, host %v (|w||x| %v)", c.typ, e, i, gotY[i], want[i], mag)
				}
			}
		}
		if err := s.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestInkling2ExpertResidencyOrderCoreFirst(t *testing.T) {
	// The released schedule: core [6,18) x3 passes over 24 layers.
	hp := model.Inkling2LoopHParams{Layers: 24, CoreStart: 6, CoreEnd: 18, Passes: 3}
	got := inkling2ExpertResidencyOrder(hp, 24)
	want := []int{6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23}
	if !slices.Equal(got, want) {
		t.Fatalf("order %v, want %v", got, want)
	}
	// Visits per token never rise along the order, so every prefix that fits a
	// VRAM budget holds the most-read expert bytes.
	visits := func(l int) int {
		if l >= hp.CoreStart && l < hp.CoreEnd {
			return hp.Passes
		}
		return 1
	}
	for i := 1; i < len(got); i++ {
		if visits(got[i]) > visits(got[i-1]) {
			t.Fatalf("layer %d (%d visits) follows layer %d (%d visits)", got[i], visits(got[i]), got[i-1], visits(got[i-1]))
		}
	}
	plain := []int{0, 1, 2, 3}
	for name, hp := range map[string]model.Inkling2LoopHParams{
		"single pass":     {Layers: 4, CoreStart: 1, CoreEnd: 3, Passes: 1},
		"empty core":      {Layers: 4, CoreStart: 2, CoreEnd: 2, Passes: 3},
		"core past model": {Layers: 4, CoreStart: 2, CoreEnd: 9, Passes: 3},
		"negative start":  {Layers: 4, CoreStart: -1, CoreEnd: 2, Passes: 3},
	} {
		if got := inkling2ExpertResidencyOrder(hp, 4); !slices.Equal(got, plain) {
			t.Fatalf("%s: order %v, want layer order", name, got)
		}
	}
}
