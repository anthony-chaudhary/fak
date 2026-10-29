package model

import (
	"encoding/binary"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

func v41CompressorHostPairFixture() *Model {
	m := &Model{Cfg: Config{HiddenSize: 7, HeadDim: 4, RMSNormEps: 1e-5}, manifest: map[string]tensorMeta{}}
	for leaf, shape := range map[string][]int{"attn.compressor.wkv.weight": {4, 7}, "attn.compressor.wgate.weight": {4, 7}, "attn.compressor.norm.weight": {4}} {
		n := 1
		for _, dim := range shape {
			n *= dim
		}
		offset := len(m.raw)
		for i := 0; i < n; i++ {
			value := float32(i%9-4) / 16
			if len(shape) == 1 {
				value = 1
			}
			var b [4]byte
			binary.LittleEndian.PutUint32(b[:], math.Float32bits(value))
			m.raw = append(m.raw, b[:]...)
		}
		m.manifest[layerName(0, leaf)] = tensorMeta{Dtype: "f32", Shape: shape, Offset: offset, Nbytes: n * 4}
	}
	return m
}

func TestV41ColdCompressorHostPairs(t *testing.T) {
	for _, n := range []int{2, 3, 8, 9} {
		m := v41CompressorHostPairFixture()
		inputs := v41DecodeRows(n, m.Cfg.HiddenSize)
		want, err := m.v41CompressedRowsWithDeviceOps(0, 2, nil, inputs, nil, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		got, handled, err := m.v41ColdCompressorHostPairs(0, inputs, &v41ProjScratch{})
		if !handled || err != nil || !reflect.DeepEqual(got, want) {
			t.Fatalf("rows%d handled%t err%v result mismatch", n, handled, err)
		}
		live, err := m.v41ColdCompressedRows(0, 2, nil, inputs, true, nil, &v41ProjScratch{})
		if err != nil || !reflect.DeepEqual(live, want) {
			t.Fatalf("live composer: %v", err)
		}
		// Every emitted group remains independently owned after later pairs.
		if len(got) > 1 {
			saved := got[1][0]
			got[0][0] = -99
			if got[1][0] != saved {
				t.Fatal("group outputs alias")
			}
		}
	}
}

func TestV41ColdCompressorHostPairFirstError(t *testing.T) {
	for _, kind := range []string{"first KV", "norm", "bad second shape", "second KV"} {
		m := v41CompressorHostPairFixture()
		inputs := v41DecodeRows(3, m.Cfg.HiddenSize)
		switch kind {
		case "first KV":
			inputs[0][0] = float32(math.NaN())
			inputs[1][0] = float32(math.Inf(1))
		case "norm":
			m.tensor(layerName(0, "attn.compressor.norm.weight"))[0] = float32(math.NaN())
			inputs[1][0] = float32(math.NaN())
		case "bad second shape":
			inputs[0][0] = float32(math.NaN())
			inputs[1] = nil
		case "second KV":
			inputs[1][0] = float32(math.NaN())
		}
		_, want := m.v41CompressedRowsWithDeviceOps(0, 2, nil, inputs, nil, nil, nil)
		_, got := m.v41ColdCompressedRows(0, 2, nil, inputs, true, nil, &v41ProjScratch{})
		if want == nil || got == nil || want.Error() != got.Error() {
			t.Fatalf("%s: got%v want%v", kind, got, want)
		}
	}
}

func TestV41ColdCompressorHostPairAdmission(t *testing.T) {
	m := v41CompressorHostPairFixture()
	inputs := v41DecodeRows(2, m.Cfg.HiddenSize)
	check := func(scratch *v41ProjScratch) {
		t.Helper()
		if _, handled, err := m.v41ColdCompressorHostPairs(0, inputs, scratch); handled || err != nil {
			t.Fatal("callback/unsupported layout admitted")
		}
	}
	check(&v41ProjScratch{projectionRoute: newV41ProjectionRouteBinding(func(int, string, []float32, int, int, int) ([]float32, v41DenseProjectionOutcome, error) {
		panic("projection callback executed during admission")
	}, nil)})
	check(&v41ProjScratch{compressorNorm: func(int, []float32, []float32, float32) ([]float32, error) {
		panic("norm callback executed during admission")
	}})
	m.v41SetExpertTimeClock(func() int64 { panic("clock executed during admission") })
	check(&v41ProjScratch{})
	m.v41SetExpertTimeClock(nil)
	computetrace.SetObserver(func(computetrace.Event) { panic("observer executed during admission") })
	check(&v41ProjScratch{})
	computetrace.SetObserver(nil)
	name := layerName(0, v41ColdCompressorLeaves[0])
	meta := m.manifest[name]
	meta.Nbytes--
	m.manifest[name] = meta
	check(&v41ProjScratch{})
}
