package ggufload

import (
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"testing"
)

// fak-test:runtime fast est=50ms lane=default
func TestKQuantReferenceGolden(t *testing.T) {
	var doc struct {
		HalfInputs   []uint32 `json:"half_input_bits"`
		HalfExpected []uint16 `json:"half_expected_bits"`
		Pin          string   `json:"upstream_pin"`
		Fixtures     []struct {
			Name  string     `json:"name"`
			Input []uint32   `json:"input_bits"`
			Q4    string     `json:"q4_k_hex"`
			Q5    string     `json:"q5_k_hex"`
			Q6    string     `json:"q6_k_hex"`
			RMSE  [3]float64 `json:"reference_rmse"`
		} `json:"fixtures"`
	}
	b, err := os.ReadFile("testdata/kquant_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatal(err)
	}
	if doc.Pin != "83078fec0db82d6b5a00d9599062c38c39145755" || len(doc.Fixtures) < 9 {
		t.Fatal("incomplete pinned C reference cohort")
	}
	if len(doc.HalfInputs) != 16 || len(doc.HalfExpected) != 16 {
		t.Fatal("missing C half-boundary oracle")
	}
	for i, bits := range doc.HalfInputs {
		if got := kQuantHalf(math.Float32frombits(bits)); got != doc.HalfExpected[i] {
			t.Fatalf("half(%08x)=%04x, C=%04x", bits, got, doc.HalfExpected[i])
		}
	}
	for _, f := range doc.Fixtures {
		t.Run(f.Name, func(t *testing.T) {
			row := make([]float32, len(f.Input))
			for i, bits := range f.Input {
				row[i] = math.Float32frombits(bits)
			}
			for i, typ := range []TensorType{TensorQ4_K, TensorQ5_K, TensorQ6_K} {
				want := []string{f.Q4, f.Q5, f.Q6}[i]
				got, err := quantizeKRow(row, typ)
				if err != nil {
					t.Fatal(err)
				}
				if hex.EncodeToString(got) != want {
					t.Fatalf("%s encoded bytes differ from pinned C oracle", typ)
				}
				decoded, err := dequantF32(TensorInfo{Name: f.Name, Dims: []uint64{uint64(len(row))}, Type: typ}, got)
				if err != nil {
					t.Fatal(err)
				}
				var sum float64
				for j, v := range decoded {
					d := float64(v) - float64(row[j])
					sum += d * d
				}
				rmse := math.Sqrt(sum / float64(len(row)))
				if math.IsNaN(rmse) || rmse > f.RMSE[i]+math.Max(1e-30, f.RMSE[i]*1e-6) {
					t.Fatalf("%s RMSE %g exceeds reference-derived bound %g", typ, rmse, f.RMSE[i])
				}
			}
		})
	}
	for _, v := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)), math.MaxFloat32} {
		row := make([]float32, 256)
		row[7] = v
		for _, typ := range []TensorType{TensorQ4_K, TensorQ5_K, TensorQ6_K} {
			if _, err := quantizeKRow(row, typ); err == nil {
				t.Fatalf("%s accepted unsupported value %g", typ, v)
			}
		}
	}
	if _, err := quantizeKRow(make([]float32, 255), TensorQ4_K); err == nil {
		t.Fatal("accepted incomplete block")
	}
}
