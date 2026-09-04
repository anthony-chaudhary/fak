package fp4runtime

import (
	"os"
	"path/filepath"
	"testing"
)

func BenchmarkFP4Runtime(b *testing.B) {
	matrixRaw, err := os.ReadFile(filepath.Join("testdata", "compatibility-matrix-v1.json"))
	if err != nil {
		b.Fatal(err)
	}
	requestRaw, err := os.ReadFile(filepath.Join("testdata", "nvfp4-blackwell-delegate.json"))
	if err != nil {
		b.Fatal(err)
	}

	var req Request
	if err := decodeStrict(requestRaw, &req); err != nil {
		b.Fatal(err)
	}
	var mat Matrix
	if err := decodeStrict(matrixRaw, &mat); err != nil {
		b.Fatal(err)
	}

	b.Run("ParseAndNegotiate", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			res, err := ParseAndNegotiate(requestRaw, matrixRaw)
			if err != nil {
				b.Fatal(err)
			}
			if res.Outcome != OutcomeDelegate {
				b.Fatalf("unexpected outcome: %v", res.Outcome)
			}
		}
	})

	b.Run("Negotiate", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			res := Negotiate(req, mat)
			if res.Outcome != OutcomeDelegate {
				b.Fatalf("unexpected outcome: %v", res.Outcome)
			}
		}
	})
}
