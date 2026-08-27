package model

import (
	"math"
	"testing"
)

func TestCompressColdKVKeepsHotAndRecoversOutliers(t *testing.T) {
	rows := make([][]float32, 12)
	for i := range rows {
		rows[i] = make([]float32, 64)
		for j := range rows[i] {
			rows[i][j] = float32(math.Sin(float64(i*64 + j)))
		}
		rows[i][7] = 100
	}
	window, receipt, err := CompressColdKV(rows, 4, 8, 0.3)
	if err != nil {
		t.Fatal(err)
	}
	got := window.Decompress()
	for i := 8; i < 12; i++ {
		for j := range rows[i] {
			if got[i][j] != rows[i][j] {
				t.Fatalf("hot row %d changed bit at %d", i, j)
			}
		}
	}
	if receipt.Engine != "fak-native" || receipt.HotRows != 4 || receipt.ColdRows != 8 || receipt.ResidualValues == 0 || receipt.MaxAbsoluteError > 0.3 {
		t.Fatalf("receipt=%+v", receipt)
	}
	if receipt.CompressedBytes >= receipt.OriginalBytes || receipt.ReadBytesAvoided <= 0 || receipt.BytesAvoidedPerAccepted <= 0 {
		t.Fatalf("no net byte win: %+v", receipt)
	}
}

func TestCompressColdKVCrossoverCanReject(t *testing.T) {
	rows := [][]float32{{-1, 1}, {-1, 1}}
	_, receipt, err := CompressColdKV(rows, 0, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	// Exact sparse residuals plus tiny-row metadata can cost more; receipt reports zero
	// avoided bytes so policy can keep the FP32 rollback instead of claiming a win.
	if receipt.CompressedBytes <= receipt.OriginalBytes || receipt.ReadBytesAvoided != 0 {
		t.Fatalf("expected reversal shape: %+v", receipt)
	}
}
