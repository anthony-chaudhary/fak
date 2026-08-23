package modelperfobs

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestKVCapacityRepositoryDogfoodReadout(t *testing.T) {
	root := filepath.Join("..", "..")
	cases := []struct {
		name    string
		file    string
		dialect KVDialect
	}{
		{name: "direct runtime", file: "kv-capacity-direct.json", dialect: KVDialectDirect},
		{name: "block runtime", file: "kv-capacity-block.json", dialect: KVDialectBlock},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join("testdata", tc.file))
			if err != nil {
				t.Fatal(err)
			}
			sample, err := DecodeKVMetricSample(data, tc.dialect)
			if err != nil {
				t.Fatal(err)
			}
			snapshot := NormalizeKVCapacity(sample, nil)
			var rendered bytes.Buffer
			if err := WriteKVCapacityMarkdown(&rendered, snapshot); err != nil {
				t.Fatal(err)
			}
			if snapshot.Normalized.ResidentTokens.Value == nil || *snapshot.Normalized.ResidentTokens.Value != 16384 || snapshot.Normalized.ResidentBytes.Value == nil || *snapshot.Normalized.ResidentBytes.Value != 2147483648 || snapshot.Normalized.Occupancy.Value == nil || *snapshot.Normalized.Occupancy.Value != 0.5 {
				t.Fatalf("normalized units = tokens=%v bytes=%v occupancy=%v", snapshot.Normalized.ResidentTokens, snapshot.Normalized.ResidentBytes, snapshot.Normalized.Occupancy)
			}
			if rendered.Len() == 0 {
				t.Fatal("empty dogfood readout")
			}
		})
	}

	readout, err := os.ReadFile(filepath.Join(root, "docs", "notes", "KV-CAPACITY-NORMALIZATION-DOGFOOD-2026-08-23.md"))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range [][]byte{[]byte("fak-kv-direct-metrics/1"), []byte("fak-kv-block-metrics/1"), []byte("16384"), []byte("2147483648"), []byte("0.5")} {
		if !bytes.Contains(readout, want) {
			t.Fatalf("dogfood readout missing %q", want)
		}
	}
}
