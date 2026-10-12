package computetrace

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// fak-test:runtime fast est=20ms lane=default
func TestActivationLlamaCppDumpDimensionOverflow(t *testing.T) {
	maxInt := int64(^uint(0) >> 1)
	for _, tc := range []struct {
		name        string
		tensor      string
		ne          [4]int64
		missingData bool
	}{
		{name: "byte_size_wrap", tensor: "qr-0", ne: [4]int64{1 << 62, 1, 1, 1}},
		{name: "width_product_wrap", tensor: "l_last-0", ne: [4]int64{1 << 62, 4, 1, 1}},
		{name: "token_product_wrap", tensor: "qr-0", ne: [4]int64{1, 1 << 62, 4, 1}},
		{name: "host_int_limit_before_read", tensor: "qr-0", ne: [4]int64{maxInt/4 + 1, 1, 1, 1}, missingData: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "tokens.json"), []byte(`{"tokens":[1]}`), 0o644); err != nil {
				t.Fatal(err)
			}
			entry, err := json.Marshal(llamaCppIndexEntry{
				Name: tc.tensor, Type: "f32", Ne: tc.ne[:], Dumped: true, File: "tensor.bin",
			})
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, "index.jsonl"), append(entry, '\n'), 0o644); err != nil {
				t.Fatal(err)
			}
			// Deliberately avoid allocating from the untrusted dimensions.
			if !tc.missingData {
				if err := os.WriteFile(filepath.Join(dir, "tensor.bin"), nil, 0o644); err != nil {
					t.Fatal(err)
				}
			}
			defer func() {
				if p := recover(); p != nil {
					t.Errorf("dimension overflow panicked: %v; want ErrLlamaCppDump", p)
				}
			}()
			art, err := ConvertLlamaCppDump(dir)
			if !errors.Is(err, ErrLlamaCppDump) {
				t.Fatalf("dimension overflow err=%v, want ErrLlamaCppDump", err)
			}
			if len(art.Events) != 0 || len(art.InputTokens) != 0 {
				t.Fatalf("malformed dimensions returned a partial artifact: %+v", art)
			}
		})
	}
}
