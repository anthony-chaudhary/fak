package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/bench"
)

func TestQuantTraceCLI(t *testing.T) {
	dir := t.TempDir()
	output := filepath.Join(dir, "receipt.json")
	input := filepath.Join(dir, "weights.raw")
	if err := os.WriteFile(input, bench.QuantTraceDemo(), 0600); err != nil {
		t.Fatal(err)
	}
	if run([]string{"--input", input, "--out", input}, &bytes.Buffer{}, &bytes.Buffer{}) == 0 {
		t.Fatal("accepted overwriting weights")
	}
	link := filepath.Join(dir, "weights-link.raw")
	if err := os.Link(input, link); err == nil {
		if run([]string{"--input", input, "--out", link}, &bytes.Buffer{}, &bytes.Buffer{}) == 0 {
			t.Fatal("accepted overwriting hardlinked weights")
		}
	}
	var stdout, stderr bytes.Buffer
	if code := run([]string{"--input", input, "--out", output, "--json"}, &stdout, &stderr); code != 0 {
		t.Fatalf("code %d: %s", code, stderr.String())
	}
	data, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(data, stdout.Bytes()) {
		t.Fatal("durable receipt differs from stdout")
	}
	var r bench.QuantTraceReceipt
	if err := json.Unmarshal(data, &r); err != nil {
		t.Fatal(err)
	}
	if !r.ExactRoundTrip || !r.ExactContraction {
		t.Fatal("failed witness")
	}
	stdout.Reset()
	if code := run(nil, &stdout, &stderr); code != 0 || !bytes.Contains(stdout.Bytes(), []byte("original pre-quantization floats")) {
		t.Fatalf("explanation: %d %s", code, stdout.String())
	}
	for _, args := range [][]string{{"--columns", "9223372036854775807"}, {"--samples", "-1"}, {"--samples", "no"}, {"--input", filepath.Join(dir, "missing")}, {"extra"}} {
		if run(args, &bytes.Buffer{}, &bytes.Buffer{}) == 0 {
			t.Fatalf("accepted %v", args)
		}
	}
}
