package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/supportgraph"
)

func writeFile(t *testing.T, dir, name string, raw []byte) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func inputs(t *testing.T) (dir, graphPath, witnessPath string, witness supportgraph.Witness) {
	t.Helper()
	dir = t.TempDir()
	graphRaw, err := os.ReadFile("../../internal/supportgraph/testdata/awq.json")
	if err != nil {
		t.Fatal(err)
	}
	observed := time.Date(2026, 8, 15, 20, 0, 0, 0, time.UTC)
	witness = supportgraph.Witness{
		Schema:         supportgraph.WitnessSchema,
		ID:             "cli-run-1",
		Tuple:          supportgraph.Tuple{Artifact: "a", Architecture: "arch", Quant: "q", Layout: "l", Backend: "b", Kernel: "k", Runtime: "cuda@12.9", Hardware: "h100-driver@580"},
		State:          supportgraph.Supported,
		Tier:           supportgraph.Witnessed,
		Authority:      "fak-lab",
		ArtifactDigest: "sha256:" + strings.Repeat("a", 64),
		Environment:    "gpu=h100,cuda=12.9,driver=580",
		Reproduce:      "go test ./internal/supportgraph",
		ObservedAt:     observed,
		Expires:        observed.Add(30 * 24 * time.Hour),
		SourceCommit:   "abc123",
	}
	witness.PayloadSHA256 = supportgraph.WitnessDigest(witness)
	witnessRaw, err := json.Marshal(witness)
	if err != nil {
		t.Fatal(err)
	}
	return dir, writeFile(t, dir, "graph.json", graphRaw), writeFile(t, dir, "witness.json", witnessRaw), witness
}

func TestRunUsageExitCodes(t *testing.T) {
	cases := map[string][]string{
		"no arguments":    nil,
		"missing witness": {"-graph", "g.json"},
		"missing graph":   {"-witness", "w.json"},
		"unknown flag":    {"-bogus"},
	}
	for name, args := range cases {
		t.Run(name, func(t *testing.T) {
			var out, errb bytes.Buffer
			if code := run(args, &out, &errb); code != 2 {
				t.Fatalf("exit=%d, want 2", code)
			}
			if out.Len() != 0 {
				t.Fatalf("stdout not empty: %q", out.String())
			}
		})
	}
}

func TestRunFailsClosedOnBadInputs(t *testing.T) {
	dir, graphPath, witnessPath, witness := inputs(t)
	tampered := witness
	tampered.PayloadSHA256 = strings.Repeat("0", 64)
	tamperedRaw, _ := json.Marshal(tampered)
	tamperedPath := writeFile(t, dir, "tampered.json", tamperedRaw)
	garbage := writeFile(t, dir, "garbage.json", []byte("{"))
	cases := map[string][]string{
		"missing graph":    {"-graph", filepath.Join(dir, "absent.json"), "-witness", witnessPath},
		"unparsable graph": {"-graph", garbage, "-witness", witnessPath},
		"missing witness":  {"-graph", graphPath, "-witness", filepath.Join(dir, "absent.json")},
		"digest mismatch":  {"-graph", graphPath, "-witness", tamperedPath},
	}
	for name, args := range cases {
		t.Run(name, func(t *testing.T) {
			var out, errb bytes.Buffer
			if code := run(args, &out, &errb); code != 1 {
				t.Fatalf("exit=%d, want 1", code)
			}
			if out.Len() != 0 {
				t.Fatalf("stdout not empty: %q", out.String())
			}
		})
	}
}

func TestRunIngestsWitnessToStdout(t *testing.T) {
	_, graphPath, witnessPath, witness := inputs(t)
	var out, errb bytes.Buffer
	if code := run([]string{"-graph", graphPath, "-witness", witnessPath}, &out, &errb); code != 0 {
		t.Fatalf("exit=%d stderr=%q", code, errb.String())
	}
	graph, err := supportgraph.Parse(out.Bytes())
	if err != nil {
		t.Fatalf("stdout is not a graph: %v", err)
	}
	if got := supportgraph.Query(graph, witness.Tuple, witness.ObservedAt.Add(time.Second)); got.State != supportgraph.Supported {
		t.Fatalf("ingested tuple state=%q", got.State)
	}
	if fields := strings.Fields(errb.String()); len(fields) < 3 || fields[0] != "INGEST" || fields[1] != witness.ID || fields[2] != "inserted=true" {
		t.Fatalf("receipt=%q", errb.String())
	}
}

func TestRunWritesGraphToOut(t *testing.T) {
	dir, graphPath, witnessPath, _ := inputs(t)
	outPath := filepath.Join(dir, "updated.json")
	var out, errb bytes.Buffer
	if code := run([]string{"-graph", graphPath, "-witness", witnessPath, "-out", outPath}, &out, &errb); code != 0 {
		t.Fatalf("exit=%d stderr=%q", code, errb.String())
	}
	if out.Len() != 0 {
		t.Fatalf("stdout not empty with -out: %q", out.String())
	}
	raw, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatal(err)
	}
	graph, err := supportgraph.Parse(raw)
	if err != nil {
		t.Fatalf("-out is not a graph: %v", err)
	}
	if len(graph.Edges) != 4 {
		t.Fatalf("edges=%d, want 4", len(graph.Edges))
	}
}
