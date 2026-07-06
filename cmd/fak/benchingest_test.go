package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"
)

// ingestFixtures is the committed snapshot corpus the pure ingestor is tested
// against; the CLI test drives the exact same files through the shell so the
// witness covers the whole path (file -> ingest -> printed registry), not just
// the library. Path is relative from cmd/fak to the benchcatalog testdata.
const ingestFixtureDir = "../../internal/benchcatalog/testdata/modelscore-ingest"

func fix(name string) string { return filepath.Join(ingestFixtureDir, name) }

// TestBenchmarkIngestCLIEmitsRegistry is the CLI half of the Done-condition
// witness: `fak bench-ingest` over the three good snapshots prints a modelscore
// registry JSON carrying the ingested, provenanced rows and exits 0.
func TestBenchmarkIngestCLIEmitsRegistry(t *testing.T) {
	var out, errb bytes.Buffer
	code := runBenchIngest(&out, &errb, []string{
		fix("terminal-bench-2.1.json"),
		fix("swe-bench-verified.json"),
		fix("frontier-swe.json"),
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, errb.String())
	}
	got := out.String()
	// The printed registry must carry the schema tag, every benchmark id, and the
	// source URLs  -  provenance survives all the way to the emitted artifact.
	for _, want := range []string{
		"fak.modelscore-registry.v1",
		"terminal-bench",
		"swe-bench-verified",
		"frontier-swe",
		"tbench.ai",
		"swebench.com",
		"frontierswe.com",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("registry JSON missing %q; got:\n%s", want, got)
		}
	}
}

// TestBenchmarkIngestCLICheckCountsRows exercises the --check arm: validate only,
// print a per-model row count, exit 0 on a clean corpus.
func TestBenchmarkIngestCLICheckCountsRows(t *testing.T) {
	var out, errb bytes.Buffer
	code := runBenchIngest(&out, &errb, []string{
		"--check",
		fix("terminal-bench-2.1.json"),
		fix("swe-bench-verified.json"),
		fix("frontier-swe.json"),
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, errb.String())
	}
	got := out.String()
	if !strings.Contains(got, "opus\t4 rows") {
		t.Fatalf("--check output missing opus row count; got:\n%s", got)
	}
	if !strings.Contains(got, "ok: ") {
		t.Fatalf("--check output missing ok summary; got:\n%s", got)
	}
}

// TestBenchmarkIngestCLIRefusesUnderProvenanced is the fail-closed CLI witness: a
// snapshot with an under-provenanced row makes the command exit nonzero and name
// the missing field on stderr, so a bad fixture fails a CI check rather than
// silently ingesting a guessed number.
func TestBenchmarkIngestCLIRefusesUnderProvenanced(t *testing.T) {
	cases := []struct {
		fixture string
		wantSub string
	}{
		{"bad-missing-source.json", "source is required"},
		{"bad-terminal-no-harness.json", "harness is required"},
		{"bad-missing-version.json", "version is required"},
	}
	for _, tc := range cases {
		t.Run(tc.fixture, func(t *testing.T) {
			var out, errb bytes.Buffer
			code := runBenchIngest(&out, &errb, []string{fix(tc.fixture)})
			if code == 0 {
				t.Fatalf("%s: exit = 0, want nonzero (under-provenanced row must be refused)", tc.fixture)
			}
			if !strings.Contains(errb.String(), tc.wantSub) {
				t.Fatalf("%s: stderr = %q, want it to mention %q", tc.fixture, errb.String(), tc.wantSub)
			}
		})
	}
}

// TestBenchmarkIngestCLINoArgs guards the zero-arg usage path: it must exit with
// a usage error, not a panic or a silent success.
func TestBenchmarkIngestCLINoArgs(t *testing.T) {
	var out, errb bytes.Buffer
	code := runBenchIngest(&out, &errb, nil)
	if code == 0 {
		t.Fatalf("exit = 0, want a nonzero usage error for no snapshot paths")
	}
	if !strings.Contains(errb.String(), "snapshot fixture path is required") {
		t.Fatalf("stderr = %q, want a missing-path usage message", errb.String())
	}
}
