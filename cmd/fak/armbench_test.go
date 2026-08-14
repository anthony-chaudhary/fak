package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/armbench"
)

func TestArmbenchCLIEndToEnd(t *testing.T) {
	dir := t.TempDir()
	var out, stderr bytes.Buffer
	if code := runArmbench(&out, &stderr, []string{"emit-demo", "--dir", dir}); code != 0 {
		t.Fatalf("emit-demo code=%d stderr=%s", code, stderr.String())
	}

	manifest := filepath.Join(dir, "manifest.json")
	corpus := filepath.Join(dir, "corpus.json")
	ledger := filepath.Join(dir, "run.json")
	out.Reset()
	stderr.Reset()
	if code := runArmbench(&out, &stderr, []string{
		"run", "--manifest", manifest, "--corpus", corpus, "--out", ledger,
	}); code != 0 {
		t.Fatalf("run code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(out.String(), "baseline") || !strings.Contains(out.String(), "fak-ctxmmu") {
		t.Fatalf("human report omitted required arms:\n%s", out.String())
	}
	b, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatal(err)
	}
	run, err := armbench.UnmarshalRun(b)
	if err != nil {
		t.Fatalf("parse ledger: %v", err)
	}
	if _, err := armbench.Summarize(run); err != nil {
		t.Fatalf("summarize ledger: %v", err)
	}
}

func TestArmbenchCLIRefusesProviderManifestMismatch(t *testing.T) {
	dir := t.TempDir()
	var out, stderr bytes.Buffer
	if code := runArmbench(&out, &stderr, []string{"emit-demo", "--dir", dir}); code != 0 {
		t.Fatal(stderr.String())
	}
	out.Reset()
	stderr.Reset()
	code := runArmbench(&out, &stderr, []string{
		"run",
		"--manifest", filepath.Join(dir, "manifest.json"),
		"--corpus", filepath.Join(dir, "corpus.json"),
		"--out", filepath.Join(dir, "run.json"),
		"--provider", "not-the-manifest-provider",
	})
	if code != 3 || !strings.Contains(stderr.String(), armbench.ReasonIncomparableManifest) {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
}

func TestArmbenchCLIImporterRequiresExplicitCavemanLicenseReview(t *testing.T) {
	var out, stderr bytes.Buffer
	code := runArmbench(&out, &stderr, []string{
		"import-fixtures",
		"--suite", "caveman",
		"--store", t.TempDir(),
		"--json",
	})
	if code != 3 {
		t.Fatalf("code=%d stdout=%s stderr=%s", code, out.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), armbench.ReasonFixtureLicenseReview) {
		t.Fatalf("missing refusal reason: %s", stderr.String())
	}
	if !strings.Contains(stderr.String(), armbench.CavemanLicenseReviewToken) {
		t.Fatalf("missing exact revision-bound review token: %s", stderr.String())
	}
}

func TestArmbenchCommittedWitnessMatchesSelfcheck(t *testing.T) {
	res, err := armbench.Selfcheck()
	if err != nil {
		t.Fatal(err)
	}
	got, err := armbench.MarshalSelfcheck(res)
	if err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile(filepath.Join("..", "..", "docs", "_witnesses", "armbench-selfcheck-2026-08-13.json"))
	if err != nil {
		t.Fatalf("read committed fake-provider witness: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("committed fake-provider witness is stale; regenerate with `fak armbench selfcheck --json`")
	}
}

func TestArmbenchCavemanFactorialSpine(t *testing.T) {
	out := t.TempDir()
	var stdout, stderr bytes.Buffer
	if code := runArmbench(&stdout, &stderr, []string{"caveman-factorial", "--out", out, "--input", filepath.Join("..", "..", "docs", "_witnesses", "armbench-caveman-native", "inputs"), "--pressures", "1,4,12"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	b, err := os.ReadFile(filepath.Join(out, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(b, []byte(`"schema": "fak-armbench-caveman-factorial/1"`)) {
		t.Fatalf("wrong manifest: %s", b)
	}
}
