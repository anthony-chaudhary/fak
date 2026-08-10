package devcmd

import (
	"bytes"
	"flag"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/blastradius"
)

// updateBlastWitness regenerates the committed estimate witness from the live shell
// output rather than hand-authoring its bytes:
//
//	go test ./cmd/fak -run TestBlastEstimateWitnessGolden -update
var updateBlastWitness = flag.Bool("update", false, "regenerate the fak blast estimate witness golden")

// The done-condition for the blast-estimate join (#2715) is a captured
// `fak blast estimate --json` over a fixture that shows the AFFECTED leases/issues
// (whose declared tree intersects the broken package's dependency radius) beside the
// disjoint ones that are free to keep running. This pins that witness as a durable,
// human-inspectable regression artifact in the tree — a stronger form than an ephemeral
// commit-message citation — over the same b->a, c->b, d->x fixture graph the pure
// package uses. held-c (internal/c, in the radius) and issue 7001 (internal/a) land in
// the hold set; free-x (internal/x) and issue 7002 (internal/z) stay disjoint.
func TestBlastEstimateWitnessGolden(t *testing.T) {
	withBlastSeams(t, testBlastGraph, func(string, time.Time) ([]blastradius.Lease, error) {
		t.Fatal("live lease source must not be read when --leases is given")
		return nil, nil
	})

	const (
		leasesPath = "testdata/blast/leases.jsonl"
		issuesPath = "testdata/blast/issues.jsonl"
		goldenPath = "testdata/blast/estimate_witness.golden.json"
	)

	var stdout, stderr bytes.Buffer
	rc := RunBlast(&stdout, &stderr, []string{
		"estimate", "internal/a", "--json",
		"--leases", leasesPath, "--issues", issuesPath,
	})
	if rc != 0 {
		t.Fatalf("runBlast rc = %d, stderr=%s", rc, stderr.String())
	}
	got := stdout.Bytes()

	if *updateBlastWitness {
		if err := os.MkdirAll(filepath.Dir(goldenPath), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(goldenPath, got, 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote witness golden %s (%d bytes)", goldenPath, len(got))
		return
	}

	want, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("read witness golden (run with -update to create): %v", err)
	}
	// Normalize CRLF so a Windows checkout compares equal to the LF-authored golden.
	lf := func(b []byte) []byte { return bytes.ReplaceAll(b, []byte("\r\n"), []byte("\n")) }
	if !bytes.Equal(lf(got), lf(want)) {
		t.Fatalf("blast estimate witness drifted from %s.\n--- got ---\n%s\n--- want ---\n%s",
			goldenPath, got, want)
	}
}
