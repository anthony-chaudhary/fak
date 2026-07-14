// Command wfmembench is the workflow-memory benchmark for issue #434. It scores
// three memory-substrate policies — full transcript, naive global summary, and
// fak's provenance-bound virtual views — over one deterministic finished session
// whose eight tool results encode the workflow-memory hazards the issue names:
// a clean result, a stale mutable source, two poisoned/sealed results, a tombstoned
// page, a multi-agent handoff, and a verified vs. unverified effect claim.
//
// It prints the comparison as JSON and, with -out DIR, writes wfmembench.json plus
// a markdown summary under DIR. The workload is model-free and deterministic, so the
// emitted comparison is byte-stable and reproducible — only the stamped lineage
// block (utc/git_commit/node, #9) varies between runs.
//
// Usage:
//
//	go run ./cmd/wfmembench                                # JSON to stdout
//	go run ./cmd/wfmembench -out experiments/wfmembench    # JSON + markdown artifacts
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/cdb"
	"github.com/anthony-chaudhary/fak/internal/contextq"
	"github.com/anthony-chaudhary/fak/internal/recall"
)

// reproCommand is the canonical command the markdown summary cites (acceptance #6).
const reproCommand = "go run ./cmd/wfmembench -out experiments/wfmembench"

func benchRequest() contextq.BenchRequest {
	return contextq.BenchRequest{
		GoalQuery:   "refund fee account",
		PoisonQuery: "sealed trust violation secret exfil",
	}
}

func main() {
	var outDir string
	flag.StringVar(&outDir, "out", "", "if set, write wfmembench.json + WFMEMBENCH-RESULTS.md under this directory")
	flag.Parse()

	if err := run(context.Background(), outDir); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// run is the benchmark body, split out of main so the deferred temp-dir reap below
// actually fires: main's error handling calls os.Exit, which skips defers, so a defer
// placed in main would leak on the marshal/write-artifacts error paths. Every exit
// leaves through run's return instead (#3508).
func run(ctx context.Context, outDir string) error {
	im, dir, err := attachFixtureImage(ctx)
	if err != nil {
		return fmt.Errorf("attach: %w", err)
	}
	// The attached image resolves entirely from memory (recall.Load reads the CAS
	// into RAM), so the on-disk wfmem-* tree can be reaped the moment the benchmark is
	// done with the image — bounding the per-invocation leak at the call site (#3508).
	defer os.RemoveAll(dir)

	// Exercise the tombstone hazard before scoring: the agent suppresses the
	// superseded preference page, so every arm but the full transcript drops it.
	tombstoneStalePreference(im)

	cmp := Compare(ctx, im, benchRequest(), reproCommand)

	js, err := benchcli.MarshalReport(cmp)
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}
	fmt.Println(string(js))

	if outDir != "" {
		if err := writeArtifacts(outDir, cmp, js); err != nil {
			return fmt.Errorf("write artifacts: %w", err)
		}
		fmt.Fprintf(os.Stderr, "wrote %s/{wfmembench.json,WFMEMBENCH-RESULTS.md}\n", outDir)
	}
	return nil
}

// attachFixtureImage ingests the embedded benchmark transcript, persists it as a
// core image, and attaches a debugger to it — the same ingest -> persist -> attach
// path a real recorded session takes. It returns the temp dir holding the persisted
// image so the caller can reap it after use; on any failure it removes the partial
// tree itself (returning dir="") so a failed attach never leaks either (#3508).
func attachFixtureImage(ctx context.Context) (im *cdb.Image, dir string, err error) {
	dir, err = os.MkdirTemp("", "wfmem-*")
	if err != nil {
		return nil, "", err
	}
	defer func() {
		if err != nil {
			os.RemoveAll(dir)
			dir = ""
		}
	}()
	src := filepath.Join(dir, "session.jsonl")
	if err = os.WriteFile(src, fixtureJSONL, 0o600); err != nil {
		return nil, "", err
	}
	rec, st, ierr := cdb.IngestSession(ctx, src, "wfmem-bench")
	if ierr != nil {
		err = ierr
		return nil, "", err
	}
	if st.Pages == 0 {
		err = fmt.Errorf("ingest recorded no pages")
		return nil, "", err
	}
	imgdir := filepath.Join(dir, "image")
	if err = rec.Persist(imgdir); err != nil {
		return nil, "", err
	}
	if im, err = cdb.Attach(imgdir); err != nil {
		return nil, "", err
	}
	return im, dir, nil
}

// recallTombstone builds a tombstone context-change request for one page step.
func recallTombstone(step int) recall.ContextChangeRequest {
	return recall.ContextChangeRequest{
		Action:      recall.ContextActionTombstone,
		Step:        step,
		Reason:      "stale preference superseded; suppress from model-visible recall",
		RequestedBy: "wfmembench",
	}
}

// writeArtifacts emits the #434 deliverable: the raw JSON and a markdown summary.
func writeArtifacts(dir string, cmp Comparison, js []byte) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "wfmembench.json"), append(js, '\n'), 0o644); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "WFMEMBENCH-RESULTS.md"), []byte(renderMarkdown(cmp)), 0o644)
}
