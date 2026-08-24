package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/benchcatalog"
)

// `fak bench-ingest`  -  the thin shell over internal/benchcatalog.Ingest: read one
// or more checked-in benchmark SNAPSHOT fixtures (official Terminal-Bench,
// SWE-bench, or FrontierSWE leaderboard exports) and fold them into a
// provenanced internal/modelscore registry, refusing any row that does not name
// its source, version, date, model, and metric. It is deliberately a shell: all
// the validation and the enum-to-confidence mapping live in the pure package, so
// this command only reads files, calls Ingest, and prints the result.
//
// It reads snapshots and NEVER touches the network  -  a snapshot is a committed
// fixture stamped with an observed_at date, not a live fetch, so this runs
// anywhere with no credentials (the same offline contract the rest of the
// benchmark surface holds).
//
//	fak bench-ingest snap1.json snap2.json ...   ingest and print the modelscore registry JSON
//	fak bench-ingest --check snap...             validate only; print a per-model row count, exit nonzero on refusal
//
// Wired as the top-level `bench-ingest` case in main.go, classified TierDev in
// internal/devindex/tiers.go with its manifest synopsis alongside.
func cmdBenchIngest(argv []string) { os.Exit(runBenchIngest(os.Stdout, os.Stderr, argv)) }

func runBenchIngest(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("bench-ingest", flag.ContinueOnError)
	fs.SetOutput(stderr)
	check := fs.Bool("check", false, "validate only: print a per-model row count and exit nonzero on any refused row, without emitting the registry")
	fs.Usage = func() {
		fmt.Fprint(stderr, `fak bench-ingest  -  fold benchmark snapshot fixtures into provenanced modelscore rows

usage:
  fak bench-ingest <snapshot.json> [more.json ...]   ingest and print the modelscore registry JSON
  fak bench-ingest --check <snapshot.json> ...        validate only; per-model row count; nonzero on refusal

Every row must name its source, benchmark version, capture date, model, and
metric; an under-provenanced row is refused loud. Snapshots are committed
fixtures (observed_at, not live) so this runs offline with no credentials.
`)
	}
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	files := fs.Args()
	if len(files) == 0 {
		fmt.Fprintln(stderr, "fak bench-ingest: at least one snapshot fixture path is required")
		fs.Usage()
		return 2
	}

	names := make([]string, 0, len(files))
	blobs := make([][]byte, 0, len(files))
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			fmt.Fprintf(stderr, "fak bench-ingest: read %s: %v\n", f, err)
			return 1
		}
		names = append(names, f)
		blobs = append(blobs, data)
	}

	reg, err := benchcatalog.IngestBytes(names, blobs...)
	if err != nil {
		// A refused row is the WHOLE point of the fail-closed gate; surface it and
		// exit nonzero so a bad fixture fails a CI check rather than half-populating.
		fmt.Fprintf(stderr, "fak bench-ingest: %v\n", err)
		return 1
	}

	models := reg.Models()
	if *check {
		for _, m := range models {
			prof, _ := reg.Profile(m)
			fmt.Fprintf(stdout, "%s\t%d rows\n", m, len(prof.Benchmarks))
		}
		fmt.Fprintf(stdout, "ok: %d model(s) ingested from %d snapshot(s)\n", len(models), len(files))
		return 0
	}

	if rc := encodeJSONOrFailPrefixed(stdout, stderr, reg, "fak bench-ingest: marshal registry"); rc != 0 {
		return rc
	}
	return 0
}
