package main

// fak eve import -- the impure shell over internal/eveimport (#2606): folds a
// SAVED Eve observability artifact — an NDJSON session stream or an OpenTelemetry
// span export carrying `eve.*` / `$eve.*` attributes — into fak's session-ledger
// row shape and prints the compact operator summary (root session, turns,
// subagents, failed steps, token totals, and the evidence source path).
//
//	fak eve import [--json] [--kind ndjson|otel] [--include-bodies] FILE
//	    FILE is the saved artifact; --kind is inferred from the extension
//	    (.ndjson -> ndjson, .json -> otel) when omitted. The text summary goes
//	    to stdout and a one-line verdict to stderr; --json emits the typed Run
//	    artifact plus the joined ledger rows instead. Message/reasoning bodies
//	    are redacted by default; --include-bodies is a fixture/debugging opt-in.
//	    Exit 0 = a legible verdict was reconstructed, INCLUDING observation
//	    "partial" or "INDETERMINATE" — those are modelled outcomes, not errors.
//	    1 = FILE unreadable, 2 = usage.
//
// All impurity lives here (os.ReadFile, flags, exit codes); the fold itself is
// pure in internal/eveimport.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/eveimport"
)

func runEveImport(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("eve import", flag.ContinueOnError)
	fs.SetOutput(stderr)
	kind := fs.String("kind", "", "input wire format: ndjson or otel (default: inferred from FILE's extension)")
	jsonOut := fs.Bool("json", false, "emit the typed Run artifact plus joined ledger rows as JSON instead of the text summary")
	includeBodies := fs.Bool("include-bodies", false, "carry message/reasoning bodies verbatim (fixture/debugging only; bodies are redacted by default)")
	// parseFlagsOrHelp (not parseFlagsRejectArgs): this verb REQUIRES exactly one
	// positional FILE, and the reject-args helper treats any positional as a usage
	// error. -h/--help still exits 0 via the shared helper.
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, "usage: fak eve import [--json] [--kind ndjson|otel] [--include-bodies] FILE")
		return 2
	}
	file := fs.Arg(0)

	wire := *kind
	if wire == "" {
		switch strings.ToLower(filepath.Ext(file)) {
		case ".ndjson":
			wire = "ndjson"
		case ".json":
			wire = "otel"
		}
	}
	if wire != "ndjson" && wire != "otel" {
		fmt.Fprintf(stderr, "fak eve import: unknown --kind %q (want ndjson or otel)\n", *kind)
		return 2
	}

	data, err := os.ReadFile(file)
	if err != nil {
		fmt.Fprintf(stderr, "fak eve import: %v\n", err)
		return 1
	}

	// IncludeBodies is an explicit fixture/debugging opt-in: off by default, the
	// redaction witness (sha256 + byte count) is what the artifact carries.
	opt := eveimport.Options{IncludeBodies: *includeBodies}
	var run eveimport.Run
	if wire == "ndjson" {
		run = eveimport.ImportNDJSON(file, data, opt)
	} else {
		run = eveimport.ImportOTelSpans(file, data, opt)
	}

	if *jsonOut {
		v := struct {
			Run    eveimport.Run         `json:"run"`
			Ledger []eveimport.LedgerRow `json:"ledger_rows"`
		}{Run: run, Ledger: eveimport.JoinLedger(run)}
		return encodeJSONOrFail(stdout, stderr, v, "fak eve import: json")
	}

	fmt.Fprint(stdout, eveimport.Summary(run))
	fmt.Fprintln(stdout)
	fmt.Fprintf(stderr, "fak eve import: observation=%s sessions=%d evidence=%s\n",
		run.Observation, run.Sessions, run.Source.Path)
	return 0
}
