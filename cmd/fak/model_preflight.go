package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

const modelPreflightUsage = "usage: fak model preflight --gguf <path|shard1> [--hf-config <config.json>] [--json]"

// runModelPreflight checks a GGUF header (every shard, no tensor bytes) for unread
// keys, HF-config mismatches, missing/orphan tensors and dtypes without a host
// kernel, all in one pass. Exit 0 ready, 3 refused, 2 usage, 1 I/O.
func runModelPreflight(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("model preflight", flag.ContinueOnError)
	fs.SetOutput(stderr)
	gguf := fs.String("gguf", "", "GGUF checkpoint path (any shard of a split checkpoint)")
	hfConfig := fs.String("hf-config", "", "optional HF config.json the GGUF keys must agree with")
	asJSON := fs.Bool("json", false, "emit the fak.model-manifest-preflight/1 report as JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	*gguf = pathutil.ExpandTilde(*gguf)
	*hfConfig = pathutil.ExpandTilde(*hfConfig)
	if *gguf == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, modelPreflightUsage)
		return 2
	}
	f, err := ggufload.OpenHeaders(*gguf)
	if err != nil {
		fmt.Fprintf(stderr, "fak model preflight: %v\n", err)
		return 1
	}
	in := ggufload.ManifestPreflightInput{Path: *gguf, File: f}
	if *hfConfig != "" {
		if in.HFConfig, err = os.ReadFile(*hfConfig); err != nil {
			fmt.Fprintf(stderr, "fak model preflight: read hf config: %v\n", err)
			return 1
		}
	}
	report := ggufload.BuildManifestPreflight(in)
	if code := emitJSONOrRender(stdout, stderr, "fak model preflight", *asJSON, report, func(w io.Writer) {
		fmt.Fprint(w, report.Render())
	}); code != 0 {
		return code
	}
	if report.Refused() {
		return 3
	}
	return 0
}
