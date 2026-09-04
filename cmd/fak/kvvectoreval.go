package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/kvvectoreval"
)

func cmdKVVectorEval(argv []string) {
	os.Exit(runKVVectorEval(os.Stdout, os.Stderr, argv))
}

func runKVVectorEval(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		kvVectorEvalUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "inspect":
		return runKVVectorEvalInspect(stdout, stderr, argv[1:])
	case "eval":
		return runKVVectorEvalEvaluate(stdout, stderr, argv[1:])
	case "verify-artifact":
		return runKVVectorEvalVerifyArtifact(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		kvVectorEvalUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak kvvectoreval: unknown subcommand %q\n", argv[0])
		kvVectorEvalUsage(stderr)
		return 2
	}
}

func runKVVectorEvalInspect(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak kvvectoreval inspect", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit output as JSON")
	fs.Usage = func() { kvVectorEvalUsage(stderr) }
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}

	req := kvvectoreval.Request{
		ContractID:        kvvectoreval.ContractID,
		PaperID:           kvvectoreval.PaperID,
		PaperPDFSHA256:    kvvectoreval.PaperPDFSHA256,
		PaperSourceSHA256: kvvectoreval.PaperSourceSHA256,
		RecipeID:          kvvectoreval.RecipeID,
		RuntimeID:         kvvectoreval.RuntimeID,
		RuntimeAvailable:  true,
	}
	res := kvvectoreval.Evaluate(req)

	if *asJSON {
		payload := struct {
			Schema    string                  `json:"schema"`
			Contract  string                  `json:"contract"`
			Paper     string                  `json:"paper"`
			Title     string                  `json:"title"`
			Recipe    string                  `json:"recipe"`
			Runtime   string                  `json:"runtime"`
			Artifacts []kvvectoreval.Artifact `json:"artifacts"`
			Metrics   []kvvectoreval.Metric   `json:"metrics"`
		}{
			Schema:    "fak.kvvectoreval.inspect/v1",
			Contract:  kvvectoreval.ContractID,
			Paper:     kvvectoreval.PaperID,
			Title:     kvvectoreval.PaperTitle,
			Recipe:    kvvectoreval.RecipeID,
			Runtime:   kvvectoreval.RuntimeID,
			Artifacts: res.Artifacts,
			Metrics:   res.Metrics,
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(payload); err != nil {
			fmt.Fprintf(stderr, "fak kvvectoreval inspect: encode json: %v\n", err)
			return 1
		}
		return 0
	}

	fmt.Fprintf(stdout, "NOVA-KV Attention-Preserving Vector-Quantization Contract: %s\n", kvvectoreval.ContractID)
	fmt.Fprintf(stdout, "  Paper:   %s (%s)\n", kvvectoreval.PaperID, kvvectoreval.PaperTitle)
	fmt.Fprintf(stdout, "  Recipe:  %s\n", kvvectoreval.RecipeID)
	fmt.Fprintf(stdout, "  Runtime: %s\n", kvvectoreval.RuntimeID)
	fmt.Fprintf(stdout, "  Artifacts (%d pinned):\n", len(res.Artifacts))
	for _, a := range res.Artifacts {
		fmt.Fprintf(stdout, "    - %s (%s)\n", a.ID, a.SHA256)
	}
	fmt.Fprintf(stdout, "  Metrics (%d entries):\n", len(res.Metrics))
	for _, m := range res.Metrics {
		fmt.Fprintf(stdout, "    - %s: %s [%s] (%s)\n", m.Name, m.Value, m.Kind, m.Provenance)
	}
	return 0
}

func runKVVectorEvalEvaluate(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak kvvectoreval eval", flag.ContinueOnError)
	fs.SetOutput(stderr)
	contractID := fs.String("contract-id", kvvectoreval.ContractID, "contract version ID")
	paperID := fs.String("paper-id", kvvectoreval.PaperID, "research paper ID")
	pdfSHA := fs.String("pdf-sha256", kvvectoreval.PaperPDFSHA256, "paper PDF SHA-256")
	sourceSHA := fs.String("source-sha256", kvvectoreval.PaperSourceSHA256, "paper source SHA-256")
	recipeID := fs.String("recipe-id", kvvectoreval.RecipeID, "recipe ID")
	runtimeID := fs.String("runtime-id", kvvectoreval.RuntimeID, "runtime ID")
	runtimeAvail := fs.Bool("runtime-available", false, "whether external runtime is available")
	asJSON := fs.Bool("json", false, "emit evaluation result as JSON")
	fs.Usage = func() { kvVectorEvalUsage(stderr) }
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}

	req := kvvectoreval.Request{
		ContractID:        *contractID,
		PaperID:           *paperID,
		PaperPDFSHA256:    *pdfSHA,
		PaperSourceSHA256: *sourceSHA,
		RecipeID:          *recipeID,
		RuntimeID:         *runtimeID,
		RuntimeAvailable:  *runtimeAvail,
	}
	res := kvvectoreval.Evaluate(req)

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(res); err != nil {
			fmt.Fprintf(stderr, "fak kvvectoreval eval: encode json: %v\n", err)
			return 1
		}
		if res.Outcome == kvvectoreval.OutcomeRefused {
			return 1
		}
		return 0
	}

	fmt.Fprintf(stdout, "outcome:  %s\n", res.Outcome)
	fmt.Fprintf(stdout, "reason:   %s\n", res.Reason)
	if res.Delegate != "" {
		fmt.Fprintf(stdout, "delegate: %s\n", res.Delegate)
	}
	if res.Outcome == kvvectoreval.OutcomeRefused {
		return 1
	}
	return 0
}

func runKVVectorEvalVerifyArtifact(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak kvvectoreval verify-artifact", flag.ContinueOnError)
	fs.SetOutput(stderr)
	artifactID := fs.String("id", "", "pinned artifact ID to verify against")
	filePath := fs.String("file", "", "path to artifact data file")
	asJSON := fs.Bool("json", false, "emit result as JSON")
	fs.Usage = func() { kvVectorEvalUsage(stderr) }
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}

	if *artifactID == "" || *filePath == "" {
		fmt.Fprintln(stderr, "fak kvvectoreval verify-artifact: --id and --file are required")
		return 2
	}

	data, err := os.ReadFile(*filePath)
	if err != nil {
		fmt.Fprintf(stderr, "fak kvvectoreval verify-artifact: read %s: %v\n", *filePath, err)
		return 1
	}

	vErr := kvvectoreval.VerifyArtifact(*artifactID, data)
	ok := vErr == nil

	if *asJSON {
		payload := struct {
			Schema   string `json:"schema"`
			Artifact string `json:"artifact"`
			Path     string `json:"path"`
			Bytes    int    `json:"bytes"`
			Valid    bool   `json:"valid"`
			Error    string `json:"error,omitempty"`
		}{
			Schema:   "fak.kvvectoreval.verify/v1",
			Artifact: *artifactID,
			Path:     *filePath,
			Bytes:    len(data),
			Valid:    ok,
		}
		if vErr != nil {
			payload.Error = vErr.Error()
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(payload)
		if !ok {
			return 1
		}
		return 0
	}

	if !ok {
		fmt.Fprintf(stderr, "VERIFY FAIL: %v\n", vErr)
		return 1
	}
	fmt.Fprintf(stdout, "VERIFY OK: %s (%d bytes)\n", *artifactID, len(data))
	return 0
}

func kvVectorEvalUsage(w io.Writer) {
	fmt.Fprintln(w, `fak kvvectoreval - NOVA-KV vector quantization research contract evaluation

  fak kvvectoreval inspect [--json]
      Inspect pinned research contract, artifacts, and evidence ledger.

  fak kvvectoreval eval [--runtime-available] [--json]
      Evaluate an interoperability request against the pinned contract.

  fak kvvectoreval verify-artifact --id <artifact-id> --file <path> [--json]
      Verify an artifact against pinned research digests.`)
}
