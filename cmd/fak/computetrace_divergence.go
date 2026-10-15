package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
	"github.com/anthony-chaudhary/fak/pkg/referenceoracle"
)

// loadDivergenceReference reads --ref as either a compute trace or a reference
// oracle; an oracle resolves to the activation trace it lists for prompt.
func loadDivergenceReference(path string, prompt int) (computetrace.Artifact, string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return computetrace.Artifact{}, "", err
	}
	var head struct {
		Schema string `json:"schema"`
	}
	if err := json.Unmarshal(data, &head); err != nil {
		return computetrace.Artifact{}, "", fmt.Errorf("%s: %v", path, err)
	}
	if !referenceoracle.IsOracleSchema(head.Schema) {
		a, err := computetrace.Read(bytes.NewReader(data))
		return a, path, err
	}
	o, err := referenceoracle.Parse(data)
	if err != nil {
		return computetrace.Artifact{}, "", err
	}
	ref, ok := o.TraceRef(prompt)
	if !ok {
		return computetrace.Artifact{}, "", fmt.Errorf("%s: oracle lists no activation trace for prompt %d", path, prompt)
	}
	tracePath := filepath.Join(filepath.Dir(path), ref.Path)
	raw, err := os.ReadFile(tracePath)
	if err != nil {
		return computetrace.Artifact{}, "", err
	}
	if want := strings.TrimPrefix(ref.SHA256, "sha256:"); want != "" {
		sum := sha256.Sum256(raw)
		if got := hex.EncodeToString(sum[:]); !strings.EqualFold(got, want) {
			return computetrace.Artifact{}, "", fmt.Errorf("%s: sha256 %s, oracle pins %s", tracePath, got, want)
		}
	}
	a, err := computetrace.Read(bytes.NewReader(raw))
	return a, tracePath, err
}

// runComputeTraceFirstDivergence reports the first (layer, stage) where an
// actual trace leaves the reference, with the full per-layer x stage profile.
func runComputeTraceFirstDivergence(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("compute-trace first-divergence", flag.ContinueOnError)
	fs.SetOutput(stderr)
	refPath := fs.String("ref", "", "reference: a compute trace, or a reference oracle whose trace_refs name one")
	gotPath := fs.String("got", "", "actual trace artifact")
	tol := fs.Float64("tol", 1e-2, "largest accepted relative L2 error per (layer, stage)")
	minCos := fs.Float64("min-cos", 0, "smallest accepted cosine similarity per (layer, stage); 0 disables")
	atol := fs.Float64("atol", 1e-3, "element-wise absolute tolerance for the strict cross-check")
	rtol := fs.Float64("rtol", 1e-2, "element-wise relative tolerance for the strict cross-check")
	prompt := fs.Int("prompt", 0, "oracle prompt index whose trace_ref is the reference")
	adopt := fs.Bool("adopt-digests", false, "adopt the actual trace's input/weight digests into the reference after its token ids match")
	stepID := fs.String("step-capture-id", "", "expected committed incremental capture ID from the current caller receipt")
	stepToken := fs.Int("step-token", -1, "expected absolute token position of the incremental capture")
	asJSON := fs.Bool("json", false, "emit the report as JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *refPath == "" || *gotPath == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak compute-trace first-divergence --ref <oracle|trace> --got <trace> [--step-capture-id ID --step-token N] [--tol <rel_l2>] [--min-cos c] [--prompt n] [--adopt-digests] [--json]")
		return 2
	}
	ref, _, err := loadDivergenceReference(*refPath, *prompt)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	got, err := readComputeTraceFile(*gotPath)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	scope := "sampled_numerical"
	var captureID string
	var captureToken *int
	var stepIDSet, stepTokenSet bool
	fs.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "step-capture-id":
			stepIDSet = true
		case "step-token":
			stepTokenSet = true
		}
	})
	wantsStep := stepIDSet || stepTokenSet
	if wantsStep || got.ActivationCapture != nil {
		capture := got.ActivationCapture
		if !stepIDSet || !stepTokenSet || strings.TrimSpace(*stepID) == "" || *stepToken < 0 || capture == nil || capture.ID != *stepID || capture.Token != *stepToken {
			fmt.Fprintln(stderr, "incremental comparison requires matching --step-capture-id and --step-token")
			return 1
		}
		if err := computetrace.ValidateActivationCapture(got); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		ref, err = computetrace.ProjectActivationReference(ref, got)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		scope, captureID = "committed_step_capture", capture.ID
		captureToken = &capture.Token
	}
	if *adopt {
		if err := computetrace.AdoptActivationDigests(&ref, got); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	}
	r := computetrace.FirstDivergence(ref, got,
		computetrace.DivergenceTolerance{RelL2: *tol, MinCosine: *minCos},
		computetrace.ActivationTolerance{Atol: *atol, Rtol: *rtol})
	if *asJSON {
		out := struct {
			computetrace.DivergenceReport
			EvidenceScope string `json:"evidence_scope"`
			CaptureID     string `json:"capture_id,omitempty"`
			CaptureToken  *int   `json:"capture_token,omitempty"`
		}{r, scope, captureID, captureToken}
		if err := json.NewEncoder(stdout).Encode(out); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "%-5s %-12s %6s %-22s %12s %10s %s\n", "layer", "verdict", "stages", "worst_stage", "max_rel_l2", "min_cos", "first_divergent_stage")
		for _, l := range r.Layers {
			fmt.Fprintf(stdout, "%-5d %-12s %6d %-22s %12.4g %10.6f %s\n", l.Layer, l.Verdict, l.Stages, l.WorstStage, l.MaxRelL2, l.MinCosine, l.FirstDivergent)
		}
		layer, stage := "none", "none"
		if r.FirstDivergent != nil {
			layer, stage = fmt.Sprint(r.FirstDivergent.Layer), r.FirstDivergent.Stage
		}
		fmt.Fprintf(stdout, "verdict=%s first_divergent_layer=%s first_divergent_stage=%s parity_layers=%d layers=%d missing=%d extra=%d strict=%s %s\n",
			r.Verdict, layer, stage, r.ParityLayers, len(r.Layers), r.Missing, r.Extra, r.Strict.Verdict, r.Reason)
		fmt.Fprintf(stdout, "scope=%s", scope)
		if captureToken != nil {
			fmt.Fprintf(stdout, " capture_id=%s capture_token=%d", captureID, *captureToken)
		}
		fmt.Fprintln(stdout)
	}
	if r.Verdict != computetrace.TraceEqual {
		return 1
	}
	return 0
}
