package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
	"github.com/anthony-chaudhary/fak/pkg/referenceoracle"
)

func runComputeTraceFromTorch(stdout, stderr io.Writer, args []string, engine string) int {
	fs := flag.NewFlagSet("compute-trace from-"+engine, flag.ContinueOnError)
	fs.SetOutput(stderr)
	out := fs.String("out", "", "reference trace output path")
	arch := fs.String("stage-map", "", "HF architecture")
	rank := fs.Int("rank", 0, "selected rank (all ranks must agree)")
	step := fs.Int("step", -1, "forward step (default: require unique prompt match)")
	adopt := fs.String("adopt-digests", "", "actual trace whose digests are adopted after token equality")
	oraclePath := fs.String("oracle", "", "existing reference oracle to attach")
	prompt := fs.Int("prompt", 0, "oracle prompt index")
	pos, rest := computeTraceSplitPositional(args)
	if err := fs.Parse(rest); err != nil {
		return 2
	}
	pos = append(pos, fs.Args()...)
	if len(pos) != 1 || *out == "" || *arch == "" || *rank < 0 || *step < -1 {
		fmt.Fprintln(stderr, "usage: fak compute-trace from-"+engine+" <dump-dir> --stage-map <HF-architecture> --out ref.json [--step N] [--rank N] [--adopt-digests trace.json] [--oracle oracle.json --prompt N]")
		return 2
	}
	opts := computetrace.TorchDumpOptions{Engine: engine, Architecture: *arch, Rank: *rank}
	if *step >= 0 {
		opts.Step = step
	}
	var oracle referenceoracle.Oracle
	var tracePath string
	if *oraclePath != "" {
		var err error
		oracle, err = referenceoracle.ReadFile(*oraclePath)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		if *prompt < 0 || *prompt >= len(oracle.Prompts) {
			fmt.Fprintln(stderr, "oracle prompt out of range")
			return 2
		}
		opts.PromptTokens = oracle.Prompts[*prompt].InputTokenIDs
		base, err := filepath.Abs(filepath.Dir(*oraclePath))
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		absolute, err := filepath.Abs(*out)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		relative, err := filepath.Rel(base, absolute)
		if err != nil || !filepath.IsLocal(relative) || absolute == filepath.Join(base, filepath.Base(*oraclePath)) {
			fmt.Fprintln(stderr, "--out must be a separate local file under the oracle directory")
			return 2
		}
		tracePath = filepath.ToSlash(relative)
	}
	result, err := computetrace.ConvertTorchDump(pos[0], opts)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	ref := result.Artifact
	if *adopt != "" {
		actual, err := readComputeTraceFile(*adopt)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		if err = computetrace.AdoptActivationDigests(&ref, actual); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	}
	ref.ClampNonFinite()
	data, err := json.Marshal(ref)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	data = append(data, '\n')
	var oracleData []byte
	if *oraclePath != "" {
		digest := sha256.Sum256(data)
		layers := 0
		for _, e := range ref.Events {
			layers = max(layers, e.Layer+1)
		}
		entry := referenceoracle.TraceRef{Prompt: *prompt, Path: tracePath, SHA256: hex.EncodeToString(digest[:]), StageMap: engine + ":" + *arch, Layers: layers, Tokens: len(ref.InputTokens)}
		refs := oracle.TraceRefs[:0]
		for _, r := range oracle.TraceRefs {
			if r.Prompt != *prompt {
				refs = append(refs, r)
			}
		}
		oracle.TraceRefs = append(refs, entry)
		oracleData, err = json.MarshalIndent(oracle, "", "  ")
		if err == nil {
			_, err = referenceoracle.Parse(oracleData)
		}
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		oracleData = append(oracleData, '\n')
	}
	if err = os.WriteFile(*out, data, 0o644); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if *oraclePath != "" {
		if err = os.WriteFile(*oraclePath, oracleData, 0o644); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	}
	unmapped, _ := json.Marshal(result.Unmapped)
	fmt.Fprintf(stdout, "wrote %s events=%d tokens=%d step=%d unmapped=%s\n", *out, len(ref.Events), len(ref.InputTokens), result.Step, unmapped)
	return 0
}
