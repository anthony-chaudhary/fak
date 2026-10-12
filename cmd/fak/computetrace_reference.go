package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

// computeTraceSplitPositional separates leading or trailing positional arguments from flags
// so `from-llamacpp <dir> --out x` and `from-llamacpp --out x <dir>` both parse.
func computeTraceSplitPositional(args []string) (pos, flags []string) {
	for len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		pos = append(pos, args[0])
		args = args[1:]
	}
	return pos, args
}

func readComputeTraceFile(path string) (computetrace.Artifact, error) {
	f, err := os.Open(path)
	if err != nil {
		return computetrace.Artifact{}, err
	}
	defer f.Close()
	return computetrace.Read(f)
}

// runComputeTraceFromLlamaCpp converts a llama.cpp tensor dump directory into a
// reference activation trace, optionally adopting a fak trace's digests.
func runComputeTraceFromLlamaCpp(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("compute-trace from-llamacpp", flag.ContinueOnError)
	fs.SetOutput(stderr)
	out := fs.String("out", "", "reference trace artifact path")
	adopt := fs.String("adopt-digests", "", "fak trace whose input/weight digests the reference adopts after its token ids match")
	pos, rest := computeTraceSplitPositional(args)
	if err := fs.Parse(rest); err != nil {
		return 2
	}
	pos = append(pos, fs.Args()...)
	if len(pos) != 1 || *out == "" {
		fmt.Fprintln(stderr, "usage: fak compute-trace from-llamacpp <dump-dir> --out ref.json [--adopt-digests fak-trace.json]")
		return 2
	}
	ref, err := computetrace.ConvertLlamaCppDump(pos[0])
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if *adopt != "" {
		src, err := readComputeTraceFile(*adopt)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		if err := computetrace.AdoptActivationDigests(&ref, src); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	}
	clamped := ref.ClampNonFinite()
	data, err := json.Marshal(ref)
	if err == nil {
		err = os.WriteFile(*out, append(data, '\n'), 0o644)
	}
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	fmt.Fprintf(stdout, "wrote %s events=%d tokens=%d samples=%d non_finite_clamped=%d digests_adopted=%t\n", *out, len(ref.Events), len(ref.InputTokens), ref.RetainedSampleValues, clamped, *adopt != "")
	return 0
}

// runComputeTraceProfile prints the per-(layer, stage) activation error profile
// of an actual trace against a reference trace.
func runComputeTraceProfile(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("compute-trace profile", flag.ContinueOnError)
	fs.SetOutput(stderr)
	refPath := fs.String("ref", "", "reference trace artifact")
	gotPath := fs.String("got", "", "actual trace artifact")
	atol := fs.Float64("atol", 1e-3, "absolute tolerance")
	rtol := fs.Float64("rtol", 1e-2, "relative tolerance (scaled by |got|)")
	asJSON := fs.Bool("json", false, "emit the profile as JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *refPath == "" || *gotPath == "" {
		fmt.Fprintln(stderr, "usage: fak compute-trace profile --ref ref.json --got fak.json [--atol 1e-3] [--rtol 1e-2] [--json]")
		return 2
	}
	ref, err := readComputeTraceFile(*refPath)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	got, err := readComputeTraceFile(*gotPath)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	p := computetrace.ActivationErrorProfile(ref, got, computetrace.ActivationTolerance{Atol: *atol, Rtol: *rtol})
	if *asJSON {
		if err := json.NewEncoder(stdout).Encode(p); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "%-5s %-20s %6s %8s %12s %12s %s\n", "layer", "stage", "tokens", "samples", "max_abs", "max_rel", "status")
		for _, r := range p.Rows {
			status := "match"
			switch {
			case !r.Comparable:
				status = "incomparable: " + r.Reason
			case !r.Match:
				status = fmt.Sprintf("DIVERGENT mismatches=%d non_finite=%d", r.Mismatches, r.NonFinite)
			}
			fmt.Fprintf(stdout, "%-5d %-20s %6d %8d %12.4g %12.4g %s\n", r.Layer, r.Stage, r.Tokens, r.ComparedSamples, r.MaxAbsError, r.MaxRelError, status)
		}
		first := "none"
		if p.FirstDivergent != nil {
			first = fmt.Sprintf("layer=%d stage=%s", p.FirstDivergent.Layer, p.FirstDivergent.Stage)
		}
		fmt.Fprintf(stdout, "verdict=%s first_divergent=%s compared=%d missing=%d extra=%d incomplete=%t %s\n", p.Verdict, first, p.ComparedSamples, p.MissingIdentities, p.ExtraIdentities, p.Incomplete, p.Reason)
	}
	if p.Verdict != computetrace.TraceEqual {
		return 1
	}
	return 0
}
