package main

// allocation-check — host-only read-only audit of a GGUF checkpoint's tensor
// inventory against its RCO allocation sidecar (#14091). It parses headers only
// (ggufload.OpenWeights reads no weight payloads), so it is safe to run on any
// development host against a local or downloaded shard set. Exit codes: 0 PASS,
// 1 typed FAIL, 2 usage error.

import (
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

const allocationCheckUsage = "usage: fak-dev allocation-check GGUF_FILE --sidecar SIDEcar_FILE [--exclude NAME]... [--bpw-tolerance F]"

// parseAllocationCheckArgs scans argv for the verb's flags in any position.
// The stdlib flag package stops at the first positional, which would swallow
// "--sidecar" when the GGUF path leads; this closed flag set is small enough
// that an explicit scan is clearer and keeps both orders working.
func parseAllocationCheckArgs(argv []string) (ggufPath, sidecarPath string, excludes []string, tolerance float64, fail, help bool) {
	i := 0
	for i < len(argv) {
		arg := argv[i]
		switch {
		case arg == "-h" || arg == "--help":
			return "", "", nil, 0, false, true
		case arg == "--sidecar":
			if i+1 >= len(argv) {
				return "", "", nil, 0, true, false
			}
			sidecarPath = argv[i+1]
			i += 2
		case strings.HasPrefix(arg, "--sidecar="):
			sidecarPath = strings.TrimPrefix(arg, "--sidecar=")
			i++
		case arg == "--bpw-tolerance":
			if i+1 >= len(argv) {
				return "", "", nil, 0, true, false
			}
			v, err := strconv.ParseFloat(argv[i+1], 64)
			if err != nil || v < 0 {
				return "", "", nil, 0, true, false
			}
			tolerance = v
			i += 2
		case strings.HasPrefix(arg, "--bpw-tolerance="):
			v, err := strconv.ParseFloat(strings.TrimPrefix(arg, "--bpw-tolerance="), 64)
			if err != nil || v < 0 {
				return "", "", nil, 0, true, false
			}
			tolerance = v
			i++
		case arg == "--exclude":
			if i+1 >= len(argv) {
				return "", "", nil, 0, true, false
			}
			excludes = append(excludes, argv[i+1])
			i += 2
		case strings.HasPrefix(arg, "--exclude="):
			excludes = append(excludes, strings.TrimPrefix(arg, "--exclude="))
			i++
		case strings.HasPrefix(arg, "-") && arg != "-":
			return "", "", nil, 0, true, false // unknown flag: refuse closed
		default:
			if ggufPath != "" {
				return "", "", nil, 0, true, false // at most one positional
			}
			ggufPath = arg
			i++
		}
	}
	if ggufPath == "" || sidecarPath == "" {
		fail = true
	}
	return ggufPath, sidecarPath, excludes, tolerance, fail, help
}

func runAllocationCheck(stdout, stderr io.Writer, argv []string) int {
	ggufPath, sidecarPath, excludes, tolerance, fail, help := parseAllocationCheckArgs(argv)
	if help {
		fmt.Fprintln(stdout, allocationCheckUsage)
		return 0
	}
	if fail {
		fmt.Fprintln(stderr, allocationCheckUsage)
		return 2
	}

	sc, err := ggufload.LoadAllocationSidecar(sidecarPath)
	if err != nil {
		fmt.Fprintf(stderr, "FAIL %s: %v\n", ggufload.AllocationErrorToken(err), err)
		return 1
	}
	ws, err := ggufload.OpenWeights(ggufPath)
	if err != nil {
		fmt.Fprintf(stderr, "FAIL GGUF_OPEN_REFUSED: %v\n", err)
		return 1
	}
	defer ws.Close()

	opts := ggufload.AllocationValidationOptions{BPWTolerance: tolerance}
	if len(excludes) > 0 {
		excluded := make(map[string]struct{}, len(excludes))
		for _, name := range excludes {
			excluded[name] = struct{}{}
		}
		opts.ExcludeBpw = func(name string) bool { _, ok := excluded[name]; return ok }
	}
	if err := ggufload.ValidateAllocation(sc, ws.File.Tensors, opts); err != nil {
		fmt.Fprintf(stderr, "FAIL %s: %v\n", ggufload.AllocationErrorToken(err), err)
		return 1
	}

	bpw, err := ggufload.EffectiveBpw(ws.File.Tensors, opts.ExcludeBpw)
	if err != nil {
		fmt.Fprintf(stderr, "FAIL %s: %v\n", ggufload.AllocationErrorToken(err), err)
		return 1
	}
	hist := make(map[ggufload.TensorType]int)
	for _, t := range ws.File.Tensors {
		hist[t.Type]++
	}
	types := make([]ggufload.TensorType, 0, len(hist))
	for typ := range hist {
		types = append(types, typ)
	}
	sort.Slice(types, func(i, j int) bool { return types[i].String() < types[j].String() })
	parts := make([]string, 0, len(types))
	for _, typ := range types {
		parts = append(parts, fmt.Sprintf("%s=%d", typ, hist[typ]))
	}
	measured := ""
	if sc.MeasuredBPW != nil {
		measured = fmt.Sprintf(" measured=%.4f", *sc.MeasuredBPW)
	}
	scope := ""
	if len(excludes) > 0 {
		scope = fmt.Sprintf(" (excluded from bpw scope: %s)", strings.Join(excludes, ", "))
	}
	fmt.Fprintf(stdout, "PASS allocation-check: %s vs %s\n", ggufPath, sidecarPath)
	fmt.Fprintf(stdout, "  tensors=%d target_bpws=%.4f%s effective_bpws=%.4f%s\n", len(ws.File.Tensors), sc.TargetBPW, measured, bpw, scope)
	fmt.Fprintf(stdout, "  histogram: %s\n", strings.Join(parts, " "))
	return 0
}
