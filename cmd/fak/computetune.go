package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/computetune"
)

// cmdComputetune is the production entry point for internal/computetune's
// storage-compute arbitration (flow ticket debtwave2-internal-computetune):
// given a KV token count and expected reuse factor, decide whether recomputing
// prompt KV on UMA DRAM beats offloading the cache to NVMe SSD.
func cmdComputetune(argv []string) {
	os.Exit(runComputetune(os.Stdout, os.Stderr, argv))
}

func runComputetune(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("computetune", flag.ContinueOnError)
	fs.SetOutput(stderr)
	tokens := fs.Int("tokens", 0, "KV token count to arbitrate (must be > 0)")
	reuse := fs.Float64("reuse", 1.0, "expected reuse factor (>= 0)")
	platform := fs.String("platform", "strix-halo", "baseline platform profile: strix-halo or strix-point (ignored when --profile is set)")
	profilePath := fs.String("profile", "", "path to a JSON computetune.PlatformProfile (overrides --platform)")
	asJSON := fs.Bool("json", false, "emit the Decision as indented JSON")

	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}

	if *tokens <= 0 {
		fmt.Fprintln(stderr, "fak computetune: --tokens must be > 0")
		return 2
	}
	if *reuse < 0 {
		fmt.Fprintln(stderr, "fak computetune: --reuse must be >= 0")
		return 2
	}

	var profile computetune.PlatformProfile
	switch {
	case *profilePath != "":
		raw, err := os.ReadFile(*profilePath)
		if err != nil {
			fmt.Fprintf(stderr, "fak computetune: read profile: %v\n", err)
			return 1
		}
		if err := json.Unmarshal(raw, &profile); err != nil {
			fmt.Fprintf(stderr, "fak computetune: decode profile: %v\n", err)
			return 1
		}
	case *platform == "strix-halo":
		profile = computetune.DefaultStrixHaloProfile()
	case *platform == "strix-point":
		profile = computetune.DefaultStrixPointProfile()
	default:
		fmt.Fprintf(stderr, "fak computetune: unknown --platform %q (valid: strix-halo, strix-point)\n", *platform)
		return 2
	}

	if err := profile.Validate(); err != nil {
		fmt.Fprintf(stderr, "fak computetune: %v\n", err)
		return 2
	}

	d := computetune.Decide(*tokens, *reuse, profile)

	if *asJSON {
		out, err := json.MarshalIndent(d, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "fak computetune: encode json: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(out))
		return 0
	}

	fmt.Fprintf(stdout, "verdict=%s tokens=%d reuse=%.2f platform=%s recompute_cost=%.4fs offload_cost=%.4fs reason=%q\n",
		d.Verdict, *tokens, *reuse, profile.Name, d.RecomputeCost, d.OffloadCost, d.Reason)
	return 0
}
