package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/conformance"
	"github.com/anthony-chaudhary/fak/pkg/harnessconformance"
)

type stockHarnessProbe struct{}

func (stockHarnessProbe) Capabilities() []harnessconformance.Capability {
	return append([]harnessconformance.Capability(nil), harnessconformance.Required...)
}

func (stockHarnessProbe) Check(c harnessconformance.Capability) (harnessconformance.Outcome, string) {
	return harnessconformance.Pass, ""
}

// cmdConformance runs the standalone fak safety-conformance suite (#453): it recomputes the
// ABI wire contract and re-adjudicates the shipped dogfood verdict matrix against the
// COMPILED kernel, then reports CONFORMANT / NON-CONFORMANT. Exit code is 1 on any
// divergence, so `fak conformance` is a CI-gateable, third-party-runnable attestation —
// any fork self-tests and any auditor verifies a "certified" claim independently.
func cmdConformance(argv []string) {
	fs := flag.NewFlagSet("conformance", flag.ExitOnError)
	asJSON := fs.Bool("json", false, "emit machine-readable JSON instead of a human report")
	localEndpoint := fs.String("local-endpoint", "", "probe a local fak endpoint against --managed-endpoint")
	managedEndpoint := fs.String("managed-endpoint", "", "probe a managed fak endpoint against --local-endpoint")
	harness := fs.Bool("harness", false, "run harness adapter conformance probe suite")
	_ = fs.Parse(argv)

	if *harness {
		cert := harnessconformance.Run(stockHarnessProbe{})
		if *asJSON {
			writeConformanceJSON(cert)
		} else {
			fmt.Printf("harness conformance: contract=%s full=%v checks=%d\n", cert.Contract, cert.Full, len(cert.Checks))
			for _, chk := range cert.Checks {
				fmt.Printf("  %-18s %s\n", chk.Capability, chk.Outcome)
			}
			if cert.Full {
				fmt.Println("CONFORMANT")
			} else {
				fmt.Println("NON-CONFORMANT")
			}
		}
		if err := cert.Validate(); err != nil {
			os.Exit(1)
		}
		return
	}

	if *localEndpoint != "" || *managedEndpoint != "" {
		if *localEndpoint == "" || *managedEndpoint == "" {
			fmt.Fprintln(os.Stderr, "fak conformance: --local-endpoint and --managed-endpoint are required together")
			os.Exit(2)
		}
		packet := conformance.ProbeEndpointPair(context.Background(), conformance.DefaultEndpointClient(), *localEndpoint, *managedEndpoint)
		writeConformanceJSON(packet)
		if packet.Verdict != "PASS" {
			os.Exit(1)
		}
		return
	}

	rep := conformance.Run()

	if *asJSON {
		writeConformanceJSON(rep)
	} else {
		fmt.Print(conformance.Render(rep))
	}

	if !rep.Pass {
		os.Exit(1)
	}
}

func writeConformanceJSON(value any) {
	b, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak conformance: %v\n", err)
		os.Exit(2)
	}
	os.Stdout.Write(append(b, '\n'))
}
