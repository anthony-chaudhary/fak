package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnessartifact"
	"github.com/anthony-chaudhary/fak/internal/harnessserver"
	"github.com/anthony-chaudhary/fak/internal/harnessverify"
)

type harnessVerifyRunCLIResult struct {
	Harness   harnessverify.Report                   `json:"harness"`
	Server    harnessserver.Verified                 `json:"server"`
	Lifecycle *harnessartifact.ModelLifecycleReceipt `json:"lifecycle,omitempty"`
}

func runHarnessVerifyRun(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness verify-run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	lockPath := fs.String("lock", "", "verified harness lock promised for the run")
	observationPath := fs.String("observation", "", "runtime capability and decision observation JSON")
	serverBindingPath := fs.String("server-binding", "", "immutable harness server binding created by harness init")
	jsonView := fs.Bool("json", false, "emit machine-readable verification JSON")
	lifecyclePath := fs.String("lifecycle-receipt", "", "model lifecycle receipt to verify")
	expectedDeclaration := fs.String("expected-declaration", "", "expected lifecycle declaration identity")
	observationTemplate := fs.Bool("print-observation-template", false, "print a runtime observation bound to --lock")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *observationTemplate {
		if *lockPath == "" {
			fmt.Fprintln(stderr, "fak harness verify-run: --print-observation-template requires --lock")
			return 2
		}
		return writeHarnessObservationTemplate(stdout, stderr, *lockPath)
	}
	if *lockPath == "" || *observationPath == "" {
		fmt.Fprintln(stderr, "fak harness verify-run: --lock and --observation are required")
		return 2
	}
	var lifecycle *harnessartifact.ModelLifecycleReceipt
	if *lifecyclePath != "" {
		receipt, lifecycleErr := harnessartifact.ReadModelLifecycleReceipt(*lifecyclePath)
		if lifecycleErr == nil {
			lifecycleErr = harnessartifact.CheckLifecycleDeclaration(receipt, *expectedDeclaration)
		}
		if lifecycleErr != nil {
			fmt.Fprintf(stderr, "fak harness verify-run: lifecycle: %s: %s\n", harnessartifact.LifecycleDiagnosticCode(lifecycleErr), harnessartifact.RedactLifecycleText(lifecycleErr.Error()))
			return 4
		}
		lifecycle = &receipt
	}
	server, err := verifyHarnessServerBinding(*serverBindingPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness verify-run: server binding: %v\n", err)
		return 1
	}
	lock, err := readHarnessPreviewLock(*lockPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness verify-run: lock: %v\n", err)
		return 1
	}
	raw, err := os.ReadFile(*observationPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness verify-run: observation: %v\n", err)
		return 1
	}
	var observation harnessverify.Observation
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&observation); err != nil {
		fmt.Fprintf(stderr, "fak harness verify-run: observation: %v\n", err)
		return 1
	}
	report, err := harnessverify.Verify(*lock, observation)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness verify-run: %v\n", err)
		return 1
	}
	if *jsonView {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		var output any = report
		if server != nil || lifecycle != nil {
			combined := harnessVerifyRunCLIResult{Harness: report, Lifecycle: lifecycle}
			if server != nil {
				combined.Server = *server
			}
			output = combined
		}
		if err := enc.Encode(output); err != nil {
			fmt.Fprintf(stderr, "fak harness verify-run: %v\n", err)
			return 1
		}
	} else {
		if lifecycle != nil {
			fmt.Fprintf(stdout, "MODEL LIFECYCLE | VERIFIED | state=%s declaration=%s readiness=%s stop=%s\n", lifecycle.State, lifecycle.Declaration.ID, lifecycle.Readiness.ID, lifecycle.Stop.ID)
		}
		if server != nil {
			fmt.Fprintf(stdout, "SERVER RECEIPT | VERIFIED | model=%s generation=%d protocol=%s/%s\n", server.ModelAlias, server.Generation, server.ProtocolFamily, server.ProtocolRevision)
		}
		fmt.Fprint(stdout, harnessverify.Render(report))
	}
	if report.Verdict == "deviation" {
		return 3
	}
	return 0
}

func writeHarnessObservationTemplate(stdout, stderr io.Writer, lockPath string) int {
	lock, err := readHarnessPreviewLock(lockPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness verify-run: lock: %v\n", err)
		return 1
	}
	observation := harnessverify.Observation{
		Schema: harnessverify.ObservationSchema,
		LockID: lock.ID,
		RunID:  "replace-with-runtime-run-id",
	}
	for _, asset := range lock.Assets {
		observation.Capabilities = append(observation.Capabilities, harnessverify.Capability{
			Capability: asset.Kind + ":" + asset.ID,
			Source:     asset.Source,
			Value:      asset.Value,
			Ref:        asset.Ref,
			Boundary:   asset.Boundary,
			Grants:     asset.Grants,
			Denies:     asset.Denies,
		})
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(observation); err != nil {
		fmt.Fprintf(stderr, "fak harness verify-run: %v\n", err)
		return 1
	}
	return 0
}
