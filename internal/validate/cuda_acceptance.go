package validate

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

const cudaComputePackage = "github.com/anthony-chaudhary/fak/internal/compute"
const cudaModelPackage = "github.com/anthony-chaudhary/fak/internal/model"

func cudaAcceptanceRequired() []validateTestTerminal {
	return []validateTestTerminal{
		{Package: cudaComputePackage, Test: "TestCUDAMatMulApproxMatchesRef", Action: "pass"},
		{Package: cudaModelPackage, Test: "TestHALDeviceForwardMatchesNative", Action: "pass"},
	}
}

// Ordinary validation records honest skips. This acceptance contract instead
// requires two specific CUDA execution witnesses, not merely a green package.
func validateCUDAAcceptanceEvents(stdout []byte) error {
	events, tests, err := decodeValidateTestEvents(stdout, []string{cudaComputePackage, cudaModelPackage})
	if err != nil {
		return err
	}
	started := map[string]bool{}
	passed := map[string]bool{}
	states := map[string]string{}
	for _, raw := range events {
		event, err := decodeCUDAAcceptanceEvent(raw)
		if err != nil {
			return err
		}
		if event.Package != cudaComputePackage && event.Package != cudaModelPackage {
			return fmt.Errorf("unexpected package %q", event.Package)
		}
		if event.Action == "start" {
			if event.Test != "" || started[event.Package] {
				return fmt.Errorf("invalid or duplicate package start: %s", event.Package)
			}
			started[event.Package] = true
			continue
		}
		if !started[event.Package] {
			return fmt.Errorf("event before package start: %s", event.Package)
		}
		key := event.Package + "\x00" + event.Test
		switch event.Action {
		case "run":
			if event.Test == "" {
				return fmt.Errorf("run without a test identity")
			}
			states[key] = "running"
		case "pause":
			if event.Test == "" || states[key] != "running" {
				return fmt.Errorf("pause without a running test: %s", event.Test)
			}
			states[key] = "paused"
		case "cont":
			if event.Test == "" || states[key] != "paused" {
				return fmt.Errorf("cont without a paused test: %s", event.Test)
			}
			states[key] = "running"
		case "output":
			if event.Test != "" && states[key] == "" {
				return fmt.Errorf("output before test run: %s", event.Test)
			}
		case "fail":
			return fmt.Errorf("Go events report failure: %s %s", event.Package, event.Test)
		case "pass", "skip":
			if event.Test != "" && states[key] != "running" {
				return fmt.Errorf("test terminal without running state: %s", event.Test)
			}
			if event.Test == "" {
				if event.Action != "pass" {
					return fmt.Errorf("package did not pass: %s", event.Package)
				}
				passed[event.Package] = true
			}
		default:
			// No benchmarks or build-only events belong to this fixed test run.
			return fmt.Errorf("unexpected Go test action %q", event.Action)
		}
	}
	for _, required := range cudaAcceptanceRequired() {
		if !passed[required.Package] {
			return fmt.Errorf("missing package pass: %s", required.Package)
		}
		found := false
		for _, test := range tests {
			if test == required {
				found = true
			}
		}
		if !found {
			return fmt.Errorf("required test did not run and pass: %s %s", required.Package, required.Test)
		}
	}
	return nil
}

// encoding/json normally accepts duplicate keys and case-insensitive aliases.
// Evidence identities must instead have one unambiguous spelling and value.
func decodeCUDAAcceptanceEvent(raw json.RawMessage) (validateEventIdentity, error) {
	var event validateEventIdentity
	decoder := json.NewDecoder(bytes.NewReader(raw))
	token, err := decoder.Token()
	if err != nil || token != json.Delim('{') {
		return event, fmt.Errorf("Go event must be an object")
	}
	seen := map[string]bool{}
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return event, err
		}
		key := token.(string)
		if seen[key] {
			return event, fmt.Errorf("duplicate Go event field %q", key)
		}
		seen[key] = true
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return event, err
		}
		if bytes.Equal(value, []byte("null")) {
			return event, fmt.Errorf("null Go event field %q", key)
		}
		switch key {
		case "Action", "Package", "Test", "Output", "FailedBuild":
			var text string
			if err := json.Unmarshal(value, &text); err != nil {
				return event, fmt.Errorf("invalid %s: %w", key, err)
			}
			switch key {
			case "Action":
				event.Action = text
			case "Package":
				event.Package = text
			case "Test":
				event.Test = text
			case "FailedBuild":
				if text != "" {
					return event, fmt.Errorf("Go events report failed build: %s", text)
				}
			}
		case "Time":
			var timestamp time.Time
			if err := json.Unmarshal(value, &timestamp); err != nil {
				return event, fmt.Errorf("invalid event time: %w", err)
			}
		case "Elapsed":
			var elapsed float64
			if err := json.Unmarshal(value, &elapsed); err != nil || elapsed < 0 {
				return event, fmt.Errorf("invalid event elapsed time")
			}
		default:
			return event, fmt.Errorf("unexpected Go event field %q", key)
		}
	}
	if _, err := decoder.Token(); err != nil {
		return event, err
	}
	if event.Action == "" || event.Package == "" {
		return event, fmt.Errorf("Go event lacks Action or Package")
	}
	return event, nil
}

// Error is diagnostic event-validation text, not an acceptance input.
type cudaAcceptanceResult struct {
	Schema             string                 `json:"schema"`
	Accepted           bool                   `json:"accepted"`
	Command            []string               `json:"command"`
	Required           []validateTestTerminal `json:"required"`
	EvidenceDirectory  string                 `json:"evidence_directory"`
	TestExitCode       int                    `json:"test_exit_code"`
	CaptureStatus      string                 `json:"capture_status"`
	CaptureReason      string                 `json:"capture_reason,omitempty"`
	Error              string                 `json:"error,omitempty"`
	StdoutTruncated    bool                   `json:"stdout_truncated"`
	StderrTruncated    bool                   `json:"stderr_truncated"`
	DescendantTeardown string                 `json:"descendant_teardown"`
}

// RunCUDAAcceptance runs the native CUDA lane, never the usual Windows-to-WSL
// test route. It does not alter compiler environment, signing or runner policy.
// Captured events attest test execution only, not broader device qualification.
func RunCUDAAcceptance(stdout, stderr io.Writer, argv []string) int {
	return runCUDAAcceptance(stdout, stderr, argv, executeValidateEventTests)
}

func runCUDAAcceptance(stdout, stderr io.Writer, argv []string, execute func(context.Context, *validateTestWitness, []string) bool) int {
	fs := flag.NewFlagSet("fak test cuda-acceptance", flag.ContinueOnError)
	fs.SetOutput(stderr)
	evidence := fs.String("evidence-dir", "", "new, exclusive directory for go-test.jsonl, go-test.stderr and result.json")
	timeout := fs.Duration("timeout", 10*time.Minute, "native test/capture deadline (positive, at most 30m)")
	asJSON := fs.Bool("json", false, "emit the acceptance result as JSON")
	dry := fs.Bool("dry-run", false, "print the fixed native command without running it")
	if err := fs.Parse(argv); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 || *timeout <= 0 || *timeout > 30*time.Minute || (!*dry && *evidence == "") {
		fmt.Fprintln(stderr, "cuda-acceptance requires --evidence-dir, no positional arguments and 0 < --timeout <= 30m")
		return 2
	}
	command := []string{"go", "test", "-json", "-tags", "cuda", "-count=1", "-timeout", timeout.String(), "-run", "CUDA|HALDevice", "./internal/compute/", "./internal/model/"}
	result := cudaAcceptanceResult{Schema: "fak.cuda_acceptance/v1", Command: command, Required: cudaAcceptanceRequired(), EvidenceDirectory: *evidence, TestExitCode: -1, CaptureStatus: "unrun", DescendantTeardown: "unproven"}
	if *dry {
		if err := json.NewEncoder(stdout).Encode(result); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		return 0
	}
	// Claim ownership before opening any output. Never truncate a previous run,
	// including failed attempts. Callers must supply a fresh path for each retry.
	if err := os.Mkdir(*evidence, 0700); err != nil {
		fmt.Fprintf(stderr, "cuda-acceptance evidence directory: %v\n", err)
		return 1
	}
	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	witness := &validateTestWitness{Candidate: &validateCandidateFingerprint{}, Argv: command}
	captured := execute(ctx, witness, []string{cudaComputePackage, cudaModelPackage})
	gateErr := validateCUDAAcceptanceEvents(witness.Stdout)
	result.Accepted = captured && witness.ExitCode == 0 && gateErr == nil
	result.TestExitCode = witness.ExitCode
	result.CaptureStatus, result.CaptureReason = witness.Status, witness.Reason
	result.StdoutTruncated, result.StderrTruncated = witness.StdoutTruncated, witness.StderrTruncated
	if gateErr != nil {
		result.Error = gateErr.Error()
	}
	code := 1
	if witness.ExitCode > 0 {
		code = witness.ExitCode
	}
	encoded, _ := json.MarshalIndent(result, "", "  ")
	for _, file := range []struct {
		name string
		data []byte
	}{{"go-test.jsonl", witness.Stdout}, {"go-test.stderr", witness.Stderr}, {"result.json", append(encoded, '\n')}} {
		if err := writeCUDAAcceptanceFile(filepath.Join(*evidence, file.name), file.data); err != nil {
			fmt.Fprintf(stderr, "cuda-acceptance evidence write: %v\n", err)
			return code
		}
	}
	if *asJSON {
		if err := json.NewEncoder(stdout).Encode(result); err != nil {
			fmt.Fprintln(stderr, err)
			return code
		}
	} else {
		fmt.Fprintf(stdout, "CUDA acceptance accepted=%t test_exit=%d evidence=%s\n", result.Accepted, result.TestExitCode, *evidence)
		if !result.Accepted {
			fmt.Fprintf(stderr, "capture=%s: %s; gate=%s\n", result.CaptureStatus, result.CaptureReason, result.Error)
		}
	}
	if result.Accepted {
		return 0
	}
	return code
}

func writeCUDAAcceptanceFile(path string, data []byte) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	_, err = f.Write(data)
	closeErr := f.Close()
	if err != nil {
		return err
	}
	return closeErr
}
