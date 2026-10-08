package devcmd

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"time"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

var runStrixSSDRestartFn = amdgpu.RunStrixSSDRestart

func runAMDStrixSSDRestart(ctx context.Context, stdout, stderr io.Writer, host string, candidate resolvedStrixCandidate, command string, timeout, admissionTimeout time.Duration, jsonOutput bool) int {
	opts := amdgpu.StrixSSDRestartOpts{
		Host: host, GitRef: candidate.archive.SourceArchiveSHA256, GitTip: candidate.tip,
		Command: command, Timeout: timeout, AdmissionTimeout: admissionTimeout,
		RequireSourceBinding: true, CandidateArchive: candidate.archive.Bytes,
		SourceArchiveSHA256: candidate.archive.SourceArchiveSHA256,
	}
	if !jsonOutput {
		fmt.Fprintf(stderr, "==> Validating native SSD restart on AMD Strix Halo target %s...\n", host)
	}
	receipt, runErr := runStrixSSDRestartFn(ctx, opts)
	if receipt == nil {
		fmt.Fprintf(stderr, "amd-strix-validate: native SSD restart failed without receipt: %v\n", runErr)
		return 1
	}
	if jsonOutput {
		data, err := json.MarshalIndent(receipt, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "amd-strix-validate: native SSD restart receipt encoding failed: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(data))
	} else {
		fmt.Fprintf(stdout, "NATIVE SSD RESTART: %s | HARDWARE: %t | RECEIPT: %s\n", receipt.Verdict, receipt.Hardware, receipt.Digest)
		for _, failure := range receipt.Failures {
			fmt.Fprintf(stdout, "  %s: %s\n", failure.Code, failure.Message)
		}
	}
	if runErr != nil {
		fmt.Fprintf(stderr, "amd-strix-validate: native SSD restart failed: %v\n", runErr)
		return 1
	}
	if err := receipt.Validate(); err != nil {
		fmt.Fprintf(stderr, "amd-strix-validate: native SSD restart receipt invalid: %v\n", err)
		return 1
	}
	return 0
}
