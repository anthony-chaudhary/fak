package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"time"

	"github.com/anthony-chaudhary/fak/internal/systembaseline"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	minimumSystemBaselineInterval = 10 * time.Millisecond
	maximumSystemBaselineInterval = 10 * time.Second
	minimumSystemBaselineDuration = 10 * time.Millisecond
	maximumSystemBaselineDuration = 60 * time.Second
	maximumSystemBaselineTimeout  = 24 * time.Hour
)

func runBenchSystemBaseline(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("bench system-baseline", flag.ContinueOnError)
	fs.SetOutput(stderr)
	interval := fs.Duration("interval", 250*time.Millisecond, "host/process sample interval (10ms..10s)")
	baselineDuration := fs.Duration("baseline-duration", time.Second, "quiet pre-command sampling duration (10ms..60s)")
	timeout := fs.Duration("timeout", 0, "kill the child and invalidate evidence after this duration (maximum 24h; 0 disables)")
	verify := fs.String("verify", "", "strictly decode and validate an existing attestation instead of running a command")
	out := fs.String("out", "", "also write the JSON attestation to this path")
	maxAmbient := fs.Float64("max-non-sut-cpu-percent", 20, "investigate above this non-SUT share of host CPU")
	maxSamplerDuty := fs.Float64("max-sampler-duty-percent", 10, "investigate above this sampler census duty (0,100]")
	requireMemory := fs.Bool("require-process-memory", false, "investigate when complete SUT/non-SUT RSS attribution is unavailable")
	topConsumers := fs.Bool("top-consumers", false, "include up to five scrubbed image+PID non-SUT consumers (local/high-cardinality evidence)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	command := fs.Args()
	if *verify != "" {
		if len(command) != 0 {
			fmt.Fprintln(stderr, "fak bench system-baseline: --verify does not accept a child command")
			return 2
		}
		return verifySystemBaseline(stdout, stderr, *verify)
	}
	if len(command) == 0 {
		fmt.Fprintln(stderr, "usage: fak bench system-baseline [flags] -- command [args...]")
		return 2
	}
	if *interval < minimumSystemBaselineInterval || *interval > maximumSystemBaselineInterval ||
		*baselineDuration < minimumSystemBaselineDuration || *baselineDuration > maximumSystemBaselineDuration ||
		*timeout < 0 || *timeout > maximumSystemBaselineTimeout || *maxAmbient <= 0 || *maxAmbient > 100 || *maxSamplerDuty <= 0 || *maxSamplerDuty > 100 {
		fmt.Fprintln(stderr, "fak bench system-baseline: interval must be 10ms..10s, baseline duration 10ms..60s, timeout 0..24h, max non-SUT CPU in (0,100], and max sampler duty in (0,100]")
		return 2
	}

	baselineSamples := captureSystemBaselineWindow(*baselineDuration, *interval)
	ctx := context.Background()
	cancel := func() {}
	if *timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, *timeout)
	}
	defer cancel()
	cmd := exec.CommandContext(ctx, command[0], command[1:]...)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Stdout, cmd.Stderr = stderr, stderr
	if err := cmd.Start(); err != nil {
		fmt.Fprintf(stderr, "fak bench system-baseline: start child: %v\n", err)
		return 1
	}
	rootPID := cmd.Process.Pid
	samples := []systembaseline.Snapshot{systembaseline.Capture()}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	ticker := time.NewTicker(*interval)
	defer ticker.Stop()
	var waitErr error
waitLoop:
	for {
		select {
		case <-ticker.C:
			samples = append(samples, systembaseline.Capture())
		case waitErr = <-done:
			break waitLoop
		}
	}
	samples = append(samples, systembaseline.Capture())
	exitCode := 0
	timedOut := errors.Is(ctx.Err(), context.DeadlineExceeded)
	if timedOut {
		exitCode = 124
	} else if waitErr != nil {
		var exitErr *exec.ExitError
		if errors.As(waitErr, &exitErr) {
			exitCode = exitErr.ExitCode()
		} else {
			exitCode = 1
		}
	}
	policy := systembaseline.DefaultPolicy()
	policy.MaximumNonSUTCPUPercent = *maxAmbient
	policy.MaximumSamplerDutyPercent = *maxSamplerDuty
	policy.RequireProcessMemory = *requireMemory
	policy.IncludeTopConsumers = *topConsumers
	report := systembaseline.Build(baselineSamples, samples, rootPID, *interval, policy, exitCode, timedOut)
	raw, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		fmt.Fprintf(stderr, "fak bench system-baseline: encode: %v\n", err)
		return 1
	}
	raw = append(raw, '\n')
	if *out != "" {
		if err := writePrivateJSON(*out, raw); err != nil {
			fmt.Fprintf(stderr, "fak bench system-baseline: write %s: %v\n", *out, err)
			return 1
		}
	}
	written, writeErr := stdout.Write(raw)
	if writeErr != nil || written != len(raw) {
		fmt.Fprintf(stderr, "fak bench system-baseline: write stdout: wrote %d of %d bytes: %v\n", written, len(raw), writeErr)
		return 1
	}
	return exitCode
}

func writePrivateJSON(path string, raw []byte) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	if err := f.Chmod(0o600); err != nil {
		_ = f.Close()
		return err
	}
	if _, err := f.Write(raw); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}

func captureSystemBaselineWindow(duration, interval time.Duration) []systembaseline.Snapshot {
	samples := []systembaseline.Snapshot{systembaseline.Capture()}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	timer := time.NewTimer(duration)
	defer timer.Stop()
	for {
		select {
		case <-ticker.C:
			samples = append(samples, systembaseline.Capture())
		case <-timer.C:
			samples = append(samples, systembaseline.Capture())
			return samples
		}
	}
}

func verifySystemBaseline(stdout, stderr io.Writer, path string) int {
	f, err := os.Open(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak bench system-baseline: verify open %s: %v\n", path, err)
		return 1
	}
	defer f.Close()
	report, err := systembaseline.Decode(f)
	if err == nil {
		err = report.Validate()
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak bench system-baseline: verify %s: %v\n", path, err)
		return 1
	}
	fmt.Fprintf(stdout, "VALID %s\n", report.Digest)
	return 0
}
