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
	"runtime"
	"time"

	"github.com/anthony-chaudhary/fak/internal/systembaseline"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	minimumSystemBaselineInterval  = 10 * time.Millisecond
	maximumSystemBaselineInterval  = 10 * time.Second
	minimumSystemBaselineDuration  = 10 * time.Millisecond
	maximumSystemBaselineDuration  = 60 * time.Second
	maximumSystemBaselineTimeout   = 24 * time.Hour
	windowsSystemBaselineWaitDelay = 250 * time.Millisecond
)

func runBenchSystemBaseline(stdout, stderr io.Writer, argv []string) int {
	return runBenchSystemBaselineWithAttributor(stdout, stderr, argv, func() systemBaselineCommandAttributor {
		return systembaseline.NewCommandAttributor()
	})
}

type systemBaselineCommandAttributor interface {
	Configure(*exec.Cmd) bool
	Active() bool
	Started(int) error
	LaunchFailed(error)
	FinishAttribution() systembaseline.CommandAttribution
}

func runBenchSystemBaselineWithAttributor(stdout, stderr io.Writer, argv []string, newAttributor func() systemBaselineCommandAttributor) int {
	fs := flag.NewFlagSet("bench system-baseline", flag.ContinueOnError)
	fs.SetOutput(stderr)
	interval := fs.Duration("interval", 250*time.Millisecond, "host/process sample interval (10ms..10s)")
	baselineDuration := fs.Duration("baseline-duration", time.Second, "quiet pre-command sampling duration (10ms..60s)")
	timeout := fs.Duration("timeout", 0, "kill the child and invalidate evidence after this duration (maximum 24h; 0 disables)")
	verify := fs.String("verify", "", "strictly decode and validate an existing attestation instead of running a command")
	out := fs.String("out", "", "also write the JSON attestation to this path")
	maxAmbient := fs.Float64("max-non-sut-cpu-percent", 20, "investigate above this non-SUT share of host CPU")
	maxSamplerDuty := fs.Float64("max-sampler-duty-percent", 10, "investigate above this sampler census duty (0,100]")
	maxPSIStall := fs.Float64("max-psi-stall-percent", 5, "investigate above this cgroup PSI stall share [0,100]")
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
		*timeout < 0 || *timeout > maximumSystemBaselineTimeout || *maxAmbient <= 0 || *maxAmbient > 100 || *maxSamplerDuty <= 0 || *maxSamplerDuty > 100 || *maxPSIStall < 0 || *maxPSIStall > 100 {
		fmt.Fprintln(stderr, "fak bench system-baseline: interval must be 10ms..10s, baseline duration 10ms..60s, timeout 0..24h, max non-SUT CPU in (0,100], max sampler duty in (0,100], and max PSI stall in [0,100]")
		return 2
	}

	cadencePolicy := systembaseline.CadencePolicy{Minimum: *interval, Maximum: 10 * *interval, MaximumDutyPercent: *maxSamplerDuty}
	baselineSamples := captureSystemBaselineWindow(*baselineDuration, cadencePolicy)
	// Arm the command timeout only after Windows has atomically placed and
	// resumed a suspended child. Starting it before Job Object setup lets a
	// short command timeout kill the process while the launch transaction is
	// still acquiring/read-checking handles, turning correct placement into a
	// nondeterministic OpenThread/NtResumeProcess failure.
	ctx, cancel := context.WithCancelCause(context.Background())
	defer cancel(context.Canceled)
	newCommand := func() *exec.Cmd {
		cmd := exec.CommandContext(ctx, command[0], command[1:]...)
		windowgate.ConfigureBackgroundCommand(cmd)
		// A job member can outlive the root while retaining os/exec's copied
		// stdout/stderr pipe handles. Bound that wait so job finalization can reap
		// the descendant instead of deadlocking behind cmd.Wait.
		if runtime.GOOS == "windows" {
			cmd.WaitDelay = windowsSystemBaselineWaitDelay
		}
		cmd.Stdout, cmd.Stderr = stderr, stderr
		return cmd
	}
	attributor := newAttributor()
	cmd := newCommand()
	configured := attributor.Configure(cmd)
	if err := cmd.Start(); err != nil {
		if configured {
			attributor.LaunchFailed(err)
			cmd = newCommand()
			err = cmd.Start()
		}
		if err != nil {
			attributor.LaunchFailed(err)
			_ = attributor.FinishAttribution()
			fmt.Fprintf(stderr, "fak bench system-baseline: start child: %v\n", err)
			return 1
		}
	}
	rootPID := cmd.Process.Pid
	if configured && attributor.Active() {
		if err := attributor.Started(rootPID); err != nil {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
			attributor.LaunchFailed(err)
			_ = attributor.FinishAttribution()
			fmt.Fprintf(stderr, "fak bench system-baseline: finalize child launch: %v\n", err)
			return 1
		}
	}
	var timeoutTimer *time.Timer
	if *timeout > 0 {
		timeoutTimer = time.AfterFunc(*timeout, func() { cancel(context.DeadlineExceeded) })
		defer timeoutTimer.Stop()
	}
	sampler := newAdaptiveSystemBaselineSampler(cadencePolicy)
	samples := []systembaseline.Snapshot{sampler.capture()}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	var waitErr error
waitLoop:
	for {
		timer := time.NewTimer(sampler.interval())
		select {
		case <-timer.C:
			samples = append(samples, sampler.capture())
		case waitErr = <-done:
			if !timer.Stop() {
				<-timer.C
			}
			break waitLoop
		}
	}
	samples = append(samples, sampler.capture())
	exitCode := 0
	timedOut := errors.Is(context.Cause(ctx), context.DeadlineExceeded)
	if timedOut {
		exitCode = 124
	} else if waitErr != nil {
		var exitErr *exec.ExitError
		if errors.Is(waitErr, exec.ErrWaitDelay) && cmd.ProcessState != nil {
			exitCode = cmd.ProcessState.ExitCode()
			if exitCode < 0 {
				exitCode = 1
			}
		} else if errors.As(waitErr, &exitErr) {
			exitCode = exitErr.ExitCode()
		} else {
			exitCode = 1
		}
	}
	policy := systembaseline.DefaultPolicy()
	policy.MaximumNonSUTCPUPercent = *maxAmbient
	policy.MaximumSamplerDutyPercent = *maxSamplerDuty
	policy.MaximumPSIStallPercent = *maxPSIStall
	policy.MinimumCensusIntervalNS = int64(cadencePolicy.Minimum)
	policy.MaximumCensusIntervalNS = int64(cadencePolicy.Maximum)
	policy.RequireProcessMemory = *requireMemory
	policy.IncludeTopConsumers = *topConsumers
	attribution := attributor.FinishAttribution()
	report := systembaseline.BuildWithCommandAttribution(baselineSamples, samples, rootPID, *interval, policy, exitCode, timedOut, &attribution)
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

type adaptiveSystemBaselineSampler struct {
	cadence *systembaseline.CadenceController
	cache   systembaseline.StableProcessCache
}

func newAdaptiveSystemBaselineSampler(policy systembaseline.CadencePolicy) *adaptiveSystemBaselineSampler {
	return &adaptiveSystemBaselineSampler{cadence: systembaseline.NewCadenceController(policy)}
}
func (s *adaptiveSystemBaselineSampler) interval() time.Duration { return s.cadence.Effective() }
func (s *adaptiveSystemBaselineSampler) capture() systembaseline.Snapshot {
	snapshot := systembaseline.Capture()
	hits, misses := s.cache.Apply(snapshot.Processes)
	snapshot.StableCacheHits, snapshot.StableCacheMisses = hits, misses
	snapshot.CensusStages = map[string]int64{"process_census": snapshot.CensusWallNS}
	snapshot.EffectiveCadenceNS = int64(s.cadence.Observe(time.Duration(snapshot.CensusWallNS)))
	snapshot.CoverageLimited = s.cadence.Overloaded()
	return snapshot
}

func captureSystemBaselineWindow(duration time.Duration, cadencePolicy systembaseline.CadencePolicy) []systembaseline.Snapshot {
	sampler := newAdaptiveSystemBaselineSampler(cadencePolicy)
	samples := []systembaseline.Snapshot{sampler.capture()}
	timer := time.NewTimer(duration)
	defer timer.Stop()
	for {
		tick := time.NewTimer(sampler.interval())
		select {
		case <-tick.C:
			samples = append(samples, sampler.capture())
		case <-timer.C:
			if !tick.Stop() {
				<-tick.C
			}
			samples = append(samples, sampler.capture())
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
