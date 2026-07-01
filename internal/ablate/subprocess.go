// Rung 2 of the ablation harness (epic #607, issue #622): the subprocess re-exec path
// that lets the sweep ablate the env-gated FAK_* features, which the kernel reads ONCE
// at process start and so cannot be flipped within one process.
//
// The shape is a parent/child fan-out:
//
//   - RunOneArm is the single-arm runner core. It replays the frozen trace through one
//     FeatureConfig in the CURRENT process and returns one AblationRun. A re-exec'd
//     child calls it (via RunArmMode) after its own process start already read the arm's
//     FAK_* env — so the env-gated feature is genuinely live in that child. The CLI
//     arm-mode (`fak ablate-arm`, DEFERRED to cmd/fak) is a thin wrapper over RunArmMode.
//
//   - SweepViaSubprocess is the parent: it computes the trace's workload hash ONCE, then
//     fans one child per config through an injected armRunner, splicing each arm's FAK_*
//     env onto the child. It asserts every child reports the parent's hash (the same
//     identical-workload guard the in-process Sweep enforces) and merges the children
//     into one Report. A child that fails to run is DROPPED with a recorded reason — a
//     logged hole, never a silent one.
//
//   - execArmRunner is the production armRunner: it re-execs `bin ablate-arm`, writes the
//     {config, trace} request to the child's stdin, sets the arm's FAK_* env, and decodes
//     the child's one AblationRun from stdout. Tests inject a FAKE armRunner instead, so
//     the package test never spawns the real binary.
//
// os/exec is fine here: the harness is a bench (tier-4 integrator), off the live
// request-path closure the architest exec-ban guards. ablate is never on a tool-call
// decision path.
package ablate

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/bench"
	"github.com/anthony-chaudhary/fak/internal/metrics"
)

// armRunner runs ONE arm and returns its AblationRun. The production implementation
// (execArmRunner) re-execs a child process with the arm's FAK_* env set; a test injects
// a fake that runs in-process, so the suite never spawns the real binary. bin is the
// path to the fak binary the child re-execs; traceJSON is the marshaled *bench.Trace the
// parent froze (handed to the child verbatim so both compute the same workload hash).
type armRunner func(ctx context.Context, bin string, c FeatureConfig, traceJSON []byte) (AblationRun, error)

// armRequest is the {config, trace} unit the parent writes to a child's stdin and the
// child decodes in RunArmMode. The trace travels as raw JSON so the child reconstructs
// the EXACT *bench.Trace the parent hashed (byte-identical workload hash on both sides).
type armRequest struct {
	Config      FeatureConfig   `json:"config"`
	EngineID    string          `json:"engine_id"`
	EngineModel string          `json:"engine_model"`
	TraceJSON   json.RawMessage `json:"trace"`
}

// DroppedArm records an arm that failed to run, so a partial sweep names exactly which
// config it could not measure and why — the fan-out never returns a silent hole.
type DroppedArm struct {
	ArmID  string `json:"arm_id"`
	Reason string `json:"reason"`
}

// SweepViaSubprocess is the rung-2 parent fan-out. It computes the trace's workload hash
// ONCE, runs each config through runner (one child per arm, with that arm's FAK_* env),
// asserts every successful child reports the parent's hash, and merges them into one
// validated Report. A child that errors is dropped with a recorded reason rather than
// failing the whole sweep or, worse, leaving a silent gap. The returned []DroppedArm is
// the honest record of what did not land.
//
// runner is injected so production passes execArmRunner (real re-exec) and tests pass a
// fake (no spawn). bin is the fak binary path the production runner re-execs; it is
// ignored by an in-process fake.
func SweepViaSubprocess(ctx context.Context, bin string, t *bench.Trace, engineID, engineModel string, configs []FeatureConfig, baseline string, runner armRunner) (*Report, []DroppedArm, error) {
	if t == nil {
		return nil, nil, fmt.Errorf("ablate: nil trace")
	}
	if runner == nil {
		return nil, nil, fmt.Errorf("ablate: nil arm runner (pass execArmRunner in production or a fake in tests)")
	}
	if len(configs) == 0 {
		return nil, nil, fmt.Errorf("ablate: need at least one feature config")
	}
	traceJSON, err := json.Marshal(t)
	if err != nil {
		return nil, nil, fmt.Errorf("ablate: marshal trace: %w", err)
	}
	// The child reconstructs its trace from traceJSON, and WorkloadHash digests the raw
	// Call.Args bytes — which json.Marshal re-emits COMPACTED (a space-bearing source
	// trace round-trips to different bytes). So the parent's authoritative hash must be
	// taken from the SAME round-tripped trace the child will see, or every child looks
	// "forked" and gets dropped. Canonicalize once, here, and hash that.
	canonTrace, err := UnmarshalTrace(json.RawMessage(traceJSON))
	if err != nil {
		return nil, nil, fmt.Errorf("ablate: canonicalize trace: %w", err)
	}
	parentHash := canonTrace.WorkloadHash()

	rep := &Report{
		Provenance:   provenanceFor(canonTrace, engineModel),
		WorkloadHash: parentHash,
		Baseline:     baseline,
	}
	var dropped []DroppedArm
	seen := make(map[string]bool, len(configs))
	for _, c := range configs {
		if c.Name == "" {
			return nil, nil, fmt.Errorf("ablate: a feature config has an empty arm name")
		}
		if seen[c.Name] {
			return nil, nil, fmt.Errorf("ablate: duplicate arm name %q", c.Name)
		}
		seen[c.Name] = true

		run, err := runner(ctx, bin, c, traceJSON)
		if err != nil {
			dropped = append(dropped, DroppedArm{ArmID: c.Name, Reason: err.Error()})
			continue
		}
		// The cross-process identical-workload guard: a child that replayed a DIFFERENT
		// workload (a forked trace, a re-exec drift) is dropped, never folded — the
		// per-feature deltas must be apples-to-apples.
		if run.WorkloadHash != parentHash {
			dropped = append(dropped, DroppedArm{
				ArmID:  c.Name,
				Reason: fmt.Sprintf("workload hash mismatch: child ran %s, parent froze %s", run.WorkloadHash, parentHash),
			})
			continue
		}
		rep.Runs = append(rep.Runs, run)
	}

	// A baseline that was itself dropped cannot anchor the table; refuse rather than
	// silently re-point it.
	if baseline != "" && rep.ArmByID(baseline) == nil {
		return nil, dropped, fmt.Errorf("ablate: baseline arm %q did not run (dropped or absent)", baseline)
	}
	if err := rep.Validate(); err != nil {
		return nil, dropped, err
	}
	return rep, dropped, nil
}

// RunOneArm is the single-arm runner core the CLI arm-mode calls. It replays the frozen
// trace through one FeatureConfig in the CURRENT process and returns one AblationRun. The
// vDSO knob is applied in-process; the env-gated FAK_* features are assumed already live
// (the re-exec'd child read them at ITS process start), so this core does not — and
// cannot — re-read them. The arm's Descriptor still records them, so the artifact names
// the full config the child ran under.
func RunOneArm(ctx context.Context, t *bench.Trace, engineID string, c FeatureConfig) (AblationRun, error) {
	if t == nil {
		return AblationRun{}, fmt.Errorf("ablate: nil trace")
	}
	if c.Name == "" {
		return AblationRun{}, fmt.Errorf("ablate: arm has an empty name")
	}
	return runArm(ctx, t, engineID, c)
}

// RunArmMode is the child entry point the production re-exec drives (and the DEFERRED
// `fak ablate-arm` CLI verb wraps). It reads one armRequest from r, runs that single arm
// in THIS process — whose FAK_* env the parent already set — via RunOneArm, and writes
// the resulting AblationRun as one JSON object to w. Splitting the wire codec out here
// keeps the cmd/fak surface a thin shim and lets the test exercise the exact child path
// without a spawn.
func RunArmMode(ctx context.Context, r io.Reader, w io.Writer, loadTrace func(json.RawMessage) (*bench.Trace, error)) error {
	raw, err := io.ReadAll(r)
	if err != nil {
		return fmt.Errorf("ablate-arm: read request: %w", err)
	}
	var req armRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		return fmt.Errorf("ablate-arm: decode request: %w", err)
	}
	tr, err := loadTrace(req.TraceJSON)
	if err != nil {
		return fmt.Errorf("ablate-arm: load trace: %w", err)
	}
	run, err := RunOneArm(ctx, tr, req.EngineID, req.Config)
	if err != nil {
		return fmt.Errorf("ablate-arm: run arm %q: %w", req.Config.Name, err)
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err := enc.Encode(run); err != nil {
		return fmt.Errorf("ablate-arm: encode result: %w", err)
	}
	return nil
}

// UnmarshalTrace reconstructs a *bench.Trace from the raw trace JSON the parent froze.
// It is the loadTrace closure RunArmMode is wired with in production (and the same
// reconstruction the test uses), so parent and child hash byte-identical workloads.
func UnmarshalTrace(raw json.RawMessage) (*bench.Trace, error) {
	var t bench.Trace
	if err := json.Unmarshal(raw, &t); err != nil {
		return nil, fmt.Errorf("ablate: unmarshal trace: %w", err)
	}
	return &t, nil
}

// execArmRunner is the PRODUCTION armRunner: it re-execs `bin ablate-arm`, splicing the
// arm's FAK_* env onto the child, writes the {config, trace} request to the child's
// stdin, and decodes the child's single AblationRun from stdout. It is the real
// subprocess boundary that makes a process-start env feature ablatable — one fresh
// process per arm, each reading its own FAK_* at start. Tests do NOT use this; they
// inject a fake runner so the suite never spawns the real binary.
func execArmRunner(ctx context.Context, bin string, c FeatureConfig, traceJSON []byte) (AblationRun, error) {
	if bin == "" {
		return AblationRun{}, fmt.Errorf("ablate: empty binary path for re-exec")
	}
	req := armRequest{
		Config:    c,
		EngineID:  "mock",
		TraceJSON: json.RawMessage(traceJSON),
	}
	reqJSON, err := json.Marshal(req)
	if err != nil {
		return AblationRun{}, fmt.Errorf("ablate: marshal arm request: %w", err)
	}

	cmd := exec.CommandContext(ctx, bin, "ablate-arm")
	cmd.Stdin = bytes.NewReader(reqJSON)
	cmd.Env = append(os.Environ(), c.childEnv()...)
	var out, errBuf bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errBuf
	if err := cmd.Run(); err != nil {
		return AblationRun{}, fmt.Errorf("ablate: arm %q child failed: %w (stderr: %s)", c.Name, err, errBuf.String())
	}

	var run AblationRun
	if err := json.Unmarshal(out.Bytes(), &run); err != nil {
		return AblationRun{}, fmt.Errorf("ablate: decode arm %q child output: %w", c.Name, err)
	}
	return run, nil
}

// ExecArmRunner is the exported production armRunner: the real subprocess re-exec that
// makes a process-start env feature ablatable (one fresh child per arm, each reading its
// own FAK_* at start). cmd/fak injects it into SweepViaSubprocess for the live rung-2 path;
// the package's own tests inject a fake instead, so the suite never spawns the real binary.
// It surfaces execArmRunner without renaming it, so the internal name — and the doc
// comments and compile-time assertion that reference it — stay intact.
func ExecArmRunner(ctx context.Context, bin string, c FeatureConfig, traceJSON []byte) (AblationRun, error) {
	return execArmRunner(ctx, bin, c, traceJSON)
}

// provenanceFor builds the report provenance for a subprocess sweep (the parent stamps
// it once; children only report their own AblationRun). Factored so both Sweep and
// SweepViaSubprocess stamp identically.
func provenanceFor(t *bench.Trace, engineModel string) metrics.Provenance {
	return metrics.Provenance{
		AppVersion:   appversion.Current(),
		Command:      "fak ablate --subprocess --suite " + t.SliceID,
		EngineModel:  engineModel,
		SliceID:      t.SliceID,
		WorkloadHash: t.WorkloadHash(),
		GoVersion:    runtime.Version(),
		OS:           runtime.GOOS,
		GeneratedBy:  "fak/internal/ablate",
	}
}

// compile-time assertion: execArmRunner satisfies the armRunner contract the parent
// fan-out and the test fake both implement.
var _ armRunner = execArmRunner
