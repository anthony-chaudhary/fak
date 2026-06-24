// Package ablate generalizes the two-arm `fak bench` (vdso_on/vdso_off) into an
// N-ARM feature sweep: replay ONE frozen tool-call trace through the kernel under a
// LIST of FeatureConfigs and emit one AblationReport binding every arm to the
// trace's single workload hash. It is the deterministic ($0, no model, no GPU) core
// of the self-ablation benchmark harness (epic #607): turn a fak feature on vs off,
// hold the model and the tool calls constant, and read the delta straight off the
// kernel counters.
//
// SCOPE (rung 1). Only RUNTIME-SETTABLE kernel knobs are swept here — for now the
// per-kernel vDSO fast path (kernel.SetVDSO), reused through bench.RunArm. The ~40
// env-gated features (FAK_NORMGATE, FAK_INKERNEL_RADIX, FAK_COMPRESSOR, …) are read
// at PROCESS START, so sweeping them means re-exec'ing one arm per child env — a
// follow-on rung tracked on the epic. FeatureConfig.Descriptor reports ONLY the
// knobs an arm actually flips, so a report never claims a feature it did not ablate.
//
// VALIDITY. Every arm replays the SAME *bench.Trace, so one Trace.WorkloadHash binds
// them all; Report.Validate refuses the report if any arm ran a different workload —
// generalizing metrics.Report.Validate from a fixed pair to N. This identical-
// workload guard is what makes the per-feature deltas apples-to-apples (the same
// guard the 2-arm bench ships). It is the load-bearing difference from a cross-AGENT
// ablation (pure fak vs Claude Code), where tool calls are nondeterministic and the
// hash guard does NOT apply — that half is a separate harness on the epic, by design.
//
// The package is a bench sibling (tier-4 integrator): it imports bench + metrics and
// is never on a live tool-call decision path.
package ablate

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/bench"
	"github.com/anthony-chaudhary/fak/internal/metrics"
)

// FeatureVDSO is the one runtime-settable feature rung 1 can sweep: the vDSO fast
// path (kernel.SetVDSO), the cache that lets Submit serve a repeated decision without
// re-adjudicating + re-calling the engine.
const FeatureVDSO = "vdso"

// KnownFeatures is the CLOSED set of features this rung knows how to flip at runtime.
// A new entry is a new knob (extend FeatureConfig + apply + Descriptor). Env-gated
// features (FAK_NORMGATE, FAK_INKERNEL_RADIX, …) read at process start and belong to
// the subprocess re-exec rung — they are deliberately NOT listed here, so `--sweep
// normgate` fails loud today instead of silently measuring nothing.
var KnownFeatures = []string{FeatureVDSO}

// knownFeature reports whether f is a feature this rung can sweep at runtime.
func knownFeature(f string) bool {
	for _, k := range KnownFeatures {
		if k == f {
			return true
		}
	}
	return false
}

// FeatureConfig is the set of RUNTIME-SETTABLE kernel knobs ONE arm runs under, plus
// the arm's name. Rung 1 exposes only the per-kernel vDSO toggle; the struct grows a
// field per knob as runtime setters land (e.g. a PolicyFloor bool once SetPolicyFloor
// exists), and Descriptor grows with it.
type FeatureConfig struct {
	Name string // the arm id (unique within a sweep)
	VDSO bool   // kernel.SetVDSO — the vDSO fast path on/off
}

// apply flips feature f on this config. Unknown features are rejected by the caller
// (BuildSweep), so apply only handles the closed KnownFeatures set.
func (c *FeatureConfig) apply(f string, on bool) {
	switch f {
	case FeatureVDSO:
		c.VDSO = on
	}
}

// Descriptor renders the {feature: on|off} map recorded in the arm's report. It lists
// ONLY the knobs this rung actually applies, so the artifact never overclaims an
// ablation it did not perform.
func (c FeatureConfig) Descriptor() map[string]string {
	return map[string]string{FeatureVDSO: onOff(c.VDSO)}
}

// onOff renders a bool as the report's on/off vocabulary.
func onOff(b bool) string {
	if b {
		return "on"
	}
	return "off"
}

// AblationRun is one arm: a FeatureConfig + the measured metrics. It generalizes
// metrics.Arm (hardwired to the vdso_on/vdso_off pair) to an arbitrary labeled config,
// and carries the per-arm workload hash (the identical-workload guard input) and the
// arm's replay wall-clock. The reserved cross-arm fields (live wall, turns, success,
// variance) populate in the live + cross-agent rungs; this rung leaves them on Arm's
// counters and the embedded metrics.
type AblationRun struct {
	ArmID        string            `json:"arm_id"`
	Features     map[string]string `json:"features"`
	WorkloadHash string            `json:"workload_hash"`
	WallSeconds  float64           `json:"wall_seconds"`
	Arm          metrics.Arm       `json:"arm"`
}

// Tokens returns the arm's total input+output tokens (a convenience for the table).
func (r AblationRun) Tokens() int64 { return r.Arm.InTokens + r.Arm.OutTokens }

// Report is the N-arm artifact: shared provenance, one AblationRun per config, and
// the single trace workload hash that binds them. Baseline names the arm used as the
// delta reference in the human table.
type Report struct {
	Provenance   metrics.Provenance `json:"provenance"`
	WorkloadHash string             `json:"workload_hash"`
	Baseline     string             `json:"baseline_arm,omitempty"`
	Runs         []AblationRun      `json:"runs"`
}

// ArmByID returns the arm with the given id, or nil.
func (r *Report) ArmByID(id string) *AblationRun {
	for i := range r.Runs {
		if r.Runs[i].ArmID == id {
			return &r.Runs[i]
		}
	}
	return nil
}

// Validate is the N-arm identical-workload guard (generalizing metrics.Report.Validate
// from a fixed pair to a slice): it refuses the report unless every arm ran the trace
// the report binds. A mismatch means an arm replayed different work, so the deltas
// would not be apples-to-apples — fail closed, never a best-effort partial.
func (r *Report) Validate() error {
	if len(r.Runs) == 0 {
		return errors.New("ablate: report has no arms")
	}
	for _, run := range r.Runs {
		if run.WorkloadHash != r.WorkloadHash {
			return fmt.Errorf("ablate: refusing to compare arms with different workload hashes (arm %q ran %s, report binds %s)",
				run.ArmID, run.WorkloadHash, r.WorkloadHash)
		}
	}
	if r.Baseline != "" && r.ArmByID(r.Baseline) == nil {
		return fmt.Errorf("ablate: baseline arm %q is not among the %d arms", r.Baseline, len(r.Runs))
	}
	return nil
}

// JSON renders the report as canonical indented JSON terminated by a newline.
func (r *Report) JSON() []byte {
	b, _ := json.MarshalIndent(r, "", "  ")
	return append(b, '\n')
}

// BuildSweep turns a list of feature names into the arm matrix: a fail-closed
// "all-off" baseline arm (every feature off), one arm per single feature turned on
// (others off — so each delta isolates exactly that knob), and, when more than one
// feature is swept, an "all-on" arm. An unknown feature fails loud (it is not a
// runtime knob this rung can flip). Duplicate feature names collapse to one arm.
func BuildSweep(features []string) ([]FeatureConfig, error) {
	seen := map[string]bool{}
	var feats []string
	for _, f := range features {
		f = strings.TrimSpace(f)
		if f == "" {
			continue
		}
		if !knownFeature(f) {
			return nil, fmt.Errorf("ablate: unknown runtime feature %q (this rung can sweep: %s; env-gated features are a subprocess rung)",
				f, strings.Join(KnownFeatures, ", "))
		}
		if !seen[f] {
			seen[f] = true
			feats = append(feats, f)
		}
	}
	if len(feats) == 0 {
		return nil, errors.New("ablate: no features to sweep (try --sweep " + strings.Join(KnownFeatures, ",") + ")")
	}

	configs := []FeatureConfig{{Name: "all-off"}} // every known knob off
	for _, f := range feats {
		c := FeatureConfig{Name: f}
		c.apply(f, true)
		configs = append(configs, c)
	}
	if len(feats) > 1 {
		all := FeatureConfig{Name: "all-on"}
		for _, f := range feats {
			all.apply(f, true)
		}
		configs = append(configs, all)
	}
	return configs, nil
}

// runArm replays the frozen trace through a fresh kernel under one config and wraps
// the resulting metrics.Arm into an AblationRun. The vDSO knob is applied via
// bench.RunArm (kernel.SetVDSO); when a second runtime knob lands, this is where the
// per-arm kernel construction moves off RunArm to apply the wider config.
func runArm(ctx context.Context, t *bench.Trace, engineID string, c FeatureConfig) (AblationRun, error) {
	t0 := time.Now()
	arm, err := bench.RunArm(ctx, t, engineID, c.VDSO, c.Name)
	if err != nil {
		return AblationRun{}, err
	}
	return AblationRun{
		ArmID:        c.Name,
		Features:     c.Descriptor(),
		WorkloadHash: t.WorkloadHash(),
		WallSeconds:  time.Since(t0).Seconds(),
		Arm:          arm,
	}, nil
}

// Sweep runs the frozen trace through every config and assembles the validated
// report. It enforces unique arm names and the N-arm identical-workload guard before
// returning — a failed guard halts assembly (no partial report).
func Sweep(ctx context.Context, t *bench.Trace, engineID, engineModel string, configs []FeatureConfig, baseline string) (*Report, error) {
	if t == nil {
		return nil, errors.New("ablate: nil trace")
	}
	if len(configs) == 0 {
		return nil, errors.New("ablate: need at least one feature config")
	}
	wh := t.WorkloadHash()
	rep := &Report{
		Provenance: metrics.Provenance{
			AppVersion:   appversion.Current(),
			Command:      "fak ablate --suite " + t.SliceID,
			EngineModel:  engineModel,
			SliceID:      t.SliceID,
			WorkloadHash: wh,
			GoVersion:    runtime.Version(),
			OS:           runtime.GOOS,
			GeneratedBy:  "fak/internal/ablate",
		},
		WorkloadHash: wh,
		Baseline:     baseline,
	}
	seen := make(map[string]bool, len(configs))
	for _, c := range configs {
		if c.Name == "" {
			return nil, errors.New("ablate: a feature config has an empty arm name")
		}
		if seen[c.Name] {
			return nil, fmt.Errorf("ablate: duplicate arm name %q", c.Name)
		}
		seen[c.Name] = true
		run, err := runArm(ctx, t, engineID, c)
		if err != nil {
			return nil, err
		}
		rep.Runs = append(rep.Runs, run)
	}
	if err := rep.Validate(); err != nil {
		return nil, err
	}
	return rep, nil
}

// FeatureKeys returns an arm descriptor's feature names in sorted order (a stable
// rendering helper for the table and tests).
func FeatureKeys(d map[string]string) []string {
	keys := make([]string, 0, len(d))
	for k := range d {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
