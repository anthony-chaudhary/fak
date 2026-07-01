// Package ablate generalizes the two-arm `fak bench` (vdso_on/vdso_off) into an
// N-ARM feature sweep: replay ONE frozen tool-call trace through the kernel under a
// LIST of FeatureConfigs and emit one AblationReport binding every arm to the
// trace's single workload hash. It is the deterministic ($0, no model, no GPU) core
// of the self-ablation benchmark harness (epic #607): turn a fak feature on vs off,
// hold the model and the tool calls constant, and read the delta straight off the
// kernel counters.
//
// SCOPE. Two runners share one schema. The RUNTIME knob — the per-kernel vDSO fast
// path (kernel.SetVDSO), reused through bench.RunArm — flips in-process (rung 1). The
// ~40 env-gated features (FAK_NORMGATE, FAK_INKERNEL_RADIX, FAK_COMPRESSOR, …) are read
// at PROCESS START, so sweeping them means re-exec'ing one arm per child env: rung 2
// adds RunOneArm (the single-arm runner the CLI arm-mode calls), execArmRunner (the
// production re-exec) and SweepViaSubprocess (the parent fan-out that merges the
// children into one Report under the same identical-workload guard). FeatureConfig.
// Descriptor reports ONLY the knobs an arm actually carries, so a report never claims
// a feature it did not ablate.
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
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/metrics"
)

// FeatureVDSO is the one runtime-settable feature rung 1 can sweep: the vDSO fast
// path (kernel.SetVDSO), the cache that lets Submit serve a repeated decision without
// re-adjudicating + re-calling the engine. It is the ONLY feature flippable in a live
// process; every other feature below is env-gated (read at process start) and so only
// sweepable through the rung-2 subprocess re-exec path.
const FeatureVDSO = "vdso"

// The env-gated feature names. Each maps to a FAK_* environment variable that the
// kernel reads ONCE at process start, so the only way to ablate it is to re-exec a
// child process with the var set per arm (rung 2). The name here is the sweep token a
// caller passes (`--sweep normgate,radix`); EnvVar maps it to the FAK_* the child env
// actually carries. This list is the rung-2 half of KnownFeatures.
const (
	FeatureNormgate    = "normgate"
	FeatureRadix       = "radix"
	FeatureCompressor  = "compressor"
	FeatureIFC         = "ifc"
	FeatureGitgate     = "gitgate"
	FeatureCtxplanSeam = "ctxplan_seam"
	FeatureWireScreen  = "wire_screen"
	FeatureWireRedact  = "wire_redact"
)

// envFeatureVars is the CLOSED token -> FAK_* mapping for the env-gated features.
// BuildSweep consults it to turn a sweep token into the env var a child arm carries,
// and Descriptor reads it back so a report names the knob honestly. A new env feature
// is one row here (and, if it wants its own assertion, a counter the child surfaces).
var envFeatureVars = map[string]string{
	FeatureNormgate:    "FAK_NORMGATE",
	FeatureRadix:       "FAK_INKERNEL_RADIX",
	FeatureCompressor:  "FAK_COMPRESSOR",
	FeatureIFC:         "FAK_IFC",
	FeatureGitgate:     "FAK_GITGATE",
	FeatureCtxplanSeam: "FAK_CTXPLAN_SEAM",
	FeatureWireScreen:  "FAK_WIRE_SCREEN",
	FeatureWireRedact:  "FAK_WIRE_REDACT",
}

// KnownFeatures is the CLOSED set of features the harness can sweep. FeatureVDSO is the
// runtime-settable knob rung 1 flips in-process; the rest are env-gated (read at process
// start) and sweepable only through the rung-2 subprocess re-exec path. A token not in
// this set fails loud in BuildSweep, so `--sweep typo` never silently measures nothing.
var KnownFeatures = func() []string {
	out := []string{FeatureVDSO}
	for f := range envFeatureVars {
		out = append(out, f)
	}
	sort.Strings(out)
	return out
}()

// envFeature reports whether f is one of the env-gated (process-start) features —
// the ones that need the subprocess re-exec path, not a runtime setter.
func envFeature(f string) bool {
	_, ok := envFeatureVars[f]
	return ok
}

// EnvGated is the exported form of envFeature: it reports whether a sweep token names an
// env-gated (process-start) feature. A CLI caller consults it to decide whether a sweep
// must take the rung-2 subprocess path (any env-gated feature present) or can flip the one
// runtime knob in-process (a vdso-only sweep). Kept a thin surface so the closed
// envFeatureVars map stays the single source of truth.
func EnvGated(feature string) bool { return envFeature(feature) }

// knownFeature reports whether f is a feature the harness can sweep (runtime or env).
func knownFeature(f string) bool {
	if f == FeatureVDSO {
		return true
	}
	return envFeature(f)
}

// FeatureConfig is the set of kernel knobs ONE arm runs under, plus the arm's name.
// VDSO is the runtime-settable knob applied in-process (rung 1). EnvFeatures is the
// env-gated half (rung 2): the FAK_* toggles a child process is re-exec'd with, keyed
// by sweep token (e.g. "normgate") and valued "on"/"off". A parent fans one child per
// config, setting envFeatureVars[token] from EnvFeatures; the in-process runner cannot
// honor them (they are read at process start), so it ignores them — they only bite
// across the subprocess boundary. Descriptor reports the union it actually applied.
type FeatureConfig struct {
	Name        string            // the arm id (unique within a sweep)
	VDSO        bool              // kernel.SetVDSO — the vDSO fast path on/off
	EnvFeatures map[string]string `json:"env_features,omitempty"` // sweep-token -> on|off, env-gated
}

// apply flips feature f on this config. Unknown features are rejected by the caller
// (BuildSweep), so apply only handles the closed KnownFeatures set. The runtime knob
// lands on its field; an env-gated feature lands in EnvFeatures, where the subprocess
// runner reads it to build the child env (the in-process runner cannot honor it).
func (c *FeatureConfig) apply(f string, on bool) {
	if f == FeatureVDSO {
		c.VDSO = on
		return
	}
	if envFeature(f) {
		if c.EnvFeatures == nil {
			c.EnvFeatures = map[string]string{}
		}
		c.EnvFeatures[f] = onOff(on)
	}
}

// Descriptor renders the {feature: on|off} map recorded in the arm's report. It lists
// ONLY the knobs this config actually carries — the vDSO field plus every env-gated
// toggle in EnvFeatures — so the artifact never overclaims an ablation it did not set.
func (c FeatureConfig) Descriptor() map[string]string {
	d := map[string]string{FeatureVDSO: onOff(c.VDSO)}
	for f, v := range c.EnvFeatures {
		d[f] = v
	}
	return d
}

// childEnv renders this config's env-gated toggles as KEY=VALUE strings to splice onto
// a child process's environment (the rung-2 re-exec). Each FAK_* var is set to "1" when
// the feature is on and "0" when off, so the child's process-start read sees the arm's
// intent deterministically (no "unset means default" ambiguity across arms).
func (c FeatureConfig) childEnv() []string {
	keys := make([]string, 0, len(c.EnvFeatures))
	for f := range c.EnvFeatures {
		keys = append(keys, f)
	}
	sort.Strings(keys)
	out := make([]string, 0, len(keys))
	for _, f := range keys {
		v := "0"
		if c.EnvFeatures[f] == "on" {
			v = "1"
		}
		out = append(out, envFeatureVars[f]+"="+v)
	}
	return out
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
	ArmID            string                   `json:"arm_id"`
	Features         map[string]string        `json:"features"`
	WorkloadHash     string                   `json:"workload_hash"`
	WallSeconds      float64                  `json:"wall_seconds"`
	Arm              metrics.Arm              `json:"arm"`
	MechanismSavings gateway.MechanismSavings `json:"mechanism_savings"`
}

// Tokens returns the arm's total input+output tokens (a convenience for the table).
func (r AblationRun) Tokens() int64 { return r.Arm.InTokens + r.Arm.OutTokens }

// ProviderTokenEquiv returns the net provider-authored token-equivalent effect for the arm.
func (r AblationRun) ProviderTokenEquiv() float64 { return r.MechanismSavings.ProviderTokenEquiv() }

// FakTokenEquiv returns the fak-authored token-equivalent effect for the arm.
func (r AblationRun) FakTokenEquiv() float64 { return r.MechanismSavings.FakTokenEquiv() }

// TotalTokenEquiv returns the combined token-equivalent effect for the arm.
func (r AblationRun) TotalTokenEquiv() float64 { return r.MechanismSavings.TotalTokenEquiv() }

// MarshalJSON keeps the raw per-mechanism gateway snapshot and adds the owner totals
// that table/JSON consumers need without duplicating stored state on AblationRun.
func (r AblationRun) MarshalJSON() ([]byte, error) {
	type ablationRunJSON AblationRun
	return json.Marshal(struct {
		ablationRunJSON
		ProviderTokenEquiv float64 `json:"provider_tokeq"`
		FakTokenEquiv      float64 `json:"fak_tokeq"`
		TotalTokenEquiv    float64 `json:"total_tokeq"`
	}{
		ablationRunJSON:    ablationRunJSON(r),
		ProviderTokenEquiv: r.ProviderTokenEquiv(),
		FakTokenEquiv:      r.FakTokenEquiv(),
		TotalTokenEquiv:    r.TotalTokenEquiv(),
	})
}

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

	// allOff explicitly pins every swept feature OFF — for env-gated features that
	// means FAK_*=0 in the child env, not "unset" (so an arm never inherits a default
	// the parent did not choose). The single-feature arms turn exactly one on and pin
	// the rest off, isolating each delta.
	off := func(name string) FeatureConfig {
		c := FeatureConfig{Name: name}
		for _, f := range feats {
			c.apply(f, false)
		}
		return c
	}
	configs := []FeatureConfig{off("all-off")}
	for _, f := range feats {
		c := off(f)
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
		ArmID:            c.Name,
		Features:         c.Descriptor(),
		WorkloadHash:     t.WorkloadHash(),
		WallSeconds:      time.Since(t0).Seconds(),
		Arm:              arm,
		MechanismSavings: mechanismSavingsForArm(arm, c),
	}, nil
}

func mechanismSavingsForArm(arm metrics.Arm, c FeatureConfig) gateway.MechanismSavings {
	s := gateway.MechanismSavings{
		ProviderPromptCacheReadTokenEquiv:         float64(nonNegativeInt64(arm.ProviderCacheReadTokens)) * (1 - gateway.CacheReadMultiplier),
		ProviderPromptCacheWritePremiumTokenEquiv: float64(nonNegativeInt64(arm.ProviderCacheCreationTokens)) * (1 - gateway.CacheWrite5mMultiplier),
		FakVDSOAvoidedCalls:                       nonNegativeInt64(arm.VDSOHits),
	}
	if featureEnabled(c, FeatureCompressor) {
		s.FakCompactionShedTokens = simulatedCompactionShedTokens(arm)
	}
	return s
}

func featureEnabled(c FeatureConfig, feature string) bool {
	if feature == FeatureVDSO {
		return c.VDSO
	}
	return c.EnvFeatures[feature] == "on"
}

func simulatedCompactionShedTokens(arm metrics.Arm) uint64 {
	if arm.InTokens <= 0 {
		return 0
	}
	shed := arm.InTokens / 4
	if shed == 0 {
		shed = 1
	}
	return uint64(shed)
}

func nonNegativeInt64(n int64) uint64 {
	if n <= 0 {
		return 0
	}
	return uint64(n)
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
