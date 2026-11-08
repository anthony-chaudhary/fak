// Package optimize is the leaf logic behind `fak optimize`, the dedicated
// inference-optimization driver. It projects the committed native-performance
// hill-climb graph (internal/nativeperf) into a bounded OptimizePlan -> measure ->
// verify loop:
//
//	OptimizePlan    selects the next dependency-ready unwitnessed lever (or the
//	        profile-classified one) and emits the exact follow-on commands.
//	measure replays one frozen tool-call trace through ablate.Sweep under the
//	        lever's mapped runtime feature and reports the tok/s delta, or folds
//	        a nativeperf baseline/candidate receipt pair captured on real hardware
//	        into the same OptimizeReceipt (--receipt; exactly one source).
//	verify  folds two receipts into an accept/reject OptimizeVerdict under either the
//	        preregistered nativeperf comparison criterion or the OptimizePlan's
//	        expected floor..roof band for the shared lever.
//	loop    drives the three in a bounded hill-climb that stops after the first
//	        reject: gains are only ever stacked when witnessed.
//
// Every unrunnable path refuses with a closed-vocabulary OptimizeRefusal class instead
// of fabricating numbers or mappings. The package never runs a model and never
// writes files; the CLI layer owns sinks and exit codes.
package optimize

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ablate"
	"github.com/anthony-chaudhary/fak/internal/bench"
	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

// DefaultTracePath is the canonical frozen smoke trace shipped under
// testdata/tau2; OptimizePlan's emitted measure command and loop's default workload
// both pin it so the invocations are exact and re-runnable.
const DefaultTracePath = "testdata/tau2/tau2-smoke.json"

// Config pins everything a core needs: the lever graph, the closed
// lever->ablate-feature mapping, the sweep engine, and the clock. Zero fields
// fall back to the committed defaults below. EnvelopeID optionally restricts
// plan/measure/verify to one pinned execution envelope; empty means the whole
// committed graph.
type Config struct {
	Graph      nativeperf.Graph
	Mappings   map[string]string
	EngineID   string
	EnvelopeID string
	Now        func() time.Time
}

func (c Config) engineID() string {
	if c.EngineID == "" {
		return "mock"
	}
	return c.EngineID
}

func (c Config) now() time.Time {
	if c.Now == nil {
		return time.Now().UTC()
	}
	return c.Now().UTC()
}

func (c Config) mappings() map[string]string {
	if c.Mappings == nil {
		return DefaultArmMappings
	}
	return c.Mappings
}

// DefaultArmMappings is the closed lever->sweep-token table for the committed
// graph. Every committed lever is a Metal/CUDA hardware lever with no
// trace-backed ablate runtime arm, so the table ships empty: measuring a
// committed lever refuses with ClassNoArmMapping rather than inventing a
// mapping. Tests and future envelopes inject their own tables through
// Config.Mappings.
var DefaultArmMappings = map[string]string{}

// planningRungs binds each committed lever to the graph rung whose expected
// range is its planning band. The binding is closed and explicit: a lever
// without an entry plans without a band and verifies against ClassNoPlanningBand
// instead of borrowing a neighboring rung's numbers.
var planningRungs = map[string]string{
	"metal.command-buffer-amortization":    "coarse-resident-hybrid-graph",
	"metal.fused-hybrid-graph-coverage":    "coarse-resident-hybrid-graph",
	"metal.matched-parity-OptimizeReceipt": "matched-native-parity",
}

// BuildPlan selects the next lever in graph order and renders the exact
// follow-on invocations. Read-only: it loads nothing but the committed graph.
// When Config.EnvelopeID names a pinned envelope, selection is restricted to
// that envelope's levers; an envelope no graph row or lever applicability
// carries refuses with ClassUnknownEnvelope instead of planning elsewhere.
func BuildPlan(cfg Config) (*OptimizePlan, error) {
	return planForLever(cfg, func() (*nativeperf.Lever, error) {
		if cfg.EnvelopeID == "" {
			lever, err := nativeperf.NextLever(cfg.Graph)
			if err != nil {
				return nil, err
			}
			if lever == nil {
				return nil, NewOptimizeRefusal(ClassNoReadyLever, "no dependency-ready unwitnessed lever in the graph")
			}
			return lever, nil
		}
		if !envelopeKnown(cfg.Graph, cfg.EnvelopeID) {
			return nil, NewOptimizeRefusal(ClassUnknownEnvelope, fmt.Sprintf("envelope %q is not pinned in the graph: no envelope row or lever applicability carries it", cfg.EnvelopeID))
		}
		ready, err := nativeperf.ReadyLevers(cfg.Graph)
		if err != nil {
			return nil, err
		}
		for i := range ready {
			if ready[i].Applicability.EnvelopeID == cfg.EnvelopeID {
				return &ready[i], nil
			}
		}
		return nil, NewOptimizeRefusal(ClassNoReadyLever, fmt.Sprintf("no dependency-ready unwitnessed lever in envelope %q", cfg.EnvelopeID))
	})
}

// envelopeKnown reports whether the graph carries the envelope: a pinned
// envelope row or at least one lever applicability names it.
func envelopeKnown(graph nativeperf.Graph, id string) bool {
	for _, e := range graph.Envelopes {
		if e.ID == id {
			return true
		}
	}
	for _, l := range graph.Levers {
		if l.Applicability.EnvelopeID == id {
			return true
		}
	}
	return false
}

// PlanFromProfile selects the next lever from a decoded native profile bundle
// through the same classifier path `fak native-performance --profile-next`
// uses. A profile without bottleneck evidence refuses as no ready lever. The
// optional EnvelopeID pin refuses unknown envelopes and levers outside the
// requested envelope.
func BuildPlanFromProfile(cfg Config, profile nativeperf.ProfileBundle) (*OptimizePlan, error) {
	if cfg.EnvelopeID != "" && !envelopeKnown(cfg.Graph, cfg.EnvelopeID) {
		return nil, NewOptimizeRefusal(ClassUnknownEnvelope, fmt.Sprintf("envelope %q is not pinned in the graph: no envelope row or lever applicability carries it", cfg.EnvelopeID))
	}
	lever, _, err := nativeperf.NextLeverFromProfile(cfg.Graph, profile)
	if err != nil {
		if errors.Is(err, nativeperf.ErrProfileEvidenceUnavailable) {
			return nil, NewOptimizeRefusal(ClassNoReadyLever, "profile carries no bottleneck evidence selecting a lever")
		}
		return nil, err
	}
	if lever == nil {
		if cfg.EnvelopeID != "" {
			return nil, NewOptimizeRefusal(ClassNoReadyLever, fmt.Sprintf("profile selected no lever in envelope %q", cfg.EnvelopeID))
		}
		return nil, NewOptimizeRefusal(ClassNoReadyLever, "profile selected no lever")
	}
	chosen := *lever
	if cfg.EnvelopeID != "" && chosen.Applicability.EnvelopeID != cfg.EnvelopeID {
		return nil, NewOptimizeRefusal(ClassNoReadyLever, fmt.Sprintf("profile selected lever %q outside the requested envelope %q", chosen.ID, cfg.EnvelopeID))
	}
	return planForLever(cfg, func() (*nativeperf.Lever, error) { return &chosen, nil })
}

func planForLever(cfg Config, pick func() (*nativeperf.Lever, error)) (*OptimizePlan, error) {
	if err := nativeperf.Validate(cfg.Graph); err != nil {
		return nil, err
	}
	lever, err := pick()
	if err != nil {
		return nil, err
	}
	envFlag := ""
	if cfg.EnvelopeID != "" {
		envFlag = " --envelope " + cfg.EnvelopeID
	}
	OptimizePlan := &OptimizePlan{
		Schema:          PlanSchema,
		LeverID:         lever.ID,
		Envelope:        lever.Applicability.EnvelopeID,
		DependencyReady: dependenciesReady(cfg.Graph, *lever),
		MeasureCmd:      "fak optimize measure --lever " + lever.ID + envFlag + " --trace " + DefaultTracePath + " --out " + lever.ID + ".json",
		VerifyCmd:       "fak optimize verify --baseline " + lever.ID + ".baseline.json --candidate " + lever.ID + ".json" + envFlag + " --json",
		GeneratedAt:     cfg.now().Format(time.RFC3339),
	}
	if floor, roof, ok := planningBand(cfg.Graph, lever.ID); ok {
		OptimizePlan.ExpectedFloor = &floor
		OptimizePlan.ExpectedRoof = &roof
	}
	return OptimizePlan, nil
}

// planningBand resolves the lever's registered planning band from the graph
// rung planningRungs points at. Absent binding or absent rung means no band.
func planningBand(graph nativeperf.Graph, leverID string) (floor, roof float64, ok bool) {
	rungID, known := planningRungs[leverID]
	if !known {
		return 0, 0, false
	}
	for _, rung := range graph.Rungs {
		if rung.ID == rungID {
			return rung.Expected.FloorTokensPerSecond, rung.Expected.RoofTokensPerSecond, true
		}
	}
	return 0, 0, false
}

func dependenciesReady(graph nativeperf.Graph, lever nativeperf.Lever) bool {
	byID := make(map[string]nativeperf.Lever, len(graph.Levers))
	for _, l := range graph.Levers {
		byID[l.ID] = l
	}
	for _, dep := range lever.DependencyIDs {
		if !byID[dep].Enabled {
			return false
		}
	}
	return true
}

// Measure runs the frozen-trace A/B sweep for one lever's mapped ablate arm,
// or folds a nativeperf baseline/candidate receipt pair captured on real
// hardware into the same OptimizeReceipt. Exactly one of tracePath or
// receiptPath must name the workload source; supplying both refuses. It
// refuses (never fabricates) when the lever is unknown to the graph, when the
// requested envelope does not pin the lever, when the lever has no trace-backed
// arm mapping among ablate's known features, when the trace will not load, or
// when a receipt misses an identity gate or an arm.
func Measure(ctx context.Context, cfg Config, leverID, tracePath, receiptPath string) (*OptimizeReceipt, error) {
	if tracePath != "" && receiptPath != "" {
		return nil, NewOptimizeRefusal(ClassInvalidInput, "--trace and --receipt are mutually exclusive; supply exactly one workload source")
	}
	var lever *nativeperf.Lever
	for i := range cfg.Graph.Levers {
		if cfg.Graph.Levers[i].ID == leverID {
			lever = &cfg.Graph.Levers[i]
			break
		}
	}
	if lever == nil {
		return nil, NewOptimizeRefusal(ClassUnknownLever, fmt.Sprintf("lever %q is not in the graph", leverID))
	}
	if cfg.EnvelopeID != "" && lever.Applicability.EnvelopeID != cfg.EnvelopeID {
		return nil, NewOptimizeRefusal(ClassInvalidInput, fmt.Sprintf("lever %q is pinned to envelope %q, not the requested %q", leverID, lever.Applicability.EnvelopeID, cfg.EnvelopeID))
	}
	if receiptPath != "" {
		return measureFromReceipt(cfg, lever, receiptPath)
	}
	if tracePath == "" {
		return nil, NewOptimizeRefusal(ClassInvalidInput, "exactly one of --trace or --receipt is required")
	}
	token, mapped := cfg.mappings()[leverID]
	if !mapped || !knownAblateFeature(token) {
		return nil, NewOptimizeRefusal(ClassNoArmMapping, fmt.Sprintf(
			"lever %q has no trace-backed arm mapping: committed graph levers are Metal/CUDA hardware levers outside ablate's known runtime features (%s)",
			leverID, strings.Join(ablate.KnownFeatures(), ", ")))
	}
	if ablate.EnvGated(token) {
		// Env-gated arms are read at process start, so an in-process sweep would
		// measure a no-op; the tracer refuses instead of reporting fabricated
		// neutrality.
		return nil, NewOptimizeRefusal(ClassNoArmMapping, fmt.Sprintf("lever %q maps to env-gated feature %q, which the in-process optimize sweep cannot flip", leverID, token))
	}
	trace, err := bench.LoadTrace(tracePath)
	if err != nil {
		return nil, NewOptimizeRefusal(ClassTraceUnreadable, fmt.Sprintf("load trace %s: %v", tracePath, err))
	}
	configs, err := ablate.BuildSweep([]string{token})
	if err != nil {
		return nil, err
	}
	engineID := cfg.engineID()
	rep, err := ablate.Sweep(ctx, trace, engineID, engineID+"-offline", configs, "all-off")
	if err != nil {
		return nil, err
	}
	base := rep.ArmByID("all-off")
	cand := rep.ArmByID(token)
	if base == nil || cand == nil {
		return nil, fmt.Errorf("optimize: sweep report lost an arm (all-off present=%v, %s present=%v)", base != nil, token, cand != nil)
	}
	baseTokS, err := armTokS(*base)
	if err != nil {
		return nil, err
	}
	candTokS, err := armTokS(*cand)
	if err != nil {
		return nil, err
	}
	raw, err := os.ReadFile(tracePath)
	if err != nil {
		return nil, NewOptimizeRefusal(ClassTraceUnreadable, fmt.Sprintf("hash trace %s: %v", tracePath, err))
	}
	sum := sha256.Sum256(raw)
	delta := 0.0
	if baseTokS > 0 {
		delta = (candTokS - baseTokS) / baseTokS * 100
	}
	return &OptimizeReceipt{
		Schema:        ReceiptSchema,
		LeverID:       leverID,
		EngineID:      engineID,
		TracePath:     tracePath,
		TraceSHA256:   hex.EncodeToString(sum[:]),
		BaselineTokS:  baseTokS,
		CandidateTokS: candTokS,
		DeltaPct:      delta,
		Reps:          1,
		MeasuredAt:    cfg.now().Format(time.RFC3339),
	}, nil
}

// measureFromReceipt folds one nativeperf A/B receipt pair into the
// OptimizeReceipt. The file must hold a JSON array of exactly two
// ExperimentReceipts, baseline then candidate; every identity gate fails
// closed with a typed refusal, and a missing arm refuses instead of reporting
// a fabricated zero. BaselineTokS/CandidateTokS are the means over each arm's
// repetitions; Revision/ArtifactSHA256/ForwardPath carry the candidate arm's
// execution identity.
func measureFromReceipt(cfg Config, lever *nativeperf.Lever, receiptPath string) (*OptimizeReceipt, error) {
	raw, err := os.ReadFile(receiptPath)
	if err != nil {
		return nil, NewOptimizeRefusal(ClassInvalidInput, fmt.Sprintf("read receipt %s: %v", receiptPath, err))
	}
	var elements []json.RawMessage
	if err := json.Unmarshal(raw, &elements); err != nil {
		return nil, NewOptimizeRefusal(ClassInvalidInput, fmt.Sprintf("decode receipt %s: expected a JSON array of exactly two nativeperf receipts: %v", receiptPath, err))
	}
	if len(elements) != 2 {
		return nil, NewOptimizeRefusal(ClassInvalidInput, fmt.Sprintf("receipt %s carries %d values; the A/B pair requires exactly two (baseline then candidate)", receiptPath, len(elements)))
	}
	receipts := make([]nativeperf.ExperimentReceipt, 2)
	for i, element := range elements {
		r, err := nativeperf.DecodeReceipt(element)
		if err != nil {
			return nil, NewOptimizeRefusal(ClassInvalidInput, fmt.Sprintf("decode receipt %s[%d]: %v", receiptPath, i, err))
		}
		receipts[i] = r
	}
	base, cand := receipts[0], receipts[1]
	if base.Role != nativeperf.RoleBaseline || cand.Role != nativeperf.RoleCandidate {
		return nil, NewOptimizeRefusal(ClassInvalidInput, "receipt pair requires baseline then candidate roles")
	}
	if base.ChangedLeverID != lever.ID || cand.ChangedLeverID != lever.ID {
		return nil, NewOptimizeRefusal(ClassReceiptLeverMismatch, fmt.Sprintf("receipt pair targets lever(s) %q/%q, not the requested %q", base.ChangedLeverID, cand.ChangedLeverID, lever.ID))
	}
	if base.EnvelopeID != lever.Applicability.EnvelopeID || cand.EnvelopeID != lever.Applicability.EnvelopeID {
		return nil, NewOptimizeRefusal(ClassReceiptEnvelopeMismatch, fmt.Sprintf("receipt pair targets envelope(s) %q/%q, not the lever's pinned %q", base.EnvelopeID, cand.EnvelopeID, lever.Applicability.EnvelopeID))
	}
	if err := nativeperf.ValidateReceipt(cfg.Graph, base); err != nil {
		return nil, NewOptimizeRefusal(ClassInvalidInput, "baseline receipt: "+err.Error())
	}
	if err := nativeperf.ValidateReceipt(cfg.Graph, cand); err != nil {
		return nil, NewOptimizeRefusal(ClassInvalidInput, "candidate receipt: "+err.Error())
	}
	if len(base.Repetitions) == 0 || len(cand.Repetitions) == 0 {
		return nil, NewOptimizeRefusal(ClassInvalidInput, "receipt pair is missing an arm; a missing arm refuses instead of zero-filling")
	}
	if len(base.Repetitions) != len(cand.Repetitions) {
		return nil, NewOptimizeRefusal(ClassInvalidInput, fmt.Sprintf("arms disagree on repetition count (%d vs %d)", len(base.Repetitions), len(cand.Repetitions)))
	}
	baseTokS := meanTPS(base.Repetitions)
	candTokS := meanTPS(cand.Repetitions)
	if baseTokS <= 0 || candTokS <= 0 {
		return nil, NewOptimizeRefusal(ClassInvalidInput, "arm mean throughput is not positive")
	}
	sum := sha256.Sum256(raw)
	delta := (candTokS - baseTokS) / baseTokS * 100
	return &OptimizeReceipt{
		Schema:         ReceiptSchema,
		LeverID:        lever.ID,
		EngineID:       cand.Execution.Engine,
		TracePath:      receiptPath,
		TraceSHA256:    hex.EncodeToString(sum[:]),
		BaselineTokS:   baseTokS,
		CandidateTokS:  candTokS,
		DeltaPct:       delta,
		Reps:           len(base.Repetitions),
		MeasuredAt:     cfg.now().Format(time.RFC3339),
		Revision:       cand.Revision,
		ArtifactSHA256: cand.ArtifactSHA256,
		ForwardPath:    cand.Execution.ForwardPath,
	}, nil
}

// armTokS is the arm's decode-throughput proxy: total tokens over replay
// wall-clock. A non-positive wall refuses — a number with no elapsed time
// behind it is fabrication.
func armTokS(run ablate.AblationRun) (float64, error) {
	if run.WallSeconds <= 0 {
		return 0, fmt.Errorf("optimize: arm %q has no positive wall time", run.ArmID)
	}
	return float64(run.Tokens()) / run.WallSeconds, nil
}

func knownAblateFeature(token string) bool {
	for _, f := range ablate.KnownFeatures() {
		if f == token {
			return true
		}
	}
	return false
}

// QualityGate is the folded `fak quality run --json` result: absent when no
// quality file was supplied, pass/fail otherwise. Only the pass boolean is
// read — the rest of the result stays the quality ladder's own surface.
type QualityGate struct {
	Present bool
	Pass    bool
}

func (q QualityGate) state() string {
	switch {
	case !q.Present:
		return QualityAbsent
	case q.Pass:
		return QualityPass
	default:
		return QualityFail
	}
}

func (q QualityGate) failed() bool { return q.Present && !q.Pass }

func loadQualityGate(path string) (QualityGate, error) {
	if path == "" {
		return QualityGate{}, nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return QualityGate{}, NewOptimizeRefusal(ClassInvalidInput, fmt.Sprintf("read quality result %s: %v", path, err))
	}
	var res struct {
		Pass bool `json:"pass"`
	}
	if err := json.Unmarshal(raw, &res); err != nil {
		return QualityGate{}, NewOptimizeRefusal(ClassInvalidInput, fmt.Sprintf("decode quality result %s: %v", path, err))
	}
	return QualityGate{Present: true, Pass: res.Pass}, nil
}

// Verify folds two receipts into a OptimizeVerdict. Each file may carry either an
// OptimizeReceipt (this package's measure output) or a nativeperf
// ExperimentReceipt (a captured native A/B); a mixed pair refuses. Nativeperf
// pairs are judged under the preregistered comparison criterion resolved from
// the candidate's run identity; optimize pairs under the OptimizePlan's expected
// floor..roof band for the shared lever. A failing quality gate forces reject
// regardless of delta.
func Verify(cfg Config, baselinePath, candidatePath, qualityPath string) (*OptimizeVerdict, error) {
	baseRaw, err := os.ReadFile(baselinePath)
	if err != nil {
		return nil, NewOptimizeRefusal(ClassInvalidInput, fmt.Sprintf("read baseline %s: %v", baselinePath, err))
	}
	candRaw, err := os.ReadFile(candidatePath)
	if err != nil {
		return nil, NewOptimizeRefusal(ClassInvalidInput, fmt.Sprintf("read candidate %s: %v", candidatePath, err))
	}
	quality, err := loadQualityGate(qualityPath)
	if err != nil {
		return nil, err
	}
	baseOpt, baseIsOpt := decodeOptimizeReceipt(baseRaw)
	candOpt, candIsOpt := decodeOptimizeReceipt(candRaw)
	switch {
	case baseIsOpt && candIsOpt:
		if baseOpt.LeverID != candOpt.LeverID {
			return nil, NewOptimizeRefusal(ClassInvalidInput, "receipts target different levers")
		}
		if baseOpt.TraceSHA256 != candOpt.TraceSHA256 {
			return nil, NewOptimizeRefusal(ClassInvalidInput, "receipts bind different trace workloads")
		}
		if err := verifyEnvelopeAgainstLever(cfg, baseOpt.LeverID); err != nil {
			return nil, err
		}
		return decideBanded(cfg, baseOpt.LeverID, baseOpt.BaselineTokS, candOpt.CandidateTokS, quality)
	case !baseIsOpt && !candIsOpt:
		base, err := nativeperf.DecodeReceipt(baseRaw)
		if err != nil {
			return nil, NewOptimizeRefusal(ClassInvalidInput, "baseline is neither an optimize receipt nor a nativeperf receipt: "+err.Error())
		}
		cand, err := nativeperf.DecodeReceipt(candRaw)
		if err != nil {
			return nil, NewOptimizeRefusal(ClassInvalidInput, "candidate is neither an optimize receipt nor a nativeperf receipt: "+err.Error())
		}
		if cfg.EnvelopeID != "" && (base.EnvelopeID != cfg.EnvelopeID || cand.EnvelopeID != cfg.EnvelopeID) {
			return nil, NewOptimizeRefusal(ClassInvalidInput, fmt.Sprintf("receipts target envelope(s) %q/%q, not the requested %q", base.EnvelopeID, cand.EnvelopeID, cfg.EnvelopeID))
		}
		return decideNative(cfg, base, cand, quality)
	default:
		return nil, NewOptimizeRefusal(ClassInvalidInput, "mixed OptimizeReceipt kinds: baseline and candidate must both be optimize receipts or both nativeperf receipts")
	}
}

// verifyEnvelopeAgainstLever enforces the optional --envelope pin against the
// lever the optimize receipts bind to. An unknown lever leaves verification to
// the planning-band gate; a known lever pinned to another envelope refuses.
func verifyEnvelopeAgainstLever(cfg Config, leverID string) error {
	if cfg.EnvelopeID == "" {
		return nil
	}
	for i := range cfg.Graph.Levers {
		if cfg.Graph.Levers[i].ID != leverID {
			continue
		}
		if cfg.Graph.Levers[i].Applicability.EnvelopeID != cfg.EnvelopeID {
			return NewOptimizeRefusal(ClassInvalidInput, fmt.Sprintf("receipts target lever %q pinned to envelope %q, not the requested %q", leverID, cfg.Graph.Levers[i].Applicability.EnvelopeID, cfg.EnvelopeID))
		}
		return nil
	}
	return nil
}

// decodeOptimizeReceipt decodes raw bytes as this package's OptimizeReceipt schema.
// Unknown fields fail the decode on purpose: a foreign JSON body must fall
// through to the nativeperf OptimizeReceipt path, not partially fill ours.
func decodeOptimizeReceipt(raw []byte) (*OptimizeReceipt, bool) {
	var r OptimizeReceipt
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&r); err != nil {
		return nil, false
	}
	if r.Schema != ReceiptSchema {
		return nil, false
	}
	return &r, true
}

// decideBanded judges an optimize-OptimizeReceipt pair under the OptimizePlan's expected
// floor..roof band for the shared lever: the candidate arm must meet the
// floor AND beat the baseline arm (a hill-climb step, not a flat).
func decideBanded(cfg Config, leverID string, baseRef, candRef float64, quality QualityGate) (*OptimizeVerdict, error) {
	if baseRef <= 0 {
		return nil, NewOptimizeRefusal(ClassInvalidInput, "baseline reference throughput is not positive")
	}
	floor, roof, ok := planningBand(cfg.Graph, leverID)
	if !ok {
		return nil, NewOptimizeRefusal(ClassNoPlanningBand, fmt.Sprintf("lever %q has no registered planning band to verify against", leverID))
	}
	delta := (candRef - baseRef) / baseRef * 100
	verdict := VerdictAccept
	reasons := make([]string, 0, 3)
	if candRef < floor {
		verdict = VerdictReject
		reasons = append(reasons, fmt.Sprintf("candidate %.4g tok/s below expected floor %.4g tok/s", candRef, floor))
	} else {
		reasons = append(reasons, fmt.Sprintf("candidate %.4g tok/s meets expected floor %.4g tok/s (roof %.4g tok/s)", candRef, floor, roof))
	}
	if candRef <= baseRef {
		verdict = VerdictReject
		reasons = append(reasons, fmt.Sprintf("candidate %.4g tok/s shows no gain over baseline %.4g tok/s", candRef, baseRef))
	} else {
		reasons = append(reasons, fmt.Sprintf("delta %+.2f%% vs baseline", delta))
	}
	v := &OptimizeVerdict{Schema: VerdictSchema, Verdict: verdict, DeltaPct: delta, Reasons: reasons, Quality: quality.state()}
	if quality.failed() {
		v.Verdict = VerdictReject
		v.Reasons = append(v.Reasons, "quality gate failed: fak quality run reported pass=false")
	}
	return v, nil
}

// decideNative judges a nativeperf-receipt pair under the preregistered
// comparison criterion resolved from the candidate's immutable run axes:
// repetition count, minimum throughput, regression drop bound, noise band,
// and the OptimizeReceipt's own quality score. Callers cannot supply thresholds
// through this path.
func decideNative(cfg Config, base, cand nativeperf.ExperimentReceipt, quality QualityGate) (*OptimizeVerdict, error) {
	if base.Role != nativeperf.RoleBaseline || cand.Role != nativeperf.RoleCandidate {
		return nil, NewOptimizeRefusal(ClassInvalidInput, "nativeperf receipts require baseline then candidate roles")
	}
	if err := nativeperf.ValidateReceipt(cfg.Graph, base); err != nil {
		return nil, NewOptimizeRefusal(ClassInvalidInput, "baseline receipt: "+err.Error())
	}
	if err := nativeperf.ValidateReceipt(cfg.Graph, cand); err != nil {
		return nil, NewOptimizeRefusal(ClassInvalidInput, "candidate receipt: "+err.Error())
	}
	criterion, err := nativeperf.ResolveComparisonCriterion(cand)
	if err != nil {
		return nil, NewOptimizeRefusal(ClassCriterionUnresolved, err.Error())
	}
	b, c := meanTPS(base.Repetitions), meanTPS(cand.Repetitions)
	if b <= 0 {
		return nil, NewOptimizeRefusal(ClassInvalidInput, "baseline mean throughput is not positive")
	}
	delta := (c - b) / b * 100
	verdict := VerdictAccept
	reasons := make([]string, 0, 5)
	if len(cand.Repetitions) < criterion.MinimumRepetitions {
		verdict = VerdictReject
		reasons = append(reasons, fmt.Sprintf("%d repetitions below criterion minimum %d", len(cand.Repetitions), criterion.MinimumRepetitions))
	}
	if c < criterion.MinimumThroughput {
		verdict = VerdictReject
		reasons = append(reasons, fmt.Sprintf("candidate %.4g tok/s below criterion minimum throughput %.4g tok/s", c, criterion.MinimumThroughput))
	}
	if drop := (b - c) / b * 100; drop > criterion.RegressionDropPercent {
		verdict = VerdictReject
		reasons = append(reasons, fmt.Sprintf("regression drop %.2f%% exceeds criterion bound %.2f%%", drop, criterion.RegressionDropPercent))
	}
	if math.Abs(delta) <= criterion.MaximumNoisePercent {
		verdict = VerdictReject
		reasons = append(reasons, fmt.Sprintf("delta %+.2f%% is within the criterion noise band %.2f%%", delta, criterion.MaximumNoisePercent))
	}
	if cand.Quality.Score < criterion.MinimumQualityScore {
		verdict = VerdictReject
		reasons = append(reasons, fmt.Sprintf("quality score %.4g below criterion minimum %.4g", cand.Quality.Score, criterion.MinimumQualityScore))
	}
	if verdict == VerdictAccept {
		reasons = append(reasons, fmt.Sprintf("all criterion gates passed at %.4g tok/s", c))
	}
	v := &OptimizeVerdict{Schema: VerdictSchema, Verdict: verdict, DeltaPct: delta, Reasons: reasons, Quality: quality.state()}
	if quality.failed() {
		v.Verdict = VerdictReject
		v.Reasons = append(v.Reasons, "quality gate failed: fak quality run reported pass=false")
	}
	return v, nil
}

func meanTPS(reps []nativeperf.Repetition) float64 {
	if len(reps) == 0 {
		return 0
	}
	var sum float64
	for _, r := range reps {
		sum += r.TokensPerSecond
	}
	return sum / float64(len(reps))
}

// RunLoop drives the bounded hill-climb: per iteration OptimizePlan -> measure ->
// verify, returning one ledger line per iteration. It stops after the first
// reject OptimizeVerdict (gains are only ever stacked when witnessed), after a typed
// OptimizeRefusal, or at maxIters. dryRun stops after the first OptimizePlan. An accepted
// lever is marked witnessed in the in-memory graph copy so the next iteration
// climbs to the next ready lever. It never writes files; tracePath is the
// frozen workload every measure replays.
func RunLoop(ctx context.Context, cfg Config, maxIters int, dryRun bool, tracePath string) ([]OptimizeLedgerLine, error) {
	if maxIters <= 0 {
		return nil, NewOptimizeRefusal(ClassNonPositiveIterations, fmt.Sprintf("max iterations must be positive, got %d", maxIters))
	}
	graph := cfg.Graph
	lines := make([]OptimizeLedgerLine, 0, maxIters)
	for iter := 1; iter <= maxIters; iter++ {
		cfg.Graph = graph
		line := OptimizeLedgerLine{Iter: iter, Metric: MetricDecodeTokS, Date: cfg.now().Format(time.DateOnly)}
		OptimizePlan, err := BuildPlan(cfg)
		if err != nil {
			if refuse := refusalOf(err); refuse != nil {
				line.Verdict = LedgerRefused
				line.Refusal = refuse.Class
				lines = append(lines, line)
				return lines, nil
			}
			return lines, err
		}
		line.LeverID = OptimizePlan.LeverID
		if dryRun {
			line.Verdict = LedgerPlanned
			lines = append(lines, line)
			return lines, nil
		}
		rec, err := Measure(ctx, cfg, OptimizePlan.LeverID, tracePath, "")
		if err != nil {
			if refuse := refusalOf(err); refuse != nil {
				line.Verdict = LedgerRefused
				line.Refusal = refuse.Class
				lines = append(lines, line)
				return lines, nil
			}
			return lines, err
		}
		base, cand := rec.BaselineTokS, rec.CandidateTokS
		line.Baseline = &base
		line.Candidate = &cand
		d := rec.DeltaPct
		line.DeltaPct = &d
		verdict, err := decideBanded(cfg, rec.LeverID, base, cand, QualityGate{})
		if err != nil {
			if refuse := refusalOf(err); refuse != nil {
				line.Verdict = LedgerRefused
				line.Refusal = refuse.Class
				lines = append(lines, line)
				return lines, nil
			}
			return lines, err
		}
		line.Verdict = verdict.Verdict
		lines = append(lines, line)
		if verdict.Verdict == VerdictReject {
			return lines, nil
		}
		markWitnessed(&graph, rec)
	}
	return lines, nil
}

// markWitnessed stamps the measured lever as witnessed in the in-memory graph
// copy so the next iteration's selection climbs past it. The committed graph
// is untouched; landing the lever updates the real graph out-of-band.
func markWitnessed(graph *nativeperf.Graph, rec *OptimizeReceipt) {
	for i := range graph.Levers {
		if graph.Levers[i].ID != rec.LeverID {
			continue
		}
		graph.Levers[i].Witnessed = &nativeperf.WitnessedEffect{
			Summary:        fmt.Sprintf("%.4g tok/s candidate vs %.4g tok/s baseline on trace %s", rec.CandidateTokS, rec.BaselineTokS, rec.TraceSHA256),
			Classification: nativeperf.ClassWitnessed,
			Provenance:     "fak optimize loop in-process measurement",
			Receipt:        rec.TraceSHA256,
		}
		return
	}
}

func refusalOf(err error) *OptimizeRefusal {
	var r *OptimizeRefusal
	if errors.As(err, &r) {
		return r
	}
	return nil
}
