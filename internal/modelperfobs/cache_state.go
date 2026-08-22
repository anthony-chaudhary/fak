package modelperfobs

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"runtime"
	"runtime/debug"
	"sync"
	"time"
)

// StateBenchmarkSchema identifies the transition-report wire format.
const StateBenchmarkSchema = "fak-model-cache-state-benchmark/1"

// StateLayer identifies the independently resettable cache boundary a receipt targets.
type StateLayer string

const (
	StateLayerProcessLocalKV   StateLayer = "process_local_kv"
	StateLayerExternalSharedKV StateLayer = "external_shared_kv"
	StateLayerProviderPrompt   StateLayer = "provider_prompt_cache"
	StateLayerWorkflow         StateLayer = "fak_workflow_cache"
)

// StateTransition is an intended, observable cache-state change.
type StateTransition string

const (
	TransitionColdStart          StateTransition = "cold_start"
	TransitionWarmAdmit          StateTransition = "warm_admit"
	TransitionExplicitInvalidate StateTransition = "explicit_invalidate"
	TransitionNaturalExpiry      StateTransition = "natural_expiry"
	TransitionPressureEvict      StateTransition = "pressure_evict"
)

// TransitionResult states whether the requested state was proved, not merely attempted.
type TransitionResult string

const (
	TransitionProved       TransitionResult = "proved"
	TransitionUnproved     TransitionResult = "transition_unproved"
	TransitionUnsupported  TransitionResult = "unsupported"
	TransitionMechanismErr TransitionResult = "mechanism_failed"
	TransitionObserveErr   TransitionResult = "observation_failed"
)

const (
	EvidenceObserved  = "observed"
	EvidenceSimulated = "simulated"
)

// BackendIdentity pins receipts to one backend instance and revision.
type BackendIdentity struct {
	Name     string `json:"name"`
	Instance string `json:"instance"`
	Revision string `json:"revision"`
}

// MetricSnapshot is one ordered reading from a stable counter epoch.
type MetricSnapshot struct {
	CapturedAt     time.Time `json:"captured_at"`
	SampleSequence uint64    `json:"sample_sequence"`
	Source         string    `json:"source"`
	CounterEpoch   string    `json:"counter_epoch"`
	Requests       uint64    `json:"requests"`
	ReuseHits      uint64    `json:"reuse_hits"`
	ReusedTokens   uint64    `json:"reused_tokens"`
	Entries        uint64    `json:"entries"`
	Invalidates    uint64    `json:"invalidations"`
	Expirations    uint64    `json:"expirations"`
	Evictions      uint64    `json:"evictions"`
}

// MechanismReceipt records the backend action separately from its postcondition.
type MechanismReceipt struct {
	Name   string `json:"name"`
	ExitOK bool   `json:"exit_ok"`
	Detail string `json:"detail,omitempty"`
}

// ProbeRequest pins the exact token prefix used to test reuse.
type ProbeRequest struct {
	TokenPrefix []int `json:"token_prefix"`
}

// ProbeObservation records whether one pinned request reused prior work.
type ProbeObservation struct {
	StartedAt          time.Time `json:"started_at"`
	CompletedAt        time.Time `json:"completed_at"`
	TokenPrefixDigest  string    `json:"token_prefix_digest"`
	PromptTokens       uint64    `json:"prompt_tokens"`
	Reused             bool      `json:"reused"`
	ReusedTokens       uint64    `json:"reused_tokens"`
	ConcurrentRequests uint64    `json:"concurrent_requests"`
}

// TransitionRequest binds an intended transition to its layer and probe.
type TransitionRequest struct {
	Layer      StateLayer      `json:"layer"`
	Transition StateTransition `json:"transition"`
	Probe      ProbeRequest    `json:"probe"`
}

// TransitionReceipt preserves the evidence used to admit or reject one arm.
type TransitionReceipt struct {
	Schema        string             `json:"schema"`
	BackendBefore BackendIdentity    `json:"backend_before"`
	BackendAfter  BackendIdentity    `json:"backend_after"`
	Layer         StateLayer         `json:"layer"`
	Transition    StateTransition    `json:"transition"`
	Mechanism     MechanismReceipt   `json:"mechanism"`
	StartedAt     time.Time          `json:"started_at"`
	EndedAt       time.Time          `json:"ended_at"`
	Result        TransitionResult   `json:"result"`
	Reason        string             `json:"reason,omitempty"`
	PreMetrics    *MetricSnapshot    `json:"pre_metrics,omitempty"`
	PostMetrics   *MetricSnapshot    `json:"post_metrics,omitempty"`
	Probes        []ProbeObservation `json:"probes,omitempty"`
}

// StateArm keeps the transition receipt beside the measurement-inclusion verdict.
type StateArm struct {
	Label               string            `json:"label"`
	MeasurementIncluded bool              `json:"measurement_included"`
	InvalidReason       string            `json:"invalid_reason,omitempty"`
	Receipt             TransitionReceipt `json:"transition_receipt"`
}

// StateProvenance bounds what environment and cache layer a report can claim.
type StateProvenance struct {
	EvidenceKind          string    `json:"evidence_kind"`
	Scope                 string    `json:"scope"`
	ExternalBackendClaims bool      `json:"external_backend_claims"`
	Command               string    `json:"command"`
	GoVersion             string    `json:"go_version"`
	GOOS                  string    `json:"goos"`
	GOARCH                string    `json:"goarch"`
	CodeRevision          string    `json:"code_revision,omitempty"`
	CodeState             string    `json:"code_state"`
	CapturedAt            time.Time `json:"captured_at"`
	Note                  string    `json:"note"`
}

// StateReport is the replay-verifiable output of one transition campaign.
type StateReport struct {
	Schema     string          `json:"schema"`
	Verdict    string          `json:"verdict"`
	Provenance StateProvenance `json:"provenance"`
	Arms       []StateArm      `json:"arms"`
}

// StateBackend is the backend-neutral transition, metric, and probe seam.
type StateBackend interface {
	Identity(context.Context) (BackendIdentity, error)
	Supports(StateLayer, StateTransition) (mechanism string, ok bool)
	Snapshot(context.Context, StateLayer) (MetricSnapshot, error)
	Apply(context.Context, TransitionRequest) (MechanismReceipt, error)
	Probe(context.Context, StateLayer, ProbeRequest) (ProbeObservation, error)
}

// StateRunner executes one transition and independently checks its postcondition.
type StateRunner struct {
	Backend StateBackend
	Now     func() time.Time
}

// RunTransition attempts and verifies one requested cache-state change.
func (r StateRunner) RunTransition(ctx context.Context, req TransitionRequest) TransitionReceipt {
	now := r.Now
	if now == nil {
		now = time.Now
	}
	receipt := TransitionReceipt{
		Schema:     "fak-cache-transition-receipt/1",
		Layer:      req.Layer,
		Transition: req.Transition,
		StartedAt:  now().UTC(),
	}
	mechanism, supported := r.Backend.Supports(req.Layer, req.Transition)
	receipt.Mechanism.Name = mechanism
	if !supported {
		receipt.Result = TransitionUnsupported
		receipt.Reason = "backend_does_not_support_transition"
		receipt.EndedAt = now().UTC()
		return receipt
	}
	before, err := r.Backend.Identity(ctx)
	if err != nil {
		return receiptObservationError(receipt, now, "backend_identity_before", err)
	}
	receipt.BackendBefore = before
	pre, err := r.Backend.Snapshot(ctx, req.Layer)
	if err != nil {
		return receiptObservationError(receipt, now, "pre_metrics", err)
	}
	receipt.PreMetrics = &pre
	mechanismReceipt, err := r.Backend.Apply(ctx, req)
	if mechanismReceipt.Name == "" {
		mechanismReceipt.Name = mechanism
	}
	receipt.Mechanism = mechanismReceipt
	if err != nil || !mechanismReceipt.ExitOK {
		receipt.Result = TransitionMechanismErr
		receipt.Reason = "mechanism_failed"
		if err != nil {
			receipt.Mechanism.Detail = err.Error()
		}
		receipt.EndedAt = now().UTC()
		return receipt
	}
	probeCount := 1
	if req.Transition == TransitionWarmAdmit {
		probeCount = 2
	}
	for i := 0; i < probeCount; i++ {
		probe, probeErr := r.Backend.Probe(ctx, req.Layer, req.Probe)
		if probeErr != nil {
			return receiptObservationError(receipt, now, "probe", probeErr)
		}
		receipt.Probes = append(receipt.Probes, probe)
	}
	post, err := r.Backend.Snapshot(ctx, req.Layer)
	if err != nil {
		return receiptObservationError(receipt, now, "post_metrics", err)
	}
	receipt.PostMetrics = &post
	after, err := r.Backend.Identity(ctx)
	if err != nil {
		return receiptObservationError(receipt, now, "backend_identity_after", err)
	}
	receipt.BackendAfter = after
	receipt.EndedAt = now().UTC()
	receipt.Result, receipt.Reason = verifyTransition(receipt)
	return receipt
}

func receiptObservationError(receipt TransitionReceipt, now func() time.Time, stage string, err error) TransitionReceipt {
	receipt.Result = TransitionObserveErr
	receipt.Reason = stage + ": " + err.Error()
	receipt.EndedAt = now().UTC()
	return receipt
}

func verifyTransition(receipt TransitionReceipt) (TransitionResult, string) {
	if receipt.PreMetrics == nil || receipt.PostMetrics == nil {
		return TransitionUnproved, "metrics_missing"
	}
	pre, post := *receipt.PreMetrics, *receipt.PostMetrics
	if receipt.BackendBefore != receipt.BackendAfter {
		return TransitionUnproved, "backend_identity_changed"
	}
	if pre.CapturedAt.Before(receipt.StartedAt) || post.CapturedAt.Before(pre.CapturedAt) || post.CapturedAt.After(receipt.EndedAt) ||
		pre.SampleSequence == 0 || post.SampleSequence <= pre.SampleSequence {
		return TransitionUnproved, "stale_metrics"
	}
	if pre.CounterEpoch == "" || post.CounterEpoch != pre.CounterEpoch || countersRegressed(pre, post) {
		return TransitionUnproved, "counter_reset"
	}
	if len(receipt.Probes) == 0 {
		return TransitionUnproved, "probe_missing"
	}
	for _, probe := range receipt.Probes {
		if probe.TokenPrefixDigest == "" || probe.PromptTokens == 0 {
			return TransitionUnproved, "probe_prefix_unpinned"
		}
		if probe.ConcurrentRequests != 0 {
			return TransitionUnproved, "concurrent_traffic_contamination"
		}
	}
	wantRequests := uint64(len(receipt.Probes))
	if post.Requests-pre.Requests != wantRequests {
		return TransitionUnproved, "concurrent_traffic_contamination"
	}
	wantHits, wantReused := uint64(0), uint64(0)
	for _, probe := range receipt.Probes {
		if probe.Reused {
			wantHits++
			wantReused += probe.ReusedTokens
		}
	}
	if post.ReuseHits-pre.ReuseHits != wantHits || post.ReusedTokens-pre.ReusedTokens != wantReused {
		return TransitionUnproved, "probe_metric_mismatch"
	}
	last := receipt.Probes[len(receipt.Probes)-1]
	switch receipt.Transition {
	case TransitionColdStart:
		if last.Reused {
			return TransitionUnproved, "reuse_observed_after_cold"
		}
	case TransitionWarmAdmit:
		if len(receipt.Probes) != 2 || receipt.Probes[0].TokenPrefixDigest != last.TokenPrefixDigest || !last.Reused || last.ReusedTokens == 0 {
			return TransitionUnproved, "warm_probe_not_reused"
		}
	case TransitionExplicitInvalidate:
		if post.Invalidates <= pre.Invalidates || last.Reused {
			return TransitionUnproved, "invalidation_postcondition_failed"
		}
	case TransitionNaturalExpiry:
		if post.Expirations <= pre.Expirations || last.Reused {
			return TransitionUnproved, "expiry_postcondition_failed"
		}
	case TransitionPressureEvict:
		if post.Evictions <= pre.Evictions || last.Reused {
			return TransitionUnproved, "eviction_postcondition_failed"
		}
	default:
		return TransitionUnproved, "unknown_transition"
	}
	return TransitionProved, ""
}

func countersRegressed(pre, post MetricSnapshot) bool {
	return post.Requests < pre.Requests || post.ReuseHits < pre.ReuseHits || post.ReusedTokens < pre.ReusedTokens ||
		post.Invalidates < pre.Invalidates || post.Expirations < pre.Expirations || post.Evictions < pre.Evictions
}

// RunStateCampaign executes transitions in order and excludes every unproved arm.
func RunStateCampaign(ctx context.Context, runner StateRunner, layer StateLayer, prefix []int, transitions []StateTransition, provenance StateProvenance) StateReport {
	report := StateReport{Schema: StateBenchmarkSchema, Verdict: "admitted", Provenance: provenance}
	for _, transition := range transitions {
		receipt := runner.RunTransition(ctx, TransitionRequest{
			Layer:      layer,
			Transition: transition,
			Probe:      ProbeRequest{TokenPrefix: append([]int(nil), prefix...)},
		})
		arm := StateArm{Label: string(transition), Receipt: receipt}
		arm.MeasurementIncluded = receipt.Result == TransitionProved
		if !arm.MeasurementIncluded {
			arm.InvalidReason = string(receipt.Result)
			if receipt.Reason != "" {
				arm.InvalidReason += ":" + receipt.Reason
			}
			report.Verdict = "invalid_arms"
		}
		report.Arms = append(report.Arms, arm)
	}
	return report
}

// RunHermeticStateBenchmark captures the observed in-process workflow-cache spine.
func RunHermeticStateBenchmark(ctx context.Context) (StateReport, error) {
	backend := NewLocalWorkflowBackend(nil)
	prefix := []int{101, 8426, 17, 23, 42, 99}
	if _, err := backend.Probe(ctx, StateLayerWorkflow, ProbeRequest{TokenPrefix: prefix}); err != nil {
		return StateReport{}, err
	}
	report := RunStateCampaign(ctx, StateRunner{Backend: backend}, StateLayerWorkflow, prefix, []StateTransition{
		TransitionColdStart,
		TransitionWarmAdmit,
		TransitionExplicitInvalidate,
		TransitionPressureEvict,
	}, runtimeStateProvenance(time.Now().UTC()))
	if err := ValidateStateReport(report); err != nil {
		return report, err
	}
	return report, nil
}

func runtimeStateProvenance(capturedAt time.Time) StateProvenance {
	p := StateProvenance{
		EvidenceKind:          EvidenceObserved,
		Scope:                 "in_process_fak_workflow_cache",
		ExternalBackendClaims: false,
		Command:               "fak model-observe cache-state-bench",
		GoVersion:             runtime.Version(),
		GOOS:                  runtime.GOOS,
		GOARCH:                runtime.GOARCH,
		CodeState:             "unknown",
		CapturedAt:            capturedAt.UTC(),
		Note:                  "Observed from the running hermetic workflow-cache backend; it does not claim external KV or provider-cache behavior.",
	}
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, setting := range info.Settings {
			switch setting.Key {
			case "vcs.revision":
				p.CodeRevision = setting.Value
			case "vcs.modified":
				if setting.Value == "true" {
					p.CodeState = "modified"
				} else if setting.Value == "false" {
					p.CodeState = "clean"
				}
			}
		}
	}
	return p
}

// ValidateStateReport replays proved receipt invariants and the provenance boundary.
func ValidateStateReport(report StateReport) error {
	if report.Schema != StateBenchmarkSchema {
		return fmt.Errorf("cache-state report schema %q, want %q", report.Schema, StateBenchmarkSchema)
	}
	if report.Provenance.EvidenceKind != EvidenceObserved && report.Provenance.EvidenceKind != EvidenceSimulated {
		return fmt.Errorf("cache-state report has unsupported provenance %q", report.Provenance.EvidenceKind)
	}
	if report.Provenance.Scope == "" || report.Provenance.Command == "" || report.Provenance.CodeState == "" || report.Provenance.CapturedAt.IsZero() || report.Provenance.Note == "" {
		return errors.New("cache-state report provenance is incomplete")
	}
	if report.Provenance.Scope == "in_process_fak_workflow_cache" && report.Provenance.ExternalBackendClaims {
		return errors.New("in-process cache-state provenance cannot claim an external backend")
	}
	if len(report.Arms) == 0 {
		return errors.New("cache-state report has no arms")
	}
	required := map[StateTransition]bool{
		TransitionColdStart:          false,
		TransitionWarmAdmit:          false,
		TransitionExplicitInvalidate: false,
		TransitionPressureEvict:      false,
	}
	allIncluded := true
	for i, arm := range report.Arms {
		if arm.Label == "" || arm.Receipt.Schema != "fak-cache-transition-receipt/1" || arm.Receipt.Mechanism.Name == "" || arm.Receipt.BackendBefore.Name == "" {
			return fmt.Errorf("cache-state arm %d has an incomplete receipt", i)
		}
		if arm.Receipt.StartedAt.IsZero() || arm.Receipt.EndedAt.Before(arm.Receipt.StartedAt) {
			return fmt.Errorf("cache-state arm %q has invalid timing", arm.Label)
		}
		if arm.MeasurementIncluded != (arm.Receipt.Result == TransitionProved) {
			return fmt.Errorf("cache-state arm %q inclusion disagrees with receipt result", arm.Label)
		}
		if arm.Receipt.Result == TransitionProved {
			result, reason := verifyTransition(arm.Receipt)
			if result != TransitionProved {
				return fmt.Errorf("cache-state arm %q receipt does not reproduce its proved result: %s/%s", arm.Label, result, reason)
			}
		}
		if _, ok := required[arm.Receipt.Transition]; ok {
			required[arm.Receipt.Transition] = arm.MeasurementIncluded
		}
		allIncluded = allIncluded && arm.MeasurementIncluded
	}
	for transition, proved := range required {
		if !proved {
			return fmt.Errorf("cache-state report lacks a proved %s arm", transition)
		}
	}
	if allIncluded != (report.Verdict == "admitted") {
		return errors.New("cache-state report verdict disagrees with arm inclusion")
	}
	return nil
}

// ReadStateReport decodes exactly one report and validates every proved receipt.
func ReadStateReport(r io.Reader) (StateReport, error) {
	var report StateReport
	decoder := json.NewDecoder(r)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&report); err != nil {
		return StateReport{}, err
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return StateReport{}, errors.New("cache-state report has trailing JSON")
		}
		return StateReport{}, err
	}
	if err := ValidateStateReport(report); err != nil {
		return StateReport{}, err
	}
	return report, nil
}

// WriteStateReport validates a report before serializing it.
func WriteStateReport(w io.Writer, report StateReport, pretty bool) error {
	if err := ValidateStateReport(report); err != nil {
		return err
	}
	encoder := json.NewEncoder(w)
	if pretty {
		encoder.SetIndent("", "  ")
	}
	return encoder.Encode(report)
}

type localWorkflowBackend struct {
	mu            sync.Mutex
	now           func() time.Time
	identity      BackendIdentity
	entries       map[string]uint64
	entryOrder    []string
	capacity      int
	counterEpoch  string
	snapshotSeq   uint64
	requests      uint64
	reuseHits     uint64
	reusedTokens  uint64
	invalidations uint64
	expirations   uint64
	evictions     uint64
}

// NewLocalWorkflowBackend returns the observed, hermetic workflow-cache adapter.
func NewLocalWorkflowBackend(now func() time.Time) StateBackend {
	if now == nil {
		now = time.Now
	}
	return &localWorkflowBackend{
		now:          now,
		identity:     BackendIdentity{Name: "modelperfobs-hermetic-workflow-cache", Instance: "in-process", Revision: "1"},
		entries:      make(map[string]uint64),
		capacity:     1,
		counterEpoch: "process-1",
	}
}

func (b *localWorkflowBackend) Identity(context.Context) (BackendIdentity, error) {
	return b.identity, nil
}

func (b *localWorkflowBackend) Supports(layer StateLayer, transition StateTransition) (string, bool) {
	if layer != StateLayerWorkflow {
		return "", false
	}
	switch transition {
	case TransitionColdStart:
		return "clear-local-workflow-store", true
	case TransitionWarmAdmit:
		return "repeat-pinned-token-prefix", true
	case TransitionExplicitInvalidate:
		return "invalidate-pinned-token-prefix", true
	case TransitionPressureEvict:
		return "admit-competing-prefix-to-one-entry-budget", true
	default:
		return "", false
	}
}

func (b *localWorkflowBackend) Snapshot(_ context.Context, layer StateLayer) (MetricSnapshot, error) {
	if layer != StateLayerWorkflow {
		return MetricSnapshot{}, fmt.Errorf("snapshot layer %q unsupported", layer)
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.snapshotSeq++
	return MetricSnapshot{
		CapturedAt:     b.now().UTC(),
		SampleSequence: b.snapshotSeq,
		Source:         "in-process-workflow-cache-counters",
		CounterEpoch:   b.counterEpoch,
		Requests:       b.requests,
		ReuseHits:      b.reuseHits,
		ReusedTokens:   b.reusedTokens,
		Entries:        uint64(len(b.entries)),
		Invalidates:    b.invalidations,
		Expirations:    b.expirations,
		Evictions:      b.evictions,
	}, nil
}

func (b *localWorkflowBackend) Apply(_ context.Context, req TransitionRequest) (MechanismReceipt, error) {
	mechanism, ok := b.Supports(req.Layer, req.Transition)
	if !ok {
		return MechanismReceipt{}, fmt.Errorf("transition %q on layer %q unsupported", req.Transition, req.Layer)
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	digest := digestTokenPrefix(req.Probe.TokenPrefix)
	switch req.Transition {
	case TransitionColdStart:
		b.entries = make(map[string]uint64)
		b.entryOrder = nil
	case TransitionWarmAdmit:
	case TransitionExplicitInvalidate:
		b.deleteEntry(digest)
		b.invalidations++
	case TransitionPressureEvict:
		pressurePrefix := append(append([]int(nil), req.Probe.TokenPrefix...), -8426)
		b.addEntry(digestTokenPrefix(pressurePrefix), uint64(len(pressurePrefix)))
	}
	detail := ""
	if req.Transition == TransitionPressureEvict {
		detail = "admitted a competing prefix into a one-entry cache budget"
	}
	return MechanismReceipt{Name: mechanism, ExitOK: true, Detail: detail}, nil
}

func (b *localWorkflowBackend) Probe(_ context.Context, layer StateLayer, req ProbeRequest) (ProbeObservation, error) {
	if layer != StateLayerWorkflow {
		return ProbeObservation{}, fmt.Errorf("probe layer %q unsupported", layer)
	}
	if len(req.TokenPrefix) == 0 {
		return ProbeObservation{}, errors.New("probe token prefix is empty")
	}
	started := b.now().UTC()
	digest := digestTokenPrefix(req.TokenPrefix)
	b.mu.Lock()
	count, reused := b.entries[digest]
	if !reused {
		count = uint64(len(req.TokenPrefix))
		b.addEntry(digest, count)
	}
	b.requests++
	if reused {
		b.reuseHits++
		b.reusedTokens += count
	}
	b.mu.Unlock()
	return ProbeObservation{
		StartedAt:         started,
		CompletedAt:       b.now().UTC(),
		TokenPrefixDigest: digest,
		PromptTokens:      uint64(len(req.TokenPrefix)),
		Reused:            reused,
		ReusedTokens:      countIf(reused, count),
	}, nil
}

func (b *localWorkflowBackend) addEntry(digest string, tokens uint64) {
	if _, exists := b.entries[digest]; exists {
		return
	}
	b.entries[digest] = tokens
	b.entryOrder = append(b.entryOrder, digest)
	for len(b.entries) > b.capacity && len(b.entryOrder) > 0 {
		oldest := b.entryOrder[0]
		b.entryOrder = b.entryOrder[1:]
		if _, exists := b.entries[oldest]; exists {
			delete(b.entries, oldest)
			b.evictions++
		}
	}
}

func (b *localWorkflowBackend) deleteEntry(digest string) {
	delete(b.entries, digest)
	for i, entry := range b.entryOrder {
		if entry == digest {
			b.entryOrder = append(b.entryOrder[:i], b.entryOrder[i+1:]...)
			return
		}
	}
}

func digestTokenPrefix(tokens []int) string {
	h := sha256.New()
	var buf [8]byte
	for _, token := range tokens {
		binary.LittleEndian.PutUint64(buf[:], uint64(int64(token)))
		_, _ = h.Write(buf[:])
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}

func countIf(ok bool, n uint64) uint64 {
	if ok {
		return n
	}
	return 0
}
