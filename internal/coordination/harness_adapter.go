package coordination

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
)

// Harness identifies the producer syntax being normalized. It is deliberately
// data, not an import of any harness package.
type Harness string

const (
	HarnessClaude    Harness = "claude"
	HarnessCodex     Harness = "codex"
	HarnessOpencode  Harness = "opencode"
	HarnessFakNative Harness = "fak_native"
)

// CapabilityState says how a neutral requirement can be provided.
type CapabilityState string

const (
	CapabilityNative      CapabilityState = "native"
	CapabilityEmulated    CapabilityState = "emulated"
	CapabilityDegraded    CapabilityState = "degraded"
	CapabilityUnsupported CapabilityState = "unsupported"
)

// HarnessWorker describes one worker without retaining harness-specific names.
type HarnessWorker struct {
	ID    string `json:"id"`
	Role  string `json:"role"`
	Model string `json:"model"`
}

type HarnessEdge struct {
	From string `json:"from"`
	To   string `json:"to"`
}

// HarnessPins distinguish hard operator constraints from reducible preferences.
type HarnessPins struct {
	Fanout      bool `json:"fanout"`
	Concurrency bool `json:"concurrency"`
	Tokens      bool `json:"tokens"`
	Cost        bool `json:"cost"`
	Duration    bool `json:"duration"`
}

type HarnessLease struct {
	Lane      string        `json:"lane"`
	Mode      string        `json:"mode"`
	TTL       time.Duration `json:"ttl"`
	Renewable bool          `json:"renewable"`
}

type HarnessBudgets struct {
	Tokens     int64         `json:"tokens"`
	CostMicros int64         `json:"costMicros"`
	Duration   time.Duration `json:"duration"`
}

type InteractionPolicy string

const (
	InteractionNone     InteractionPolicy = "none"
	InteractionAsk      InteractionPolicy = "ask"
	InteractionPeer     InteractionPolicy = "peer"
	InteractionOperator InteractionPolicy = "operator"
)

type CancellationPolicy string

const (
	CancellationFailFast CancellationPolicy = "fail_fast"
	CancellationDrain    CancellationPolicy = "drain"
	CancellationIsolate  CancellationPolicy = "isolate"
)

type ExhaustionPolicy string

const (
	ExhaustionDelay      ExhaustionPolicy = "delay"
	ExhaustionReduce     ExhaustionPolicy = "reduce"
	ExhaustionInfeasible ExhaustionPolicy = "infeasible"
)

type WitnessPolicy string

const (
	WitnessRequired    WitnessPolicy = "required"
	WitnessIndependent WitnessPolicy = "independent"
)

type DegradationPolicy string

const (
	DegradationAllowEmulated DegradationPolicy = "allow_emulated"
	DegradationAllowDegraded DegradationPolicy = "allow_degraded"
	DegradationForbid        DegradationPolicy = "forbid"
)

// NeutralRequirements are sufficient inputs for downstream context/cache,
// placement, and serving adapters. Values name capabilities rather than APIs.
type NeutralRequirements struct {
	Context   []string `json:"context"`
	Cache     []string `json:"cache"`
	Placement []string `json:"placement"`
	Serve     []string `json:"serve"`
}

// HarnessWorkflow is the small common ingestion shape for Claude, Codex,
// opencode, and fak_native. Exactly two workers are required by this contract.
type HarnessWorkflow struct {
	Harness       Harness                    `json:"harness"`
	Coordination  bool                       `json:"coordination"`
	WorkID        string                     `json:"workId"`
	CorrelationID string                     `json:"correlationId"`
	Workers       []HarnessWorker            `json:"workers"`
	DAG           []HarnessEdge              `json:"dag"`
	Fanout        int                        `json:"fanout"`
	Concurrency   int                        `json:"concurrency"`
	Pins          HarnessPins                `json:"pins"`
	Lease         HarnessLease               `json:"lease"`
	Budgets       HarnessBudgets             `json:"budgets"`
	Interaction   InteractionPolicy          `json:"interaction"`
	Cancellation  CancellationPolicy         `json:"cancellation"`
	Exhaustion    ExhaustionPolicy           `json:"exhaustion"`
	Witness       WitnessPolicy              `json:"witness"`
	Degradation   DegradationPolicy          `json:"degradation"`
	Capabilities  map[string]CapabilityState `json:"capabilities"`
	Requirements  NeutralRequirements        `json:"requirements"`
}

// HarnessIntent is the canonical, harness-neutral workflow.
type NeutralHarnessIntent struct {
	WorkID        string                     `json:"workId"`
	CorrelationID string                     `json:"correlationId"`
	Workers       []HarnessWorker            `json:"workers"`
	DAG           []HarnessEdge              `json:"dag"`
	Fanout        int                        `json:"fanout"`
	Concurrency   int                        `json:"concurrency"`
	Pins          HarnessPins                `json:"pins"`
	Lease         HarnessLease               `json:"lease"`
	Budgets       HarnessBudgets             `json:"budgets"`
	Interaction   InteractionPolicy          `json:"interaction"`
	Cancellation  CancellationPolicy         `json:"cancellation"`
	Exhaustion    ExhaustionPolicy           `json:"exhaustion"`
	Witness       WitnessPolicy              `json:"witness"`
	Degradation   DegradationPolicy          `json:"degradation"`
	Capabilities  map[string]CapabilityState `json:"capabilities"`
	Requirements  NeutralRequirements        `json:"requirements"`
}

type HarnessPressure struct {
	Concurrency  int                        `json:"concurrency"`
	Tokens       int64                      `json:"tokens"`
	CostMicros   int64                      `json:"costMicros"`
	Duration     time.Duration              `json:"duration"`
	Capabilities map[string]CapabilityState `json:"capabilities"`
	Cancelled    bool                       `json:"cancelled"`
	Exhausted    bool                       `json:"exhausted"`
}

type HarnessDecisionKind string

const (
	HarnessDecisionExecute    HarnessDecisionKind = "execute"
	HarnessDecisionReduce     HarnessDecisionKind = "reduce"
	HarnessDecisionDelay      HarnessDecisionKind = "delay"
	HarnessDecisionInfeasible HarnessDecisionKind = "infeasible"
	HarnessDecisionDelegate   HarnessDecisionKind = "delegate"
)

type HarnessDecision struct {
	Kind       HarnessDecisionKind  `json:"kind"`
	Reason     string               `json:"reason"`
	Intent     NeutralHarnessIntent `json:"intent"`
	Delegation string               `json:"delegation,omitempty"`
}

// HarnessProjection carries only neutral requirements and stable identity.
type HarnessProjection struct {
	WorkID        string                     `json:"workId"`
	CorrelationID string                     `json:"correlationId"`
	Context       []string                   `json:"context"`
	Cache         []string                   `json:"cache"`
	Placement     []string                   `json:"placement"`
	Serve         []string                   `json:"serve"`
	Capabilities  map[string]CapabilityState `json:"capabilities"`
}

type HarnessAdapter struct{}

func NewHarnessAdapter() *HarnessAdapter { return &HarnessAdapter{} }

// Normalize rejects unknown values and canonicalizes ordering, so equivalent
// harness descriptions produce byte-identical intent and derived identities.
func (a *HarnessAdapter) Normalize(workflow HarnessWorkflow) (NeutralHarnessIntent, error) {
	if a == nil {
		return NeutralHarnessIntent{}, errors.New("coordination: nil harness adapter")
	}
	if !workflow.Coordination {
		return NeutralHarnessIntent{WorkID: workflow.WorkID, CorrelationID: workflow.CorrelationID}, nil
	}
	if !validHarness(workflow.Harness) {
		return NeutralHarnessIntent{}, fmt.Errorf("coordination: unknown harness %q", workflow.Harness)
	}
	intent := NeutralHarnessIntent{
		WorkID: strings.TrimSpace(workflow.WorkID), CorrelationID: strings.TrimSpace(workflow.CorrelationID),
		Workers: append([]HarnessWorker(nil), workflow.Workers...), DAG: append([]HarnessEdge(nil), workflow.DAG...),
		Fanout: workflow.Fanout, Concurrency: workflow.Concurrency, Pins: workflow.Pins, Lease: workflow.Lease,
		Budgets: workflow.Budgets, Interaction: workflow.Interaction, Cancellation: workflow.Cancellation,
		Exhaustion: workflow.Exhaustion, Witness: workflow.Witness, Degradation: workflow.Degradation,
		Capabilities: cloneCapabilities(workflow.Capabilities), Requirements: cloneRequirements(workflow.Requirements),
	}
	canonicalizeIntent(&intent)
	if err := validateIntent(intent); err != nil {
		return NeutralHarnessIntent{}, err
	}
	identity := intent
	identity.WorkID, identity.CorrelationID = "", ""
	digest, err := stableDigest(identity)
	if err != nil {
		return NeutralHarnessIntent{}, err
	}
	if intent.WorkID == "" {
		intent.WorkID = "work-" + digest[:24]
	}
	if intent.CorrelationID == "" {
		sum := sha256.Sum256([]byte("correlation:" + intent.WorkID))
		intent.CorrelationID = "corr-" + hex.EncodeToString(sum[:12])
	}
	return intent, nil
}

// Decide translates current pressure without changing graph, roles, witness
// policy, or identity. Hard pins turn impossible reductions into infeasible.
func (a *HarnessAdapter) Decide(workflow HarnessWorkflow, pressure HarnessPressure) HarnessDecision {
	if !workflow.Coordination {
		return HarnessDecision{Kind: HarnessDecisionDelegate, Reason: "coordination disabled", Delegation: "legacy:#5964", Intent: NeutralHarnessIntent{WorkID: workflow.WorkID, CorrelationID: workflow.CorrelationID}}
	}
	intent, err := a.Normalize(workflow)
	if err != nil {
		return HarnessDecision{Kind: HarnessDecisionInfeasible, Reason: err.Error()}
	}
	if err := validatePressure(pressure); err != nil {
		return HarnessDecision{Kind: HarnessDecisionInfeasible, Reason: err.Error(), Intent: intent}
	}
	if pressure.Cancelled {
		return HarnessDecision{Kind: HarnessDecisionDelay, Reason: "cancelled", Intent: intent}
	}
	missing, degraded := capabilityPressure(intent, pressure.Capabilities)
	if missing != "" {
		return HarnessDecision{Kind: HarnessDecisionInfeasible, Reason: "unsupported capability: " + missing, Intent: intent}
	}
	if degraded != "" && intent.Degradation == DegradationForbid {
		return HarnessDecision{Kind: HarnessDecisionInfeasible, Reason: "degradation forbidden: " + degraded, Intent: intent}
	}

	effective := intent
	reduced := false
	if pressure.Concurrency < effective.Concurrency {
		if pressure.Concurrency < 1 || effective.Pins.Concurrency || (effective.Pins.Fanout && pressure.Concurrency < effective.Fanout) {
			return HarnessDecision{Kind: HarnessDecisionInfeasible, Reason: "pinned concurrency unavailable", Intent: intent}
		}
		effective.Concurrency, reduced = pressure.Concurrency, true
	}
	if pressure.Tokens < effective.Budgets.Tokens {
		if effective.Pins.Tokens {
			return HarnessDecision{Kind: HarnessDecisionInfeasible, Reason: "pinned token budget unavailable", Intent: intent}
		}
		effective.Budgets.Tokens, reduced = pressure.Tokens, true
	}
	if pressure.CostMicros < effective.Budgets.CostMicros {
		if effective.Pins.Cost {
			return HarnessDecision{Kind: HarnessDecisionInfeasible, Reason: "pinned cost budget unavailable", Intent: intent}
		}
		effective.Budgets.CostMicros, reduced = pressure.CostMicros, true
	}
	if pressure.Duration < effective.Budgets.Duration {
		if effective.Pins.Duration {
			return HarnessDecision{Kind: HarnessDecisionInfeasible, Reason: "pinned duration unavailable", Intent: intent}
		}
		effective.Budgets.Duration, reduced = pressure.Duration, true
	}
	if pressure.Exhausted {
		switch effective.Exhaustion {
		case ExhaustionDelay:
			return HarnessDecision{Kind: HarnessDecisionDelay, Reason: "resources exhausted", Intent: intent}
		case ExhaustionInfeasible:
			return HarnessDecision{Kind: HarnessDecisionInfeasible, Reason: "resources exhausted", Intent: intent}
		case ExhaustionReduce:
			reduced = true
		}
	}
	if reduced || degraded != "" {
		return HarnessDecision{Kind: HarnessDecisionReduce, Reason: "pressure requires an allowed reduction", Intent: effective}
	}
	return HarnessDecision{Kind: HarnessDecisionExecute, Reason: "requirements satisfied", Intent: effective}
}

func (a *HarnessAdapter) Project(intent NeutralHarnessIntent) (HarnessProjection, error) {
	if a == nil {
		return HarnessProjection{}, errors.New("coordination: nil harness adapter")
	}
	if err := validateIntent(intent); err != nil {
		return HarnessProjection{}, err
	}
	return HarnessProjection{WorkID: intent.WorkID, CorrelationID: intent.CorrelationID, Context: append([]string(nil), intent.Requirements.Context...), Cache: append([]string(nil), intent.Requirements.Cache...), Placement: append([]string(nil), intent.Requirements.Placement...), Serve: append([]string(nil), intent.Requirements.Serve...), Capabilities: cloneCapabilities(intent.Capabilities)}, nil
}

type HarnessAdapterSelfCheck struct {
	OK            bool   `json:"ok"`
	Deterministic bool   `json:"deterministic"`
	ContentFree   bool   `json:"contentFree"`
	Digest        string `json:"digest"`
	Error         string `json:"error,omitempty"`
}

// SelfCheck uses fixed synthetic metadata and returns only a digest and status.
func (a *HarnessAdapter) SelfCheck() HarnessAdapterSelfCheck {
	base := HarnessWorkflow{Harness: HarnessClaude, Coordination: true, WorkID: "self-check", CorrelationID: "self-check", Workers: []HarnessWorker{{ID: "a", Role: "worker", Model: "m"}, {ID: "b", Role: "reviewer", Model: "m"}}, DAG: []HarnessEdge{{From: "a", To: "b"}}, Fanout: 2, Concurrency: 2, Lease: HarnessLease{Lane: "neutral", Mode: "exclusive", TTL: time.Second}, Budgets: HarnessBudgets{Tokens: 2, CostMicros: 2, Duration: time.Second}, Interaction: InteractionNone, Cancellation: CancellationFailFast, Exhaustion: ExhaustionDelay, Witness: WitnessIndependent, Degradation: DegradationForbid, Capabilities: map[string]CapabilityState{"witness": CapabilityNative}}
	one, err := a.Normalize(base)
	if err != nil {
		return HarnessAdapterSelfCheck{ContentFree: true, Error: err.Error()}
	}
	base.Harness = HarnessCodex
	two, err := a.Normalize(base)
	if err != nil {
		return HarnessAdapterSelfCheck{ContentFree: true, Error: err.Error()}
	}
	digest, err := stableDigest(one)
	if err != nil {
		return HarnessAdapterSelfCheck{ContentFree: true, Error: err.Error()}
	}
	deterministic := one.WorkID == two.WorkID && one.CorrelationID == two.CorrelationID && digest != ""
	return HarnessAdapterSelfCheck{OK: deterministic, Deterministic: deterministic, ContentFree: true, Digest: digest}
}

func validHarness(v Harness) bool {
	return v == HarnessClaude || v == HarnessCodex || v == HarnessOpencode || v == HarnessFakNative
}

func validateIntent(v NeutralHarnessIntent) error {
	if len(v.Workers) != 2 || v.Fanout != 2 {
		return errors.New("coordination: two workers and fanout 2 are required")
	}
	if v.Concurrency < 1 || v.Concurrency > v.Fanout {
		return errors.New("coordination: concurrency must be within fanout")
	}
	ids := map[string]bool{}
	for _, worker := range v.Workers {
		if worker.ID == "" || worker.Role == "" || worker.Model == "" || ids[worker.ID] {
			return errors.New("coordination: workers require unique ids, roles, and models")
		}
		ids[worker.ID] = true
	}
	for _, edge := range v.DAG {
		if edge.From == edge.To || !ids[edge.From] || !ids[edge.To] {
			return errors.New("coordination: invalid DAG edge")
		}
	}
	if cyclic(v.Workers, v.DAG) {
		return errors.New("coordination: DAG contains a cycle")
	}
	if v.Lease.Lane == "" || (v.Lease.Mode != "shared" && v.Lease.Mode != "exclusive") || v.Lease.TTL <= 0 {
		return errors.New("coordination: invalid lease")
	}
	if v.Budgets.Tokens <= 0 || v.Budgets.CostMicros <= 0 || v.Budgets.Duration <= 0 {
		return errors.New("coordination: budgets must be positive")
	}
	if !oneOf(string(v.Interaction), "none", "ask", "peer", "operator") || !oneOf(string(v.Cancellation), "fail_fast", "drain", "isolate") || !oneOf(string(v.Exhaustion), "delay", "reduce", "infeasible") || !oneOf(string(v.Witness), "required", "independent") || !oneOf(string(v.Degradation), "allow_emulated", "allow_degraded", "forbid") {
		return errors.New("coordination: unknown policy value")
	}
	for name, state := range v.Capabilities {
		if strings.TrimSpace(name) == "" || !oneOf(string(state), "native", "emulated", "degraded", "unsupported") {
			return errors.New("coordination: invalid capability")
		}
		if state == CapabilityUnsupported {
			return fmt.Errorf("coordination: unsupported required capability %q", name)
		}
		if state == CapabilityEmulated && v.Degradation == DegradationForbid || state == CapabilityDegraded && v.Degradation != DegradationAllowDegraded {
			return fmt.Errorf("coordination: capability %q violates degradation policy", name)
		}
	}
	if strings.TrimSpace(v.WorkID) == "" || strings.TrimSpace(v.CorrelationID) == "" {
		return errors.New("coordination: stable identity is required")
	}
	return nil
}

func validatePressure(v HarnessPressure) error {
	if v.Concurrency < 0 || v.Tokens < 0 || v.CostMicros < 0 || v.Duration < 0 {
		return errors.New("coordination: invalid pressure value")
	}
	for name, state := range v.Capabilities {
		if name == "" || !oneOf(string(state), "native", "emulated", "degraded", "unsupported") {
			return errors.New("coordination: invalid pressure capability")
		}
	}
	return nil
}

func capabilityPressure(intent NeutralHarnessIntent, available map[string]CapabilityState) (string, string) {
	for name := range intent.Capabilities {
		state, ok := available[name]
		if !ok || state == CapabilityUnsupported {
			return name, ""
		}
		if state == CapabilityEmulated && intent.Degradation == DegradationForbid || state == CapabilityDegraded && intent.Degradation != DegradationAllowDegraded {
			return "", name
		}
	}
	return "", ""
}

func canonicalizeIntent(v *NeutralHarnessIntent) {
	for i := range v.Workers {
		v.Workers[i].ID, v.Workers[i].Role, v.Workers[i].Model = strings.TrimSpace(v.Workers[i].ID), strings.TrimSpace(v.Workers[i].Role), strings.TrimSpace(v.Workers[i].Model)
	}
	sort.Slice(v.Workers, func(i, j int) bool { return v.Workers[i].ID < v.Workers[j].ID })
	sort.Slice(v.DAG, func(i, j int) bool {
		if v.DAG[i].From == v.DAG[j].From {
			return v.DAG[i].To < v.DAG[j].To
		}
		return v.DAG[i].From < v.DAG[j].From
	})
	v.Lease.Lane, v.Lease.Mode = strings.TrimSpace(v.Lease.Lane), strings.TrimSpace(v.Lease.Mode)
	v.Requirements = cloneRequirements(v.Requirements)
}

func cloneRequirements(v NeutralRequirements) NeutralRequirements {
	v.Context = canonicalStrings(v.Context)
	v.Cache = canonicalStrings(v.Cache)
	v.Placement = canonicalStrings(v.Placement)
	v.Serve = canonicalStrings(v.Serve)
	return v
}

func canonicalStrings(v []string) []string {
	set := map[string]struct{}{}
	for _, item := range v {
		if item = strings.TrimSpace(item); item != "" {
			set[item] = struct{}{}
		}
	}
	out := make([]string, 0, len(set))
	for item := range set {
		out = append(out, item)
	}
	sort.Strings(out)
	return out
}

func cloneCapabilities(v map[string]CapabilityState) map[string]CapabilityState {
	out := make(map[string]CapabilityState, len(v))
	for key, value := range v {
		out[strings.TrimSpace(key)] = value
	}
	return out
}

func cyclic(workers []HarnessWorker, edges []HarnessEdge) bool {
	indegree := make(map[string]int, len(workers))
	next := make(map[string][]string, len(workers))
	for _, w := range workers {
		indegree[w.ID] = 0
	}
	for _, e := range edges {
		indegree[e.To]++
		next[e.From] = append(next[e.From], e.To)
	}
	queue := make([]string, 0, len(workers))
	for id, n := range indegree {
		if n == 0 {
			queue = append(queue, id)
		}
	}
	seen := 0
	for len(queue) > 0 {
		id := queue[0]
		queue = queue[1:]
		seen++
		for _, to := range next[id] {
			indegree[to]--
			if indegree[to] == 0 {
				queue = append(queue, to)
			}
		}
	}
	return seen != len(workers)
}

func stableDigest(v any) (string, error) {
	data, err := json.Marshal(v)
	if err != nil {
		return "", fmt.Errorf("coordination: canonical encoding: %w", err)
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

func oneOf(value string, allowed ...string) bool {
	for _, candidate := range allowed {
		if value == candidate {
			return true
		}
	}
	return false
}
