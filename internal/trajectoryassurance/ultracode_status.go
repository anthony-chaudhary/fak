package trajectoryassurance

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"
)

const UltracodeStatusSchema = "fak.ultracode_status.v1"

const (
	ReasonUltracodeVerified              = "ULTRACODE_DELEGATION_VERIFIED"
	ReasonUltracodeSchemaUnsupported     = "ULTRACODE_SCHEMA_UNSUPPORTED"
	ReasonUltracodeIdentityMissing       = "ULTRACODE_IDENTITY_MISSING"
	ReasonUltracodeReceiptStale          = "ULTRACODE_RECEIPT_STALE"
	ReasonUltracodeOutcomeMissing        = "ULTRACODE_OUTCOME_EVIDENCE_MISSING"
	ReasonUltracodeOutcomeFailed         = "ULTRACODE_OUTCOME_EVIDENCE_FAILED"
	ReasonUltracodeBudgetIncomplete      = "ULTRACODE_BUDGET_INCOMPLETE"
	ReasonUltracodeBudgetFailed          = "ULTRACODE_BUDGET_FAILED"
	ReasonUltracodeChildMissing          = "ULTRACODE_CHILD_EVIDENCE_MISSING"
	ReasonUltracodeChildActivationFailed = "ULTRACODE_CHILD_ACTIVATION_FAILED"
	ReasonUltracodeChildCompletionFailed = "ULTRACODE_CHILD_COMPLETION_FAILED"
)

type ultracodeOutcome struct {
	Verdict            string `json:"verdict"`
	EffectReadback     string `json:"effect_readback"`
	IndependentWitness string `json:"independent_witness"`
	Reconciliation     string `json:"reconciliation"`
	Reason             string `json:"reason"`
}

type ultracodeActivationSummary struct {
	Schema   string                     `json:"schema"`
	Total    int                        `json:"total"`
	Active   int                        `json:"active"`
	Inactive int                        `json:"inactive"`
	Degraded int                        `json:"degraded"`
	Unknown  int                        `json:"unknown"`
	Verified int                        `json:"verified"`
	Ratio    float64                    `json:"ratio"`
	Children []ultracodeActivationChild `json:"children"`
}

type ultracodeActivationChild struct {
	RunID   string `json:"run_id"`
	ChildID string `json:"child_id"`
	Harness string `json:"harness"`
	State   string `json:"state"`
}

type ultracodeBudget struct {
	Schema          string                 `json:"schema"`
	DeclaredTokens  int64                  `json:"declared_tokens"`
	WallBudgetMS    int64                  `json:"wall_budget_ms"`
	StartedAt       time.Time              `json:"started_at"`
	DeadlineAt      time.Time              `json:"deadline_at"`
	Authority       string                 `json:"authority"`
	CoveredChildren int                    `json:"covered_children"`
	TotalChildren   int                    `json:"total_children"`
	ConsumedTokens  int64                  `json:"consumed_tokens"`
	RemainingTokens int64                  `json:"remaining_tokens"`
	ConsumedWallMS  int64                  `json:"consumed_wall_ms"`
	RemainingWallMS int64                  `json:"remaining_wall_ms"`
	TokenOverrun    bool                   `json:"token_overrun"`
	WallOverrun     bool                   `json:"wall_overrun"`
	Overrun         bool                   `json:"overrun"`
	Complete        bool                   `json:"complete"`
	Admitted        bool                   `json:"admitted"`
	Reason          string                 `json:"reason,omitempty"`
	Children        []ultracodeBudgetChild `json:"children"`
}

type ultracodeBudgetChild struct {
	ChildID        string `json:"child_id"`
	ReservedTokens int64  `json:"reserved_tokens"`
	ProviderTokens int64  `json:"provider_tokens"`
	Authority      string `json:"authority"`
	Covered        bool   `json:"covered"`
	Overrun        bool   `json:"overrun"`
}

type ultracodeWorker struct {
	ChildID            string    `json:"child_id"`
	State              string    `json:"state"`
	TurnsStarted       int       `json:"turns_started"`
	TurnsCompleted     int       `json:"turns_completed"`
	LastEvent          string    `json:"last_event,omitempty"`
	Activation         string    `json:"activation"`
	ActivationVerdict  string    `json:"activation_verdict"`
	ActivationReason   string    `json:"activation_reason,omitempty"`
	ActivationAgeMS    int64     `json:"activation_age_ms"`
	ActivationDeadline time.Time `json:"activation_deadline_at,omitempty"`
}

type ultracodeStatus struct {
	Schema           string                     `json:"schema"`
	SessionID        string                     `json:"session_id"`
	RunID            string                     `json:"run_id"`
	RequestedProfile string                     `json:"requested_profile"`
	ResolvedProfile  string                     `json:"resolved_profile"`
	State            string                     `json:"state"`
	Outcome          ultracodeOutcome           `json:"outcome"`
	Activation       ultracodeActivationSummary `json:"activation"`
	Budget           ultracodeBudget            `json:"budget"`
	BudgetPhase      string                     `json:"budget_phase"`
	Workers          []ultracodeWorker          `json:"workers"`
}

// DecodeUltracodeStatus strictly decodes the public status schema and maps it
// to delegation evidence. It performs no callbacks or state changes.
func DecodeUltracodeStatus(r io.Reader, now time.Time) (Input, error) {
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	var status ultracodeStatus
	if err := dec.Decode(&status); err != nil {
		return Input{}, fmt.Errorf("trajectory assurance: decode ultracode status: %w", err)
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return Input{}, fmt.Errorf("trajectory assurance: decode ultracode status: multiple JSON values")
		}
		return Input{}, fmt.Errorf("trajectory assurance: decode ultracode status: %w", err)
	}
	observation := assessUltracodeStatus(status, now.UTC())
	return Input{
		SessionID:           status.SessionID,
		RunID:               status.RunID,
		TrajectoryID:        status.RunID,
		ObservationWindow:   observation.Evidence.Freshness,
		DelegationIntegrity: observation,
	}, nil
}

func assessUltracodeStatus(s ultracodeStatus, now time.Time) Observation {
	e := Evidence{Source: UltracodeStatusSchema, Provenance: "status-receipt", Authority: "orchestration-status"}
	if s.Schema != UltracodeStatusSchema {
		e.ReasonToken, e.Reason = ReasonUltracodeSchemaUnsupported, "unsupported ultracode status schema"
		return Observation{State: Unknown, Evidence: e}
	}
	fail := func(token, reason string) Observation {
		e.ReasonToken, e.Reason = token, reason
		return Observation{State: Fail, Evidence: e}
	}
	unknown := func(token, reason string) Observation {
		e.ReasonToken, e.Reason = token, reason
		return Observation{State: Unknown, Evidence: e}
	}

	// Evaluate explicit producer failures before incomplete evidence so a
	// not_observed outcome cannot mask a budget or child failure.
	if outcomePolarity(s.Outcome.Verdict) == Fail {
		return fail(ReasonUltracodeOutcomeFailed, "ultracode outcome explicitly failed")
	}
	if s.Budget.Overrun || s.Budget.TokenOverrun || s.Budget.WallOverrun || (s.Budget.Complete && !s.Budget.Admitted) {
		return fail(ReasonUltracodeBudgetFailed, "budget evidence explicitly rejects admission or reports overrun")
	}
	for _, child := range s.Activation.Children {
		if child.State == "degraded" || child.State == "inactive" {
			return fail(ReasonUltracodeChildActivationFailed, "child activation explicitly failed")
		}
	}
	for _, worker := range s.Workers {
		if worker.ActivationVerdict == "failed" || worker.Activation == "degraded" || worker.Activation == "inactive" {
			return fail(ReasonUltracodeChildActivationFailed, "child activation explicitly failed")
		}
		if worker.State == "failed" || worker.State == "exited" || worker.State == "invalid" || worker.State == "terminal" {
			return fail(ReasonUltracodeChildCompletionFailed, "child execution explicitly failed")
		}
	}

	if s.Budget.Schema != "fak.ultracode_budget_receipt.v1" || s.BudgetPhase != "final" || !s.Budget.Complete || !s.Budget.Admitted || s.Budget.Authority != "provider-reported" || s.Budget.TotalChildren < 2 || s.Budget.CoveredChildren != s.Budget.TotalChildren || len(s.Budget.Children) != s.Budget.TotalChildren {
		return unknown(ReasonUltracodeBudgetIncomplete, "final admitted provider-authoritative budget coverage is incomplete")
	}
	covered := make(map[string]bool, len(s.Budget.Children))
	for _, child := range s.Budget.Children {
		if child.ChildID == "" || !child.Covered || child.Authority != "provider-reported" {
			return unknown(ReasonUltracodeBudgetIncomplete, "per-child budget coverage is incomplete")
		}
		covered[child.ChildID] = true
	}
	if s.Activation.Schema != "fak.ultracode_activation.v1" || len(s.Activation.Children) != s.Budget.TotalChildren || len(s.Workers) != s.Budget.TotalChildren || s.Activation.Total != s.Budget.TotalChildren {
		return unknown(ReasonUltracodeChildMissing, "activation or completion evidence is incomplete")
	}
	seen := make(map[string]bool, len(s.Workers))
	for _, worker := range s.Workers {
		if worker.ChildID == "" || seen[worker.ChildID] || !covered[worker.ChildID] || worker.TurnsStarted < 1 || worker.TurnsCompleted != worker.TurnsStarted {
			return unknown(ReasonUltracodeChildMissing, "per-child execution evidence is incomplete")
		}
		seen[worker.ChildID] = true
	}
	if !s.Budget.DeadlineAt.IsZero() && now.After(s.Budget.DeadlineAt) {
		return unknown(ReasonUltracodeReceiptStale, "status evidence is stale after the run deadline")
	}

	// v1 currently emits only unverified/not_observed terminal outcome fields.
	// Keep the adapter useful for identity, budget, and child failure readback,
	// but never manufacture PASS from values the producer cannot emit.
	return unknown(ReasonUltracodeOutcomeMissing, "producer has no positive terminal outcome contract; see issue #8834")
}

// outcomePolarity recognizes only values emitted by the v1 producer. The
// producer has no positive terminal outcome yet, so an adapter must not invent
// one from plausible-looking aliases.
func outcomePolarity(value string) State {
	switch strings.TrimSpace(value) {
	case "invalid":
		return Fail
	case "unverified":
		return Unknown
	default:
		return Unknown
	}
}

// evidencePolarity recognizes the sole v1 producer value. Positive
// effect-readback, witness, and reconciliation enums are not yet part of the
// producer contract; unsupported values therefore fail closed as UNKNOWN.
func evidencePolarity(value string) State {
	switch strings.TrimSpace(value) {
	case "not_observed":
		return Unknown
	default:
		return Unknown
	}
}
