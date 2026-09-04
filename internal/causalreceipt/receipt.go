// Package causalreceipt records privacy-safe whole-turn causal evidence without requiring an external tracing SDK.
//
// Invariants and contracts:
//   - Invariant: A causal receipt enforces strict fail-closed validation over causal identity and phase hierarchy.
//   - Invariant: Privacy guard rejects any sensitive content-bearing attributes (e.g. prompts, outputs, arguments, screenshots).
//   - Invariant: Every referenced resource must be explicitly released (lifecycle closure guard).
//   - Invariant: Orphan phases lacking valid parentage or carrying ambiguous native engines are rejected.
package causalreceipt

import (
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"
)

// Schema is the canonical schema identifier for causal receipts.
const Schema = "fak.causal-receipt/1"

// IDs captures the causal identity scope across work, turn, graph, request, and model session.
type IDs struct {
	Work         string `json:"work"`
	Turn         string `json:"turn"`
	Graph        string `json:"graph"`
	Request      string `json:"request"`
	ModelSession string `json:"model_session,omitempty"`
}

// Decision records an individual causal decision and its outcome.
type Decision struct {
	Kind    string `json:"kind"`
	ID      string `json:"id"`
	Reason  string `json:"reason,omitempty"`
	Planned string `json:"planned,omitempty"`
	Actual  string `json:"actual"`
}

// Phase represents a bounded execution phase in the turn graph.
type Phase struct {
	ID              string    `json:"id"`
	ParentID        string    `json:"parent_id,omitempty"`
	Kind            string    `json:"kind"`
	Engine          string    `json:"engine"`
	Backend         string    `json:"backend"`
	Outcome         string    `json:"outcome"`
	Reason          string    `json:"reason,omitempty"`
	Started         time.Time `json:"started,omitempty"`
	Ended           time.Time `json:"ended,omitempty"`
	Tokens          int64     `json:"tokens,omitempty"`
	Bytes           int64     `json:"bytes,omitempty"`
	CacheReuseBytes int64     `json:"cache_reuse_bytes,omitempty"`
	QueueNS         int64     `json:"queue_ns,omitempty"`
	LoadNS          int64     `json:"load_ns,omitempty"`
	TransferNS      int64     `json:"transfer_ns,omitempty"`
	RecomputeNS     int64     `json:"recompute_ns,omitempty"`
	RecoveryNS      int64     `json:"recovery_ns,omitempty"`
	VerificationNS  int64     `json:"verification_ns,omitempty"`
	EnergyMicroJ    int64     `json:"energy_micro_j,omitempty"`
	Quality         string    `json:"quality,omitempty"`
	ResourceIDs     []string  `json:"resource_ids,omitempty"`
	OperationIDs    []string  `json:"operation_ids,omitempty"`
}

// Resource tracks an allocated resource and its lifecycle state.
type Resource struct {
	ID              string `json:"id"`
	Kind            string `json:"kind"`
	State           string `json:"state"`
	PlannedLocality string `json:"planned_locality,omitempty"`
	ActualLocality  string `json:"actual_locality,omitempty"`
	Bytes           int64  `json:"bytes,omitempty"`
	TransferBytes   int64  `json:"transfer_bytes,omitempty"`
	RecomputeBytes  int64  `json:"recompute_bytes,omitempty"`
	Released        bool   `json:"released"`
}

// Receipt records whole-turn causal evidence across IDs, phases, resources, decisions, and attributes.
type Receipt struct {
	Schema         string            `json:"schema"`
	IDs            IDs               `json:"ids"`
	Phases         []Phase           `json:"phases"`
	Resources      []Resource        `json:"resources,omitempty"`
	Decisions      []Decision        `json:"decisions,omitempty"`
	ModuleVersions map[string]string `json:"module_versions,omitempty"`
	Attributes     map[string]string `json:"attributes,omitempty"`
}

// Metrics aggregates whole-turn resource overheads, token usage, and outcomes.
type Metrics struct {
	PhaseCount      int            `json:"phase_count"`
	Tokens          int64          `json:"tokens"`
	Bytes           int64          `json:"bytes"`
	CacheReuseBytes int64          `json:"cache_reuse_bytes"`
	OverheadNS      int64          `json:"overhead_ns"`
	Outcomes        map[string]int `json:"outcomes"`
}

// ErrInvalid signals an ill-formed or privacy-violating causal receipt.
var ErrInvalid = errors.New("causalreceipt: invalid receipt")

var sensitive = regexp.MustCompile(`(?i)(prompt|output|argument|result|content|screenshot|file[_-]?path)`)

// Validate checks that a receipt conforms to the causal schema and contains no sensitive content.
func Validate(r Receipt) error {
	if r.Schema != Schema || r.IDs.Work == "" || r.IDs.Turn == "" || r.IDs.Graph == "" || r.IDs.Request == "" {
		return fmt.Errorf("%w: missing causal identity", ErrInvalid)
	}
	for k := range r.Attributes {
		if sensitive.MatchString(k) {
			return fmt.Errorf("%w: content-bearing field %s", ErrInvalid, k)
		}
	}
	phase := map[string]bool{}
	for _, p := range r.Phases {
		if p.ID == "" || phase[p.ID] || p.Engine == "" || p.Outcome == "" {
			return fmt.Errorf("%w: phase", ErrInvalid)
		}
		phase[p.ID] = true
		if p.ParentID != "" && !phase[p.ParentID] {
			return fmt.Errorf("%w: orphan phase %s", ErrInvalid, p.ID)
		}
		if strings.Contains(p.Engine, "native") && p.Engine != "fak-native" {
			return fmt.Errorf("%w: ambiguous native engine", ErrInvalid)
		}
	}
	for _, x := range r.Resources {
		if x.ID == "" || !x.Released {
			return fmt.Errorf("%w: unreleased resource", ErrInvalid)
		}
	}
	for _, d := range r.Decisions {
		if d.ID == "" || d.Kind == "" || d.Actual == "" {
			return fmt.Errorf("%w: decision", ErrInvalid)
		}
	}
	return nil
}

// DeriveMetrics computes aggregated turn metrics after validating the receipt.
func DeriveMetrics(r Receipt) (Metrics, error) {
	if e := Validate(r); e != nil {
		return Metrics{}, e
	}
	m := Metrics{PhaseCount: len(r.Phases), Outcomes: map[string]int{}}
	for _, p := range r.Phases {
		m.Tokens += p.Tokens
		m.Bytes += p.Bytes
		m.CacheReuseBytes += p.CacheReuseBytes
		m.OverheadNS += p.QueueNS + p.LoadNS + p.TransferNS + p.RecomputeNS + p.RecoveryNS + p.VerificationNS
		m.Outcomes[p.Outcome]++
	}
	return m, nil
}

// MetricLabels returns bounded low-cardinality labels; causal IDs remain only in receipts.
func MetricLabels(r Receipt) map[string]string {
	out := map[string]string{"schema": r.Schema}
	if len(r.Phases) > 0 {
		out["engine"] = r.Phases[0].Engine
		out["outcome"] = r.Phases[len(r.Phases)-1].Outcome
	}
	return out
}

// IncidentAnswers extracts incident and failure signatures from phase latencies and evicted resources.
func IncidentAnswers(r Receipt) []string {
	answers := []string{}
	for _, p := range r.Phases {
		if p.LoadNS > 0 {
			answers = append(answers, "reload:"+p.ID)
		}
		if p.TransferNS > 0 {
			answers = append(answers, "transfer:"+p.ID)
		}
		if p.Reason != "" {
			answers = append(answers, "reason:"+p.Reason)
		}
	}
	for _, x := range r.Resources {
		if x.State == "evicted" {
			answers = append(answers, "evicted:"+x.ID)
		}
	}
	sort.Strings(answers)
	return answers
}
