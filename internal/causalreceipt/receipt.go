// Package causalreceipt records privacy-safe whole-turn causal evidence without requiring an external tracing SDK.
package causalreceipt

import (
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"
)

// Schema identifies the versioned causal receipt specification format.
const Schema = "fak.causal-receipt/1"

// IDs tracks correlated identity across a turn execution, tying work, turn,
// causal graph, request, and model session together without telemetry leakage.
type IDs struct{ Work, Turn, Graph, Request, ModelSession string }

// Decision records an intentional scheduling, routing, or policy choice made
// during turn execution, capturing both planned and actual outcomes.
type Decision struct{ Kind, ID, Reason, Planned, Actual string }

// Phase captures timing, resource consumption, and outcome metrics for an
// execution segment within the causal graph.
type Phase struct {
	ID, ParentID, Kind, Engine, Backend, Outcome, Reason                                                               string
	Started, Ended                                                                                                     time.Time
	Tokens, Bytes, CacheReuseBytes, QueueNS, LoadNS, TransferNS, RecomputeNS, RecoveryNS, VerificationNS, EnergyMicroJ int64
	Quality                                                                                                            string
	ResourceIDs, OperationIDs                                                                                          []string
}

// Resource tracks durable or transient artifacts (e.g., model weights, kv cache)
// utilized by phases, ensuring lifecycles are properly released.
type Resource struct {
	ID, Kind, State, PlannedLocality, ActualLocality string
	Bytes, TransferBytes, RecomputeBytes             int64
	Released                                         bool
}

// Receipt represents whole-turn causal evidence joining execution IDs, phases,
// resources, decisions, module versions, and redaction-safe attributes.
type Receipt struct {
	Schema         string
	IDs            IDs
	Phases         []Phase
	Resources      []Resource
	Decisions      []Decision
	ModuleVersions map[string]string
	Attributes     map[string]string
}

// Metrics provides rolled-up execution statistics across all phases in a receipt.
type Metrics struct {
	PhaseCount                                 int
	Tokens, Bytes, CacheReuseBytes, OverheadNS int64
	Outcomes                                   map[string]int
}

// ErrInvalid indicates that a causal receipt failed structural, schema, or security validation.
var ErrInvalid = errors.New("causalreceipt: invalid receipt")

var sensitive = regexp.MustCompile(`(?i)(prompt|output|argument|result|content|screenshot|file[_-]?path)`)

// Validate verifies structural integrity, causal ancestry, engine constraints,
// and privacy guarantees across the entire receipt.
//
// Invariant: causal receipt graph is acyclic and privacy-safe: phases reference
// strictly prior phases and attributes contain no sensitive content-bearing fields.
// Guard: fail-closed on missing causal identity or sensitive keys (prompts, outputs,
// file paths, or arguments).
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

// DeriveMetrics computes aggregate token, byte, and overhead timings from a valid receipt.
// Returns an error if the receipt fails validation.
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

// Summarize validates the receipt and returns aggregate execution metrics.
// It is the primary summary entry point for whole-turn causal metrics.
func Summarize(r Receipt) (Metrics, error) {
	return DeriveMetrics(r)
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

// IncidentAnswers extracts root-cause diagnostic markers (reloads, transfers,
// evictions, and failure reasons) to speed up post-incident analysis.
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
