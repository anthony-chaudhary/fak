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

const Schema = "fak.causal-receipt/1"

type IDs struct{ Work, Turn, Graph, Request, ModelSession string }
type Decision struct{ Kind, ID, Reason, Planned, Actual string }
type Phase struct {
	ID, ParentID, Kind, Engine, Backend, Outcome, Reason                                                               string
	Started, Ended                                                                                                     time.Time
	Tokens, Bytes, CacheReuseBytes, QueueNS, LoadNS, TransferNS, RecomputeNS, RecoveryNS, VerificationNS, EnergyMicroJ int64
	Quality                                                                                                            string
	ResourceIDs, OperationIDs                                                                                          []string
}
type Resource struct {
	ID, Kind, State, PlannedLocality, ActualLocality string
	Bytes, TransferBytes, RecomputeBytes             int64
	Released                                         bool
}
type Receipt struct {
	Schema         string
	IDs            IDs
	Phases         []Phase
	Resources      []Resource
	Decisions      []Decision
	ModuleVersions map[string]string
	Attributes     map[string]string
}
type Metrics struct {
	PhaseCount                                 int
	Tokens, Bytes, CacheReuseBytes, OverheadNS int64
	Outcomes                                   map[string]int
}

var ErrInvalid = errors.New("causalreceipt: invalid receipt")
var sensitive = regexp.MustCompile(`(?i)(prompt|output|argument|result|content|screenshot|file[_-]?path)`)

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
