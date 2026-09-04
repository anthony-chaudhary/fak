// Package opensweharder runs a deterministic, reversible closed-loop evaluation
// over frozen software-engineering tasks. It models Open SWE orchestration and a
// harder SWE-smith task source without claiming that either live system ran.
package opensweharder

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
)

// EvidenceMode states whether a result came from a frozen simulation or a live run.
type EvidenceMode string

const (
	ModeSimulated EvidenceMode = "simulated_frozen_fixture"
	ModeLive      EvidenceMode = "live_external_run"
)

// Task is one accepted or rejected unit offered to the loop.
type Task struct {
	ID            string `json:"id"`
	Source        string `json:"source"`
	Difficulty    int    `json:"difficulty"`
	Accepted      bool   `json:"accepted"`
	BaselinePass  bool   `json:"baseline_pass"`
	CandidatePass bool   `json:"candidate_pass"`
	ReversalPass  bool   `json:"reversal_pass"`
}

// Fixture is the frozen input contract.
type Fixture struct {
	Schema string       `json:"schema"`
	Mode   EvidenceMode `json:"mode"`
	Tasks  []Task       `json:"tasks"`
}

// Hypothesis captures the baseline and the falsifiable counter-hypothesis.
type Hypothesis struct {
	Baseline          string `json:"baseline"`
	CounterHypothesis string `json:"counter_hypothesis"`
}

// Score reports passes over accepted tasks only.
type Score struct {
	Passed   int `json:"passed"`
	Accepted int `json:"accepted"`
}

// Report is the closed-loop result. Reversed must reproduce Baseline exactly.
type Report struct {
	Schema        string       `json:"schema"`
	Mode          EvidenceMode `json:"mode"`
	Hypothesis    Hypothesis   `json:"hypothesis"`
	OfferedTasks  int          `json:"offered_tasks"`
	AcceptedTasks int          `json:"accepted_tasks"`
	Baseline      Score        `json:"baseline"`
	Candidate     Score        `json:"candidate"`
	Reversed      Score        `json:"reversed"`
	Decision      string       `json:"decision"`
	ReversalOK    bool         `json:"reversal_ok"`
	AcceptedIDs   []string     `json:"accepted_ids"`
}

// LoadFixture loads a frozen fixture and rejects dishonest or malformed evidence.
func LoadFixture(path string) (Fixture, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Fixture{}, err
	}
	var f Fixture
	if err := json.Unmarshal(b, &f); err != nil {
		return Fixture{}, err
	}
	if f.Schema != "fak-openswe-harder-fixture/1" {
		return Fixture{}, fmt.Errorf("unsupported schema %q", f.Schema)
	}
	if f.Mode != ModeSimulated && f.Mode != ModeLive {
		return Fixture{}, fmt.Errorf("unsupported evidence mode %q", f.Mode)
	}
	if len(f.Tasks) == 0 {
		return Fixture{}, errors.New("fixture has no tasks")
	}
	seen := map[string]bool{}
	for _, t := range f.Tasks {
		if t.ID == "" || t.Source == "" {
			return Fixture{}, errors.New("task id and source are required")
		}
		if seen[t.ID] {
			return Fixture{}, fmt.Errorf("duplicate task %q", t.ID)
		}
		seen[t.ID] = true
		if t.Difficulty < 1 {
			return Fixture{}, fmt.Errorf("task %q has invalid difficulty", t.ID)
		}
		if t.Accepted && t.ReversalPass != t.BaselinePass {
			return Fixture{}, fmt.Errorf("task %q does not reverse to baseline", t.ID)
		}
	}
	return f, nil
}

// Invariant: OpenSWE-Harder evaluations are fail-closed and preserve baseline bounds.
// Guard: Discrepancy between baseline pass and reversal pass triggers immediate rejection.
// Contract: Fixtures must contain verified tasks with valid schema versions and non-zero difficulty.

// Run evaluates baseline, candidate, and reversal against the same accepted-task denominator.
func Run(f Fixture) (Report, error) {
	if len(f.Tasks) == 0 {
		return Report{}, errors.New("fixture has no tasks")
	}
	r := Report{
		Schema: "fak-openswe-harder-report/1", Mode: f.Mode,
		Hypothesis: Hypothesis{
			Baseline:          "Open SWE-style orchestration on the accepted frozen task set",
			CounterHypothesis: "adding harder SWE-smith-derived tasks improves accepted-task resolves without hiding rejects or surviving reversal",
		},
		OfferedTasks: len(f.Tasks),
	}
	for _, t := range f.Tasks {
		if !t.Accepted {
			continue
		}
		r.AcceptedTasks++
		r.AcceptedIDs = append(r.AcceptedIDs, t.ID)
		if t.BaselinePass {
			r.Baseline.Passed++
		}
		if t.CandidatePass {
			r.Candidate.Passed++
		}
		if t.ReversalPass {
			r.Reversed.Passed++
		}
	}
	if r.AcceptedTasks == 0 {
		return Report{}, errors.New("no accepted tasks")
	}
	sort.Strings(r.AcceptedIDs)
	r.Baseline.Accepted, r.Candidate.Accepted, r.Reversed.Accepted = r.AcceptedTasks, r.AcceptedTasks, r.AcceptedTasks
	r.ReversalOK = r.Reversed == r.Baseline
	switch {
	case !r.ReversalOK:
		r.Decision = "reject_reversal_failed"
	case r.Candidate.Passed > r.Baseline.Passed:
		r.Decision = "accept_counter_hypothesis"
	default:
		r.Decision = "retain_baseline"
	}
	return r, nil
}
