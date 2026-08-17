package valuechain

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
)

const Schema = "fak-value-chain/1"

type Manifest struct {
	Schema   string    `json:"schema"`
	Name     string    `json:"name"`
	Stages   []Stage   `json:"stages"`
	Arms     []Arm     `json:"arms"`
	Outcomes []Outcome `json:"outcomes"`
}

type Stage struct {
	ID        string   `json:"id"`
	Kind      string   `json:"kind"`
	DependsOn []string `json:"depends_on,omitempty"`
}

type Arm struct {
	ID      string `json:"id"`
	Default bool   `json:"default,omitempty"`
}
type Outcome struct {
	ID   string `json:"id"`
	Unit string `json:"unit"`
}

type Observation struct {
	ID         string             `json:"id"`
	TraceID    string             `json:"trace_id"`
	SessionID  string             `json:"session_id,omitempty"`
	PairID     string             `json:"pair_id,omitempty"`
	StageID    string             `json:"stage_id"`
	Arm        string             `json:"arm"`
	Turns      int64              `json:"turns,omitempty"`
	CostUSD    *float64           `json:"cost_usd,omitempty"`
	CostKey    string             `json:"cost_key,omitempty"`
	Outcomes   map[string]float64 `json:"outcomes,omitempty"`
	Provenance string             `json:"provenance"`
}

type Input struct {
	Schema       string        `json:"schema"`
	Observations []Observation `json:"observations"`
}

type Coverage struct {
	Covered int64   `json:"covered"`
	Total   int64   `json:"total"`
	Ratio   float64 `json:"ratio"`
}
type ArmReport struct {
	Arm             string             `json:"arm"`
	Default         bool               `json:"default,omitempty"`
	Traces          int                `json:"traces"`
	Sessions        int                `json:"sessions"`
	Turns           int64              `json:"turns"`
	CostUSD         *float64           `json:"cost_usd,omitempty"`
	BillingEvidence Coverage           `json:"billing_evidence"`
	Outcomes        map[string]float64 `json:"outcomes"`
	CostPerTurn     *float64           `json:"cost_per_turn_usd,omitempty"`
	CostPerOutcome  map[string]float64 `json:"cost_per_outcome_usd,omitempty"`
}
type LinkStatus struct {
	Stage        string `json:"stage"`
	Kind         string `json:"kind"`
	Status       string `json:"status"`
	Observations int    `json:"observations"`
}
type Comparison struct {
	Baseline            string   `json:"baseline"`
	Candidate           string   `json:"candidate"`
	Design              string   `json:"design"`
	PairedTraces        int      `json:"paired_traces"`
	CostPerTurnDeltaPct *float64 `json:"cost_per_turn_delta_pct,omitempty"`
}
type Report struct {
	Schema     string       `json:"schema"`
	Name       string       `json:"name"`
	Arms       []ArmReport  `json:"arms"`
	Inventory  []LinkStatus `json:"inventory"`
	Comparison *Comparison  `json:"comparison,omitempty"`
	Warnings   []string     `json:"warnings,omitempty"`
}

func Read(manifestPath, observationsPath string) (Manifest, Input, error) {
	var m Manifest
	var in Input
	if err := decodeFile(manifestPath, &m); err != nil {
		return m, in, fmt.Errorf("manifest: %w", err)
	}
	if err := decodeFile(observationsPath, &in); err != nil {
		return m, in, fmt.Errorf("observations: %w", err)
	}
	return m, in, nil
}
func decodeFile(path string, dst any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err = json.Unmarshal(b, dst); err != nil {
		return err
	}
	return nil
}

func Audit(m Manifest, in Input) (Report, error) {
	if m.Schema != Schema || in.Schema != Schema {
		return Report{}, fmt.Errorf("schema must be %q", Schema)
	}
	if strings.TrimSpace(m.Name) == "" {
		return Report{}, errors.New("manifest name is required")
	}
	stages := map[string]Stage{}
	arms := map[string]Arm{}
	outcomes := map[string]Outcome{}
	for _, s := range m.Stages {
		if s.ID == "" {
			return Report{}, errors.New("stage id is required")
		}
		if _, ok := stages[s.ID]; ok {
			return Report{}, fmt.Errorf("duplicate stage %q", s.ID)
		}
		stages[s.ID] = s
	}
	for _, a := range m.Arms {
		if a.ID == "" {
			return Report{}, errors.New("arm id is required")
		}
		if _, ok := arms[a.ID]; ok {
			return Report{}, fmt.Errorf("duplicate arm %q", a.ID)
		}
		arms[a.ID] = a
	}
	for _, o := range m.Outcomes {
		if o.ID == "" || o.Unit == "" {
			return Report{}, errors.New("outcome id and unit are required")
		}
		if _, ok := outcomes[o.ID]; ok {
			return Report{}, fmt.Errorf("duplicate outcome %q", o.ID)
		}
		outcomes[o.ID] = o
	}
	if err := validateDAG(stages); err != nil {
		return Report{}, err
	}
	type accum struct {
		traces, sessions map[string]bool
		turns, covered   int64
		cost             float64
		costKnown        bool
		outcomes         map[string]float64
		obs              int
	}
	acc := map[string]*accum{}
	seenObs := map[string]bool{}
	costOwners := map[string]string{}
	paired := map[string]map[string]bool{}
	stageCount := map[string]int{}
	for _, o := range in.Observations {
		if o.ID == "" || seenObs[o.ID] {
			return Report{}, fmt.Errorf("duplicate or empty observation id %q", o.ID)
		}
		seenObs[o.ID] = true
		if _, ok := stages[o.StageID]; !ok {
			return Report{}, fmt.Errorf("observation %q references unknown stage %q", o.ID, o.StageID)
		}
		if _, ok := arms[o.Arm]; !ok {
			return Report{}, fmt.Errorf("observation %q references unknown arm %q", o.ID, o.Arm)
		}
		if o.TraceID == "" || o.Provenance == "" {
			return Report{}, fmt.Errorf("observation %q requires trace_id and provenance", o.ID)
		}
		if o.Turns < 0 {
			return Report{}, fmt.Errorf("observation %q has negative turns", o.ID)
		}
		if o.CostUSD != nil && *o.CostUSD < 0 {
			return Report{}, fmt.Errorf("observation %q has negative cost_usd", o.ID)
		}
		for id, value := range o.Outcomes {
			if value < 0 {
				return Report{}, fmt.Errorf("observation %q has negative outcome %q", o.ID, id)
			}
		}
		a := acc[o.Arm]
		if a == nil {
			a = &accum{traces: map[string]bool{}, sessions: map[string]bool{}, outcomes: map[string]float64{}}
			acc[o.Arm] = a
		}
		a.traces[o.TraceID] = true
		if o.SessionID != "" {
			a.sessions[o.SessionID] = true
		}
		a.turns += o.Turns
		a.obs++
		stageCount[o.StageID]++
		if o.CostUSD != nil {
			key := o.CostKey
			if key == "" {
				key = o.ID
			}
			if owner, ok := costOwners[key]; ok && owner != o.Arm {
				return Report{}, fmt.Errorf("cost_key %q is ambiguous across arms %q and %q", key, owner, o.Arm)
			}
			if _, ok := costOwners[key]; !ok {
				costOwners[key] = o.Arm
				a.cost += *o.CostUSD
				a.costKnown = true
			}
			a.covered += o.Turns
		}
		for id, v := range o.Outcomes {
			if _, ok := outcomes[id]; !ok {
				return Report{}, fmt.Errorf("observation %q references unknown outcome %q", o.ID, id)
			}
			a.outcomes[id] += v
		}
		if o.PairID != "" {
			if paired[o.PairID] == nil {
				paired[o.PairID] = map[string]bool{}
			}
			paired[o.PairID][o.Arm] = true
		}
	}
	rep := Report{Schema: Schema, Name: m.Name}
	for _, s := range m.Stages {
		status := "ABSENT"
		if stageCount[s.ID] > 0 {
			status = "PRESENT"
		}
		rep.Inventory = append(rep.Inventory, LinkStatus{Stage: s.ID, Kind: s.Kind, Status: status, Observations: stageCount[s.ID]})
	}
	for _, arm := range m.Arms {
		a := acc[arm.ID]
		ar := ArmReport{Arm: arm.ID, Default: arm.Default, Outcomes: map[string]float64{}, CostPerOutcome: map[string]float64{}}
		if a != nil {
			ar.Traces = len(a.traces)
			ar.Sessions = len(a.sessions)
			ar.Turns = a.turns
			ar.Outcomes = a.outcomes
			ar.BillingEvidence = Coverage{Covered: a.covered, Total: a.turns}
			if a.turns > 0 {
				ar.BillingEvidence.Ratio = float64(a.covered) / float64(a.turns)
			}
			if a.costKnown {
				c := a.cost
				ar.CostUSD = &c
				// A partial billing join is evidence about the observed subtotal, not the
				// whole arm. Keep derived rates absent until every measured turn has cost
				// evidence so missing telemetry can never masquerade as zero cost.
				if a.turns > 0 && a.covered == a.turns {
					x := c / float64(a.turns)
					ar.CostPerTurn = &x
					for id, v := range a.outcomes {
						if v > 0 {
							ar.CostPerOutcome[id] = c / v
						}
					}
				}
			}
		}
		if len(ar.CostPerOutcome) == 0 {
			ar.CostPerOutcome = nil
		}
		rep.Arms = append(rep.Arms, ar)
	}
	if len(m.Arms) == 2 {
		base, cand := m.Arms[0], m.Arms[1]
		if cand.Default {
			base, cand = cand, base
		}
		c := &Comparison{Baseline: base.ID, Candidate: cand.ID, Design: "observational"}
		for _, xs := range paired {
			if xs[base.ID] && xs[cand.ID] {
				c.PairedTraces++
			}
		}
		if c.PairedTraces > 0 {
			c.Design = "paired"
		}
		ba, ca := armByID(rep.Arms, base.ID), armByID(rep.Arms, cand.ID)
		if ba != nil && ca != nil && ba.CostPerTurn != nil && ca.CostPerTurn != nil && *ba.CostPerTurn != 0 {
			x := (*ca.CostPerTurn - *ba.CostPerTurn) / *ba.CostPerTurn * 100
			c.CostPerTurnDeltaPct = &x
		}
		rep.Comparison = c
	}
	sort.Slice(rep.Inventory, func(i, j int) bool { return rep.Inventory[i].Stage < rep.Inventory[j].Stage })
	return rep, nil
}
func armByID(xs []ArmReport, id string) *ArmReport {
	for i := range xs {
		if xs[i].Arm == id {
			return &xs[i]
		}
	}
	return nil
}
func validateDAG(stages map[string]Stage) error {
	state := map[string]int{}
	var visit func(string) error
	visit = func(id string) error {
		if state[id] == 1 {
			return fmt.Errorf("stage dependency cycle at %q", id)
		}
		if state[id] == 2 {
			return nil
		}
		state[id] = 1
		for _, d := range stages[id].DependsOn {
			if _, ok := stages[d]; !ok {
				return fmt.Errorf("stage %q depends on unknown stage %q", id, d)
			}
			if err := visit(d); err != nil {
				return err
			}
		}
		state[id] = 2
		return nil
	}
	for id := range stages {
		if err := visit(id); err != nil {
			return err
		}
	}
	return nil
}
