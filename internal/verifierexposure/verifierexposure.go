// Package verifierexposure ranks the gameability of fak's verification gates.
package verifierexposure

import (
	"fmt"
	"math"
	"sort"
)

const (
	Schema        = "fak-verifier-exposure/1"
	DebtThreshold = 0.35
)

type Kind string

const (
	Deterministic Kind = "deterministic"
	LLMJudge      Kind = "llm_judge"
	SelfReport    Kind = "self_report"
)

// Gate is one declared verification surface and the independently reviewable
// signals used by the pinned scoring heuristic.
type SignalProbe struct {
	Path     string `json:"path"`
	Contains string `json:"contains"`
}

type Gate struct {
	Name                    string        `json:"name"`
	Kind                    Kind          `json:"kind"`
	Sources                 []string      `json:"sources"`
	SignalProbes            []SignalProbe `json:"signal_probes,omitempty"`
	CheckerBytesPinned      bool          `json:"checker_bytes_pinned"`
	SchemaPinned            bool          `json:"schema_pinned"`
	TemperatureZero         bool          `json:"temperature_zero"`
	FailsOpen               bool          `json:"fails_open"`
	InjectableProse         bool          `json:"injectable_prose"`
	IndependentlyRemeasured bool          `json:"independently_remeasured"`
}

type GateExposure struct {
	Gate
	Exposure float64  `json:"exposure"`
	Reasons  []string `json:"reasons"`
}

type Report struct {
	Schema               string         `json:"schema"`
	GateCount            int            `json:"gate_count"`
	VerifierExposureDebt int            `json:"verifier_exposure_debt"`
	DebtThreshold        float64        `json:"debt_threshold"`
	Grade                string         `json:"grade"`
	Worklist             []GateExposure `json:"worklist"`
	InventoryErrors      []string       `json:"inventory_errors,omitempty"`
}

// Score applies a deliberately small, pinned heuristic. Kind establishes the
// attack surface; hardening signals subtract exposure and soft failure / prose
// injection add it. Values are clamped to [0,1].
func Score(g Gate) GateExposure {
	var score float64
	var reasons []string
	switch g.Kind {
	case Deterministic:
		score = 0.10
		reasons = append(reasons, "deterministic base +0.10")
	case LLMJudge:
		score = 0.55
		reasons = append(reasons, "LLM judge base +0.55")
	case SelfReport:
		score = 0.80
		reasons = append(reasons, "self-report base +0.80")
	default:
		score = 1
		reasons = append(reasons, "unknown kind +1.00")
	}
	if g.CheckerBytesPinned {
		score -= 0.15
		reasons = append(reasons, "checker bytes pinned -0.15")
	}
	if g.SchemaPinned {
		score -= 0.10
		reasons = append(reasons, "schema pinned -0.10")
	}
	if g.TemperatureZero {
		score -= 0.05
		reasons = append(reasons, "temperature zero -0.05")
	}
	if g.FailsOpen {
		score += 0.15
		reasons = append(reasons, "fails open +0.15")
	}
	if g.InjectableProse {
		score += 0.15
		reasons = append(reasons, "injectable prose +0.15")
	}
	if g.IndependentlyRemeasured {
		score -= 0.15
		reasons = append(reasons, "independently remeasured -0.15")
	}
	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}
	score = math.Round(score*100) / 100
	return GateExposure{Gate: g, Exposure: score, Reasons: reasons}
}

func Fold(gates []Gate, inventoryErrors []string) Report {
	work := make([]GateExposure, 0, len(gates))
	debt := 0
	for _, g := range gates {
		e := Score(g)
		if e.Exposure >= DebtThreshold {
			debt++
		}
		work = append(work, e)
	}
	sort.Slice(work, func(i, j int) bool {
		if work[i].Exposure != work[j].Exposure {
			return work[i].Exposure > work[j].Exposure
		}
		return work[i].Name < work[j].Name
	})
	return Report{Schema: Schema, GateCount: len(gates), VerifierExposureDebt: debt, DebtThreshold: DebtThreshold, Grade: grade(debt, len(gates), len(inventoryErrors)), Worklist: work, InventoryErrors: append([]string(nil), inventoryErrors...)}
}

func grade(debt, total, inventoryErrors int) string {
	if inventoryErrors > 0 || total == 0 {
		return "F"
	}
	ratio := float64(debt) / float64(total)
	switch {
	case debt == 0:
		return "A"
	case ratio <= .25:
		return "B"
	case ratio <= .50:
		return "C"
	default:
		return "D"
	}
}

func Render(r Report) string {
	out := fmt.Sprintf("verifier exposure: debt=%d/%d threshold=%.2f grade=%s", r.VerifierExposureDebt, r.GateCount, r.DebtThreshold, r.Grade)
	for _, g := range r.Worklist {
		out += fmt.Sprintf("\n  %.2f  %-28s %s", g.Exposure, g.Name, g.Kind)
	}
	for _, e := range r.InventoryErrors {
		out += "\n  INVENTORY_ERROR " + e
	}
	return out
}
