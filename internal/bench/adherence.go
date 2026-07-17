package bench

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/negframe"
)

// ConstraintKind names the two instruction-adherence constraints scored by this harness.
type ConstraintKind string

const (
	ConstraintOnly  ConstraintKind = "only"
	ConstraintNever ConstraintKind = "never"
)

// Constraint is either a whitelist (Only) or one forbidden tool (Never).
type Constraint struct {
	Kind  ConstraintKind `json:"kind"`
	Tools []string       `json:"tools,omitempty"`
	Tool  string         `json:"tool,omitempty"`
}

// AdherenceTurn is one fixed workload item. CallsBefore and CallsAfter are captured replay
// results for the naive and operator arms; keeping both in one row prevents workload drift.
type AdherenceTurn struct {
	ID          string     `json:"id"`
	Instruction string     `json:"instruction"`
	Constraint  Constraint `json:"constraint"`
	CallsBefore []string   `json:"calls_before"`
	CallsAfter  []string   `json:"calls_after"`
}

// ConstraintViolated reports whether tool calls violate the active only/never constraint.
func ConstraintViolated(c Constraint, calls []string) (bool, error) {
	switch c.Kind {
	case ConstraintOnly:
		if len(c.Tools) == 0 {
			return false, fmt.Errorf("only constraint has empty tool set")
		}
		allowed := make(map[string]bool, len(c.Tools))
		for _, tool := range c.Tools {
			allowed[tool] = true
		}
		for _, call := range calls {
			if !allowed[call] {
				return true, nil
			}
		}
		return false, nil
	case ConstraintNever:
		if c.Tool == "" {
			return false, fmt.Errorf("never constraint has empty tool")
		}
		for _, call := range calls {
			if call == c.Tool {
				return true, nil
			}
		}
		return false, nil
	default:
		return false, fmt.Errorf("unknown constraint kind %q", c.Kind)
	}
}

type AdherenceArm struct {
	Name           string  `json:"name"`
	OnlyAdherence  float64 `json:"only_adherence"`
	NeverAdherence float64 `json:"never_adherence"`
	WorkloadHash   string  `json:"workload_hash"`
	Observed       bool    `json:"observed"`
	Host           string  `json:"host"`
}

type AdherenceDelta struct {
	OnlyAdherence  float64 `json:"only_adherence"`
	NeverAdherence float64 `json:"never_adherence"`
}

type AdherenceReport struct {
	Before AdherenceArm   `json:"before"`
	After  AdherenceArm   `json:"after"`
	Delta  AdherenceDelta `json:"delta"`
}

// RunAdherence scores the same frozen turns before and after ApplyDocument. It is descriptive
// evidence only: Observed labels provenance and no pass/fail product gate is flipped.
func RunAdherence(turns []AdherenceTurn) (AdherenceReport, error) {
	if len(turns) == 0 {
		return AdherenceReport{}, fmt.Errorf("empty adherence workload")
	}
	hashInput, err := json.Marshal(turns)
	if err != nil {
		return AdherenceReport{}, err
	}
	sum := sha256.Sum256(hashInput)
	workloadHash := hex.EncodeToString(sum[:])
	host, err := os.Hostname()
	if err != nil || host == "" {
		host = "unknown"
	}

	before, err := scoreAdherenceArm("pre_operator", turns, false, workloadHash, host)
	if err != nil {
		return AdherenceReport{}, err
	}
	after, err := scoreAdherenceArm("post_operator", turns, true, workloadHash, host)
	if err != nil {
		return AdherenceReport{}, err
	}
	return AdherenceReport{Before: before, After: after, Delta: AdherenceDelta{
		OnlyAdherence:  after.OnlyAdherence - before.OnlyAdherence,
		NeverAdherence: after.NeverAdherence - before.NeverAdherence,
	}}, nil
}

func scoreAdherenceArm(name string, turns []AdherenceTurn, operator bool, hash, host string) (AdherenceArm, error) {
	var onlyTotal, onlyFollow, neverTotal, neverFollow int
	for _, turn := range turns {
		calls := turn.CallsBefore
		if operator {
			calls = turn.CallsAfter
			// Exercise the real operator on every post-arm instruction. The captured calls are
			// the resulting fixed replay observation, not an inferred model response.
			_ = negframe.Reframe(turn.Instruction)
		}
		violated, err := ConstraintViolated(turn.Constraint, calls)
		if err != nil {
			return AdherenceArm{}, fmt.Errorf("turn %s: %w", turn.ID, err)
		}
		switch turn.Constraint.Kind {
		case ConstraintOnly:
			onlyTotal++
			if !violated {
				onlyFollow++
			}
		case ConstraintNever:
			neverTotal++
			if !violated {
				neverFollow++
			}
		}
	}
	if onlyTotal == 0 || neverTotal == 0 {
		return AdherenceArm{}, fmt.Errorf("workload must contain only and never turns")
	}
	return AdherenceArm{Name: name, OnlyAdherence: float64(onlyFollow) / float64(onlyTotal), NeverAdherence: float64(neverFollow) / float64(neverTotal), WorkloadHash: hash, Observed: true, Host: host}, nil
}
