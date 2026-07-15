package gateway

import (
	"errors"
	"sort"
	"strings"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/negframe"
)

var ErrReseedMissingInvariant = errors.New("reseed missing originating-task invariant")

var reseedCarriedInvariantTotal atomic.Uint64
var reseedRefusedTotal atomic.Uint64

// ReseedRequest is the lean restart material assembled after a wrong turn.
type ReseedRequest struct {
	CorrectedGoal string `json:"corrected_goal"`
	Invariant     string `json:"originating_task_invariant"`
	PriorContext  string `json:"prior_context,omitempty"`
}

type ReseedResult struct {
	Query            string `json:"query,omitempty"`
	CarriedInvariant bool   `json:"carried_invariant"`
	Refused          bool   `json:"refused"`
	Reason           string `json:"reason,omitempty"`
}

// AssembleInvariantReseed builds a fresh positive query. It never emits an
// amnesiac seed: absent or token-incomplete invariants are refused so callers
// can fall back to a fuller carry.
func AssembleInvariantReseed(req ReseedRequest) (ReseedResult, error) {
	goal := strings.TrimSpace(req.CorrectedGoal)
	invariant := strings.TrimSpace(req.Invariant)
	if goal == "" || invariant == "" {
		reseedRefusedTotal.Add(1)
		return ReseedResult{Refused: true, Reason: ErrReseedMissingInvariant.Error()}, ErrReseedMissingInvariant
	}

	// Deliberately exclude PriorContext: restart-over-negate constructs current
	// positive state instead of broadcasting an "ignore the above" ledger.
	candidate := "Current goal: " + goal + "\nOriginating-task invariant: " + invariant
	query := negframe.Reframe(candidate)
	if !containsAllReseedTokens(query, invariant) {
		reseedRefusedTotal.Add(1)
		return ReseedResult{Refused: true, Reason: ErrReseedMissingInvariant.Error()}, ErrReseedMissingInvariant
	}
	reseedCarriedInvariantTotal.Add(1)
	return ReseedResult{Query: query, CarriedInvariant: true}, nil
}

func containsAllReseedTokens(text, required string) bool {
	have := make(map[string]struct{})
	for _, token := range reseedTokens(text) {
		have[token] = struct{}{}
	}
	for _, token := range reseedTokens(required) {
		if _, ok := have[token]; !ok {
			return false
		}
	}
	return true
}

func reseedTokens(text string) []string {
	fields := strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '_' || r == '-')
	})
	sort.Strings(fields)
	return fields
}

type ReseedMetrics struct {
	CarriedInvariant uint64 `json:"reseed_carried_invariant"`
	Refused          uint64 `json:"reseed_refused"`
}

func InvariantReseedMetrics() ReseedMetrics {
	return ReseedMetrics{CarriedInvariant: reseedCarriedInvariantTotal.Load(), Refused: reseedRefusedTotal.Load()}
}

func InvariantReseedPrometheus() string {
	m := InvariantReseedMetrics()
	return "fak_reseed_total{verdict=\"carried_invariant\"} " + utoaGateway(m.CarriedInvariant) + "\n" +
		"fak_reseed_total{verdict=\"refused\"} " + utoaGateway(m.Refused) + "\n"
}

func utoaGateway(n uint64) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}
