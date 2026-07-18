package bench

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"unicode"

	"github.com/anthony-chaudhary/fak/internal/metrics"
)

const NegationABObservedProvenance = "OBSERVED / DETERMINISTIC CORPUS REPLAY"

type NegationABItem struct {
	ID            string `json:"id"`
	Rung          string `json:"rung"`
	NaiveText     string `json:"naive_negation"`
	OperatorText  string `json:"operator"`
	ExpectedClass string `json:"expected_class"`
}

type NegationABArm struct {
	metrics.Arm
	Rung            string  `json:"rung"`
	WorkloadHash    string  `json:"workload_hash"`
	InversionTokens int     `json:"inversion_tax_tokens"`
	Correct         int     `json:"correct"`
	Total           int     `json:"total"`
	Score           float64 `json:"classification_score"`
}

type NegationABRung struct {
	Rung                 string        `json:"rung"`
	Naive                NegationABArm `json:"naive_negation"`
	Operator             NegationABArm `json:"operator"`
	InversionTokensSaved int           `json:"inversion_tax_tokens_saved"`
	ClassificationDelta  float64       `json:"classification_score_delta"`
}

type NegationABArtifact struct {
	Schema            string           `json:"schema"`
	Provenance        string           `json:"provenance"`
	Host              string           `json:"host"`
	AffectsAcceptance bool             `json:"affects_acceptance"`
	Rungs             []NegationABRung `json:"rungs"`
}

func ReadNegationABCorpus(r io.Reader) ([]NegationABItem, error) {
	var items []NegationABItem
	if err := json.NewDecoder(r).Decode(&items); err != nil {
		return nil, fmt.Errorf("decode corpus: %w", err)
	}
	seen := map[string]bool{}
	for i, item := range items {
		if item.ID == "" || item.Rung == "" || item.NaiveText == "" || item.OperatorText == "" || item.ExpectedClass == "" {
			return nil, fmt.Errorf("item %d incomplete", i)
		}
		if seen[item.ID] {
			return nil, fmt.Errorf("duplicate item %q", item.ID)
		}
		seen[item.ID] = true
	}
	if len(items) == 0 {
		return nil, fmt.Errorf("empty corpus")
	}
	return items, nil
}

func BuildNegationAB(items []NegationABItem, host string) (NegationABArtifact, error) {
	if host == "" {
		return NegationABArtifact{}, fmt.Errorf("host is required")
	}
	byRung := map[string][]NegationABItem{}
	for _, item := range items {
		byRung[item.Rung] = append(byRung[item.Rung], item)
	}
	names := make([]string, 0, len(byRung))
	for name := range byRung {
		names = append(names, name)
	}
	sort.Strings(names)
	art := NegationABArtifact{Schema: "fak-negation-ab/1", Provenance: NegationABObservedProvenance, Host: host, AffectsAcceptance: false}
	for _, rung := range names {
		group := byRung[rung]
		hash := negationABWorkloadHash(group)
		naive := scoreNegationABArm(rung, "naive_negation", hash, group, func(i NegationABItem) string { return i.NaiveText })
		op := scoreNegationABArm(rung, "operator", hash, group, func(i NegationABItem) string { return i.OperatorText })
		if naive.WorkloadHash != op.WorkloadHash {
			return NegationABArtifact{}, fmt.Errorf("rung %s workload mismatch", rung)
		}
		art.Rungs = append(art.Rungs, NegationABRung{Rung: rung, Naive: naive, Operator: op, InversionTokensSaved: naive.InversionTokens - op.InversionTokens, ClassificationDelta: op.Score - naive.Score})
	}
	return art, nil
}

func scoreNegationABArm(rung, name, hash string, items []NegationABItem, text func(NegationABItem) string) NegationABArm {
	arm := NegationABArm{Arm: metrics.Arm{Label: name, Calls: len(items)}, Rung: rung, WorkloadHash: hash, Total: len(items)}
	for _, item := range items {
		value := text(item)
		arm.InversionTokens += negationTokenCount(value)
		if classifyNegationAB(value) == item.ExpectedClass {
			arm.Correct++
		}
	}
	arm.Score = float64(arm.Correct) / float64(arm.Total)
	return arm
}

func negationABWorkloadHash(items []NegationABItem) string {
	type identity struct{ ID, Rung, Expected string }
	ids := make([]identity, 0, len(items))
	for _, i := range items {
		ids = append(ids, identity{i.ID, i.Rung, i.ExpectedClass})
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i].ID < ids[j].ID })
	b, _ := json.Marshal(ids)
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func negationTokenCount(s string) int {
	count := 0
	for _, tok := range strings.FieldsFunc(strings.ToLower(s), func(r rune) bool { return !unicode.IsLetter(r) }) {
		switch tok {
		case "not", "no", "never", "without", "avoid", "don", "dont", "only":
			count++
		}
	}
	return count
}

func classifyNegationAB(s string) string {
	lower := strings.ToLower(s)
	for _, field := range strings.Fields(lower) {
		if strings.HasPrefix(field, "state=") {
			return strings.Trim(strings.TrimPrefix(field, "state="), ".,;:[]()")
		}
	}
	if strings.Contains(lower, "do not delete") || strings.Contains(lower, "don't delete") {
		return "preserve"
	}
	if strings.Contains(lower, "only read") || strings.Contains(lower, "read only") {
		return "readonly"
	}
	if strings.Contains(lower, "never send") {
		return "withhold"
	}
	return "unknown"
}
