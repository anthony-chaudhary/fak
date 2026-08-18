package bench

import (
	"bufio"
	"encoding/json"
	"errors"
	"math"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/mathx"
	"github.com/anthony-chaudhary/fak/internal/metrics"
)

const NegatedQAOfflineProvenance = "MODELED / OFFLINE PROXY"
const NegatedQALiveProvenance = "OBSERVED / LIVE WITNESS"

type NegatedQAItem struct {
	ID          string   `json:"id"`
	LowPrompt   string   `json:"low_prompt"`
	HighPrompt  string   `json:"high_prompt"`
	Forbidden   []string `json:"forbidden"`
	LowFailure  float64  `json:"low_failure"`
	HighFailure float64  `json:"high_failure"`
}

type NegatedQAWitness struct {
	ID     string `json:"id"`
	Arm    string `json:"arm"`
	Output string `json:"output"`
	Failed *bool  `json:"failed,omitempty"`
}

type NegatedQAReport struct {
	Provenance      string  `json:"provenance"`
	Pairs           int     `json:"pairs"`
	Correlation     float64 `json:"correlation"`
	ImprovedPairs   int     `json:"improved_pairs"`
	WorsenedPairs   int     `json:"worsened_pairs"`
	Ties            int     `json:"ties"`
	SignTestP       float64 `json:"sign_test_p"`
	DirectionalPass bool    `json:"directional_pass"`
	Significant     bool    `json:"significant"`
	Verdict         string  `json:"verdict"`
}

func LoadNegatedQAFixture(path string) ([]NegatedQAItem, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var items []NegatedQAItem
	if err := json.Unmarshal(b, &items); err != nil {
		return nil, err
	}
	if len(items) < 4 {
		return nil, errors.New("negated-QA fixture requires at least four pairs")
	}
	return items, nil
}

func AnalyzeNegatedQA(items []NegatedQAItem, witnessPath string) (NegatedQAReport, error) {
	provenance := NegatedQAOfflineProvenance
	if strings.TrimSpace(witnessPath) != "" {
		if err := applyNegatedQAWitness(items, witnessPath); err != nil {
			return NegatedQAReport{}, err
		}
		provenance = NegatedQALiveProvenance
	}
	var tax, failure []float64
	improved, worsened, ties := 0, 0, 0
	for _, item := range items {
		lowTax := float64(metrics.PromptNegationTax(item.LowPrompt))
		highTax := float64(metrics.PromptNegationTax(item.HighPrompt))
		tax = append(tax, lowTax, highTax)
		failure = append(failure, item.LowFailure, item.HighFailure)
		deltaTax := highTax - lowTax
		deltaFailure := item.HighFailure - item.LowFailure
		switch {
		case deltaTax*deltaFailure > 0:
			improved++
		case deltaTax*deltaFailure < 0:
			worsened++
		default:
			ties++
		}
	}
	corr := mathx.Pearson(tax, failure)
	p := twoSidedSignP(improved, worsened)
	directional := corr > 0 && improved > worsened
	significant := p <= 0.05
	verdict := "not-yet-proven"
	if directional && significant {
		verdict = "load-bearing (predicts degradation)"
	}
	return NegatedQAReport{Provenance: provenance, Pairs: len(items), Correlation: corr, ImprovedPairs: improved, WorsenedPairs: worsened, Ties: ties, SignTestP: p, DirectionalPass: directional, Significant: significant, Verdict: verdict}, nil
}

func applyNegatedQAWitness(items []NegatedQAItem, path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	byID := make(map[string]*NegatedQAItem, len(items))
	for i := range items {
		byID[items[i].ID] = &items[i]
	}
	seen := make(map[string]bool)
	s := bufio.NewScanner(f)
	for s.Scan() {
		var row NegatedQAWitness
		if err := json.Unmarshal(s.Bytes(), &row); err != nil {
			return err
		}
		item := byID[row.ID]
		if item == nil || (row.Arm != "low" && row.Arm != "high") {
			return errors.New("unknown negated-QA witness row")
		}
		failed := containsForbidden(row.Output, item.Forbidden)
		if row.Failed != nil {
			failed = *row.Failed
		}
		if row.Arm == "low" {
			if failed {
				item.LowFailure = 1
			} else {
				item.LowFailure = 0
			}
		} else {
			if failed {
				item.HighFailure = 1
			} else {
				item.HighFailure = 0
			}
		}
		seen[row.ID+"/"+row.Arm] = true
	}
	if err := s.Err(); err != nil {
		return err
	}
	if len(seen) != len(items)*2 {
		return errors.New("live witness must contain both arms for every item")
	}
	return nil
}

func containsForbidden(output string, forbidden []string) bool {
	output = strings.ToLower(output)
	for _, token := range forbidden {
		if strings.Contains(output, strings.ToLower(token)) {
			return true
		}
	}
	return false
}

func twoSidedSignP(positive, negative int) float64 {
	n := positive + negative
	if n == 0 {
		return 1
	}
	k := positive
	if negative < k {
		k = negative
	}
	var sum float64
	for i := 0; i <= k; i++ {
		sum += binomial(n, i)
	}
	p := 2 * sum / math.Pow(2, float64(n))
	if p > 1 {
		return 1
	}
	return p
}
func binomial(n, k int) float64 {
	if k < 0 || k > n {
		return 0
	}
	if k > n-k {
		k = n - k
	}
	v := 1.0
	for i := 1; i <= k; i++ {
		v *= float64(n-k+i) / float64(i)
	}
	return v
}

func SortNegatedQA(items []NegatedQAItem) {
	sort.Slice(items, func(i, j int) bool { return items[i].ID < items[j].ID })
}
