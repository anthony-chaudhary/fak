package bench

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"sort"
)

const (
	DepthLatencyNaiveArm    = "naive_negation"
	DepthLatencyOperatorArm = "positive_operator"
)

// DepthLatencyRow is one arm of a matched negation cost observation. Depth is
// the zero-based crystallization depth from model.CrystallizationDepth;
// LatencyMS is the measured wall time for the named replay surface.
type DepthLatencyRow struct {
	Schema      string  `json:"schema"`
	PairID      string  `json:"pair_id"`
	Condition   string  `json:"condition"`
	Depth       int     `json:"depth"`
	LatencyMS   float64 `json:"latency_ms"`
	Accuracy    float64 `json:"accuracy"`
	Model       string  `json:"model"`
	Host        string  `json:"host"`
	Surface     string  `json:"surface"`
	Provenance  string  `json:"provenance"`
	Repetitions int     `json:"repetitions"`
}

type DepthLatencyDelta struct {
	PairID         string  `json:"pair_id"`
	DepthDelta     int     `json:"depth_delta"`
	LatencyDeltaMS float64 `json:"latency_delta_ms"`
	AccuracyDelta  float64 `json:"accuracy_delta"`
}

// ReadDepthLatencyJSONL decodes the auditable one-row-per-condition witness.
func ReadDepthLatencyJSONL(r io.Reader) ([]DepthLatencyRow, error) {
	scanner := bufio.NewScanner(r)
	var rows []DepthLatencyRow
	for line := 1; scanner.Scan(); line++ {
		if len(scanner.Bytes()) == 0 {
			continue
		}
		var row DepthLatencyRow
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			return nil, fmt.Errorf("depth/latency row %d: %w", line, err)
		}
		rows = append(rows, row)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return rows, nil
}

// ValidateDepthLatencyTax checks matched workload identity and computes naive
// minus operator cost. Every operator arm must preserve accuracy and strictly
// reduce depth or latency; no aggregate can hide a losing pair.
func ValidateDepthLatencyTax(rows []DepthLatencyRow) ([]DepthLatencyDelta, error) {
	type pair struct{ naive, operator *DepthLatencyRow }
	pairs := make(map[string]*pair)
	for i := range rows {
		row := &rows[i]
		if row.Schema != "fak-negation-depth-latency/1" || row.PairID == "" || row.Model == "" || row.Host == "" || row.Surface == "" || row.Provenance == "" {
			return nil, fmt.Errorf("row %d has incomplete witness identity", i)
		}
		if row.Depth < 0 || row.LatencyMS <= 0 || row.Accuracy < 0 || row.Accuracy > 1 || row.Repetitions <= 0 {
			return nil, fmt.Errorf("row %d has invalid metrics", i)
		}
		p := pairs[row.PairID]
		if p == nil {
			p = &pair{}
			pairs[row.PairID] = p
		}
		switch row.Condition {
		case DepthLatencyNaiveArm:
			if p.naive != nil {
				return nil, fmt.Errorf("pair %q duplicates naive arm", row.PairID)
			}
			p.naive = row
		case DepthLatencyOperatorArm:
			if p.operator != nil {
				return nil, fmt.Errorf("pair %q duplicates operator arm", row.PairID)
			}
			p.operator = row
		default:
			return nil, fmt.Errorf("pair %q has unknown condition %q", row.PairID, row.Condition)
		}
	}
	ids := make([]string, 0, len(pairs))
	for id := range pairs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	deltas := make([]DepthLatencyDelta, 0, len(ids))
	for _, id := range ids {
		p := pairs[id]
		if p.naive == nil || p.operator == nil {
			return nil, fmt.Errorf("pair %q is missing an arm", id)
		}
		if p.naive.Model != p.operator.Model || p.naive.Host != p.operator.Host || p.naive.Surface != p.operator.Surface || p.naive.Repetitions != p.operator.Repetitions {
			return nil, fmt.Errorf("pair %q violates identical-workload identity", id)
		}
		depthDelta := p.naive.Depth - p.operator.Depth
		latencyDelta := p.naive.LatencyMS - p.operator.LatencyMS
		accuracyDelta := p.operator.Accuracy - p.naive.Accuracy
		if accuracyDelta < 0 {
			return nil, fmt.Errorf("pair %q operator accuracy regressed by %.3f", id, -accuracyDelta)
		}
		if depthDelta <= 0 && latencyDelta <= 0 {
			return nil, fmt.Errorf("pair %q operator reduced neither depth nor latency", id)
		}
		deltas = append(deltas, DepthLatencyDelta{PairID: id, DepthDelta: depthDelta, LatencyDeltaMS: latencyDelta, AccuracyDelta: accuracyDelta})
	}
	if len(deltas) == 0 {
		return nil, fmt.Errorf("empty depth/latency witness")
	}
	return deltas, nil
}
