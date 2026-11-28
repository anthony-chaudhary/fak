// Adapted from microsoft/MInference
// (Apache-2.0): per-head sparse-pattern classifier using an offline last-64-q
// probe scored against stream_llm, vertical_and_slash, and block_sparse
// candidates, producing a per-layer per-head JSON config applied at model load.
package model

import (
	"encoding/json"
	"errors"
	"fmt"
)

// SparsePatternKind is the closed set of MInference per-head attention patterns.
type SparsePatternKind string

const (
	SparsePatternStreamLLM        SparsePatternKind = "stream_llm"
	SparsePatternVerticalAndSlash SparsePatternKind = "vertical_and_slash"
	SparsePatternBlockSparse      SparsePatternKind = "block_sparse"
)

// SparseHeadProbeScores holds offline last-64-q probe recall/fidelity scores for
// one (layer, head) pair against each candidate sparse pattern.
type SparseHeadProbeScores struct {
	Layer            int     `json:"layer"`
	Head             int     `json:"head"`
	StreamLLM        float64 `json:"stream_llm"`
	VerticalAndSlash float64 `json:"vertical_and_slash"`
	BlockSparse      float64 `json:"block_sparse"`
}

// SparseHeadPatternConfig is one selected per-layer, per-head sparse pattern.
type SparseHeadPatternConfig struct {
	Layer   int               `json:"layer"`
	Head    int               `json:"head"`
	Pattern SparsePatternKind `json:"pattern"`
	Score   float64           `json:"score"`
}

// SparsePatternModelConfig is the JSON-serializable per-layer per-head pattern
// map applied at model load.
type SparsePatternModelConfig struct {
	Schema    string                    `json:"schema"`
	ProbeRows int                       `json:"probe_rows"` // e.g. 64 for last-64-q
	Heads     []SparseHeadPatternConfig `json:"heads"`
}

// ClassifySparsePatterns selects the highest-scoring candidate pattern per
// (layer, head) from offline last-64-q probe scores.
func ClassifySparsePatterns(probeRows int, probes []SparseHeadProbeScores) (SparsePatternModelConfig, error) {
	if probeRows <= 0 {
		return SparsePatternModelConfig{}, fmt.Errorf("model: probeRows %d must be > 0", probeRows)
	}
	if len(probes) == 0 {
		return SparsePatternModelConfig{}, errors.New("model: empty sparse pattern probe list")
	}
	cfg := SparsePatternModelConfig{
		Schema:    "fak-minference-pattern/1",
		ProbeRows: probeRows,
		Heads:     make([]SparseHeadPatternConfig, 0, len(probes)),
	}
	for _, p := range probes {
		if p.Layer < 0 || p.Head < 0 {
			return SparsePatternModelConfig{}, fmt.Errorf("model: invalid layer=%d head=%d", p.Layer, p.Head)
		}
		bestKind := SparsePatternStreamLLM
		bestScore := p.StreamLLM
		if p.VerticalAndSlash > bestScore {
			bestKind = SparsePatternVerticalAndSlash
			bestScore = p.VerticalAndSlash
		}
		if p.BlockSparse > bestScore {
			bestKind = SparsePatternBlockSparse
			bestScore = p.BlockSparse
		}
		cfg.Heads = append(cfg.Heads, SparseHeadPatternConfig{
			Layer:   p.Layer,
			Head:    p.Head,
			Pattern: bestKind,
			Score:   bestScore,
		})
	}
	return cfg, nil
}

// ParseSparsePatternConfigJSON parses and validates a model-load sparse pattern JSON config.
func ParseSparsePatternConfigJSON(data []byte) (SparsePatternModelConfig, error) {
	var cfg SparsePatternModelConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return SparsePatternModelConfig{}, err
	}
	if cfg.Schema != "fak-minference-pattern/1" || cfg.ProbeRows <= 0 {
		return SparsePatternModelConfig{}, errors.New("model: invalid sparse pattern config header")
	}
	for _, h := range cfg.Heads {
		switch h.Pattern {
		case SparsePatternStreamLLM, SparsePatternVerticalAndSlash, SparsePatternBlockSparse:
		default:
			return SparsePatternModelConfig{}, fmt.Errorf("model: unknown sparse pattern %q", h.Pattern)
		}
	}
	return cfg, nil
}
