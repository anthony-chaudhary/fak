package model

import (
	"encoding/json"
	"testing"
)

// fak-test:runtime fast est=50ms lane=default
func TestSparsePatternClassifier(t *testing.T) {
	probes := []SparseHeadProbeScores{
		{Layer: 0, Head: 0, StreamLLM: 0.96, VerticalAndSlash: 0.81, BlockSparse: 0.72},
		{Layer: 0, Head: 1, StreamLLM: 0.60, VerticalAndSlash: 0.94, BlockSparse: 0.88},
		{Layer: 1, Head: 0, StreamLLM: 0.55, VerticalAndSlash: 0.70, BlockSparse: 0.95},
	}
	cfg, err := ClassifySparsePatterns(64, probes)
	if err != nil {
		t.Fatalf("ClassifySparsePatterns: %v", err)
	}
	if cfg.Heads[0].Pattern != SparsePatternStreamLLM ||
		cfg.Heads[1].Pattern != SparsePatternVerticalAndSlash ||
		cfg.Heads[2].Pattern != SparsePatternBlockSparse {
		t.Fatalf("unexpected pattern classification: %+v", cfg.Heads)
	}

	raw, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	parsed, err := ParseSparsePatternConfigJSON(raw)
	if err != nil || len(parsed.Heads) != 3 {
		t.Fatalf("ParseSparsePatternConfigJSON = (%+v, %v)", parsed, err)
	}
	if _, err := ParseSparsePatternConfigJSON([]byte(`{"schema":"fak-minference-pattern/1","probe_rows":64,"heads":[{"layer":0,"head":0,"pattern":"unknown"}]}`)); err == nil {
		t.Fatal("expected error for unknown sparse pattern")
	}
}
