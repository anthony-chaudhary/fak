package ultracodebench

import (
	"crypto/sha256"
	"fmt"
	"testing"
)

func TestEvaluateToolResultShapeFindsCrossover(t *testing.T) {
	got, err := EvaluateToolResultShape(toolResultShapeFixture())
	if err != nil {
		t.Fatal(err)
	}
	if got.Verdict != "GAIN" || got.Crossovers["irrelevant"] != "small" || got.Crossovers["relevant"] != "large" {
		t.Fatalf("report = %+v", got)
	}
	for _, cell := range got.Cells {
		if cell.Omitted.Role != 0 || cell.Omitted.Repository != 0 || cell.Omitted.History != 0 {
			t.Fatalf("role/repository/history were not held constant: %+v", cell)
		}
	}
}

func TestEvaluateToolResultShapeAbstainsOnFactLoss(t *testing.T) {
	campaign := toolResultShapeFixture()
	campaign.Cells[0].RetainedFactsDigest = "lost-fact"
	got, err := EvaluateToolResultShape(campaign)
	if err != nil {
		t.Fatal(err)
	}
	if got.Verdict != "ABSTAIN" || got.Crossovers != nil {
		t.Fatalf("report = %+v", got)
	}
}

func toolResultShapeFixture() ToolResultShapeCampaign {
	c := ToolResultShapeCampaign{Schema: ToolResultShapeCampaignSchema, EvidenceKind: "synthetic_fixture", SourceArtifact: "fixture", Model: "qwen2.5:0.5b", Runtime: "Ollama/llama.cpp", Tokenizer: "whitespace-v1", TaskDigest: "sha256:task", CachePosture: "cold", CampaignVersion: "issue8677-v1", AcceptedEffectDigest: "sha256:effect", RequiredFactsDigest: "sha256:facts", RoleTokens: 80, RepositoryTokens: 120, HistoryTokens: 40}
	for _, x := range []struct {
		size, relevance           string
		tokens, omitted, overhead int64
	}{
		{"small", "relevant", 20, 5, 10}, {"large", "relevant", 200, 40, 10},
		{"small", "irrelevant", 20, 15, 10}, {"large", "irrelevant", 200, 180, 10},
	} {
		toolResult := repeatedWords(int(x.tokens), x.relevance)
		receipt := fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(toolResult)))
		c.Cells = append(c.Cells, ToolResultShapeCell{Size: x.size, Relevance: x.relevance, ToolResult: toolResult, ToolResultTokens: x.tokens, Accepted: true, EffectDigest: c.AcceptedEffectDigest, RetainedFactsDigest: c.RequiredFactsDigest, Omitted: ToolResultOmission{ToolResults: x.omitted}, ScopingOverheadTokens: x.overhead, SourceReceipt: receipt})
	}
	return c
}

func repeatedWords(n int, prefix string) string {
	result := prefix
	for i := 1; i < n; i++ {
		result += fmt.Sprintf(" token%d", i)
	}
	return result
}
