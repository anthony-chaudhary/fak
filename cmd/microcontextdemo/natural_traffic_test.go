package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeTrafficFixture(t *testing.T, dir string) string {
	t.Helper()
	c := naturalTrafficCorpus{Schema: naturalTrafficCorpusSchema, CreatedAt: "2026-08-16T00:00:00Z", Source: "fixture", Selection: "hash ranked; natural prior", SplitRule: "hash modulo five", Requested: 100, Eligible: 100, ReadbackClassPrior: map[string]int{}}
	for i := 0; i < 100; i++ {
		split := "tune"
		if i%5 == 0 {
			split = "test"
		}
		labels := []string{"repo_search"}
		if i%3 == 0 {
			labels = append(labels, "docs_read")
		}
		c.Records = append(c.Records, naturalTrafficRecord{ID: fmt.Sprintf("r-%03d", i), Split: split, SourceIssue: 6372, SourceSHA256: fmt.Sprintf("%064d", i), Question: "natural question", Context: "bounded context", ReadbackLabels: labels})
	}
	p := filepath.Join(dir, "corpus.json")
	if err := writeJSONFile(p, c); err != nil {
		t.Fatal(err)
	}
	return p
}
func writeTrafficJudgment(t *testing.T, p, corpusSHA, who, model string, records []naturalTrafficRecord) {
	t.Helper()
	j := naturalTrafficJudgments{Schema: naturalTrafficJudgmentSchema, CorpusSHA256: corpusSHA, Adjudicator: who, Model: model}
	for _, r := range records {
		j.Records = append(j.Records, naturalTrafficDecision{ID: r.ID, Labels: r.ReadbackLabels, Confidence: .9})
	}
	if err := writeJSONFile(p, j); err != nil {
		t.Fatal(err)
	}
}
func TestNaturalTrafficFoldAndVerifierAcceptHundredRecordMultiLabelCorpus(t *testing.T) {
	d := t.TempDir()
	cp := writeTrafficFixture(t, d)
	c, sha, e := loadNaturalTrafficCorpus(cp)
	if e != nil {
		t.Fatal(e)
	}
	a, b := filepath.Join(d, "a.json"), filepath.Join(d, "b.json")
	writeTrafficJudgment(t, a, sha, "model-a", "a", c.Records)
	writeTrafficJudgment(t, b, sha, "model-b", "b", c.Records)
	fold := filepath.Join(d, "fold.json")
	if e = foldNaturalTraffic(cp, a, b, fold); e != nil {
		t.Fatal(e)
	}
	if e = verifyNaturalTraffic(cp, fold); e != nil {
		t.Fatal(e)
	}
	var f naturalTrafficFold
	raw, _ := os.ReadFile(fold)
	json.Unmarshal(raw, &f)
	if f.Counts["test"] != 20 || len(f.Records) != 100 {
		t.Fatalf("counts=%v records=%d", f.Counts, len(f.Records))
	}
}
func TestNaturalTrafficFoldRefusesSameModel(t *testing.T) {
	d := t.TempDir()
	cp := writeTrafficFixture(t, d)
	c, sha, _ := loadNaturalTrafficCorpus(cp)
	a, b := filepath.Join(d, "a.json"), filepath.Join(d, "b.json")
	writeTrafficJudgment(t, a, sha, "one", "same", c.Records)
	writeTrafficJudgment(t, b, sha, "two", "same", c.Records)
	if e := foldNaturalTraffic(cp, a, b, filepath.Join(d, "fold.json")); e == nil {
		t.Fatal("same-model fold accepted")
	}
}
func TestNaturalTrafficReportVerifierRequiresWinningAndFalsifyingRegimes(t *testing.T) {
	d := t.TempDir()
	cp := writeTrafficFixture(t, d)
	_, sha, _ := loadNaturalTrafficCorpus(cp)
	r := naturalTrafficReport{Schema: naturalTrafficReportSchema, CorpusSHA256: sha, Receipts: []naturalTrafficReceipt{{ID: "r", Split: "test"}}, Policies: make([]naturalTrafficPolicyRow, 8), BreakEven: []naturalTrafficBreakEven{{Wins: true}, {Wins: false}, {}, {}, {}, {}}}
	p := filepath.Join(d, "report.json")
	writeJSONFile(p, r)
	if e := verifyNaturalTraffic(cp, p); e != nil {
		t.Fatal(e)
	}
}

func TestNaturalTrafficChatHonorsInjectedTransport(t *testing.T) {
	old := http.DefaultClient
	defer func() { http.DefaultClient = old }()

	called := false
	http.DefaultClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		called = true
		return &http.Response{
			StatusCode: http.StatusOK,
			Status:     "200 OK",
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader(`{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":3,"completion_tokens":1}}`)),
			Request:    r,
		}, nil
	})}

	got, usage, err := callNaturalTrafficChat(context.Background(), "https://model.invalid/v1", "key", "model", "prompt")
	if err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Fatal("injected default transport was bypassed")
	}
	if got != "ok" || usage.PromptTokens != 3 || usage.CompletionTokens != 1 {
		t.Fatalf("response = %q %+v", got, usage)
	}
}
