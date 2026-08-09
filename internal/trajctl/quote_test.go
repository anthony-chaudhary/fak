package trajctl

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func fixtureCorpus(t *testing.T) []QuoteObservation {
	t.Helper()
	f, err := os.Open(filepath.Join("testdata", "repo_question_corpus.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	v, err := ReadQuoteCorpus(f)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func TestRepoQuestionQuoteSixFixtureSpine(t *testing.T) {
	obs := fixtureCorpus(t)
	if len(obs) != 6 {
		t.Fatalf("fixtures=%d want 6", len(obs))
	}
	_, err := NewRepoQuestionQuote(parseTime(obs[0].At), obs[0].Capability, obs[0].Index, obs[0].Quality, obs[0].Route, nil)
	if !errors.Is(err, ErrUnsupportedColdStart) {
		t.Fatalf("cold start err=%v", err)
	}
	q, err := NewRepoQuestionQuote(parseTime(obs[2].At), obs[2].Capability, obs[2].Index, obs[2].Quality, obs[2].Route, obs[:2])
	if err != nil {
		t.Fatal(err)
	}
	if q.Envelope.P50 != 10 || q.Envelope.P80 != 20 || q.Envelope.P95 != 20 {
		t.Fatalf("envelope=%+v", q.Envelope)
	}
	rev := ReviseForCapabilityFailure(q, 1, time.Date(2026, 1, 3, 1, 0, 0, 0, time.UTC), "route_failure")
	if rev.Envelope.P95 <= q.Envelope.P95 || rev.QuoteID != q.QuoteID {
		t.Fatalf("revision did not widen/bind: %+v", rev)
	}
	if q.Envelope.P95 != 20 {
		t.Fatal("original quote mutated")
	}
	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(struct {
		Quote    Quote         `json:"initial_quote"`
		Revision QuoteRevision `json:"revision"`
	}{q, rev}); err != nil {
		t.Fatal(err)
	}
	golden := filepath.Join("testdata", "repo_question_quote_revision.json")
	if os.Getenv("FAK_UPDATE_GOLDEN") == "1" {
		if err := os.WriteFile(golden, b.Bytes(), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(golden)
	if err != nil {
		t.Fatal(err)
	}
	var gotV, wantV any
	if json.Unmarshal(b.Bytes(), &gotV) != nil || json.Unmarshal(want, &wantV) != nil {
		t.Fatal("invalid artifact")
	}
	gb, _ := json.Marshal(gotV)
	wb, _ := json.Marshal(wantV)
	if !bytes.Equal(gb, wb) {
		t.Fatalf("artifact drift; run FAK_UPDATE_GOLDEN=1 go test ./internal/trajctl -run TestRepoQuestionQuoteSixFixtureSpine")
	}
}

func TestRepoQuestionQuoteBacktestChronologicalAndCensored(t *testing.T) {
	rep := BacktestRepoQuestion(fixtureCorpus(t))
	if rep.Tested != 4 || rep.ColdStartRefusals != 2 || rep.Censored != 1 {
		t.Fatalf("report=%+v", rep)
	}
	if len(rep.Coverage) != 3 || rep.Coverage[0].Quantile != "p50" || rep.Coverage[2].Quantile != "p95" {
		t.Fatalf("coverage=%+v", rep.Coverage)
	}
	b, _ := json.MarshalIndent(rep, "", "  ")
	b = append(b, '\n')
	golden := filepath.Join("testdata", "repo_question_backtest.json")
	if os.Getenv("FAK_UPDATE_GOLDEN") == "1" {
		if err := os.WriteFile(golden, b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(golden)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(b, want) {
		t.Fatalf("backtest drift; run update golden")
	}
}

func TestQuoteLedgerAppendOnlyBindsCompletion(t *testing.T) {
	obs := fixtureCorpus(t)
	q, _ := NewRepoQuestionQuote(parseTime(obs[2].At), obs[2].Capability, obs[2].Index, obs[2].Quality, obs[2].Route, obs[:2])
	path := filepath.Join(t.TempDir(), "quote.jsonl")
	if err := AppendQuoteRecord(path, QuoteLedgerRecord{Kind: "quote", Quote: &q}); err != nil {
		t.Fatal(err)
	}
	c := QuoteCompletion{Schema: QuoteSchema, QuoteID: q.QuoteID, CreatedAt: obs[2].At, QualityScore: obs[2].QualityScore, QualityWitness: obs[2].QualityWitness, Quality: q.Quality, RawRealizedCost: obs[2].RawRealizedCost, CostUnit: obs[2].CostUnit}
	if err := AppendQuoteRecord(path, QuoteLedgerRecord{Kind: "completion", Completion: &c}); err != nil {
		t.Fatal(err)
	}
	data, _ := os.ReadFile(path)
	if bytes.Count(data, []byte("\n")) != 2 || !bytes.Contains(data, []byte("witness-3")) {
		t.Fatalf("ledger=%s", data)
	}
}
