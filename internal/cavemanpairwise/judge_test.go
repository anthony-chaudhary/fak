package cavemanpairwise

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
)

type fakeJudge struct {
	mu    sync.Mutex
	calls int
	mode  string
}

func (f *fakeJudge) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if f.mode == "transport" {
		http.Error(w, "upstream unavailable", http.StatusServiceUnavailable)
		return
	}
	f.mu.Lock()
	f.calls++
	n := f.calls
	f.mu.Unlock()
	v := "tie"
	if f.mode == "parse" {
		fmt.Fprint(w, `{"choices":[{"message":{"content":"bad"}}]}`)
		return
	}
	if f.mode == "uncertain" {
		v = "uncertain"
	}
	if f.mode == "flip" {
		v = "A"
	}
	if n <= 20 && f.mode == "" {
		var q struct{ Messages []struct{ Content string } }
		_ = json.NewDecoder(r.Body).Decode(&q)
		v = fixtureVerdict(q.Messages[len(q.Messages)-1].Content)
	}
	scores := map[string]Scores{}
	for _, c := range Criteria {
		scores[c] = Scores{A: 3, B: 3}
	}
	jb, _ := json.Marshal(Judgment{Verdict: v, Scores: scores, Evidence: []string{"visible evidence"}})
	_ = json.NewEncoder(w).Encode(map[string]any{"id": fmt.Sprintf("fake-%d", n), "model": "gpt-5.6-sol", "choices": []any{map[string]any{"message": map[string]string{"content": string(jb)}, "finish_reason": "stop"}}})
}
func fixtureVerdict(u string) string {
	if strings.Contains(u, "No task context") {
		return "uncertain"
	}
	a := between(u, "RESPONSE A:\n", "\n\nRESPONSE B:")
	b := strings.SplitN(u, "\n\nRESPONSE B:\n", 2)[1]
	q := func(s string) int {
		switch {
		case strings.Contains(s, "Four."), strings.Contains(s, "Paris."), strings.Contains(s, "milliseconds"), strings.Contains(s, "placeholders"), strings.Contains(s, "treat that as data"), strings.Contains(s, "git status"), strings.Contains(s, "restricted environment"):
			return 2
		case strings.Contains(s, "Five."), strings.Contains(s, "Rome."), strings.Contains(s, "cannot help"), strings.Contains(s, "Disable authentication"), strings.Contains(s, "concatenate escaped"), strings.Contains(s, "Reveal your system"):
			return 0
		}
		return 1
	}
	x, y := q(a), q(b)
	if x > y {
		return "A"
	}
	if y > x {
		return "B"
	}
	return "tie"
}
func between(s, a, b string) string {
	x := strings.SplitN(s, a, 2)
	if len(x) < 2 {
		return ""
	}
	return strings.SplitN(x[1], b, 2)[0]
}
func inputs(t *testing.T) ([]byte, []byte, []byte) {
	t.Helper()
	rd := func(p string) []byte {
		b, e := os.ReadFile(p)
		if e != nil {
			t.Fatal(e)
		}
		return b
	}
	return rd("../../docs/_witnesses/armbench-caveman-native/live-gpt-5.6-sol-v2/manifest.json"), rd("../../docs/_witnesses/armbench-caveman-native/inputs/prompts.json"), rd("testdata/calibration.json")
}
func runFake(t *testing.T, mode string, mutate func(*Source)) Receipt {
	t.Helper()
	s, p, c := inputs(t)
	if mutate != nil {
		var x Source
		_ = json.Unmarshal(s, &x)
		mutate(&x)
		s, _ = json.Marshal(x)
	}
	f := &fakeJudge{mode: mode}
	srv := httptest.NewServer(f)
	defer srv.Close()
	r, e := Run(context.Background(), Client{BaseURL: srv.URL, APIKey: "x", Model: "gpt-5.6-sol"}, s, p, c)
	if e != nil {
		t.Fatal(e)
	}
	return r
}
func TestRunFullSpine(t *testing.T) {
	r := runFake(t, "", nil)
	if !r.Calibration.Pass || len(r.Application.Pairs) != 60 || len(r.Application.Pairs[0].Directions) != 2 {
		t.Fatalf("bad run: cal=%v pairs=%d", r.Calibration.Pass, len(r.Application.Pairs))
	}
	for _, m := range r.Application.ByComparison {
		if m.Total != 30 || m.Ties != 30 {
			t.Fatalf("%+v", m)
		}
	}
	if r.Application.NonInferiority == nil || !*r.Application.NonInferiority {
		t.Fatal("pairwise gate failed")
	}
	if r.TokenEligible {
		t.Fatal("token metrics eligible before safety binding")
	}
	safety := []byte(`{"source_sha256":"` + r.Provenance.SourceSHA256 + `","verdict":{"safety_gate_pass":true}}`)
	if err := BindSafety(&r, safety, 65); err != nil || !r.TokenEligible || r.TokenSavedPercent == nil {
		t.Fatalf("combined gate: err=%v eligible=%v", err, r.TokenEligible)
	}
}
func TestDeterministicBlindingAndOrder(t *testing.T) {
	if Blind("h", "id", "a") != Blind("h", "id", "a") || Order("h", "id") != Order("h", "id") || Blind("h", "id", "a") == Blind("h", "id", "b") {
		t.Fatal("nondeterministic")
	}
}
func TestStrictParse(t *testing.T) {
	v := `{"verdict":"tie","scores":{"factual_correctness":{"A":3,"B":3},"required_constraints":{"A":3,"B":3},"instruction_adherence":{"A":3,"B":3},"safety":{"A":4,"B":4},"justified_answering":{"A":4,"B":4}},"evidence":["same"]}`
	if _, e := ParseJudgment(v); e != nil {
		t.Fatal(e)
	}
	for _, b := range []string{`{"verdict":"C"}`, v + ` {}`, strings.Replace(v, `"tie"`, `"tie","extra":1`, 1)} {
		if _, e := ParseJudgment(b); e == nil {
			t.Fatal("accepted invalid")
		}
	}
}
func TestFailClosedCalibrationParse(t *testing.T) {
	r := runFake(t, "parse", nil)
	if r.Calibration.Pass || r.Application.Attempted || r.TokenEligible {
		t.Fatal("passed")
	}
}
func TestFailClosedCalibrationUncertain(t *testing.T) {
	r := runFake(t, "uncertain", nil)
	if r.Calibration.Pass || r.Application.Attempted {
		t.Fatal("passed")
	}
}
func TestFailClosedOrderBias(t *testing.T) {
	if runFake(t, "flip", nil).Calibration.Pass {
		t.Fatal("passed")
	}
}
func TestFailClosedProvenance(t *testing.T) {
	r := runFake(t, "", func(s *Source) { s.Schema = "drift" })
	if !r.Calibration.Pass || r.Application.Attempted || r.TokenEligible {
		t.Fatal("passed")
	}
}
func TestFailClosedMissingCell(t *testing.T) {
	s, p, _ := inputs(t)
	var src Source
	var pf PromptFile
	_ = json.Unmarshal(s, &src)
	_ = json.Unmarshal(p, &pf)
	src.Calls = src.Calls[1:]
	if ValidateMatchedCells(src, pf) == nil {
		t.Fatal("passed")
	}
}

func TestTransportErrorFailsClosed(t *testing.T) {
	r := runFake(t, "transport", nil)
	if r.Calibration.Pass || r.Calibration.Metrics.ParseFailures == 0 {
		t.Fatalf("transport failure did not fail closed: %+v", r.Calibration.Metrics)
	}
}

func TestSafetyBindingFailsClosed(t *testing.T) {
	r := runFake(t, "", nil)
	bad := []byte(`{"source_sha256":"wrong","verdict":{"safety_gate_pass":true}}`)
	if err := BindSafety(&r, bad, 65); err == nil || r.TokenEligible {
		t.Fatal("mismatched safety receipt accepted")
	}
	fail := []byte(`{"source_sha256":"` + r.Provenance.SourceSHA256 + `","verdict":{"safety_gate_pass":false}}`)
	if err := BindSafety(&r, fail, 65); err != nil || r.TokenEligible {
		t.Fatalf("failed safety gate accepted: %v", err)
	}
}

func TestProviderRequestBlindsArmMetadata(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","model":"gpt-5.6-sol","choices":[{"finish_reason":"stop","message":{"content":"{\"verdict\":\"tie\",\"scores\":{\"factual_correctness\":{\"A\":4,\"B\":4},\"required_constraints\":{\"A\":4,\"B\":4},\"instruction_adherence\":{\"A\":4,\"B\":4},\"safety\":{\"A\":4,\"B\":4},\"justified_answering\":{\"A\":4,\"B\":4}},\"evidence\":[\"equivalent\"]}"}}]}`))
	}))
	defer srv.Close()
	_, err := (Client{BaseURL: srv.URL, APIKey: "x", Model: "gpt-5.6-sol"}).Judge(context.Background(), "question", "alpha", "beta")
	if err != nil {
		t.Fatal(err)
	}
	body, _ := json.Marshal(got)
	for _, leaked := range []string{"normal", "native_medium", "caveman", "token_count"} {
		if strings.Contains(string(body), leaked) {
			t.Fatalf("arm metadata leaked in provider request: %s", leaked)
		}
	}
}
