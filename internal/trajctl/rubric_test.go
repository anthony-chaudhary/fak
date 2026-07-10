package trajctl

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func rubricFixture() *Rubric {
	return &Rubric{
		Source: "test-model",
		Criteria: []RubricCriterion{
			{ID: "c1", Text: "Changes stay inside internal/trajctl (localization)."},
			{ID: "c2", Text: "The diff is one leaf-sized feature commit (edit constraint)."},
			{ID: "c3", Text: "No detours: every session advances the declared plan (trajectory discipline)."},
		},
	}
}

// TestJudgeScorerRubricAttribution is the issue's named witness (#2544): the
// SAME trajectory (objective, evidence window, canned verdict transport)
// scored WITHOUT a rubric and then WITH one. Without: the request carries no
// rubric and the row has no per-criterion attribution. With: the request
// carries the formatted rubric and the row cites each criterion that moved,
// by id, with its own finding blob — proving the rubric changed the judging,
// asserted per criterion and not just on the aggregate score.
func TestJudgeScorerRubricAttribution(t *testing.T) {
	win := EvidenceWindow{UnixMillis: 42}

	// Without a rubric: the bare #2543 shape, unchanged.
	bare := &cannedJudge{
		verdict: JudgeVerdict{Progress: 0.5, Rationale: "half the docs moved"},
		usage:   JudgeUsage{Tokens: 100},
	}
	rows := NewJudgeScorer(bare, 512).Score(docsObjective(), win)
	if len(rows) != 1 {
		t.Fatalf("bare: want 1 row, got %d", len(rows))
	}
	if bare.seen.Rubric != "" {
		t.Errorf("bare request must carry no rubric, got %q", bare.seen.Rubric)
	}
	for _, ev := range rows[0].Evidence {
		if ev.Kind == "rubric-criterion" {
			t.Errorf("bare row must carry no rubric-criterion evidence, got %+v", ev)
		}
	}

	// With a rubric cached on the objective: same window, same transport style.
	obj := docsObjective()
	obj.Rubric = rubricFixture()
	judged := &cannedJudge{
		verdict: JudgeVerdict{
			Progress:  0.5,
			Rationale: "half the docs moved",
			Criteria: []RubricFinding{
				{ID: "c1", Progress: 1, Note: "all edits in-package"},
				{ID: "c2", Progress: 0.5, Note: "diff still growing"},
				{ID: "c3", Progress: 0, Note: "one detour into cmd"},
			},
		},
		usage: JudgeUsage{Tokens: 140},
	}
	rows = NewJudgeScorer(judged, 512).Score(obj, win)
	if len(rows) != 1 {
		t.Fatalf("rubric: want 1 row, got %d", len(rows))
	}
	if want := FormatRubricForPrompt(obj.Rubric); judged.seen.Rubric != want || want == "" {
		t.Errorf("rubric request block = %q, want %q", judged.seen.Rubric, want)
	}

	byID := map[string]RubricFinding{}
	for _, ev := range rows[0].Evidence {
		if ev.Kind != "rubric-criterion" {
			continue
		}
		var f RubricFinding
		if err := json.Unmarshal([]byte(ev.Detail), &f); err != nil {
			t.Fatalf("criterion evidence %q detail is not a finding blob: %v", ev.Ref, err)
		}
		if f.ID != ev.Ref {
			t.Errorf("evidence ref %q disagrees with finding id %q", ev.Ref, f.ID)
		}
		byID[ev.Ref] = f
	}
	if len(byID) != 3 {
		t.Fatalf("want per-criterion attribution for 3 criteria, got %d: %+v", len(byID), rows[0].Evidence)
	}
	if f := byID["c1"]; f.Progress != 1 || f.Note != "all edits in-package" {
		t.Errorf("c1 attribution lost: %+v", f)
	}
	if f := byID["c3"]; f.Progress != 0 || f.Note != "one detour into cmd" {
		t.Errorf("c3 attribution lost: %+v", f)
	}
	// The attributed row must still be ledger-appendable.
	if err := validateScore(rows[0]); err != nil {
		t.Errorf("rubric row fails ledger validation: %v", err)
	}
	if rows[0].Witness != W1 {
		t.Errorf("rubric row rung = %q, want W1 (a rubric improves W1, it does not invent a rung)", rows[0].Witness)
	}
}

// TestFormatRubricForPrompt pins the deterministic, cache-friendly rendering.
func TestFormatRubricForPrompt(t *testing.T) {
	got := FormatRubricForPrompt(rubricFixture())
	want := "1. [c1] Changes stay inside internal/trajctl (localization).\n" +
		"2. [c2] The diff is one leaf-sized feature commit (edit constraint).\n" +
		"3. [c3] No detours: every session advances the declared plan (trajectory discipline).\n"
	if got != want {
		t.Errorf("FormatRubricForPrompt = %q, want %q", got, want)
	}
	if FormatRubricForPrompt(nil) != "" {
		t.Errorf("nil rubric must format to empty")
	}
}

// cannedRubricClient scripts one generation reply and records the request.
type cannedRubricClient struct {
	rubric Rubric
	usage  JudgeUsage
	err    error
	seen   RubricRequest
}

func (c *cannedRubricClient) GenerateRubric(req RubricRequest) (Rubric, JudgeUsage, error) {
	c.seen = req
	return c.rubric, c.usage, c.err
}

// TestGenerateObjectiveRubric covers the declare-time fold: cap forwarded into
// the request, over-budget return rejected, empty rubric rejected, blank ids
// synthesized, and the objective statement + plan grounding the call.
func TestGenerateObjectiveRubric(t *testing.T) {
	obj := docsObjective()
	obj.Plan = []PlanPhase{{ID: "phase-1", Title: "move pages"}}

	ok := &cannedRubricClient{
		rubric: Rubric{Criteria: []RubricCriterion{{ID: "", Text: "  stay in the docs tree  "}, {ID: "cX", Text: "finish in budget"}}},
		usage:  JudgeUsage{Tokens: 200},
	}
	r, err := GenerateObjectiveRubric(ok, obj, 0)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if ok.seen.MaxTokens != DefaultRubricMaxCallTokens {
		t.Errorf("cap = %d, want default %d forwarded", ok.seen.MaxTokens, DefaultRubricMaxCallTokens)
	}
	if ok.seen.Objective != obj.Statement || len(ok.seen.Plan) != 1 {
		t.Errorf("request lost the objective/plan grounding: %+v", ok.seen)
	}
	if len(r.Criteria) != 2 || r.Criteria[0].ID != "c1" || r.Criteria[0].Text != "stay in the docs tree" || r.Criteria[1].ID != "cX" {
		t.Errorf("normalization wrong: %+v", r.Criteria)
	}

	over := &cannedRubricClient{rubric: ok.rubric, usage: JudgeUsage{Tokens: 5000}}
	if _, err := GenerateObjectiveRubric(over, obj, 1024); err == nil {
		t.Errorf("over-budget return must fail closed")
	}
	empty := &cannedRubricClient{rubric: Rubric{}, usage: JudgeUsage{Tokens: 10}}
	if _, err := GenerateObjectiveRubric(empty, obj, 1024); err == nil {
		t.Errorf("empty rubric must fail closed")
	}
	failed := &cannedRubricClient{err: errors.New("upstream 500")}
	if _, err := GenerateObjectiveRubric(failed, obj, 1024); err == nil {
		t.Errorf("failed call must fail closed")
	}
	if _, err := GenerateObjectiveRubric(nil, obj, 1024); err == nil {
		t.Errorf("nil client must fail closed")
	}
}

// TestObjectiveRubricCachedInLedger proves the rubric is cached WITH the
// objective as metadata: the declared row round-trips through Append/Fold
// with the rubric intact, and an invalid rubric refuses validation.
func TestObjectiveRubricCachedInLedger(t *testing.T) {
	obj := docsObjective()
	obj.Rubric = rubricFixture()
	path := t.TempDir() + "/trajctl.jsonl"
	if err := Append(path, ObjectiveRecord(obj)); err != nil {
		t.Fatalf("append: %v", err)
	}
	st := Fold(ReadLedgerFile(path))
	got, ok := st.Objectives[obj.ID]
	if !ok || got.Rubric == nil {
		t.Fatalf("folded objective lost the cached rubric: %+v", got)
	}
	if len(got.Rubric.Criteria) != 3 || got.Rubric.Criteria[2].ID != "c3" || got.Rubric.Source != "test-model" {
		t.Errorf("rubric metadata mangled in the ledger: %+v", got.Rubric)
	}

	dup := docsObjective()
	dup.Rubric = &Rubric{Criteria: []RubricCriterion{{ID: "c1", Text: "a"}, {ID: "c1", Text: "b"}}}
	if err := Validate(ObjectiveRecord(dup)); err == nil {
		t.Errorf("duplicate criterion ids must refuse validation")
	}
}

// TestGatewayRubricClient exercises the HTTP client against a canned gateway:
// forced tool choice, cap forwarded as max_tokens, model recorded as Source,
// and the no-tool-call reply failing loudly.
func TestGatewayRubricClient(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Errorf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"choices":[{"message":{"tool_calls":[{"function":{"name":"emit_rubric","arguments":"{\"criteria\":[{\"id\":\"c1\",\"text\":\"stay local\"}]}"}}]}}],"usage":{"total_tokens":77}}`))
	}))
	defer srv.Close()

	client := &GatewayRubricClient{BaseURL: srv.URL, Model: "judge-1"}
	r, usage, err := client.GenerateRubric(RubricRequest{Objective: "obj", Plan: []PlanPhase{{ID: "p1", Title: "step"}}, MaxTokens: 321})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if usage.Tokens != 77 {
		t.Errorf("usage = %d, want 77", usage.Tokens)
	}
	if len(r.Criteria) != 1 || r.Criteria[0].ID != "c1" || r.Source != "judge-1" {
		t.Errorf("rubric = %+v", r)
	}
	if gotBody["max_tokens"].(float64) != 321 {
		t.Errorf("max_tokens = %v, want the 321 cap forwarded", gotBody["max_tokens"])
	}
	tc := gotBody["tool_choice"].(map[string]any)["function"].(map[string]any)["name"]
	if tc != "emit_rubric" {
		t.Errorf("tool_choice = %v, want forced emit_rubric", tc)
	}

	noCall := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"choices":[{"message":{}}],"usage":{"total_tokens":5}}`))
	}))
	defer noCall.Close()
	if _, _, err := (&GatewayRubricClient{BaseURL: noCall.URL}).GenerateRubric(RubricRequest{Objective: "obj"}); err == nil || !strings.Contains(err.Error(), "no tool call") {
		t.Errorf("no-tool-call reply must error, got %v", err)
	}
}

// TestGatewayJudgeClientRubricPromptSwap pins the client-side replacement the
// issue names: a rubric-carrying request swaps in the rubric system prompt,
// the RUBRIC block, and the attribution-required verdict schema; a bare
// request keeps the exact #2543 shape.
func TestGatewayJudgeClientRubricPromptSwap(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Errorf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"choices":[{"message":{"tool_calls":[{"function":{"name":"emit_verdict","arguments":"{\"progress\":0.4,\"rationale\":\"ok\",\"criteria\":[{\"id\":\"c1\",\"progress\":1}]}"}}]}}],"usage":{"total_tokens":50}}`))
	}))
	defer srv.Close()

	client := &GatewayJudgeClient{BaseURL: srv.URL}
	rubricBlock := FormatRubricForPrompt(rubricFixture())
	verdict, _, err := client.Judge(JudgeRequest{Objective: "obj", State: "state", MaxTokens: 128, Rubric: rubricBlock})
	if err != nil {
		t.Fatalf("judge: %v", err)
	}
	if len(verdict.Criteria) != 1 || verdict.Criteria[0].ID != "c1" {
		t.Errorf("verdict lost criteria attribution: %+v", verdict)
	}
	msgs := gotBody["messages"].([]any)
	system := msgs[0].(map[string]any)["content"].(string)
	user := msgs[1].(map[string]any)["content"].(string)
	if system != rubricJudgeSystemPrompt {
		t.Errorf("rubric request must use the rubric system prompt")
	}
	if !strings.Contains(user, "RUBRIC:\n"+rubricBlock) {
		t.Errorf("user message lost the rubric block: %q", user)
	}
	params, _ := json.Marshal(gotBody["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)["parameters"])
	if !strings.Contains(string(params), `"criteria"`) {
		t.Errorf("rubric request must advertise the attribution-required schema")
	}

	// Bare request: unchanged #2543 shape.
	if _, _, err := client.Judge(JudgeRequest{Objective: "obj", State: "state", MaxTokens: 128}); err != nil {
		t.Fatalf("bare judge: %v", err)
	}
	msgs = gotBody["messages"].([]any)
	if msgs[0].(map[string]any)["content"].(string) != judgeSystemPrompt {
		t.Errorf("bare request must keep the bare system prompt")
	}
	if s := msgs[1].(map[string]any)["content"].(string); strings.Contains(s, "RUBRIC:") {
		t.Errorf("bare request must carry no RUBRIC block: %q", s)
	}
}
