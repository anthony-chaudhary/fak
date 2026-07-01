package selfquery

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/capindex"
	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/memq"
)

func writeRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"dos.toml": `[lanes.trees]
gateway = ["internal/gateway/**"] # gateway surface
cmd = ["cmd/**"] # command shells
docs = ["docs/**"] # docs
`,
		"INDEX.md": `# INDEX
- [Gateway guide](docs/gateway.md) - MCP and gateway docs.
`,
		"docs/gateway.md": "# Gateway guide\nUse the MCP gateway.\n",
		"CLAIMS.md": `# CLAIMS
## Gateway
- [SHIPPED] internal/gateway exposes MCP tools.
`,
		"cmd/fak/main.go": `package main
func main() {
	switch os.Args[1] {
	case "index":
		cmdIndex(os.Args[2:])
	case "feature":
		cmdFeature(os.Args[2:])
	default:
	}
}
`,
	}
	for path, body := range files {
		full := filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func testTools() []ToolDescriptor {
	return []ToolDescriptor{
		{Name: "fak_memory_drivers", Description: "List memory drivers.", InputSchema: json.RawMessage(`{"type":"object"}`)},
		{Name: "fak_memory_explain", Description: "Explain a memory query.", InputSchema: json.RawMessage(`{"type":"object","properties":{"driver":{"type":"string"}}}`)},
		{Name: "fak_memory_run", Description: "Run a memory query.", InputSchema: json.RawMessage(`{"type":"object","properties":{"driver":{"type":"string"},"apply":{"type":"boolean"}}}`)},
		{Name: "deny_delete", Description: "Synthetic ordinary tool for guarded request tests.", InputSchema: json.RawMessage(`{"type":"object"}`)},
	}
}

func TestQueryMemoryReturnsToolsAndDrivers(t *testing.T) {
	cat, err := Load(writeRepo(t), Options{Tools: testTools()})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := cat.Query(Request{Query: "memory", Plane: PlaneAll})
	if err != nil {
		t.Fatal(err)
	}
	names := namesOf(resp.Cards)
	for _, want := range []string{"fak_memory_drivers", "fak_memory_explain", "fak_memory_run", "memory-driver:recall", "memory-driver:clean"} {
		if !names[want] {
			t.Fatalf("memory query missing %s; got %v", want, sortedNames(resp.Cards))
		}
	}
	for _, c := range resp.Cards {
		if c.Name == "memory-driver:clean" {
			if c.Effect != EffectPropose || c.RequiresCap == "" {
				t.Fatalf("clean card = %+v, want proposal-only mutation signal", c)
			}
		}
		if c.Name == "fak_memory_run" && c.Request.Executed {
			t.Fatalf("feature query must not execute fak_memory_run: %+v", c.Request)
		}
	}
}

func TestEveryRegisteredMemoryDriverHasFeatureCard(t *testing.T) {
	cat, err := Load(writeRepo(t), Options{Tools: testTools()})
	if err != nil {
		t.Fatal(err)
	}
	names := namesOf(cat.Cards(PlaneLive))
	for _, d := range memq.Drivers() {
		want := "memory-driver:" + d.Name
		if !names[want] {
			t.Fatalf("registered memq driver %q missing from self-feature catalog", d.Name)
		}
	}
}

func TestDynamicallyRegisteredMemoryDriverHasFeatureCard(t *testing.T) {
	const name = "selfquery-dynamic-witness"
	memq.Register(memq.Driver{
		Name: name,
		Doc:  "dynamic driver registered by a freshness witness",
		Build: func(p memq.Params) memq.Query {
			return memq.Query{
				Intent: p.Intent,
				Ops: []memq.Op{
					{Kind: memq.OpScan},
					{Kind: memq.OpRender},
				},
			}
		},
	})

	cat, err := Load(writeRepo(t), Options{Tools: testTools()})
	if err != nil {
		t.Fatal(err)
	}
	names := namesOf(cat.Cards(PlaneLive))
	if !names["memory-driver:"+name] {
		t.Fatalf("dynamically registered memq driver %q missing from self-feature catalog", name)
	}
}

func TestEveryIndexedVerbHasFeatureCard(t *testing.T) {
	root := writeRepo(t)
	cat, err := Load(root, Options{Tools: testTools()})
	if err != nil {
		t.Fatal(err)
	}
	names := namesOf(cat.Cards(PlaneDev))
	for _, v := range cat.dev.Verbs() {
		want := "fak " + v.Name
		if !names[want] {
			t.Fatalf("indexed fak verb %q missing from self-feature catalog", v.Name)
		}
	}
}

func TestCapindexCardsAreLoweredWhenProvided(t *testing.T) {
	capCard := capindex.CapCard{
		Ref:       capindex.CapRef{Kind: capindex.CapKindSkill, Name: "memory-helper", Version: "v1"},
		Trigger:   "helps inspect memory plans",
		Tags:      []string{"memory", "skill"},
		CardBytes: []byte(`{"name":"memory-helper"}`),
	}
	cat, err := Load(writeRepo(t), Options{Tools: testTools(), CapCards: []capindex.CapCard{capCard}})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := cat.Query(Request{Query: "memory helper", Plane: PlaneLive, Detail: "skill:memory-helper@v1"})
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, c := range resp.Cards {
		if c.Name == "skill:memory-helper" && c.Source == "capindex" {
			found = true
		}
	}
	if !found {
		t.Fatalf("capindex card missing from query results: %v", sortedNames(resp.Cards))
	}
	if resp.Detail == nil || resp.Detail.CardBytes == "" {
		t.Fatalf("capindex detail did not fault card bytes: %+v", resp.Detail)
	}
}

func TestRootSkillCapabilitiesAreLoadedThroughCapindex(t *testing.T) {
	root := writeRepo(t)
	skillPath := filepath.Join(root, ".claude", "skills", "memory-helper", "SKILL.md")
	if err := os.MkdirAll(filepath.Dir(skillPath), 0o755); err != nil {
		t.Fatal(err)
	}
	body := `---
name: memory-helper
version: v1
description: helps inspect memory plans
tags: [memory, skill]
---

# Memory Helper

Faulted capability body.
`
	if err := os.WriteFile(skillPath, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	cat, err := Load(root, Options{Tools: testTools()})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := cat.Query(Request{Query: "memory helper", Plane: PlaneLive, Detail: "skill:memory-helper@v1"})
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, c := range resp.Cards {
		if c.Name == "skill:memory-helper" && c.Source == "capindex" {
			found = true
			if c.Witness == "" || c.DetailRef != "skill:memory-helper@v1" {
				t.Fatalf("capindex skill card missing witness/detail ref: %+v", c)
			}
		}
	}
	if !found {
		t.Fatalf("root .claude skill capability missing from query results: %v", sortedNames(resp.Cards))
	}
	if resp.Detail == nil || !strings.Contains(resp.Detail.CardBytes, "Faulted capability body.") {
		t.Fatalf("capindex skill detail did not fault SKILL.md body: %+v", resp.Detail)
	}
}

func TestStableOrderingAndSourceAttribution(t *testing.T) {
	capsA := []capindex.CapCard{
		{Ref: capindex.CapRef{Kind: capindex.CapKindSkill, Name: "zeta"}, Trigger: "later skill", Tags: []string{"skill"}},
		{Ref: capindex.CapRef{Kind: capindex.CapKindSkill, Name: "alpha"}, Trigger: "earlier skill", Tags: []string{"skill"}},
	}
	capsB := []capindex.CapCard{capsA[1], capsA[0]}
	toolsA := []ToolDescriptor{
		{Name: "zz_tool", Description: "last test tool"},
		{Name: "aa_tool", Description: "first test tool"},
	}
	toolsB := []ToolDescriptor{toolsA[1], toolsA[0]}

	catA, err := Load(writeRepo(t), Options{Tools: toolsA, CapCards: capsA})
	if err != nil {
		t.Fatal(err)
	}
	catB, err := Load(writeRepo(t), Options{Tools: toolsB, CapCards: capsB})
	if err != nil {
		t.Fatal(err)
	}
	namesA := sortedNames(catA.Cards(PlaneAll))
	namesB := sortedNames(catB.Cards(PlaneAll))
	if !reflect.DeepEqual(namesA, namesB) {
		t.Fatalf("card ordering drifted with input order:\nA=%v\nB=%v", namesA, namesB)
	}

	sources := map[string]string{}
	for _, c := range catA.Cards(PlaneAll) {
		sources[c.Name] = c.Source
	}
	for name, want := range map[string]string{
		"fak index lane":       "devindex",
		"memory-driver:recall": "memq",
		"aa_tool":              "gateway.tools",
		"skill:alpha":          "capindex",
	} {
		if sources[name] != want {
			t.Fatalf("%s source = %q, want %q", name, sources[name], want)
		}
	}
}

func TestAssumptionConfidenceFeatureCard(t *testing.T) {
	cat, err := Load(writeRepo(t), Options{Tools: testTools()})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := cat.Query(Request{Query: "assumption confidence stale unknown witnessed", Plane: PlaneLive})
	if err != nil {
		t.Fatal(err)
	}
	var found *FeatureCard
	for i := range resp.Cards {
		if resp.Cards[i].Name == "context-plan:assumptions" {
			found = &resp.Cards[i]
			break
		}
	}
	if found == nil {
		t.Fatalf("assumption confidence query missing context-plan card: %v", sortedNames(resp.Cards))
	}
	if found.Source != "ctxplan" || found.DetailRef != "internal/ctxplan/assumption.go" {
		t.Fatalf("assumption card source/ref = %s %s, want ctxplan assumption source", found.Source, found.DetailRef)
	}
	if found.Effect != EffectRead || found.Request.Executed {
		t.Fatalf("assumption card request = effect %s request %+v, want read-only discovery", found.Effect, found.Request)
	}
	tags := map[string]bool{}
	for _, tag := range found.Tags {
		tags[tag] = true
	}
	for _, want := range []string{"user_stated", "witnessed", "inferred", "stale", "unknown", "confidence"} {
		if !tags[want] {
			t.Fatalf("assumption card tags missing %q: %+v", want, found.Tags)
		}
	}
}

// TestAskPolicyFeatureCard pins that #1580's general ask-vs-assume policy is
// discoverable through the same feature catalog as every other selfquery surface —
// the "grounded in a real call site" half of the issue: a caller (or `fak feature
// query`) can find selfquery.ShouldAsk without already knowing the file exists.
func TestAskPolicyFeatureCard(t *testing.T) {
	cat, err := Load(writeRepo(t), Options{Tools: testTools()})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := cat.Query(Request{Query: "ask policy stakes reversibility confidence", Plane: PlaneLive})
	if err != nil {
		t.Fatal(err)
	}
	var found *FeatureCard
	for i := range resp.Cards {
		if resp.Cards[i].Name == "ask-policy:should-ask" {
			found = &resp.Cards[i]
			break
		}
	}
	if found == nil {
		t.Fatalf("ask policy query missing ask-policy card: %v", sortedNames(resp.Cards))
	}
	if found.Source != "selfquery" || found.DetailRef != "internal/selfquery/ask_policy.go" {
		t.Fatalf("ask policy card source/ref = %s %s, want selfquery ask_policy.go", found.Source, found.DetailRef)
	}
	if found.Effect != EffectRead || found.Request.Executed {
		t.Fatalf("ask policy card request = effect %s request %+v, want read-only discovery", found.Effect, found.Request)
	}
	tags := map[string]bool{}
	for _, tag := range found.Tags {
		tags[tag] = true
	}
	for _, want := range []string{"ask", "assume", "stakes", "reversibility", "confidence", "policy"} {
		if !tags[want] {
			t.Fatalf("ask policy card tags missing %q: %+v", want, found.Tags)
		}
	}
}

func TestQueryClarificationBrokerTurnsMissingContextIntoBoundedQuestion(t *testing.T) {
	cat, err := Load(writeRepo(t), Options{Tools: testTools()})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := cat.Query(Request{
		Query:          "deploy",
		Plane:          PlaneLive,
		MissingContext: []string{"deploy-target", "approval-ticket"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Clarifications == nil {
		t.Fatal("missing context should produce a clarification plan")
	}
	plan := resp.Clarifications
	if !plan.Bounded || plan.MaxQuestions != 3 || plan.MaxBudgetTokens != 120 {
		t.Fatalf("clarification plan bounds = %+v", plan)
	}
	if len(plan.Questions) != 2 {
		t.Fatalf("questions=%+v, want one question per missing key", plan.Questions)
	}
	q := plan.Questions[0]
	if q.Key != "approval-ticket" || q.Reason != ClarificationMissingContext {
		t.Fatalf("first clarification = %+v, want stable missing-context question", q)
	}
	if q.DefaultChoice != "provide_value" || q.BudgetTokens <= 0 || q.BudgetTokens > plan.MaxBudgetTokens {
		t.Fatalf("clarification budget/default = %+v within plan %+v", q, plan)
	}
	if len(q.Choices) != 3 {
		t.Fatalf("clarification choices = %+v, want bounded option set", q.Choices)
	}
}

func TestPlanClarificationsBoundsQuestionsAndBudget(t *testing.T) {
	report := ctxplan.AssessAssumptions([]ctxplan.Assumption{
		{Key: "a", Source: ctxplan.AssumptionUnknown},
		{Key: "b", Source: ctxplan.AssumptionUnknown},
		{Key: "c", Source: ctxplan.AssumptionUnknown},
		{Key: "d", Source: ctxplan.AssumptionUnknown},
	}, ctxplan.DefaultAssumptionPolicy())
	bounded := PlanClarifications(report, ClarificationOptions{
		MaxQuestions:    1,
		MaxBudgetTokens: 40,
	})
	if len(bounded.Questions) != 1 || bounded.Omitted != 3 {
		t.Fatalf("bounded plan = %+v, want one included and three omitted", bounded)
	}
}

func TestStaleClarificationDefaultsToRefreshSource(t *testing.T) {
	report := ctxplan.AssessAssumptions([]ctxplan.Assumption{
		{Key: "account-tier", Source: ctxplan.AssumptionStale, Statement: "gold in recalled page", SourceRef: "recall:page:4:trust_epoch"},
	}, ctxplan.DefaultAssumptionPolicy())
	plan := PlanClarifications(report, ClarificationOptions{})
	if len(plan.Questions) != 1 {
		t.Fatalf("stale assumption should produce one clarification question: %+v", plan)
	}
	q := plan.Questions[0]
	if q.Reason != ClarificationStaleContext || q.DefaultChoice != "refresh_source" {
		t.Fatalf("stale clarification = %+v, want refresh_source default", q)
	}
	if q.SourceRef != "recall:page:4:trust_epoch" || q.BudgetTokens <= 0 {
		t.Fatalf("stale clarification should carry source ref and budget: %+v", q)
	}
}

func TestDetailFaultsOnlySelectedSchemaOrPlan(t *testing.T) {
	cat, err := Load(writeRepo(t), Options{Tools: testTools()})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := cat.Query(Request{Query: "memory", Detail: "fak_memory_run"})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Detail == nil || len(resp.Detail.Schema) == 0 {
		t.Fatalf("detail did not fault selected tool schema: %+v", resp.Detail)
	}
	for _, c := range resp.Cards {
		if c.Name == "fak_memory_explain" && len(resp.Detail.Schema) > 0 && resp.Detail.Card.Name != "fak_memory_run" {
			t.Fatal("unexpected detail card")
		}
	}
	resp, err = cat.Query(Request{Query: "memory", Detail: "memory-driver:compact"})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Detail == nil || resp.Detail.Plan == nil || len(resp.Detail.Plan.Mutations) == 0 {
		t.Fatalf("memory-driver detail did not fault explain plan: %+v", resp.Detail)
	}
}

func TestLightweightQueryDoesNotInlineSchemasOrPlans(t *testing.T) {
	cat, err := Load(writeRepo(t), Options{Tools: testTools()})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := cat.Query(Request{Query: "memory"})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Detail != nil {
		t.Fatalf("lightweight query faulted detail: %+v", resp.Detail)
	}
	b, err := json.Marshal(resp)
	if err != nil {
		t.Fatal(err)
	}
	if hasJSONKey(b, "schema") || hasJSONKey(b, "plan") || hasJSONKey(b, "inputSchema") {
		t.Fatalf("lightweight response inlined detail/schema bytes: %s", string(b))
	}
}

func TestMemoryMutationDetailExplainsWithoutExecution(t *testing.T) {
	cat, err := Load(writeRepo(t), Options{Tools: testTools()})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := cat.Query(Request{Query: "memory", Detail: "memory-driver:dream"})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Detail == nil || resp.Detail.Plan == nil || len(resp.Detail.Plan.Mutations) == 0 {
		t.Fatalf("dream detail = %+v, want explain plan with proposed mutations", resp.Detail)
	}
	if resp.Detail.Card.Effect != EffectPropose || resp.Detail.Card.Request.Executed {
		t.Fatalf("dream request = card %+v request %+v, want proposal-only without execution", resp.Detail.Card, resp.Detail.Card.Request)
	}
}

func TestQueryIndexDocsUsesDevIndexSource(t *testing.T) {
	cat, err := Load(writeRepo(t), Options{Tools: testTools()})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := cat.Query(Request{Query: "index docs", Plane: PlaneDev})
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, c := range resp.Cards {
		if c.Kind == "dev-doc" && c.Source == "devindex" {
			found = true
		}
	}
	if !found {
		t.Fatalf("index docs query did not return devindex doc cards: %v", sortedNames(resp.Cards))
	}
}

func TestQueryCommitStampPointsAtIndexLane(t *testing.T) {
	cat, err := Load(writeRepo(t), Options{Tools: testTools()})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := cat.Query(Request{Query: "commit stamp", Plane: PlaneDev, Limit: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Cards) != 1 || resp.Cards[0].Name != "fak index lane" {
		t.Fatalf("commit stamp top card = %v, want fak index lane", sortedNames(resp.Cards))
	}
}

func TestEmptyQueryFailsClosed(t *testing.T) {
	cat, err := Load(writeRepo(t), Options{Tools: testTools()})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := cat.Query(Request{Query: "   "}); err == nil {
		t.Fatal("empty feature query returned a menu; want fail-closed error")
	}
}

func TestOrdinaryToolCardReturnsAdjudicationRequestShape(t *testing.T) {
	cat, err := Load(writeRepo(t), Options{Tools: testTools()})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := cat.Query(Request{Query: "deny delete", Detail: "deny_delete"})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Detail == nil {
		t.Fatal("missing detail")
	}
	req := resp.Detail.Card.Request
	if req.Route != "adjudicate-before-execute" || req.MCPTool != "fak_adjudicate" || req.Executed {
		t.Fatalf("ordinary tool request = %+v, want adjudication shape without execution", req)
	}
}

func namesOf(cards []FeatureCard) map[string]bool {
	out := map[string]bool{}
	for _, c := range cards {
		out[c.Name] = true
	}
	return out
}

func sortedNames(cards []FeatureCard) []string {
	out := make([]string, len(cards))
	for i, c := range cards {
		out[i] = c.Name
	}
	return out
}

func hasJSONKey(b []byte, key string) bool {
	var v any
	if err := json.Unmarshal(b, &v); err != nil {
		return false
	}
	return hasKey(v, key)
}

func hasKey(v any, key string) bool {
	switch x := v.(type) {
	case map[string]any:
		for k, child := range x {
			if k == key || hasKey(child, key) {
				return true
			}
		}
	case []any:
		for _, child := range x {
			if hasKey(child, key) {
				return true
			}
		}
	}
	return false
}
