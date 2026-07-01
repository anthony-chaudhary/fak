package main

import (
	"bytes"
	"encoding/json"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/selfquery"
)

func TestFeatureQueryMemoryJSON(t *testing.T) {
	root := writeIndexRepo(t)
	var out, errb bytes.Buffer
	if rc := runFeature(&out, &errb, []string{"query", "--root", root, "memory", "--json"}); rc != 0 {
		t.Fatalf("runFeature query rc=%d stderr=%s", rc, errb.String())
	}
	var resp struct {
		Cards []struct {
			Name    string `json:"name"`
			Source  string `json:"source"`
			Request struct {
				MCPTool  string `json:"mcp_tool"`
				Executed bool   `json:"executed"`
			} `json:"request"`
		} `json:"cards"`
	}
	if err := json.Unmarshal(out.Bytes(), &resp); err != nil {
		t.Fatalf("feature query --json invalid: %v\n%s", err, out.String())
	}
	seen := map[string]bool{}
	for _, c := range resp.Cards {
		seen[c.Name] = true
		if strings.HasPrefix(c.Name, "fak_memory_") && c.Request.Executed {
			t.Fatalf("memory tool card executed during discovery: %+v", c)
		}
	}
	for _, want := range []string{"fak_memory_drivers", "fak_memory_explain", "fak_memory_run", "memory-driver:recall"} {
		if !seen[want] {
			t.Fatalf("memory feature query missing %s; got %v", want, seen)
		}
	}
}

func TestFeatureQueryCommitStamp(t *testing.T) {
	root := writeIndexRepo(t)
	var out, errb bytes.Buffer
	if rc := runFeature(&out, &errb, []string{"query", "--root", root, "--limit", "1", "commit", "stamp"}); rc != 0 {
		t.Fatalf("runFeature query rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "fak index lane") {
		t.Fatalf("commit stamp query should point at fak index lane, got:\n%s", out.String())
	}
}

func TestFeatureQueryDetailFaultsSelectedOnly(t *testing.T) {
	root := writeIndexRepo(t)
	var out, errb bytes.Buffer
	if rc := runFeature(&out, &errb, []string{"query", "--root", root, "memory", "--detail", "fak_memory_run", "--json"}); rc != 0 {
		t.Fatalf("runFeature query detail rc=%d stderr=%s", rc, errb.String())
	}
	var resp struct {
		Detail *struct {
			Card struct {
				Name string `json:"name"`
			} `json:"card"`
			Schema json.RawMessage `json:"schema"`
			Plan   json.RawMessage `json:"plan"`
		} `json:"detail"`
	}
	if err := json.Unmarshal(out.Bytes(), &resp); err != nil {
		t.Fatalf("feature query detail invalid: %v\n%s", err, out.String())
	}
	if resp.Detail == nil || resp.Detail.Card.Name != "fak_memory_run" || len(resp.Detail.Schema) == 0 {
		t.Fatalf("detail = %+v, want selected fak_memory_run schema", resp.Detail)
	}
	if len(resp.Detail.Plan) != 0 {
		t.Fatalf("tool schema detail should not fault a memory-driver plan: %s", string(resp.Detail.Plan))
	}
}

func TestFeatureQueryCLIUsesLiveDescriptorCatalog(t *testing.T) {
	root := writeIndexRepo(t)
	var out, errb bytes.Buffer
	if rc := runFeature(&out, &errb, []string{"query", "--root", root, "--plane", "live", "memory", "--json"}); rc != 0 {
		t.Fatalf("runFeature query rc=%d stderr=%s", rc, errb.String())
	}
	var cli selfquery.Response
	if err := json.Unmarshal(out.Bytes(), &cli); err != nil {
		t.Fatalf("feature query --json invalid: %v\n%s", err, out.String())
	}
	cat, err := selfquery.Load(root, selfquery.Options{
		Tools: selfquery.ToolDescriptorsFromMaps(gateway.ToolDescriptorsForResolver()),
	})
	if err != nil {
		t.Fatal(err)
	}
	direct, err := cat.Query(selfquery.Request{Query: "memory", Plane: selfquery.PlaneLive})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := featureSharedFacts(cli.Cards), featureSharedFacts(direct.Cards); !reflect.DeepEqual(got, want) {
		t.Fatalf("CLI feature query drifted from live descriptor catalog:\ngot  %v\nwant %v", got, want)
	}
}

func TestFeatureQueryMissingContextProducesClarification(t *testing.T) {
	root := writeIndexRepo(t)
	var out, errb bytes.Buffer
	if rc := runFeature(&out, &errb, []string{
		"query",
		"--root", root,
		"--missing-context", "deploy-target",
		"deploy",
		"--json",
	}); rc != 0 {
		t.Fatalf("runFeature query rc=%d stderr=%s", rc, errb.String())
	}
	var resp selfquery.Response
	if err := json.Unmarshal(out.Bytes(), &resp); err != nil {
		t.Fatalf("feature query --json invalid: %v\n%s", err, out.String())
	}
	if resp.Clarifications == nil {
		t.Fatalf("missing context did not produce clarifications:\n%s", out.String())
	}
	plan := resp.Clarifications
	if !plan.Bounded || len(plan.Questions) != 1 {
		t.Fatalf("clarification plan = %+v, want one bounded question", plan)
	}
	q := plan.Questions[0]
	if q.Key != "deploy-target" || q.Reason != selfquery.ClarificationMissingContext {
		t.Fatalf("clarification question = %+v, want missing deploy-target", q)
	}
	if q.DefaultChoice != "provide_value" || q.BudgetTokens <= 0 || len(q.Choices) != 3 {
		t.Fatalf("clarification question is not bounded/actionable: %+v", q)
	}
}

func featureSharedFacts(cards []selfquery.FeatureCard) []string {
	var out []string
	for _, c := range cards {
		if c.Source != "gateway.tools" && c.Source != "memq" {
			continue
		}
		out = append(out, string(c.Effect)+" "+c.Source+" "+c.Name)
	}
	sort.Strings(out)
	return out
}
