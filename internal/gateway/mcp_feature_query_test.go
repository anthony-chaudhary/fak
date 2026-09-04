package gateway

import (
	"context"
	"encoding/json"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/selfquery"
)

func TestMCPCapabilitiesRanksWIPAdministrationFirst(t *testing.T) {
	srv := newTestServer(t)
	for _, query := range []string{"WIP", "worktree", "unlanded work"} {
		t.Run(query, func(t *testing.T) {
			resp := callMCPTool[selfquery.CapabilitiesResponse](t, srv, "fak_capabilities", map[string]any{
				"query": query,
				"limit": 3,
			})
			if len(resp.Cards) == 0 {
				t.Fatalf("fak_capabilities query %q returned no cards", query)
			}
			got := resp.Cards[0]
			if got.Name != "Administer worktrees and unlanded WIP" {
				t.Fatalf("fak_capabilities query %q ranked %q first; want WIP administration", query, got.Name)
			}
			wantCommand := []string{"fak", "wip", "queue", "--json"}
			if !reflect.DeepEqual(got.Request.Command, wantCommand) {
				t.Fatalf("WIP administration command = %#v, want %#v", got.Request.Command, wantCommand)
			}
			if got.Request.Executed {
				t.Fatal("fak_capabilities must return an unexecuted exact command")
			}
			for _, want := range []string{"fak wip inventory --json", "fak worktree worker list --json", "fak-wip-action-queue/1"} {
				if !strings.Contains(got.DetailRef, want) {
					t.Errorf("WIP administration detail %q missing %q", got.DetailRef, want)
				}
			}
		})
	}
}

func TestMCPFeatureQueryMemoryCards(t *testing.T) {
	srv := newTestServer(t)
	root := writeMCPIndexRepo(t)

	resp := callMCPTool[selfquery.Response](t, srv, "fak_feature_query", map[string]any{
		"root":  root,
		"query": "memory",
		"plane": "all",
		"limit": 10,
	})
	names := map[string]bool{}
	for _, c := range resp.Cards {
		names[c.Name] = true
		if c.Name == "fak_memory_run" {
			if c.Effect != selfquery.EffectPropose || c.Request.Executed {
				t.Fatalf("fak_memory_run card = %+v, want proposal request without execution", c)
			}
		}
	}
	for _, want := range []string{"fak_memory_drivers", "fak_memory_explain", "fak_memory_run", "memory-driver:recall"} {
		if !names[want] {
			t.Fatalf("fak_feature_query memory missing %s; got %v", want, names)
		}
	}
}

func TestMCPFeatureQueryDetailFaultsOneSchema(t *testing.T) {
	srv := newTestServer(t)
	root := writeMCPIndexRepo(t)

	resp := callMCPTool[selfquery.Response](t, srv, "fak_feature_query", map[string]any{
		"root":   root,
		"query":  "memory",
		"detail": "fak_memory_run",
	})
	if resp.Detail == nil || resp.Detail.Card.Name != "fak_memory_run" || len(resp.Detail.Schema) == 0 {
		t.Fatalf("detail = %+v, want selected fak_memory_run schema", resp.Detail)
	}
	if resp.Detail.Plan != nil {
		t.Fatalf("tool schema detail faulted an unrelated memory plan: %+v", resp.Detail.Plan)
	}
}

func TestMCPFeatureQueryMatchesCatalogFromLiveDescriptors(t *testing.T) {
	srv := newTestServer(t)
	root := writeMCPIndexRepo(t)

	live := callMCPTool[selfquery.Response](t, srv, "fak_feature_query", map[string]any{
		"root":  root,
		"query": "memory",
		"plane": "live",
	})
	cat, err := selfquery.Load(root, selfquery.Options{
		Tools: selfquery.ToolDescriptorsFromMaps(toolDescriptors()),
	})
	if err != nil {
		t.Fatal(err)
	}
	direct, err := cat.Query(selfquery.Request{Query: "memory", Plane: selfquery.PlaneLive})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := sharedCardFacts(live.Cards), sharedCardFacts(direct.Cards); !reflect.DeepEqual(got, want) {
		t.Fatalf("MCP feature query drifted from live descriptor catalog:\ngot  %v\nwant %v", got, want)
	}
}

func TestMCPFeatureQueryRequiresNonEmptyQuery(t *testing.T) {
	srv := newTestServer(t)
	params, _ := json.Marshal(map[string]any{
		"name":      "fak_feature_query",
		"arguments": map[string]any{"query": "   "},
	})
	if _, rerr := srv.callTool(context.Background(), params); rerr == nil || rerr.Code != rpcInvalidParams {
		t.Fatalf("empty fak_feature_query should be InvalidParams, got %+v", rerr)
	}
}

func TestEveryGatewayToolHasFeatureCard(t *testing.T) {
	root := writeMCPIndexRepo(t)
	cat, err := selfquery.Load(root, selfquery.Options{
		Tools: selfquery.ToolDescriptorsFromMaps(toolDescriptors()),
	})
	if err != nil {
		t.Fatal(err)
	}
	names := map[string]bool{}
	for _, c := range cat.Cards(selfquery.PlaneLive) {
		names[c.Name] = true
	}
	for _, td := range toolDescriptors() {
		name, _ := td["name"].(string)
		if name != "" && !names[name] {
			t.Fatalf("gateway tool %q missing from self-feature catalog", name)
		}
	}
}

func TestFeatureQueryDoesNotBypassDeniedAdjudication(t *testing.T) {
	srv := newTestServer(t)
	root := writeMCPIndexRepo(t)
	resp := callMCPTool[selfquery.Response](t, srv, "fak_feature_query", map[string]any{
		"root":   root,
		"query":  "syscall",
		"detail": "fak_syscall",
	})
	if resp.Detail == nil || resp.Detail.Card.Name != "fak_syscall" {
		t.Fatalf("expected fak_syscall detail, got %+v", resp.Detail)
	}

	adj := callMCPTool[SyscallResponse](t, srv, "fak_adjudicate", map[string]any{
		"tool":      "deny_delete",
		"arguments": map[string]any{},
	})
	if adj.Verdict.Kind != "DENY" || adj.Verdict.Reason != "POLICY_BLOCK" {
		t.Fatalf("denied tool after feature discovery = %+v, want POLICY_BLOCK DENY", adj.Verdict)
	}
}

func TestSelectedOrdinaryToolRequestShapeRemainsPolicyDenied(t *testing.T) {
	srv := newTestServer(t)
	root := writeMCPIndexRepo(t)
	cat, err := selfquery.Load(root, selfquery.Options{
		Tools: []selfquery.ToolDescriptor{{
			Name:        "deny_delete",
			Description: "ordinary external delete tool",
			InputSchema: json.RawMessage(`{"type":"object"}`),
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := cat.Query(selfquery.Request{Query: "delete", Plane: selfquery.PlaneLive, Detail: "deny_delete"})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Detail == nil || resp.Detail.Card.Request.MCPTool != "fak_adjudicate" || resp.Detail.Card.Request.Executed {
		t.Fatalf("ordinary tool card did not produce adjudication request shape: %+v", resp.Detail)
	}
	tool, _ := resp.Detail.Card.Request.Arguments["tool"].(string)
	args, _ := resp.Detail.Card.Request.Arguments["arguments"].(map[string]any)
	adj := callMCPTool[SyscallResponse](t, srv, "fak_adjudicate", map[string]any{
		"tool":      tool,
		"arguments": args,
	})
	if adj.Verdict.Kind != "DENY" || adj.Verdict.Reason != "POLICY_BLOCK" {
		t.Fatalf("selected ordinary tool adjudication = %+v, want POLICY_BLOCK DENY", adj.Verdict)
	}
}

func sharedCardFacts(cards []selfquery.FeatureCard) []string {
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
