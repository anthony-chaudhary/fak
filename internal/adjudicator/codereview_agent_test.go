package adjudicator_test

import (
	"context"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// #3291 (epic #3256, workstream E): the flagship reference app for the governed
// runtime is a governed code-review agent. Its least-agency floor ships as
// examples/code-review-bot-policy.json — a real fak-policy/v1 manifest — but until
// now nothing bound that manifest to the adjudicator, so it was an orphan artifact:
// a policy no test proved actually governs anything.
//
// This is the gen/next foundation for that showcase: a hermetic contract that loads
// the SHIPPED policy through the SAME loader (policy.Parse) and engine (adjudicator)
// the runtime uses, and locks the reviewer's core governance loop — read-only work
// is ALLOWED, every dangerous repo mutation is DENIED live, and an unlisted tool
// fails closed. No model, no network: the adjudication a real run would show, proven
// deterministically. The captured LIVE run (audit tail + cost accounting) named in
// the issue's acceptance is the promotion step that moves this toward "now"; see the
// final report for what remains (a tool-capable `fak serve` host, #D1 quickstart,
// #D2 `fak init agent` template).

// codeReviewPolicy loads and parses the shipped governed-code-review floor exactly as
// the runtime would. Kept in one helper so every assertion reads the same manifest.
func codeReviewPolicy(t *testing.T) adjudicator.Policy {
	t.Helper()
	b, err := os.ReadFile("../../examples/code-review-bot-policy.json")
	if err != nil {
		t.Fatalf("read code-review policy: %v", err)
	}
	p, err := policy.Parse(b)
	if err != nil {
		t.Fatalf("parse code-review policy: %v", err)
	}
	return p
}

func codeReviewCall(tool, jsonArgs string) *abi.ToolCall {
	return &abi.ToolCall{
		Tool: tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(jsonArgs)},
		Meta: map[string]string{"readOnlyHint": "true"},
	}
}

// The reviewer's day job — reading the diff, the PR, the comments, and searching the
// tree — must be affirmatively ALLOWED, both by name and by the read-only allow_prefix
// set. If the floor ever stops permitting these, the agent can't do its work at all.
func TestCodeReviewAgentAllowsReadOnlyWork(t *testing.T) {
	a := adjudicator.New(codeReviewPolicy(t))
	ctx := context.Background()
	allowed := []struct{ tool, args string }{
		{"read_diff", `{"pr":42}`},             // explicit allow
		{"get_pull_request", `{"pr":42}`},      // explicit allow
		{"list_review_comments", `{"pr":42}`},  // explicit allow
		{"read_file", `{"path":"main.go"}`},    // explicit allow
		{"search_code", `{"q":"test"}`},        // allow_prefix search_
		{"find_symbol", `{"name":"New"}`},      // allow_prefix find_
		{"lookup_owner", `{"path":"main.go"}`}, // allow_prefix lookup_
	}
	for _, c := range allowed {
		v := a.Adjudicate(ctx, codeReviewCall(c.tool, c.args))
		if v.Kind != abi.VerdictAllow {
			t.Errorf("%s: got %v/%s, want Allow", c.tool, v.Kind, abi.ReasonName(v.Reason))
		}
	}
}

// The flagship witness the issue centers on: the governed reviewer proposes a
// dangerous repo mutation and the runtime DENIES it live with POLICY_BLOCK. A review
// bot reads and comments; it never merges, pushes, force-pushes, deletes branches,
// dispatches CI, ships releases, rotates creds, or drops to a shell. Every one of the
// manifest's deny entries is locked here — drop or weaken one in the policy and this
// fails by the tool's name.
func TestCodeReviewAgentDeniesDangerousActions(t *testing.T) {
	a := adjudicator.New(codeReviewPolicy(t))
	ctx := context.Background()
	dangerous := []string{
		"merge_pull_request",
		"git_push",
		"force_push",
		"delete_branch",
		"workflow_dispatch",
		"rerun_workflow",
		"publish_release",
		"rotate_credentials",
		"shell",
	}
	for _, tool := range dangerous {
		v := a.Adjudicate(ctx, codeReviewCall(tool, `{}`))
		if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
			t.Errorf("%s: got %v/%s, want Deny/POLICY_BLOCK", tool, v.Kind, abi.ReasonName(v.Reason))
		}
	}
}

// fail_closed is the whole point of a least-agency floor: a tool the manifest never
// named is refused with DEFAULT_DENY, not quietly allowed. This is what makes the
// allow-list trustworthy — capability the operator never granted does not leak in.
func TestCodeReviewAgentFailsClosedOnUnlistedTool(t *testing.T) {
	a := adjudicator.New(codeReviewPolicy(t))
	ctx := context.Background()
	for _, tool := range []string{"wire_money", "exfiltrate_secrets", "send_email"} {
		v := a.Adjudicate(ctx, codeReviewCall(tool, `{}`))
		if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonDefaultDeny {
			t.Errorf("%s: got %v/%s, want Deny/DEFAULT_DENY", tool, v.Kind, abi.ReasonName(v.Reason))
		}
	}
}
