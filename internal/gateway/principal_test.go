package gateway

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestPeerConfirmRefused is the #2412 acceptance witness: a `_fak_confirm` arriving
// via a peer/A2A-labeled message is refused with a closed reason and the original
// REQUIRE_WITNESS stands, while a human-labeled confirm on the SAME token passes.
// It composes the real gateway principal gate with the real adjudicator reversibility
// rung so the "who consented" type-check is proven end to end, not stubbed.
func TestPeerConfirmRefused(t *testing.T) {
	ctx := context.Background()
	a := adjudicator.New(adjudicator.Policy{Allow: map[string]bool{"Bash": true}})

	// The confirm token the reversibility rung minted for an irreversible call.
	env := adjudicator.ClassifyReversibility("Bash", map[string]any{"command": "rm -rf build"})
	if env.ConfirmToken == "" {
		t.Fatalf("reversibility envelope minted no confirm token: %+v", env)
	}
	confirmedArgs := `{"command":"rm -rf build","_fak_confirm":"` + env.ConfirmToken + `"}`
	call := agent.ToolCall{ID: "c1", Type: "function", Function: agent.Func{Name: "Bash", Arguments: confirmedArgs}}

	adjudicate := func(args string) abi.Verdict {
		return a.Adjudicate(ctx, &abi.ToolCall{
			Tool: "Bash",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(args)},
			Meta: map[string]string{"readOnlyHint": "true"},
		})
	}

	// Sanity: the same token, presented by the HUMAN, is a real approval — the gate is
	// a no-op passthrough and the reversibility rung consumes the token and TRANSFORMs
	// (strips it, dispatches).
	humanCalls, humanRefusals := gateInboundAuthority(PrincipalHuman, []agent.ToolCall{call})
	if len(humanRefusals) != 0 {
		t.Fatalf("human confirm drew a refusal: %+v", humanRefusals)
	}
	if humanCalls[0].Function.Arguments != confirmedArgs {
		t.Fatalf("human gate mutated the confirm: %q", humanCalls[0].Function.Arguments)
	}
	if v := adjudicate(humanCalls[0].Function.Arguments); v.Kind != abi.VerdictTransform || v.Meta["reversibility_confirmed"] != "true" {
		t.Fatalf("human-confirmed call: got %v/%v, want Transform w/ reversibility_confirmed", v.Kind, v.Meta)
	}

	// The load-bearing case: the SAME token arriving under a peer/A2A principal is
	// refused with the closed reason, the confirm is stripped, and the original
	// REQUIRE_WITNESS hold stands.
	peerCalls, peerRefusals := gateInboundAuthority(PrincipalPeerAgent, []agent.ToolCall{call})
	if len(peerRefusals) != 1 || peerRefusals[0].Reason != ReasonPrincipalNotHuman || peerRefusals[0].Principal != PrincipalPeerAgent {
		t.Fatalf("peer confirm refusal = %+v, want one PRINCIPAL_NOT_HUMAN under peer-agent", peerRefusals)
	}
	if strings.Contains(peerCalls[0].Function.Arguments, "_fak_confirm") {
		t.Fatalf("peer gate left the confirm token in args: %q", peerCalls[0].Function.Arguments)
	}
	// The command itself is preserved — only the approval was stripped.
	var got map[string]any
	if err := json.Unmarshal([]byte(peerCalls[0].Function.Arguments), &got); err != nil {
		t.Fatalf("peer-gated args are not JSON: %q: %v", peerCalls[0].Function.Arguments, err)
	}
	if got["command"] != "rm -rf build" {
		t.Fatalf("peer gate changed the command: %v", got)
	}
	if v := adjudicate(peerCalls[0].Function.Arguments); v.Kind != abi.VerdictRequireWitness {
		t.Fatalf("peer-stripped call: got %v/%s, want REQUIRE_WITNESS to still stand", v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestClassifyPrincipal_FailsClosed pins the decode-time classification: the direct
// interactive wire (no header) is human, recognized relay classes and their aliases
// map through, and an unrecognized non-empty label fails closed to unknown — an
// ambiguous source is NEVER defaulted to human.
func TestClassifyPrincipal_FailsClosed(t *testing.T) {
	cases := map[string]Principal{
		"":             PrincipalHuman,
		"human":        PrincipalHuman,
		"HUMAN":        PrincipalHuman,
		"self-model":   PrincipalSelfModel,
		"peer-agent":   PrincipalPeerAgent,
		"peer":         PrincipalPeerAgent,
		"a2a":          PrincipalPeerAgent,
		"timer":        PrincipalTimer,
		"cron":         PrincipalTimer,
		"network-tool": PrincipalNetworkTool,
		"webhook":      PrincipalNetworkTool,
		"whoknows":     PrincipalUnknown,
	}
	for raw, want := range cases {
		if got := classifyPrincipal(raw); got != want {
			t.Errorf("classifyPrincipal(%q) = %q, want %q", raw, got, want)
		}
		if want != PrincipalHuman && want.IsHuman() {
			t.Errorf("non-human label %q reported IsHuman", want)
		}
	}
}

// TestGateInboundAuthority_HumanIsNoOp guards the common path: a human turn's calls
// pass through untouched (same backing slice header semantics, no allocation of a
// refusal), so the floor costs nothing on the direct interactive wire.
func TestGateInboundAuthority_HumanIsNoOp(t *testing.T) {
	calls := []agent.ToolCall{{Function: agent.Func{Name: "Bash", Arguments: `{"command":"ls","_fak_confirm":"tok"}`}}}
	got, refusals := gateInboundAuthority(PrincipalHuman, calls)
	if refusals != nil {
		t.Fatalf("human refusals = %+v, want none", refusals)
	}
	if got[0].Function.Arguments != calls[0].Function.Arguments {
		t.Fatalf("human path mutated args: %q", got[0].Function.Arguments)
	}
}
