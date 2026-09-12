package adjudicator

import (
	"bytes"
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestOpenCodeLowercaseToolsUseDefaultPolicyWithoutWeakeningWriteFloor(t *testing.T) {
	a := New(DefaultPolicy())
	ctx := context.Background()

	for _, tc := range []struct {
		name string
		tool string
		args string
	}{
		{name: "bash go test", tool: "bash", args: `{"command":"go test ./internal/adjudicator"}`},
		{name: "edit ordinary file", tool: "edit", args: `{"filePath":"notes.txt","oldString":"before","newString":"after"}`},
		{name: "write ordinary file", tool: "write", args: `{"filePath":"notes.txt","content":"after"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			call := inlineCall(tc.tool, tc.args)
			call.Meta = nil
			before := append([]byte(nil), call.Args.Inline...)
			verdict := a.Adjudicate(ctx, call)
			if verdict.Kind != abi.VerdictAllow {
				t.Fatalf("%s: verdict=%v/%s, want allow", tc.tool, verdict.Kind, abi.ReasonName(verdict.Reason))
			}
			if !bytes.Equal(call.Args.Inline, before) {
				t.Fatalf("%s: args changed from %q to %q", tc.tool, before, call.Args.Inline)
			}
		})
	}

	for _, tc := range []struct {
		name string
		tool string
		args string
	}{
		{name: "edit protected linux path", tool: "edit", args: `{"filePath":"internal/adjudicator/decide.go","oldString":"before","newString":"after"}`},
		{name: "write protected windows path", tool: "write", args: `{"filePath":"internal\\adjudicator\\decide.go","content":"after"}`},
		{name: "bash protected linux path", tool: "bash", args: `{"command":"printf after > internal/adjudicator/decide.go"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			call := inlineCall(tc.tool, tc.args)
			call.Meta = nil
			verdict := a.Adjudicate(ctx, call)
			if verdict.Kind != abi.VerdictDeny || verdict.Reason != abi.ReasonSelfModify {
				t.Fatalf("%s: verdict=%v/%s, want deny/SELF_MODIFY", tc.tool, verdict.Kind, abi.ReasonName(verdict.Reason))
			}
		})
	}

	destructive := inlineCall("bash", `{"command":"rm -rf internal/adjudicator"}`)
	destructive.Meta = nil
	if verdict := a.Adjudicate(ctx, destructive); verdict.Kind != abi.VerdictDeny {
		t.Fatalf("destructive bash self-modification verdict=%v/%s, want deny", verdict.Kind, abi.ReasonName(verdict.Reason))
	}

	unknown := inlineCall("write_text_file", `{"filePath":"notes.txt","content":"after"}`)
	unknown.Meta = nil
	if verdict := a.Adjudicate(ctx, unknown); verdict.Kind != abi.VerdictDeny || verdict.Reason != abi.ReasonDefaultDeny {
		t.Fatalf("unknown write-shaped tool verdict=%v/%s, want deny/DEFAULT_DENY", verdict.Kind, abi.ReasonName(verdict.Reason))
	}
}

func TestOpenCodeWindowsSelfModifyGlobPreservesConfiguredWitness(t *testing.T) {
	const configuredGlob = `internal\adjudicator\`
	a := New(Policy{
		Allow:           map[string]bool{"write": true},
		SelfModifyGlobs: []string{configuredGlob},
	})
	call := inlineCall("write", `{"filePath":"internal\\adjudicator\\decide.go","content":"after"}`)
	call.Meta = nil

	verdict := a.Adjudicate(context.Background(), call)
	if verdict.Kind != abi.VerdictDeny || verdict.Reason != abi.ReasonSelfModify {
		t.Fatalf("verdict=%v/%s, want deny/SELF_MODIFY", verdict.Kind, abi.ReasonName(verdict.Reason))
	}
	witness, ok := verdict.Payload.(abi.WitnessPayload)
	if !ok {
		t.Fatalf("payload type=%T, want abi.WitnessPayload", verdict.Payload)
	}
	if witness.Claim != configuredGlob {
		t.Fatalf("witness claim=%q, want original configured glob %q", witness.Claim, configuredGlob)
	}
}
