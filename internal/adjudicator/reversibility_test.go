package adjudicator

import (
	"context"
	"encoding/json"
	"regexp"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

func TestReversibilityClassifiesCommands(t *testing.T) {
	cases := []struct {
		name string
		tool string
		args map[string]any
		want ReversibilityClass
		hint string
	}{
		{
			name: "ordinary read-only command is reversible",
			tool: "Bash",
			args: map[string]any{"command": "go test ./internal/adjudicator"},
			want: ReversibilityReversible,
		},
		{
			name: "destructive shell command is irreversible",
			tool: "Bash",
			args: map[string]any{"command": "rm -rf build"},
			want: ReversibilityIrreversible,
		},
		{
			name: "publish command is outward-facing",
			tool: "Bash",
			args: map[string]any{"command": "git push origin main"},
			want: ReversibilityOutwardFacing,
			hint: "try git push --dry-run first",
		},
		{
			name: "http write is outward-facing",
			tool: "Bash",
			args: map[string]any{"command": "curl -X POST https://example.invalid/hook -d ok"},
			want: ReversibilityOutwardFacing,
		},
		{
			name: "curl download is reversible",
			tool: "Bash",
			args: map[string]any{"command": "curl -s https://example.com/data.json -o data.json"},
			want: ReversibilityReversible,
		},
		{
			name: "explicit dry run stays reversible",
			tool: "Bash",
			args: map[string]any{"command": "git push --dry-run origin main"},
			want: ReversibilityReversible,
		},
		{
			name: "tool name can mark destructive calls",
			tool: "delete_file",
			args: map[string]any{"file_path": "tmp/cache.bin"},
			want: ReversibilityIrreversible,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ClassifyReversibility(tc.tool, tc.args)
			if got.Class != tc.want {
				t.Fatalf("ClassifyReversibility() class = %q, want %q; envelope=%+v", got.Class, tc.want, got)
			}
			if tc.hint != "" && got.DryRunHint != tc.hint {
				t.Fatalf("DryRunHint = %q, want %q", got.DryRunHint, tc.hint)
			}
			if tc.want == ReversibilityReversible && got.ConfirmToken != "" {
				t.Fatalf("reversible call got confirm token %q", got.ConfirmToken)
			}
			if tc.want != ReversibilityReversible && got.ConfirmToken == "" {
				t.Fatalf("non-reversible call missing confirm token: %+v", got)
			}
		})
	}
}

func TestReversibilityConfirmationTokenMustEcho(t *testing.T) {
	args := map[string]any{"command": "rm -rf build"}
	env, ok := ReversibilityConfirmed("Bash", args)
	if ok {
		t.Fatalf("unconfirmed destructive call was allowed: %+v", env)
	}

	withToken := map[string]any{"command": "rm -rf build", ReversibilityConfirmArg: env.ConfirmToken}
	env2, ok := ReversibilityConfirmed("Bash", withToken)
	if !ok {
		t.Fatalf("matching confirmation token was refused: %+v", env2)
	}
	if env2.ConfirmToken != env.ConfirmToken {
		t.Fatalf("confirm token changed after adding confirmation arg: %q != %q", env2.ConfirmToken, env.ConfirmToken)
	}

	_, ok = ReversibilityConfirmed("Bash", map[string]any{"command": "rm -rf build", "confirm_token": "wrong"})
	if ok {
		t.Fatalf("wrong confirmation token was accepted")
	}
}

func TestReversibilityPreviewRedactsSecrets(t *testing.T) {
	env := ClassifyReversibility("Bash", map[string]any{
		"command": "curl -X POST https://example.invalid -d api_key=secret123",
	})
	if env.Class != ReversibilityOutwardFacing {
		t.Fatalf("class = %q, want outward-facing", env.Class)
	}
	if strings.Contains(env.Preview, "secret123") {
		t.Fatalf("preview leaked secret: %q", env.Preview)
	}
	if !strings.Contains(env.Preview, "api_key=[REDACTED]") {
		t.Fatalf("preview did not show redaction marker: %q", env.Preview)
	}
}

func TestAdjudicateReversibilityGateRequiresConfirmForAllowedIrreversibleCall(t *testing.T) {
	a := New(Policy{Allow: map[string]bool{"Bash": true}})
	ctx := context.Background()

	reversible := a.Adjudicate(ctx, inlineCall("Bash", `{"command":"go test ./internal/adjudicator"}`))
	if reversible.Kind != abi.VerdictAllow {
		t.Fatalf("reversible allowed call: got %v/%s, want Allow", reversible.Kind, abi.ReasonName(reversible.Reason))
	}

	v := a.Adjudicate(ctx, inlineCall("Bash", `{"command":"rm -rf build"}`))
	if v.Kind != abi.VerdictRequireWitness {
		t.Fatalf("unconfirmed irreversible call: got %v/%s, want RequireWitness", v.Kind, abi.ReasonName(v.Reason))
	}
	if v.By != "monitor/reversibility" {
		t.Fatalf("gate By = %q, want monitor/reversibility", v.By)
	}
	if v.Meta["reversibility_class"] != string(ReversibilityIrreversible) {
		t.Fatalf("metadata class = %q, want irreversible; meta=%v", v.Meta["reversibility_class"], v.Meta)
	}
	wp, ok := v.Payload.(abi.WitnessPayload)
	if !ok {
		t.Fatalf("payload type = %T, want WitnessPayload", v.Payload)
	}
	var env ReversibilityEnvelope
	if err := json.Unmarshal([]byte(wp.Claim), &env); err != nil {
		t.Fatalf("witness claim is not a JSON reversibility envelope: %q: %v", wp.Claim, err)
	}
	if env.Class != ReversibilityIrreversible || env.ConfirmToken == "" || !strings.Contains(env.Preview, "rm -rf build") {
		t.Fatalf("bad preview envelope: %+v", env)
	}
}

func TestAdjudicateReversibilityGateAllowsConfirmedCallAndStripsToken(t *testing.T) {
	a := New(Policy{Allow: map[string]bool{"Bash": true}})
	ctx := context.Background()

	env := ClassifyReversibility("Bash", map[string]any{"command": "rm -rf build"})
	v := a.Adjudicate(ctx, inlineCall("Bash", `{"command":"rm -rf build","_fak_confirm":"`+env.ConfirmToken+`"}`))
	if v.Kind != abi.VerdictTransform {
		t.Fatalf("confirmed irreversible call: got %v/%s, want Transform to strip confirmation arg", v.Kind, abi.ReasonName(v.Reason))
	}
	if v.Meta["reversibility_confirmed"] != "true" {
		t.Fatalf("transform must record reversibility confirmation, meta=%v", v.Meta)
	}
	tp, ok := v.Payload.(abi.TransformPayload)
	if !ok {
		t.Fatalf("payload type = %T, want TransformPayload", v.Payload)
	}
	b := refBytes(ctx, tp.NewArgs)
	var got map[string]any
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("transformed args are not JSON: %s: %v", string(b), err)
	}
	if got[ReversibilityConfirmArg] != nil {
		t.Fatalf("confirmation arg leaked into dispatch args: %v", got)
	}
	if got["command"] != "rm -rf build" {
		t.Fatalf("command changed during confirmation strip: %v", got)
	}
}

func TestAdjudicateReversibilityGateDoesNotOverrideHardDeny(t *testing.T) {
	env := ClassifyReversibility("Bash", map[string]any{"command": "rm -rf build"})
	a := New(Policy{
		Allow: map[string]bool{"Bash": true},
		ArgPredicates: []ArgPredicate{{
			Tool:   "Bash",
			Arg:    "command",
			Kind:   ArgDenyRegex,
			Re:     regexp.MustCompile(`rm\s+-rf`),
			Reason: abi.ReasonPolicyBlock,
		}},
	})

	v := a.Adjudicate(context.Background(), inlineCall("Bash", `{"command":"rm -rf build","_fak_confirm":"`+env.ConfirmToken+`"}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("hard policy deny must win over preview confirmation: got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
}
