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
			// git push stays gated, but the recovery hint now NAMES the compiled
			// sidestep (fak sync push) first — generalizing #2651's fak-issue-create
			// pattern to the git-push family so the agent isn't sent into the
			// confirm-token loop (docs/notes/CONFIRM-GATE-DEADLOCK-2026-07-04.md).
			name: "publish command is outward-facing and names fak sync push",
			tool: "Bash",
			args: map[string]any{"command": "git push origin main"},
			want: ReversibilityOutwardFacing,
			hint: "push with the safe compiled verb: fak sync push (a trusted-binary non-force push the kernel admits), or preview first with git push --dry-run",
		},
		{
			// The sidestep the hint names must actually clear the gate: `fak sync
			// push` has command head `fak`, not `git push`, so it is reversible.
			name: "fak sync push sidesteps the git-push gate",
			tool: "Bash",
			args: map[string]any{"command": "fak sync push"},
			want: ReversibilityReversible,
		},
		{
			// A git push whose branch merely CONTAINS "slack" keeps its git-push
			// hint — the slack case is segment-head-guarded, not substring-matched.
			name: "git push of a slack-named branch keeps the git-push hint",
			tool: "Bash",
			args: map[string]any{"command": "git push origin fix/slack-integration"},
			want: ReversibilityOutwardFacing,
			hint: "push with the safe compiled verb: fak sync push (a trusted-binary non-force push the kernel admits), or preview first with git push --dry-run",
		},
		{
			// slack HEAD is escalated and now names the compiled fak slack send verb.
			name: "slack send is outward-facing and names fak slack send",
			tool: "Bash",
			args: map[string]any{"command": "slack send -c ops 'deploy done'"},
			want: ReversibilityOutwardFacing,
			hint: "send it with the sanctioned compiled verb: fak slack send (a trusted-binary path the kernel admits)",
		},
		{
			// Floor preservation: mail/mutt have NO fak verb, so naming verbs for
			// slack/git-push must not relax their escalation nor invent a hint.
			name: "mail head stays outward-facing with no fak-verb hint",
			tool: "Bash",
			args: map[string]any{"command": "echo done | mail -s ci ops@example.invalid"},
			want: ReversibilityOutwardFacing,
		},
		{
			// gh is operator-relaxed (2026-07-05): every gh write targets the
			// operator's own authenticated GitHub and is reversible in practice, so
			// the outbound preview-confirm pause no longer fires for the gh family.
			name: "gh issue create is reversible (gh surface relaxed)",
			tool: "Bash",
			args: map[string]any{"command": `gh issue create --title "bug" --body "repro"`},
			want: ReversibilityReversible,
		},
		{
			name: "gh issue close is reversible",
			tool: "Bash",
			args: map[string]any{"command": "gh issue close 123 --reason completed"},
			want: ReversibilityReversible,
		},
		{
			name: "gh pr merge is reversible (gh surface relaxed)",
			tool: "Bash",
			args: map[string]any{"command": "gh pr merge 42 --squash"},
			want: ReversibilityReversible,
		},
		{
			name: "gh release create is reversible (gh surface relaxed)",
			tool: "Bash",
			args: map[string]any{"command": "gh release create v1.2.3 --notes done"},
			want: ReversibilityReversible,
		},
		{
			name: "gh api write is reversible (gh surface relaxed)",
			tool: "Bash",
			args: map[string]any{"command": "gh api -X POST /repos/o/r/issues -f title=x"},
			want: ReversibilityReversible,
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
		{
			name: "grep pattern mentioning git push is reversible",
			tool: "Bash",
			args: map[string]any{"command": `grep -rn "git push" docs/`},
			want: ReversibilityReversible,
		},
		{
			name: "commit message mentioning push is reversible",
			tool: "Bash",
			args: map[string]any{"command": `git commit -m "docs: explain when to push"`},
			want: ReversibilityReversible,
		},
		{
			name: "grep for mail is reversible",
			tool: "Bash",
			args: map[string]any{"command": "grep -c mail internal/gateway/*.go"},
			want: ReversibilityReversible,
		},
		{
			name: "echo mentioning rm -rf is reversible",
			tool: "Bash",
			args: map[string]any{"command": `echo "never run rm -rf blindly"`},
			want: ReversibilityReversible,
		},
		{
			name: "git log grep for npm publish is reversible",
			tool: "Bash",
			args: map[string]any{"command": `git log --grep "npm publish"`},
			want: ReversibilityReversible,
		},
		{
			name: "docker run --rm is reversible",
			tool: "Bash",
			args: map[string]any{"command": "docker run --rm -it ubuntu bash"},
			want: ReversibilityReversible,
		},
		{
			name: "push in second sequence segment is outward-facing",
			tool: "Bash",
			args: map[string]any{"command": `git commit -m "fix: gate" && git push`},
			want: ReversibilityOutwardFacing,
		},
		{
			name: "env assignment and sudo wrapper still expose rm",
			tool: "Bash",
			args: map[string]any{"command": "FOO=1 sudo rm -rf x"},
			want: ReversibilityIrreversible,
		},
		{
			name: "mail in second pipe segment is outward-facing",
			tool: "Bash",
			args: map[string]any{"command": "echo hi | mail bob"},
			want: ReversibilityOutwardFacing,
		},
		{
			name: "sql drop payload is still irreversible",
			tool: "Bash",
			args: map[string]any{"command": `psql -c "drop database x"`},
			want: ReversibilityIrreversible,
		},
		{
			name: "git reset hard is still irreversible",
			tool: "Bash",
			args: map[string]any{"command": "git reset --hard HEAD~1"},
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

// TestRedirectComesFromTheBlockingFamily pins the drift #2748 removes: the
// redirect must derive from the family entry that actually blocked, so a
// trigger phrase inside a quoted payload cannot attach another family's hint
// to an unrelated escalation. Here the curl write escalates (http-write
// family, no sanctioned sidestep) while the payload merely MENTIONS git push;
// the git-push redirect must not leak onto it.
func TestRedirectComesFromTheBlockingFamily(t *testing.T) {
	env := ClassifyReversibility("Bash", map[string]any{
		"command": `curl -X POST https://example.invalid/hook -d "git push"`,
	})
	if env.Class != ReversibilityOutwardFacing {
		t.Fatalf("curl write must stay outward-facing, got %q", env.Class)
	}
	if env.DryRunHint != "" {
		t.Fatalf("http-write escalation borrowed another family's redirect: %q", env.DryRunHint)
	}
}

// TestFamilyTableSingleEntryYieldsBlockAndRedirect is the #2748 witness:
// declaring ONE hypothetical family entry is sufficient to get BOTH its
// escalation and its redirect — there is no second switch to keep in sync.
func TestFamilyTableSingleEntryYieldsBlockAndRedirect(t *testing.T) {
	const cmd = "carrierpigeon deliver --to ops 'release note'"
	if class, hint := classifyAgainstFamilies(reversibilityFamilies, "Bash", cmd); class != ReversibilityReversible || hint != "" {
		t.Fatalf("hypothetical family already classified by the real table: %q/%q", class, hint)
	}
	withFamily := append(append([]reversibilityFamily{}, reversibilityFamilies...), reversibilityFamily{
		name:  "carrier-pigeon",
		class: ReversibilityOutwardFacing,
		heads: []string{"carrierpigeon"},
		hint:  "send it with the sanctioned compiled verb: fak pigeon send",
	})
	class, hint := classifyAgainstFamilies(withFamily, "Bash", cmd)
	if class != ReversibilityOutwardFacing {
		t.Fatalf("single table entry did not escalate: class=%q", class)
	}
	if hint != "send it with the sanctioned compiled verb: fak pigeon send" {
		t.Fatalf("single table entry did not carry its redirect: hint=%q", hint)
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
