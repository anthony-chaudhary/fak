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
		name         string
		tool         string
		args         map[string]any
		want         ReversibilityClass
		hint         string
		hintContains []string
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
			// Floor preservation: mail/mutt have no fak verb, so naming verbs for
			// slack/git-push must not relax their escalation; they still get a
			// concrete review path instead of an empty redirect.
			name:         "mail head stays outward-facing with a concrete review hint",
			tool:         "Bash",
			args:         map[string]any{"command": "echo done | mail -s ci ops@example.invalid"},
			want:         ReversibilityOutwardFacing,
			hintContains: []string{"recipient", "body"},
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
			// #3560: gh api WRITES are the carve-out from the gh relaxation — a
			// `gh api -X POST` authors via the REST/Git Data API against ARBITRARY
			// repos, a larger blast radius than the gated git-CLI writes, so it is
			// re-escalated to git push's outward-facing class instead of slipping
			// under the relaxed gh surface.
			name: "gh api short-form write is outward-facing (carve-out)",
			tool: "Bash",
			args: map[string]any{"command": "gh api -X POST /repos/o/r/issues -f title=x"},
			want: ReversibilityOutwardFacing,
		},
		{
			name: "gh api --method POST is outward-facing",
			tool: "Bash",
			args: map[string]any{"command": "gh api --method POST /repos/o/r/pulls -f title=x"},
			want: ReversibilityOutwardFacing,
		},
		{
			name: "gh api -X PATCH is outward-facing",
			tool: "Bash",
			args: map[string]any{"command": "gh api -X PATCH /repos/o/r -f name=renamed"},
			want: ReversibilityOutwardFacing,
		},
		{
			// Flag-order robustness: --method after the endpoint still escalates.
			name: "gh api write method after endpoint is outward-facing",
			tool: "Bash",
			args: map[string]any{"command": "gh api /repos/o/r/git/refs --method DELETE"},
			want: ReversibilityOutwardFacing,
		},
		{
			// A gh api READ (GET is the default; no --method) stays reversible.
			name: "gh api GET read stays reversible",
			tool: "Bash",
			args: map[string]any{"command": "gh api repos/o/r/contents/x"},
			want: ReversibilityReversible,
		},
		{
			name: "gh api explicit --method GET stays reversible",
			tool: "Bash",
			args: map[string]any{"command": "gh api --method GET /repos/o/r/pulls"},
			want: ReversibilityReversible,
		},
		{
			name: "gh repo rename is outward-facing (always mutating)",
			tool: "Bash",
			args: map[string]any{"command": "gh repo rename newname"},
			want: ReversibilityOutwardFacing,
		},
		{
			name: "gh repo delete is outward-facing",
			tool: "Bash",
			args: map[string]any{"command": "gh repo delete o/r --yes"},
			want: ReversibilityOutwardFacing,
		},
		{
			name: "gh repo fork is outward-facing",
			tool: "Bash",
			args: map[string]any{"command": "gh repo fork o/r --clone=false"},
			want: ReversibilityOutwardFacing,
		},
		{
			// gh repo READS (view/clone) are not the fork/rename/delete carve-out and
			// stay reversible.
			name: "gh repo view stays reversible",
			tool: "Bash",
			args: map[string]any{"command": "gh repo view o/r --json name"},
			want: ReversibilityReversible,
		},
		{
			// Scope guard: the carve-out is gh api/gh repo only — the rest of the gh
			// surface stays operator-relaxed. gh pr create / gh issue create remain
			// reversible (own authenticated GitHub, reversible in practice), so the
			// fix did not sweep them back under the gate.
			name: "gh pr create stays reversible (relaxation intact)",
			tool: "Bash",
			args: map[string]any{"command": `gh pr create --title fix --body details`},
			want: ReversibilityReversible,
		},
		{
			// Regression guard: the git-CLI write path this issue is about stays
			// blocked-class (outward-facing) — the carve-out lifts gh api/gh repo UP
			// to match it, it must not accidentally relax git push.
			name: "git push stays blocked-class (regression)",
			tool: "Bash",
			args: map[string]any{"command": "git push origin main"},
			want: ReversibilityOutwardFacing,
		},
		{
			// A quoted MENTION of a gh api write (inside a commit message) must not
			// classify — the carve-out reads the payload-scan view, like the other
			// families.
			name: "commit message mentioning gh api POST is reversible",
			tool: "Bash",
			args: map[string]any{"command": `git commit -m "docs: run gh api -X POST to open a PR"`},
			want: ReversibilityReversible,
		},
		{
			// A leading sudo/env wrapper must not hide the gh api write.
			name: "sudo-wrapped gh api write is outward-facing",
			tool: "Bash",
			args: map[string]any{"command": "sudo gh api -X PUT /repos/o/r/contents/f"},
			want: ReversibilityOutwardFacing,
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
			// The short -n is the documented equivalent of --dry-run above; the
			// long form was already reversible, the short form must be too.
			name: "git push short dry-run is reversible",
			tool: "Bash",
			args: map[string]any{"command": "git push -n origin main"},
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
			for _, want := range tc.hintContains {
				if !strings.Contains(got.DryRunHint, want) {
					t.Fatalf("DryRunHint = %q, want substring %q", got.DryRunHint, want)
				}
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
	if env.DryRunHint == "" {
		t.Fatalf("http-write escalation must carry its own redirect")
	}
	if strings.Contains(env.DryRunHint, "fak sync push") {
		t.Fatalf("http-write escalation borrowed the git-push redirect: %q", env.DryRunHint)
	}
}

func TestReversibilityFamiliesDeclareRedirects(t *testing.T) {
	for _, family := range reversibilityFamilies {
		if strings.TrimSpace(family.hint) == "" {
			t.Fatalf("reversibility family %q escalates without a redirect hint", family.name)
		}
	}
}

func TestNonReversibleRepresentativeCorpusCarriesRedirect(t *testing.T) {
	cases := []struct {
		name    string
		tool    string
		args    map[string]any
		want    ReversibilityClass
		wantFak string
	}{
		{
			name:    "bash git push names fak sync push",
			tool:    "Bash",
			args:    map[string]any{"command": "git push origin main"},
			want:    ReversibilityOutwardFacing,
			wantFak: "fak sync push",
		},
		{
			name:    "mcp issue create names fak issue create",
			tool:    "create_issue",
			args:    map[string]any{"title": "bug", "body": "repro"},
			want:    ReversibilityOutwardFacing,
			wantFak: "fak issue create",
		},
		{
			name:    "bash slack names fak slack send",
			tool:    "Bash",
			args:    map[string]any{"command": "slack send -c ops 'done'"},
			want:    ReversibilityOutwardFacing,
			wantFak: "fak slack send",
		},
		{
			name: "npm publish",
			tool: "Bash",
			args: map[string]any{"command": "npm publish"},
			want: ReversibilityOutwardFacing,
		},
		{
			name: "mail head",
			tool: "Bash",
			args: map[string]any{"command": "echo done | mail ops@example.invalid"},
			want: ReversibilityOutwardFacing,
		},
		{
			name: "webhook command",
			tool: "Bash",
			args: map[string]any{"command": "webhook notify ops"},
			want: ReversibilityOutwardFacing,
		},
		{
			name: "registry publish",
			tool: "Bash",
			args: map[string]any{"command": "docker push registry.example.invalid/app:latest"},
			want: ReversibilityOutwardFacing,
		},
		{
			name: "http write",
			tool: "Bash",
			args: map[string]any{"command": "curl -X POST https://example.invalid/hook -d ok"},
			want: ReversibilityOutwardFacing,
		},
		{
			name: "messaging tool",
			tool: "send_email",
			args: map[string]any{"to": "ops@example.invalid", "body": "done"},
			want: ReversibilityOutwardFacing,
		},
		{
			name: "pr create tool",
			tool: "create_pr",
			args: map[string]any{"title": "fix", "body": "details"},
			want: ReversibilityOutwardFacing,
		},
		{
			name: "filesystem destroy",
			tool: "Bash",
			args: map[string]any{"command": "rm -rf build"},
			want: ReversibilityIrreversible,
		},
		{
			name: "git destroy",
			tool: "Bash",
			args: map[string]any{"command": "git reset --hard HEAD~1"},
			want: ReversibilityIrreversible,
		},
		{
			name: "infra destroy",
			tool: "Bash",
			args: map[string]any{"command": "terraform destroy -auto-approve"},
			want: ReversibilityIrreversible,
		},
		{
			name: "sql drop",
			tool: "Bash",
			args: map[string]any{"command": `psql -c "drop table users"`},
			want: ReversibilityIrreversible,
		},
		{
			name: "raw device write",
			tool: "Bash",
			args: map[string]any{"command": "dd if=image.bin of=/dev/sda"},
			want: ReversibilityIrreversible,
		},
		{
			name: "destructive tool",
			tool: "delete_file",
			args: map[string]any{"path": "tmp/cache.bin"},
			want: ReversibilityIrreversible,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ClassifyReversibility(tc.tool, tc.args)
			if got.Class != tc.want {
				t.Fatalf("ClassifyReversibility() class = %q, want %q; envelope=%+v", got.Class, tc.want, got)
			}
			if strings.TrimSpace(got.DryRunHint) == "" {
				t.Fatalf("non-reversible call carried no redirect hint: %+v", got)
			}
			if tc.wantFak != "" && !strings.Contains(got.DryRunHint, tc.wantFak) {
				t.Fatalf("DryRunHint = %q, want sanctioned verb %q", got.DryRunHint, tc.wantFak)
			}
		})
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

// TestReversibilityConfirmTokenIgnoresDescriptionDrift is the regression witness
// for the confirm-loop that wedged session f0e7ac0f (and the 2026-07-04 deadlock
// note): the model re-proposed a byte-identical destructive *command* but reworded
// the Bash "description" annotation each turn, so the confirm token rotated and the
// advertised "re-propose byte-identical + add _fak_confirm" recovery never
// converged. The token must bind to the effect-bearing command only — a confirm
// issued under one description must be accepted when echoed under another.
func TestReversibilityConfirmTokenIgnoresDescriptionDrift(t *testing.T) {
	const cmd = "rm tools/new_leaf.py tools/new_leaf_test.py"

	// First proposal: the gate refuses and issues a token bound to this call.
	first := map[string]any{"command": cmd, "description": "Delete the leftover new_leaf files"}
	env, ok := ReversibilityConfirmed("Bash", first)
	if ok {
		t.Fatalf("unconfirmed destructive call was allowed: %+v", env)
	}
	if env.ConfirmToken == "" {
		t.Fatalf("gate issued no confirm token: %+v", env)
	}

	// Re-proposal: same command, echoed token, but a REWORDED description — exactly
	// the drift that rotated the token before. It must now confirm.
	reproposed := map[string]any{
		"command":               cmd,
		"description":           "Remove the untracked new_leaf tool files",
		ReversibilityConfirmArg: env.ConfirmToken,
	}
	if _, ok := ReversibilityConfirmed("Bash", reproposed); !ok {
		t.Fatalf("confirm token rotated on description drift — the f0e7ac0f loop is not fixed")
	}

	// The token itself must be identical across the two descriptions.
	t1 := ReversibilityConfirmToken(ReversibilityIrreversible, "Bash", first)
	t2 := ReversibilityConfirmToken(ReversibilityIrreversible, "Bash", reproposed)
	if t1 != t2 {
		t.Fatalf("confirm token depends on description: %q != %q", t1, t2)
	}

	// Binding preserved: a DIFFERENT command must still get a different token, so a
	// confirm cannot be transplanted from a benign call onto a more dangerous one.
	other := map[string]any{"command": "rm -rf /", "description": "Delete the leftover new_leaf files"}
	if ReversibilityConfirmToken(ReversibilityIrreversible, "Bash", other) == t1 {
		t.Fatalf("confirm token failed to bind to the command text")
	}
}

// TestReversibilityConfirmTokenIgnoresCommandWhitespaceDrift is the second half of
// #2777: after description exclusion, a command reflowed with cosmetic whitespace —
// extra spaces between the SAME tokens, or leading/trailing padding — must still
// hash to the same confirm token so the re-proposal converges. Whitespace INSIDE a
// quoted span stays significant (a genuinely different action), so the binding is
// not weakened. This test fails on the per-arg hash that kept the command verbatim
// and passes once the command is whitespace-normalized for the token.
func TestReversibilityConfirmTokenIgnoresCommandWhitespaceDrift(t *testing.T) {
	base := map[string]any{"command": "rm -rf build/cache"}
	reflowed := map[string]any{"command": "rm   -rf    build/cache  "}

	t1 := ReversibilityConfirmToken(ReversibilityIrreversible, "Bash", base)
	t2 := ReversibilityConfirmToken(ReversibilityIrreversible, "Bash", reflowed)
	if t1 != t2 {
		t.Fatalf("confirm token rotated on cosmetic command whitespace: %q != %q", t1, t2)
	}

	// End to end: the reflowed re-proposal echoing the first token must confirm,
	// not draw a fresh refusal — the exact loop #2777 describes.
	env, ok := ReversibilityConfirmed("Bash", base)
	if ok {
		t.Fatalf("unconfirmed destructive call was allowed: %+v", env)
	}
	if env.ConfirmToken == "" {
		t.Fatalf("gate issued no confirm token: %+v", env)
	}
	reproposed := map[string]any{
		"command":               "rm   -rf    build/cache",
		ReversibilityConfirmArg: env.ConfirmToken,
	}
	if _, ok := ReversibilityConfirmed("Bash", reproposed); !ok {
		t.Fatalf("confirm token rotated on command whitespace drift — #2777 not fixed")
	}

	// Binding preserved: whitespace INSIDE a quoted span is significant, so two
	// commands that differ ONLY there must still get different tokens — a confirm
	// cannot be transplanted between semantically distinct commands.
	q1 := ReversibilityConfirmToken(ReversibilityIrreversible, "Bash", map[string]any{"command": `rm "a  b"`})
	q2 := ReversibilityConfirmToken(ReversibilityIrreversible, "Bash", map[string]any{"command": `rm "a b"`})
	if q1 == q2 {
		t.Fatalf("confirm token ignored quoted whitespace — binding weakened")
	}
}

// TestReversibilityConfirmTokenIgnoresLineContinuationReflow covers the wrapped
// re-proposal the issue names alongside re-spacing (#2777): a long command the
// model re-emits across lines with a backslash-newline continuation is the SAME
// action the shell joins back into one line, so it must hash to the same confirm
// token. A bare unquoted newline is a command SEPARATOR, not a continuation, so it
// stays significant — folding it would merge two commands and could transplant a
// confirm across distinct actions.
func TestReversibilityConfirmTokenIgnoresLineContinuationReflow(t *testing.T) {
	flat := map[string]any{"command": "terraform destroy -auto-approve -target=aws_s3_bucket.logs"}
	wrapped := map[string]any{"command": "terraform destroy \\\n  -auto-approve \\\n  -target=aws_s3_bucket.logs"}
	if ReversibilityConfirmToken(ReversibilityIrreversible, "Bash", flat) !=
		ReversibilityConfirmToken(ReversibilityIrreversible, "Bash", wrapped) {
		t.Fatalf("confirm token rotated on a backslash-newline line-continuation reflow — #2777 not fixed")
	}

	// A BARE unquoted newline separates two commands and must keep the token
	// distinct from the single-command form — the continuation fold cannot merge
	// them.
	twoCmds := map[string]any{"command": "rm -rf a\nrm -rf b"}
	oneCmd := map[string]any{"command": "rm -rf a rm -rf b"}
	if ReversibilityConfirmToken(ReversibilityIrreversible, "Bash", twoCmds) ==
		ReversibilityConfirmToken(ReversibilityIrreversible, "Bash", oneCmd) {
		t.Fatalf("confirm token folded a bare newline separator into a space — binding weakened")
	}
}

// TestAdjudicateReversibilityTokenStableAcrossDescriptionDrift drives the FULL
// adjudicate path (rung ordering -> wouldAdmit -> ClassifyReversibility ->
// reversibilityGateVerdict -> Meta["confirm_token"]) for an outward-facing git push
// whose Bash "description" is reworded between the refusal and the re-proposal. It is
// the RUNTIME guard #3306 asked for: the pure-function TestReversibility...Drift test
// above cannot catch a regression where a future rung reorder or envelope change
// folds the per-turn description back into the gated arg map. The token surfaced in
// the gate verdict must be identical across the drift, and echoing the first token
// under the reworded call must CONFIRM (Transform), not re-refuse — the exact loop
// that wedged session f0e7ac0f. A plain command (no shell metacharacters) sidesteps
// the json.Marshal HTML-escaping the issue flagged for `2>&1`.
func TestAdjudicateReversibilityTokenStableAcrossDescriptionDrift(t *testing.T) {
	a := New(Policy{Allow: map[string]bool{"Bash": true}})
	ctx := context.Background()

	v1 := a.Adjudicate(ctx, inlineCall("Bash", `{"command":"git push origin main","description":"push the auth fix"}`))
	if v1.Kind != abi.VerdictRequireWitness {
		t.Fatalf("git push not gated: got %v/%s, want REQUIRE_WITNESS", v1.Kind, abi.ReasonName(v1.Reason))
	}
	tok1 := v1.Meta["confirm_token"]
	if tok1 == "" {
		t.Fatalf("gate verdict carried no confirm_token: %+v", v1.Meta)
	}

	v2 := a.Adjudicate(ctx, inlineCall("Bash", `{"command":"git push origin main","description":"ship the corrected auth handler"}`))
	if v2.Kind != abi.VerdictRequireWitness {
		t.Fatalf("re-proposal not gated: got %v/%s", v2.Kind, abi.ReasonName(v2.Reason))
	}
	if tok1 != v2.Meta["confirm_token"] {
		t.Fatalf("confirm token rotated on description drift through the live adjudicate path: %q != %q", tok1, v2.Meta["confirm_token"])
	}

	// Echo the FIRST token under the REWORDED call — the advertised recovery must
	// converge to a confirmed dispatch, not a fresh refusal.
	confirmed := a.Adjudicate(ctx, inlineCall("Bash", `{"command":"git push origin main","description":"ship the corrected auth handler","`+ReversibilityConfirmArg+`":"`+tok1+`"}`))
	if confirmed.Kind == abi.VerdictRequireWitness {
		t.Fatalf("echoed token under a reworded description did not confirm — the live loop is not fixed: %+v", confirmed)
	}
}

// TestAdjudicateReversibilityTokenStableAcrossCommandWhitespaceDrift is the
// whitespace sibling of the description-drift runtime guard above (#2777). It
// drives the FULL adjudicate path for an outward-facing git push whose COMMAND is
// reflowed with cosmetic extra spaces between the refusal and the re-proposal. The
// pure-function TestReversibilityConfirmTokenIgnoresCommandWhitespaceDrift cannot
// catch a regression where a rung reorder or envelope change folds the verbatim
// command back into the gated arg map ahead of normalization; this asserts the
// gate-surfaced token is identical across the drift and that echoing the first
// token under the reflowed command CONFIRMS instead of drawing a fresh refusal.
func TestAdjudicateReversibilityTokenStableAcrossCommandWhitespaceDrift(t *testing.T) {
	a := New(Policy{Allow: map[string]bool{"Bash": true}})
	ctx := context.Background()

	v1 := a.Adjudicate(ctx, inlineCall("Bash", `{"command":"git push origin main"}`))
	if v1.Kind != abi.VerdictRequireWitness {
		t.Fatalf("git push not gated: got %v/%s, want REQUIRE_WITNESS", v1.Kind, abi.ReasonName(v1.Reason))
	}
	tok1 := v1.Meta["confirm_token"]
	if tok1 == "" {
		t.Fatalf("gate verdict carried no confirm_token: %+v", v1.Meta)
	}

	v2 := a.Adjudicate(ctx, inlineCall("Bash", `{"command":"git   push    origin main"}`))
	if v2.Kind != abi.VerdictRequireWitness {
		t.Fatalf("reflowed re-proposal not gated: got %v/%s", v2.Kind, abi.ReasonName(v2.Reason))
	}
	if tok1 != v2.Meta["confirm_token"] {
		t.Fatalf("confirm token rotated on command whitespace drift through the live adjudicate path: %q != %q", tok1, v2.Meta["confirm_token"])
	}

	// Echo the FIRST token under the REFLOWED command — the advertised recovery
	// must converge to a confirmed dispatch, not a fresh refusal.
	confirmed := a.Adjudicate(ctx, inlineCall("Bash", `{"command":"git   push    origin main","`+ReversibilityConfirmArg+`":"`+tok1+`"}`))
	if confirmed.Kind == abi.VerdictRequireWitness {
		t.Fatalf("echoed token under a reflowed command did not confirm — the live loop is not fixed: %+v", confirmed)
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

func TestDatabaseDestinationFence(t *testing.T) {
	cases := []struct {
		name        string
		tool        string
		args        map[string]any
		wantBlocked bool
		wantHost    string
	}{
		{
			name:        "local localhost url",
			tool:        "bash",
			args:        map[string]any{"command": "psql postgres://user:pass@localhost:5432/app"},
			wantBlocked: false,
		},
		{
			name:        "local 127.0.0.1 url",
			tool:        "bash",
			args:        map[string]any{"command": "psql postgres://user:pass@127.0.0.1:5432/app"},
			wantBlocked: false,
		},
		{
			name:        "local container alias db",
			tool:        "bash",
			args:        map[string]any{"command": "mysql -h db -u root -p secret app"},
			wantBlocked: false,
		},
		{
			name:        "remote internal host blocked",
			tool:        "bash",
			args:        map[string]any{"command": "psql postgres://user:pass@prod-db.internal:5432/app"},
			wantBlocked: true,
			wantHost:    "prod-db.internal",
		},
		{
			name:        "remote amazon rds host blocked",
			tool:        "bash",
			args:        map[string]any{"command": "psql -h remote.rds.amazonaws.com -U admin app"},
			wantBlocked: true,
			wantHost:    "remote.rds.amazonaws.com",
		},
		{
			name:        "structured argument remote url",
			tool:        "db_query",
			args:        map[string]any{"database_url": "postgres://user:pass@staging-db.corp.net:5432/app"},
			wantBlocked: true,
			wantHost:    "staging-db.corp.net",
		},
		{
			name:        "structured argument local url",
			tool:        "db_query",
			args:        map[string]any{"database_url": "postgres://user:pass@localhost:5432/app"},
			wantBlocked: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			host, blocked, reason := ClassifyDatabaseDestination(tc.tool, tc.args)
			if blocked != tc.wantBlocked {
				t.Fatalf("blocked = %v, want %v (host=%s, reason=%s)", blocked, tc.wantBlocked, host, reason)
			}
			if tc.wantBlocked && host != tc.wantHost {
				t.Errorf("host = %q, want %q", host, tc.wantHost)
			}
			if tc.wantBlocked && reason != ReasonProductionDBBlock {
				t.Errorf("reason = %q, want %q", reason, ReasonProductionDBBlock)
			}
		})
	}
}

func TestDatabaseMutationReversibility(t *testing.T) {
	cases := []struct {
		name      string
		cmd       string
		wantClass ReversibilityClass
		wantHint  string
	}{
		{
			name:      "truncate table is irreversible",
			cmd:       "psql -c 'TRUNCATE TABLE users;'",
			wantClass: ReversibilityIrreversible,
			wantHint:  "TRUNCATE",
		},
		{
			name:      "alter table is irreversible",
			cmd:       "mysql -e 'ALTER TABLE accounts ADD COLUMN status VARCHAR(20);'",
			wantClass: ReversibilityIrreversible,
			wantHint:  "ALTER",
		},
		{
			name:      "prisma reset is irreversible",
			cmd:       "npx prisma migrate reset --force",
			wantClass: ReversibilityIrreversible,
			wantHint:  "snapshot",
		},
		{
			name:      "alembic downgrade is irreversible",
			cmd:       "alembic downgrade -1",
			wantClass: ReversibilityIrreversible,
			wantHint:  "snapshot",
		},
		{
			name:      "goose down is irreversible",
			cmd:       "goose down",
			wantClass: ReversibilityIrreversible,
			wantHint:  "snapshot",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			env := ClassifyReversibility("bash", map[string]any{"command": tc.cmd})
			if env.Class != tc.wantClass {
				t.Errorf("class = %v, want %v", env.Class, tc.wantClass)
			}
			if !strings.Contains(env.DryRunHint, tc.wantHint) {
				t.Errorf("hint %q does not contain %q", env.DryRunHint, tc.wantHint)
			}
		})
	}
}
