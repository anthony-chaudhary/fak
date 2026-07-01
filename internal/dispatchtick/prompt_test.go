package dispatchtick

import (
	"strings"
	"testing"
)

func sampleIssuePrompt() IssuePromptInput {
	return IssuePromptInput{
		Number:    465,
		Title:     "obs: arm the DOS verdict-journal auto-emit",
		Body:      "The trust floor's own decisions should be observable.",
		Labels:    []string{"enhancement", "trust-floor"},
		Lane:      "docs",
		Workspace: "C:/work/fak",
	}
}

func TestIssuePromptCitesIssueNumberAsCloseLink(t *testing.T) {
	p := RenderIssuePrompt(sampleIssuePrompt())
	for _, want := range []string{"#465", "commit subject", "never closes"} {
		if !strings.Contains(p, want) {
			t.Fatalf("prompt missing %q:\n%s", want, p)
		}
	}
}

func TestIssuePromptEmbedsIssueFacts(t *testing.T) {
	in := sampleIssuePrompt()
	in.Lane = "gateway"
	p := RenderIssuePrompt(in)
	for _, want := range []string{"auto-emit", "observable", "enhancement, trust-floor", "`gateway` lane"} {
		if !strings.Contains(p, want) {
			t.Fatalf("prompt missing %q:\n%s", want, p)
		}
	}
}

func TestIssuePromptExtractsAgentIssueBrief(t *testing.T) {
	in := sampleIssuePrompt()
	in.Body = strings.Join([]string{
		"## Work unit",
		"leaf",
		"## Expected steps",
		"4",
		"## Working spine",
		"source row -> scoped issue -> dispatch worker",
		"## Assumptions",
		"- Existing marker dedupe is available.",
		"## Confusion risks",
		"- Do not turn this leaf into the parent epic.",
		"## Coordination notes",
		"- Serialize with issuecontract parser edits.",
		"## Trigger",
		"Verified handoff proposed this next leaf.",
		"## Batch policy",
		"At most two follow-up issues per handoff; update existing marker.",
		"## In scope",
		"Render a compact worker brief.",
		"## Out of scope",
		"Do not change route picking.",
		"## Done condition",
		"Prompt includes the parsed brief.",
		"## Witness",
		"go test ./internal/dispatchtick",
		"## Acceptance gate",
		"go test ./internal/dispatchtick",
	}, "\n")
	p := RenderIssuePrompt(in)
	for _, want := range []string{
		"agent issue brief (parsed from standard sections):",
		"- Work unit: leaf",
		"- Expected steps: 4",
		"- Assumptions: Existing marker dedupe is available.",
		"- Confusion risks: Do not turn this leaf into the parent epic.",
		"- Coordination: Serialize with issuecontract parser edits.",
		"- Batch policy: At most two follow-up issues per handoff; update existing marker.",
		"- Acceptance gate: go test ./internal/dispatchtick",
	} {
		if !strings.Contains(p, want) {
			t.Fatalf("prompt missing %q:\n%s", want, p)
		}
	}
}

func TestIssuePromptOmitsAgentBriefWithoutStandardSections(t *testing.T) {
	p := RenderIssuePrompt(sampleIssuePrompt())
	if strings.Contains(p, "agent issue brief") {
		t.Fatalf("prompt should not emit an empty agent brief:\n%s", p)
	}
}

func TestIssuePromptStatesGitLawsAndHonestBlock(t *testing.T) {
	p := RenderIssuePrompt(sampleIssuePrompt())
	for _, want := range []string{"main", "git add -A", "git commit -s", "OFF_TRUNK", "final report", "fabricate"} {
		if !strings.Contains(p, want) {
			t.Fatalf("prompt missing %q:\n%s", want, p)
		}
	}
}

func TestIssuePromptIncludesProofByDefaultChecklist(t *testing.T) {
	p := RenderIssuePrompt(sampleIssuePrompt())
	for _, want := range []string{
		"Proof by default checklist:",
		"visual/TUI bugs need a captured render or screenshot witness",
		"logic/behavior bugs need a failing-before and passing-after repro test",
		"docs/operator changes need a lint, render, or exact-output fixture",
		"shipped/done claims need a witnessed commit tied to `#465` and `(fak docs)`",
		"Do not stop on narrative alone.",
	} {
		if !strings.Contains(p, want) {
			t.Fatalf("prompt missing proof checklist item %q:\n%s", want, p)
		}
	}
	lower := strings.ToLower(p)
	for _, forbidden := range []string{
		"just say done",
		"report that it is done",
		"looks fixed",
	} {
		if strings.Contains(lower, forbidden) {
			t.Fatalf("prompt contains broad self-report wording %q:\n%s", forbidden, p)
		}
	}
}

func TestIssuePromptRedactsPrivateControlDetails(t *testing.T) {
	in := sampleIssuePrompt()
	in.Title = "dispatch: route gpu-lab-01 capacity"
	in.Body = strings.Join([]string{
		"## Working spine",
		"Keep public workers away from fak-private control details.",
		"## In scope",
		"Replace Slack control bridge references and docs/private-comms-channel.md links.",
		"## Done condition",
		"gpu-lab-01 and GPU-server reservation details are not visible in the public prompt.",
		"## Witness",
		"go test ./internal/dispatchtick",
	}, "\n")
	p := RenderIssuePrompt(in)
	lower := strings.ToLower(p)
	for _, forbidden := range []string{
		"fak-private",
		"slack control",
		"private-control",
		"private control bridge",
		"docs/private-comms-channel.md",
		"gpu-lab-01",
		"gpu-server reservation",
	} {
		if strings.Contains(lower, forbidden) {
			t.Fatalf("prompt leaked private-control token %q:\n%s", forbidden, p)
		}
	}
	for _, want := range []string{"[companion repo boundary]", "[companion repo control path]", "GPU/cloud capacity", "GPU/cloud host"} {
		if !strings.Contains(p, want) {
			t.Fatalf("prompt missing redaction replacement %q:\n%s", want, p)
		}
	}
}

func TestIssuePromptLocksTrunkOnlyAndForbidsBranchEscape(t *testing.T) {
	p := RenderIssuePrompt(sampleIssuePrompt())
	for _, want := range []string{
		"Work on the configured development branch `main` ONLY.",
		"Never branch / new-worktree (the OFF_TRUNK guard refuses it).",
		"No push / tag / force-push / history-rewrite / reset / clean / checkout-of-tracked-files.",
	} {
		if !strings.Contains(p, want) {
			t.Fatalf("prompt missing trunk-only guard %q:\n%s", want, p)
		}
	}
	lower := strings.ToLower(p)
	for _, forbidden := range []string{
		"feature branch",
		"side branch",
		"git checkout -b",
		"git switch -c",
		"git branch ",
		"git worktree add",
		"create a branch",
		"create a new worktree",
		"open a branch",
		"open a new worktree",
	} {
		if strings.Contains(lower, forbidden) {
			t.Fatalf("prompt contains branch-escape wording %q:\n%s", forbidden, p)
		}
	}
}

func TestIssuePromptUsesConfiguredDevelopmentBranch(t *testing.T) {
	in := sampleIssuePrompt()
	in.DevelopmentBranch = "dev"
	p := RenderIssuePrompt(in)
	for _, want := range []string{
		"configured development branch `dev`",
		"Just commit on `dev`.",
		"a committed change on the configured development branch `dev`",
	} {
		if !strings.Contains(p, want) {
			t.Fatalf("prompt missing %q:\n%s", want, p)
		}
	}
	for _, stale := range []string{
		"ship it on `main`",
		"Work on `main` ONLY",
		"Just commit on main",
		"a committed change on `main`",
	} {
		if strings.Contains(p, stale) {
			t.Fatalf("prompt contains stale branch wording %q:\n%s", stale, p)
		}
	}
}

func TestIssuePromptTruncatesLongBody(t *testing.T) {
	in := sampleIssuePrompt()
	in.Body = strings.Repeat("x", 5000)
	p := RenderIssuePrompt(in)
	if !strings.Contains(p, "truncated") {
		t.Fatalf("prompt missing truncation marker:\n%s", p)
	}
	if strings.Contains(p, strings.Repeat("x", 2000)) {
		t.Fatalf("prompt embedded an overlong body without truncation")
	}
}

func TestIssuePromptMissingBodyStillRenders(t *testing.T) {
	in := IssuePromptInput{Number: 7, Title: "t", Lane: "docs", Workspace: "repo"}
	rec := BuildIssuePrompt(in)
	if rec.Schema != PromptSchema || rec.Issue != 7 || rec.PromptChars != len(rec.Prompt) {
		t.Fatalf("record = %+v, want stable schema/issue/char count", rec)
	}
	if !strings.Contains(rec.Prompt, "#7") || !strings.Contains(rec.Prompt, "no body") {
		t.Fatalf("prompt missing fallback body or issue:\n%s", rec.Prompt)
	}
}
