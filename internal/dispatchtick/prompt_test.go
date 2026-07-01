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
