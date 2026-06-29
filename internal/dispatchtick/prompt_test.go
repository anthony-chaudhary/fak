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

func TestIssuePromptStatesGitLawsAndHonestBlock(t *testing.T) {
	p := RenderIssuePrompt(sampleIssuePrompt())
	for _, want := range []string{"main", "git add -A", "git commit -s", "OFF_TRUNK", "final report", "fabricate"} {
		if !strings.Contains(p, want) {
			t.Fatalf("prompt missing %q:\n%s", want, p)
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
