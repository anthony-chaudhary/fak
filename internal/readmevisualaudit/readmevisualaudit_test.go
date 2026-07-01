package readmevisualaudit

import "testing"

func TestDetectors(t *testing.T) {
	txt := "intro\n\n```mermaid\nflowchart LR\n  A --> B\n```\n"
	if !HasMermaid(txt) || !AuditText(txt).HasVisual {
		t.Fatalf("mermaid not detected")
	}
	if HasMermaid("```bash\necho hi\n```") {
		t.Fatalf("plain fence counted as mermaid")
	}
	img := "![turn-tax curves](visuals/60-hero-turntax-curves.png)"
	if got := DiagramImages(img); len(got) != 1 || got[0] != "visuals/60-hero-turntax-curves.png" || !AuditText(img).HasVisual {
		t.Fatalf("diagram image not counted: %v", got)
	}
	if len(DiagramImages("![gate](../../visuals/46-two-gate-security-model.svg)")) == 0 {
		t.Fatalf("relative visuals image not counted")
	}
	for _, badge := range []string{
		"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](x)",
		"![build](https://img.shields.io/badge/ci-green.svg)",
	} {
		if len(DiagramImages(badge)) != 0 || AuditText(badge).HasVisual {
			t.Fatalf("badge counted as visual: %s", badge)
		}
	}
	box := "```text\n┌─────────┐     ┌─────────┐\n│ client  │ ──▶ │ kernel  │\n└─────────┘     └─────────┘\n```\n"
	if ASCIIDiagramBlocks(box) != 1 || !AuditText(box).HasVisual {
		t.Fatalf("box diagram not counted")
	}
	bar := "```text\n  durable  ████████████████············ 10\n  usable   █████████████··············· 8\n  coverage [████████████████████████████] 100%\n```\n"
	if ASCIIDiagramBlocks(bar) != 1 || !AuditText(bar).HasVisual {
		t.Fatalf("bar chart not counted")
	}
	arrows := "```\nclient --> gate\ngate   --> upstream\n```\n"
	if ASCIIDiagramBlocks(arrows) != 1 {
		t.Fatalf("ascii arrow diagram not counted")
	}
	if ASCIIDiagramBlocks("```bash\nfoo --> bar.txt   # just a redirect-ish note\necho done\n```") != 0 {
		t.Fatalf("single arrow line counted")
	}
	if AuditText("| a | b |\n|---|---|\n| 1 | 2 |\n").HasVisual {
		t.Fatalf("markdown table counted")
	}
}

func TestPayload(t *testing.T) {
	ok := BuildPayload(".", []Check{{Check: "README.md", Status: "OK", Detail: "has image×4"}}, "")
	if !ok.OK || ok.Verdict != "OK" {
		t.Fatalf("ok payload = %+v", ok)
	}
	fail := BuildPayload(".", []Check{
		{Check: "README.md", Status: "OK", Detail: "has image×4"},
		{Check: "docs/x/README.md", Status: "FAIL", Detail: "text-only"},
	}, "")
	if fail.OK || fail.Verdict != "ACTION" || fail.Finding != "readmes_text_only" || fail.Reason == "" {
		t.Fatalf("fail payload = %+v", fail)
	}
	err := BuildPayload(".", nil, "no tracked README.md found")
	if err.OK || err.Finding != "tooling_error" {
		t.Fatalf("error payload = %+v", err)
	}
}
