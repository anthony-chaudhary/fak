package memoryindex

import "testing"

func TestParseFrontmatterDecodesQuotedScalars(t *testing.T) {
	got := ParseFrontmatter(`---
name: 'owner''s-memory'
description: "Use a colon: safely and say \"hello\" from C:\\tools."
metadata:
  type: 'project''s'
---
`)
	want := Frontmatter{
		Present:     true,
		Terminated:  true,
		Name:        "owner's-memory",
		Description: `Use a colon: safely and say "hello" from C:\tools.`,
		Type:        "project's",
	}
	if got != want {
		t.Fatalf("ParseFrontmatter = %+v, want %+v", got, want)
	}
}

func TestParseFrontmatterPreservesMalformedQuotedScalars(t *testing.T) {
	got := ParseFrontmatter(`---
name: "unterminated
description: 'also unterminated
metadata:
  type: "bad\q"
---
`)
	want := Frontmatter{
		Present:     true,
		Terminated:  true,
		Name:        `"unterminated`,
		Description: `'also unterminated`,
		Type:        `"bad\q"`,
	}
	if got != want {
		t.Fatalf("ParseFrontmatter = %+v, want %+v", got, want)
	}
}
