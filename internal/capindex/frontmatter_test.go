package capindex

import (
	"reflect"
	"testing"
)

func TestParseFrontmatterDecodesQuotedScalars(t *testing.T) {
	got := parseFrontmatter([]byte(`---
name: "quoted\\skill"
version: 'author''s-v1'
description: "Use a colon: safely and say \"hello\" from C:\\tools."
intent: 'Bob''s trigger'
tags: ['owner''s', "C:\\tools"]
---
`))

	if got.name != `quoted\skill` {
		t.Errorf("name = %q, want %q", got.name, `quoted\skill`)
	}
	if got.version != "author's-v1" {
		t.Errorf("version = %q, want %q", got.version, "author's-v1")
	}
	if got.description != `Use a colon: safely and say "hello" from C:\tools.` {
		t.Errorf("description = %q", got.description)
	}
	if got.intent != "Bob's trigger" {
		t.Errorf("intent = %q, want %q", got.intent, "Bob's trigger")
	}
	if want := []string{"owner's", `C:\tools`}; !reflect.DeepEqual(got.tags, want) {
		t.Errorf("tags = %#v, want %#v", got.tags, want)
	}
}

func TestParseFrontmatterPreservesMalformedQuotedScalars(t *testing.T) {
	got := parseFrontmatter([]byte(`---
name: valid
description: "unterminated
intent: 'also unterminated
tags: ["bad\q"]
---
`))
	if got.name != "valid" {
		t.Fatalf("name = %q, want valid", got.name)
	}
	if got.description != `"unterminated` || got.intent != `'also unterminated` {
		t.Fatalf("malformed quoted scalars were not preserved: %+v", got)
	}
	if want := []string{`"bad\q"`}; !reflect.DeepEqual(got.tags, want) {
		t.Fatalf("tags = %#v, want malformed source preserved as %#v", got.tags, want)
	}
}
