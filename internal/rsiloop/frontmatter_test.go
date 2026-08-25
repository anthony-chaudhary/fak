package rsiloop

import "testing"

func TestParseQuarantineCandidateDecodesQuotedScalars(t *testing.T) {
	got := ParseQuarantineCandidate(`---
name: 'owner''s-skill'
description: "Use a colon: safely and say \"hello\" from C:\\tools."
---
body
`, nil)
	if got.Name != "owner's-skill" {
		t.Errorf("name = %q, want %q", got.Name, "owner's-skill")
	}
	if got.Description != `Use a colon: safely and say "hello" from C:\tools.` {
		t.Errorf("description = %q", got.Description)
	}
	if got.Body != "body\n" {
		t.Errorf("body = %q, want %q", got.Body, "body\n")
	}
}

func TestParseQuarantineCandidatePreservesMalformedQuotedScalars(t *testing.T) {
	got := ParseQuarantineCandidate(`---
name: 'unterminated
description: "bad\q"
---
body
`, nil)
	if got.Name != `'unterminated` || got.Description != `"bad\q"` {
		t.Fatalf("malformed quoted scalars were not preserved: %+v", got)
	}
}
