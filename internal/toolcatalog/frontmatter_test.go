package toolcatalog

import "testing"

func TestFrontmatterDecodesQuotedScalars(t *testing.T) {
	name, description, err := frontmatter([]byte(`---
name: 'owner''s-tool'
description: "Use a colon: safely and say \"hello\" from C:\\tools."
---
`))
	if err != nil {
		t.Fatal(err)
	}
	if name != "owner's-tool" {
		t.Errorf("name = %q, want %q", name, "owner's-tool")
	}
	if description != `Use a colon: safely and say "hello" from C:\tools.` {
		t.Errorf("description = %q", description)
	}
}

func TestFrontmatterPreservesMalformedQuotedScalars(t *testing.T) {
	name, description, err := frontmatter([]byte(`---
name: valid
description: "unterminated
---
`))
	if err != nil {
		t.Fatal(err)
	}
	if name != "valid" || description != `"unterminated` {
		t.Fatalf("frontmatter = name=%q description=%q, want malformed source preserved", name, description)
	}

	name, description, err = frontmatter([]byte(`---
name: 'unterminated
description: valid
---
`))
	if err != nil {
		t.Fatal(err)
	}
	if name != `'unterminated` || description != "valid" {
		t.Fatalf("frontmatter = name=%q description=%q, want malformed source preserved", name, description)
	}
}
