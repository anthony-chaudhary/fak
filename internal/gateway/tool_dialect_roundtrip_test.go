package gateway

import (
	"strings"
	"testing"
)

// TestToolDialectFormatNormalizeRoundTrip pins the fak#13074 invariant: for every known
// canonical tool and every supported dialect, FormatToolDialect followed by
// NormalizeToolDialect returns the SAME canonical tool name. This is the client-facing
// namespace contract, so a silent naming drift (a new dialect or a suffix-matching change)
// must fail here rather than in a downstream wiring bug.
//
// The round-trip is a property, not a fixed example table: it is driven from
// knownCanonicalTools x {every ToolDialect}, so it grows automatically as tools are added.
func TestToolDialectFormatNormalizeRoundTrip(t *testing.T) {
	dialects := []ToolDialect{
		DialectBare,
		DialectClaudeMCPCanonical,
		DialectClaudeMCP,
		DialectOpenAIFunc,
		DialectOpenCode,
		DialectColonMCP,
	}
	if len(knownCanonicalTools) == 0 {
		t.Fatal("knownCanonicalTools is empty; the round-trip property would be vacuous")
	}
	for canonical := range knownCanonicalTools {
		for _, d := range dialects {
			wire := FormatToolDialect(canonical, d)
			if wire == "" {
				t.Fatalf("FormatToolDialect(%q, %s) returned empty", canonical, d)
			}
			gotCanonical, _, _ := NormalizeToolDialect(wire)
			if gotCanonical != canonical {
				t.Errorf("round-trip drift: canonical=%q dialect=%s wire=%q -> %q (want %q)",
					canonical, d, wire, gotCanonical, canonical)
			}
		}
	}
}

// TestToolDialectOpenCodeSuffixMatchMultiUnderscore covers the ambiguity-prone OpenCode
// single-prefix suffix-match path (tool_dialect.go branch 8) for a multi-underscore
// canonical name: "<server>_<canonical>" must resolve to the canonical tool and the server,
// not to a longer/shorter suffix or a harness-excluded prefix.
func TestToolDialectOpenCodeSuffixMatchMultiUnderscore(t *testing.T) {
	const canonical = "fak_context_restore" // five underscore-separated segments
	wire := "myserver_" + canonical
	got, server, dialect := NormalizeToolDialect(wire)
	if got != canonical {
		t.Fatalf("suffix-match canonical = %q, want %q", got, canonical)
	}
	if server != "myserver" {
		t.Fatalf("suffix-match server = %q, want %q", server, "myserver")
	}
	if dialect != DialectOpenCode {
		t.Fatalf("suffix-match dialect = %s, want %s", dialect, DialectOpenCode)
	}

	// A harness-excluded prefix must NOT be stripped to a canonical tool: it stays bare so
	// an allow_/deny_/selfmod_/transform_/witness_ name is never mistaken for a real tool.
	for _, excluded := range []string{"allow", "deny", "selfmod", "transform", "witness"} {
		exWire := excluded + "_" + canonical
		gotEx, _, _ := NormalizeToolDialect(exWire)
		if gotEx == canonical {
			t.Errorf("excluded prefix %q_ was stripped to canonical tool %q", excluded, canonical)
		}
	}
}

// TestToolDialectFormatPreservesBareLegalName is the negative-space control: a bare
// canonical name must normalize to itself with no server, so the property above cannot pass
// by accident through a prefix that happens to strip to the same string.
func TestToolDialectFormatPreservesBareLegalName(t *testing.T) {
	const canonical = "fak_read"
	got, server, dialect := NormalizeToolDialect(canonical)
	if got != canonical || server != "" || dialect != DialectBare {
		t.Fatalf("bare normalize(%q) = (%q,%q,%s), want (%q,\"\",%s)",
			canonical, got, server, dialect, canonical, DialectBare)
	}
	if wire := FormatToolDialect(canonical, DialectBare); !strings.EqualFold(wire, canonical) {
		t.Fatalf("FormatToolDialect(%q, bare) = %q, want %q", canonical, wire, canonical)
	}
}
