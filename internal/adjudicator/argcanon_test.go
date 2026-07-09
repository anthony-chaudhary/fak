package adjudicator

import (
	"context"
	"encoding/json"
	"regexp"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// inlineCallJSON builds a tool call whose args are the JSON encoding of m,
// avoiding hand-escaped JSON literals for values containing quotes/backslashes.
func inlineCallJSON(t *testing.T, tool string, m map[string]any) *abi.ToolCall {
	t.Helper()
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}
	return &abi.ToolCall{
		Tool: tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: b},
		Meta: map[string]string{"readOnlyHint": "true"},
	}
}

// TestCanonicalArgVariantsHitOneRule is the #2407 witness: >=12 spelled
// variants of the SAME credential path — backslash separators, redundant
// dot-segments, "$HOME"/"${HOME}"/"~" aliases, and single/double-quote
// wrapping (including combined with an alias) — all resolve to the identical
// canonical path and so all hit the SAME deny rule a raw-string match would
// have caught only for the one spelling it was written against.
func TestCanonicalArgVariantsHitOneRule(t *testing.T) {
	t.Setenv("HOME", "/home/me")

	a := New(Policy{
		Allow: map[string]bool{"read_credential_file": true},
		ArgPredicates: []ArgPredicate{{
			Tool: "read_credential_file", Arg: "path", Kind: ArgDenyRegex,
			Re: regexp.MustCompile(`\.ssh/id_(rsa|ed25519)$`), Reason: abi.ReasonPolicyBlock,
		}},
	})
	ctx := context.Background()

	const canonical = "/home/me/.ssh/id_rsa"
	variants := []string{
		canonical,
		`\home\me\.ssh\id_rsa`,        // backslash separators
		"/home/me/foo/../.ssh/id_rsa", // redundant dot-segment
		"/home/me/./.ssh/id_rsa",      // redundant "." segment
		"//home/me/.ssh/id_rsa",       // doubled leading slash
		"$HOME/.ssh/id_rsa",           // env alias
		"${HOME}/.ssh/id_rsa",         // braced env alias
		"~/.ssh/id_rsa",               // tilde alias
		`'` + canonical + `'`,         // single-quoted
		`"` + canonical + `"`,         // double-quoted
		`'~/.ssh/id_rsa'`,             // quoted + tilde alias
		`"$HOME/.ssh/id_rsa"`,         // quoted + env alias
		`\home\me\foo\..\.ssh\id_rsa`, // backslash + dot-segment
	}
	if len(variants) < 12 {
		t.Fatalf("need >=12 spelled variants, have %d", len(variants))
	}

	for _, val := range variants {
		v := a.Adjudicate(ctx, inlineCallJSON(t, "read_credential_file", map[string]any{"path": val}))
		if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
			t.Errorf("variant %q: got %v/%s, want Deny/POLICY_BLOCK", val, v.Kind, abi.ReasonName(v.Reason))
		}
	}

	// A path genuinely outside the rule's reach stays allowed — canonicalization
	// must not widen the rule into a false positive.
	if v := a.Adjudicate(ctx, inlineCallJSON(t, "read_credential_file", map[string]any{"path": "/home/me/notes.txt"})); v.Kind != abi.VerdictAllow {
		t.Fatalf("benign path: got %v/%s, want Allow", v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestCanonicalArgUndecodableFailsMalformed pins the fail-closed half: a
// value the canonicalizer cannot decode (an unterminated quote) denies
// MALFORMED rather than silently skipping the rule because the raw string
// never matched the canonical-form regex.
func TestCanonicalArgUndecodableFailsMalformed(t *testing.T) {
	a := New(Policy{
		Allow: map[string]bool{"read_credential_file": true},
		ArgPredicates: []ArgPredicate{{
			Tool: "read_credential_file", Arg: "path", Kind: ArgDenyRegex,
			Re: regexp.MustCompile(`\.ssh/id_(rsa|ed25519)$`), Reason: abi.ReasonPolicyBlock,
		}},
	})
	v := a.Adjudicate(context.Background(), inlineCallJSON(t, "read_credential_file", map[string]any{"path": `'/home/me/.ssh/id_rsa`}))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonMalformed {
		t.Fatalf("unterminated quote: got %v/%s, want Deny/MALFORMED", v.Kind, abi.ReasonName(v.Reason))
	}
	// The MALFORMED verdict must DISCLOSE what was rejected (#2771): the Claim
	// names the specific decode failure and the bounded Meta["fix"] rides the
	// remedy seam so the agent is told how to fix it — never a silent MALFORMED
	// that reads as a broken tool.
	wp, ok := v.Payload.(abi.WitnessPayload)
	if !ok || !strings.Contains(wp.Claim, "unterminated quote") {
		t.Fatalf("MALFORMED claim must name the decode failure, got %q", wp.Claim)
	}
	if v.Meta["fix"] == "" {
		t.Fatalf("MALFORMED verdict must carry a bounded Meta[fix] disclosure, got %+v", v.Meta)
	}
}

// TestCanonicalArgQuotedFirstTokenCommandNotMalformed is the #2771 witness: a
// well-formed multi-token shell command whose FIRST token is a closed quoted
// word (a quoted program path, a `"$HOME/..."` invocation) must NOT be refused
// as MALFORMED. The old unwrapQuotes read any value STARTING with a quote as a
// whole-value wrap and failed closed the moment it did not also END with that
// quote — so `"$HOME/go/bin/fak" build ./... 2>&1 | tee log` (a benign build
// command) was refused MALFORMED with no disclosure, the false positive the
// guard complaint reported. A quoted first token that closes in the interior is
// a command line, not an unterminated quote: it is admitted (and still matched
// against the canonical form by whatever rule targets it).
func TestCanonicalArgQuotedFirstTokenCommandNotMalformed(t *testing.T) {
	t.Setenv("HOME", "/home/me")

	a := New(Policy{
		Allow: map[string]bool{"Bash": true},
		ArgPredicates: []ArgPredicate{{
			Tool: "Bash", Arg: "command", Kind: ArgDenyRegex,
			Re: regexp.MustCompile(`\bsudo\b`), Reason: abi.ReasonPolicyBlock,
		}},
	})
	ctx := context.Background()

	// Every one of these begins with a quoted token and does not end in a quote —
	// the exact shape the old classifier failed closed on. None names a blocked
	// construct, so each must be ADMITTED, never MALFORMED.
	benign := []string{
		`"$HOME/go/bin/fak" build ./... 2>&1 | tee log`, // quoted program path + redirect + pipe
		`'/usr/local/bin/go' test ./internal/... -count=1`, // single-quoted first token, compound go
		`"$HOME/bin/tool" --flag && git status`,            // quoted token + compound git
		`'my program' run`,                                 // quoted token with a space inside
	}
	for _, cmd := range benign {
		v := a.Adjudicate(ctx, inlineCallJSON(t, "Bash", map[string]any{"command": cmd}))
		if v.Kind != abi.VerdictAllow {
			t.Errorf("benign quoted-first-token command %q: got %v/%s, want Allow (no MALFORMED false positive)",
				cmd, v.Kind, abi.ReasonName(v.Reason))
		}
	}

	// The canonical form is still reached: a quoted-first-token command that DOES
	// name a blocked construct is still denied by the rule (canonicalization was
	// widened, not disabled).
	v := a.Adjudicate(ctx, inlineCallJSON(t, "Bash", map[string]any{"command": `"$HOME/bin/tool" && sudo rm x`}))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("quoted-first-token command naming a blocked construct: got %v/%s, want Deny/POLICY_BLOCK",
			v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestCanonicalArgPreservesCommandShapedTraversal is the regression this
// change must not introduce: a multi-token shell command containing a literal
// "../.." traversal must still match a traversal rule byte-for-byte — the
// canonicalizer skips dot-segment resolution for anything but a bare,
// single-token path/URL value precisely so it never eats the command text
// around the dots (see dogfood_manifest_test.go's "redirect traversal" / "cp
// traversal" cases, which pin the same behavior end-to-end).
func TestCanonicalArgPreservesCommandShapedTraversal(t *testing.T) {
	a := New(Policy{
		Allow: map[string]bool{"Bash": true},
		ArgPredicates: []ArgPredicate{{
			Tool: "Bash", Arg: "command", Kind: ArgDenyRegex,
			Re: regexp.MustCompile(`\.\./\.\.`), Reason: abi.ReasonPolicyBlock,
		}},
	})
	v := a.Adjudicate(context.Background(), inlineCallJSON(t, "Bash", map[string]any{"command": "echo x >> ../../tmp/exfil"}))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("command-shaped traversal: got %v/%s, want Deny/POLICY_BLOCK", v.Kind, abi.ReasonName(v.Reason))
	}
}
