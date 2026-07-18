package policy

import (
	"strings"
	"testing"
)

// The hiring-app catalog from issue #5153, in the Anthropic wire shape:
// one read-shaped tool and three verb-y ones, with string-typed arguments.
const anthropicCatalog = `[
  {"name": "lookup_order", "description": "Look up an order",
   "input_schema": {"type": "object", "properties": {
     "order_id": {"type": "string"},
     "limit": {"type": "integer"}}}},
  {"name": "screen_candidate", "input_schema": {"type": "object", "properties": {
     "candidate_id": {"type": "string"}}}},
  {"name": "send_offer", "input_schema": {"type": "object", "properties": {
     "candidate_id": {"type": "string"},
     "salary": {"type": "number"}}}},
  {"name": "reject_applicant", "input_schema": {"type": "object", "properties": {
     "candidate_id": {"type": "string"},
     "note": {"type": "string"}}}}
]`

// The same catalog in the OpenAI wire shape, wrapped under a "tools" key the
// way it sits in a chat-completions request body.
const openAICatalog = `{"tools": [
  {"type": "function", "function": {"name": "lookup_order",
   "parameters": {"type": "object", "properties": {
     "order_id": {"type": "string"},
     "limit": {"type": "integer"}}}}},
  {"type": "function", "function": {"name": "screen_candidate",
   "parameters": {"type": "object", "properties": {
     "candidate_id": {"type": "string"}}}}},
  {"type": "function", "function": {"name": "send_offer",
   "parameters": {"type": "object", "properties": {
     "candidate_id": {"type": "string"},
     "salary": {"type": "number"}}}}},
  {"type": "function", "function": {"name": "reject_applicant",
   "parameters": {"type": "object", "properties": {
     "candidate_id": {"type": "string"},
     "note": {"type": "string"}}}}}
]}`

// goldenScaffold is the exact manifest ScaffoldFromTools must emit for the
// catalog above, from EITHER wire shape — the known-schema → known-manifest
// golden the issue's acceptance requires. Deterministic: tools and argument
// stubs are sorted, so a byte-level compare is stable.
const goldenScaffold = `{
  "version": "fak-policy/v1",
  "posture": "fail_closed",
  "allow": [
    "lookup_order"
  ],
  "deny": {
    "reject_applicant": "POLICY_BLOCK",
    "screen_candidate": "POLICY_BLOCK",
    "send_offer": "POLICY_BLOCK"
  },
  "arg_rules": [
    {
      "tool": "lookup_order",
      "arg": "order_id",
      "deny_regex": "\\A__FROM_TOOLS_TODO_FORBIDDEN_VALUE__\\z",
      "reason": "POLICY_BLOCK",
      "fix": "TODO: replace the placeholder deny_regex for lookup_order.order_id with the real forbidden-value pattern (or delete this rule); the placeholder matches nothing"
    },
    {
      "tool": "reject_applicant",
      "arg": "candidate_id",
      "deny_regex": "\\A__FROM_TOOLS_TODO_FORBIDDEN_VALUE__\\z",
      "reason": "POLICY_BLOCK",
      "fix": "TODO: replace the placeholder deny_regex for reject_applicant.candidate_id with the real forbidden-value pattern (or delete this rule); the placeholder matches nothing"
    },
    {
      "tool": "reject_applicant",
      "arg": "note",
      "deny_regex": "\\A__FROM_TOOLS_TODO_FORBIDDEN_VALUE__\\z",
      "reason": "POLICY_BLOCK",
      "fix": "TODO: replace the placeholder deny_regex for reject_applicant.note with the real forbidden-value pattern (or delete this rule); the placeholder matches nothing"
    },
    {
      "tool": "screen_candidate",
      "arg": "candidate_id",
      "deny_regex": "\\A__FROM_TOOLS_TODO_FORBIDDEN_VALUE__\\z",
      "reason": "POLICY_BLOCK",
      "fix": "TODO: replace the placeholder deny_regex for screen_candidate.candidate_id with the real forbidden-value pattern (or delete this rule); the placeholder matches nothing"
    },
    {
      "tool": "send_offer",
      "arg": "candidate_id",
      "deny_regex": "\\A__FROM_TOOLS_TODO_FORBIDDEN_VALUE__\\z",
      "reason": "POLICY_BLOCK",
      "fix": "TODO: replace the placeholder deny_regex for send_offer.candidate_id with the real forbidden-value pattern (or delete this rule); the placeholder matches nothing"
    }
  ]
}
`

// TestScaffoldFromToolsGolden asserts the known catalog produces the known
// manifest byte-for-byte, from both the Anthropic and the OpenAI wire shape —
// the deterministic schema→manifest contract of `fak policy --from-tools`.
func TestScaffoldFromToolsGolden(t *testing.T) {
	for _, tc := range []struct {
		shape, catalog string
	}{
		{"anthropic", anthropicCatalog},
		{"openai", openAICatalog},
	} {
		m, err := ScaffoldFromTools([]byte(tc.catalog))
		if err != nil {
			t.Fatalf("%s: ScaffoldFromTools: %v", tc.shape, err)
		}
		if got := string(m.JSON()); got != goldenScaffold {
			t.Errorf("%s: scaffold mismatch\n--- got ---\n%s\n--- want ---\n%s", tc.shape, got, goldenScaffold)
		}
	}
}

// TestScaffoldFromToolsRoundTripsCheck asserts the scaffold loads through the
// SAME validator `fak policy --check` uses, unchanged — every tool name from
// the input is on the resolved floor (allow or explicit deny), and the floor
// stays fail-closed for the verb-y tools.
func TestScaffoldFromToolsRoundTripsCheck(t *testing.T) {
	m, err := ScaffoldFromTools([]byte(anthropicCatalog))
	if err != nil {
		t.Fatalf("ScaffoldFromTools: %v", err)
	}
	rt, err := m.ToRuntime()
	if err != nil {
		t.Fatalf("scaffold does not round-trip --check validation: %v", err)
	}
	p := rt.Adjudicator
	if !p.Allow["lookup_order"] {
		t.Errorf("read-shaped lookup_order is not on the allow floor")
	}
	for _, verb := range []string{"screen_candidate", "send_offer", "reject_applicant"} {
		if p.Allow[verb] {
			t.Errorf("verb-shaped %s is silently allowed; scaffold must stay fail-closed", verb)
		}
		if !p.NeverAdmits(verb) {
			t.Errorf("verb-shaped %s is not explicitly denied", verb)
		}
	}
}

// TestScaffoldFromToolsFailLoud pins the loud-failure edges: an empty catalog,
// a nameless tool, and bytes that are neither shape all refuse with a
// from-tools error instead of emitting a hollow floor.
func TestScaffoldFromToolsFailLoud(t *testing.T) {
	for _, tc := range []struct{ name, in string }{
		{"empty array", `[]`},
		{"empty tools key", `{"tools": []}`},
		{"nameless tool", `[{"input_schema": {"type": "object"}}]`},
		{"not a catalog", `{"model": "gpt-4o"}`},
	} {
		if _, err := ScaffoldFromTools([]byte(tc.in)); err == nil {
			t.Errorf("%s: expected an error, got a manifest", tc.name)
		} else if !strings.Contains(err.Error(), "from-tools") {
			t.Errorf("%s: error does not name from-tools: %v", tc.name, err)
		}
	}
}

// TestScaffoldFromToolsReadShapeBoundary pins the word-boundary rule: verb
// prefixes qualify only at `_`/`-`/`.`/camelCase boundaries, so `getaway_car`
// cannot ride the `get` prefix onto the allow-list.
func TestScaffoldFromToolsReadShapeBoundary(t *testing.T) {
	for name, want := range map[string]bool{
		"get_user":    true,
		"getUser":     true,
		"list":        true,
		"search.docs": true,
		"getaway_car": false,
		"delete_user": false,
		"Checkout":    false,
	} {
		if got := isReadShaped(name); got != want {
			t.Errorf("isReadShaped(%q) = %v, want %v", name, got, want)
		}
	}
}
