package vdso

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// call builds a ToolCall with the given tool name and meta hints.
func call(tool string, meta map[string]string) *abi.ToolCall {
	return &abi.ToolCall{Tool: tool, Meta: meta}
}

func roIdem(extra ...string) map[string]string {
	m := map[string]string{"readOnlyHint": "true", "idempotentHint": "true"}
	for i := 0; i+1 < len(extra); i += 2 {
		m[extra[i]] = extra[i+1]
	}
	return m
}

func TestSpeculatable_DefaultDenyOnEffects(t *testing.T) {
	cases := []struct {
		name string
		c    *abi.ToolCall
		want SpecReason
	}{
		{"nil call fails closed", nil, SpecRefusedNilCall},
		{"no hints at all fails closed", call("get_weather", nil), SpecRefusedNotReadOnly},
		{"read-only but no idempotent", call("get_weather", map[string]string{"readOnlyHint": "true"}), SpecRefusedNotIdempotent},
		{"idempotent but no read-only", call("get_weather", map[string]string{"idempotentHint": "true"}), SpecRefusedNotReadOnly},
		{"read-only + idempotent pure read is OK", call("get_weather", roIdem()), SpecOK},
		{"explicit destructive meta refuses even when read+idem", call("get_weather", roIdem("destructive", "true")), SpecRefusedDestructive},
		{"write-shaped tool name refuses even when read+idem", call("write_file", roIdem()), SpecRefusedDestructive},
		{"delete-shaped tool name refuses", call("delete_record", roIdem()), SpecRefusedDestructive},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := Speculatable(tc.c)
			if got != tc.want {
				t.Fatalf("Speculatable reason = %s, want %s", got, tc.want)
			}
			if wantOK := tc.want == SpecOK; ok != wantOK {
				t.Fatalf("Speculatable ok = %v, want %v", ok, wantOK)
			}
			if CanSpeculate(tc.c) != (tc.want == SpecOK) {
				t.Fatalf("CanSpeculate disagreed with Speculatable for %q", tc.name)
			}
			if got.Refused() == (tc.want == SpecOK) {
				t.Fatalf("Refused() = %v inconsistent with reason %s", got.Refused(), got)
			}
		})
	}
}

// TestSpeculatable_MatchesCacheAdmission pins the load-bearing invariant: the
// speculation gate admits EXACTLY the calls the vDSO cache gate admits to its pure
// tiers (readOnlyHint && idempotentHint && !destructive). If these two predicates
// ever drift, a call could be speculated but not cached (or the reverse), opening
// the gap the shared predicate exists to close.
func TestSpeculatable_MatchesCacheAdmission(t *testing.T) {
	tools := []string{"get_weather", "search_kb", "write_file", "delete_x", "edit_doc", "lookup"}
	hintSets := []map[string]string{
		nil,
		{"readOnlyHint": "true"},
		{"idempotentHint": "true"},
		{"readOnlyHint": "true", "idempotentHint": "true"},
		{"readOnlyHint": "true", "idempotentHint": "true", "destructive": "true"},
	}
	for _, tool := range tools {
		for _, h := range hintSets {
			c := call(tool, h)
			// The exact gate vdso.Lookup uses to enter its pure tiers.
			cacheAdmits := metaTrue(c, "readOnlyHint") && metaTrue(c, "idempotentHint") && !destructive(c)
			if CanSpeculate(c) != cacheAdmits {
				t.Fatalf("drift: tool=%q hints=%v cacheAdmits=%v CanSpeculate=%v",
					tool, h, cacheAdmits, CanSpeculate(c))
			}
		}
	}
}

func TestSpecReason_StringStable(t *testing.T) {
	for r := SpecOK; r <= SpecRefusedDestructive; r++ {
		if r.String() == "SPEC_REFUSED_UNCLASSIFIED" {
			t.Fatalf("reason %d has no stable token", r)
		}
	}
}
