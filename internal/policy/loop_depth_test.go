package policy

import (
	"encoding/json"
	"testing"
)

func TestLoopDepthClamp(t *testing.T) {
	rule, err := CompileLoopDepth(&LoopDepthRule{
		MaxLoopDepth: 4,
		Scopes:       map[string]int{"classify": 2, "deep": 8},
	})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	cases := []struct {
		name      string
		scope     string
		requested int
		want      int
		clamped   bool
	}{
		{"default under cap", "other", 3, 3, false},
		{"default at cap", "other", 4, 4, false},
		{"default over cap", "other", 9, 4, true},
		{"model default resolves to cap", "other", 0, 4, true},
		{"negative resolves to cap", "other", -1, 4, true},
		{"scope tighter than default", "classify", 3, 2, true},
		{"scope looser than default", "deep", 6, 6, false},
		{"scope over its cap", "deep", 12, 8, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, clamped := rule.ClampLoopDepth(tc.scope, tc.requested)
			if got != tc.want || clamped != tc.clamped {
				t.Fatalf("ClampLoopDepth(%q,%d) = (%d,%v), want (%d,%v)", tc.scope, tc.requested, got, clamped, tc.want, tc.clamped)
			}
		})
	}
}

func TestLoopDepthNoCapPassesThrough(t *testing.T) {
	var nilRule *LoopDepthRule
	if got, clamped := nilRule.ClampLoopDepth("any", 7); got != 7 || clamped {
		t.Fatalf("nil rule = (%d,%v), want (7,false)", got, clamped)
	}
	scopeOnly, err := CompileLoopDepth(&LoopDepthRule{Scopes: map[string]int{"classify": 2}})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	if got, clamped := scopeOnly.ClampLoopDepth("unlisted", 0); got != 0 || clamped {
		t.Fatalf("unlisted scope without default = (%d,%v), want (0,false)", got, clamped)
	}
	if got, clamped := scopeOnly.ClampLoopDepth("classify", 5); got != 2 || !clamped {
		t.Fatalf("listed scope = (%d,%v), want (2,true)", got, clamped)
	}
}

func TestLoopDepthCompileValidation(t *testing.T) {
	if got, err := CompileLoopDepth(nil); got != nil || err != nil {
		t.Fatalf("nil => (nil,nil), got (%v,%v)", got, err)
	}
	bad := []*LoopDepthRule{
		{},
		{MaxLoopDepth: -1},
		{MaxLoopDepth: 4, Scopes: map[string]int{"x": 0}},
		{MaxLoopDepth: 4, Scopes: map[string]int{"": 2}},
	}
	for i, r := range bad {
		if _, err := CompileLoopDepth(r); err == nil {
			t.Errorf("case %d %+v: want load error", i, r)
		}
	}
}

func TestLoopDepthJSONFieldName(t *testing.T) {
	var r LoopDepthRule
	if err := json.Unmarshal([]byte(`{"max_loop_depth":3,"scopes":{"classify":1}}`), &r); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if r.MaxLoopDepth != 3 || r.Scopes["classify"] != 1 {
		t.Fatalf("decoded %+v", r)
	}
}
