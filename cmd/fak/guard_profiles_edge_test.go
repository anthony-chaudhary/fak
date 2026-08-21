package main

import "testing"

func TestGuardProfileEdgeAndAdversarialInputs(t *testing.T) {
	cases := []struct {
		name, profile string
		wantOK        bool
	}{
		{"empty", "", false}, {"blank", "   ", false}, {"unknown", "hostile; rm -rf", false},
		{"caveman exact", "caveman", true}, {"ponytail exact", "ponytail", true},
		{"no prefix widening", "ponytail-extra", false}, {"no case confusion", "PONYTAIL", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, ok := guardBuiltinProfiles()[tc.profile]
			if ok != tc.wantOK {
				t.Fatalf("profile %q accepted=%v, want %v", tc.profile, ok, tc.wantOK)
			}
		})
	}
}
