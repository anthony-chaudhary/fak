package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/syspromptmmu"
)

func TestGuardProfileEdgeAndAdversarialInputs(t *testing.T) {
	cases := []struct {
		name, profile string
		wantOK        bool
	}{
		{"empty", "", true}, {"blank", "   ", true}, {"unknown", "hostile; rm -rf", false},
		{"caveman exact", "caveman", false}, {"ponytail exact", "ponytail", false},
		{"caveman qualified", "caveman:medium", true}, {"ponytail qualified", "ponytail:native:medium", true},
		{"no prefix widening", "ponytail-extra", false}, {"no case confusion", "PONYTAIL", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ok := syspromptmmu.DescribeStyle(tc.profile).Known || syspromptmmu.DescribeWorkProfile(tc.profile).Known
			if ok != tc.wantOK {
				t.Fatalf("profile %q accepted=%v, want %v", tc.profile, ok, tc.wantOK)
			}
		})
	}
}
