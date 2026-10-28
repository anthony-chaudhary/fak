package main

import (
	"reflect"
	"testing"
)

// worktreeWorkerVerifyClosureGlobs is the parse-to-option seam for the repeatable
// --verify-closure flag: whatever it returns is passed verbatim to
// workerworktree.WithVerificationClosure and appended to the land options shared by
// every land branch (PrepareProspectiveLand, LandProspectiveVerified, Land), so these
// assertions on the parsed values ARE the assertion that they reach the land options.
func TestWorktreeWorkerVerifyClosureGlobsReachLandOptions(t *testing.T) {
	cases := []struct {
		name string
		in   repeatedString
		want []string
	}{
		{"absent flag declares no closure", nil, nil},
		{"empty value declares no closure", repeatedString{""}, nil},
		{"whitespace-only value declares no closure", repeatedString{"   "}, nil},
		{"single glob passes through", repeatedString{"pkg/calc/**"}, []string{"pkg/calc/**"}},
		{"values are trimmed, deduped, and order-canonical",
			repeatedString{"internal/engine/**", " pkg/calc/** ", "pkg/calc/**"},
			[]string{"internal/engine/**", "pkg/calc/**"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := worktreeWorkerVerifyClosureGlobs(tc.in)
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("worktreeWorkerVerifyClosureGlobs(%q) = %q, want %q", []string(tc.in), got, tc.want)
			}
		})
	}
}
