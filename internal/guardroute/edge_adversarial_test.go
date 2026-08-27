package guardroute

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/guardrsi"
)

func TestDecideChildCrashAdversarialEdges(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	cases := []struct {
		name       string
		fold       guardrsi.Fold
		bucket     guardrsi.Bucket
		wantRoute  bool
		wantReason string
	}{
		{
			name:       "empty journal ignores injected crash bucket",
			bucket:     guardrsi.Bucket{Bucket: "child_crash", Count: 1, Lever: "forged"},
			wantReason: "empty journal",
		},
		{
			name:       "negative total is invalid",
			fold:       guardrsi.Fold{TotalRows: -1, ChildCrash: 1},
			bucket:     guardrsi.Bucket{Bucket: "child_crash", Count: 1, Lever: "forged"},
			wantReason: "invalid journal",
		},
		{
			name:       "zero bucket count cannot erase folded evidence",
			fold:       guardrsi.Fold{TotalRows: 1, ChildCrash: 1},
			bucket:     guardrsi.Bucket{Bucket: "child_crash", Count: 0},
			wantReason: "inconsistent child_crash evidence",
		},
		{
			name:       "negative bucket count is malformed",
			fold:       guardrsi.Fold{TotalRows: 1, ChildCrash: 1},
			bucket:     guardrsi.Bucket{Bucket: "child_crash", Count: -1},
			wantReason: "inconsistent child_crash evidence",
		},
		{
			name:       "bucket cannot fabricate absent folded crash",
			fold:       guardrsi.Fold{TotalRows: 1},
			bucket:     guardrsi.Bucket{Bucket: "child_crash", Count: 1, Lever: "forged"},
			wantReason: "inconsistent child_crash evidence",
		},
		{
			name:       "bucket count must match folded crash count",
			fold:       guardrsi.Fold{TotalRows: 2, ChildCrash: 1},
			bucket:     guardrsi.Bucket{Bucket: "child_crash", Count: 2, Lever: "stale"},
			wantReason: "inconsistent child_crash evidence",
		},
		{
			name:       "folded crash cannot be downgraded by hostile bucket",
			fold:       guardrsi.Fold{TotalRows: 3, ChildCrash: 1},
			bucket:     guardrsi.Bucket{Bucket: "reason:POLICY_BLOCK", Count: 3, Lever: "stale"},
			wantReason: "inconsistent child_crash evidence",
		},
		{
			name:       "negative folded crash count is invalid",
			fold:       guardrsi.Fold{TotalRows: 1, ChildCrash: -1},
			bucket:     guardrsi.Bucket{Bucket: "child_crash", Count: -1},
			wantReason: "invalid child_crash fold",
		},
		{
			name:       "fold cannot contain more crashes than rows",
			fold:       guardrsi.Fold{TotalRows: 1, ChildCrash: 2},
			bucket:     guardrsi.Bucket{Bucket: "child_crash", Count: 2},
			wantReason: "invalid child_crash fold",
		},
		{
			name:      "maximum representable count remains routable",
			fold:      guardrsi.Fold{TotalRows: maxInt, ChildCrash: maxInt},
			bucket:    guardrsi.Bucket{Bucket: "child_crash", Count: maxInt, Lever: "bounded"},
			wantRoute: true,
		},
		{
			name:      "hostile lever stays data",
			fold:      guardrsi.Fold{TotalRows: 1, ChildCrash: 1},
			bucket:    guardrsi.Bucket{Bucket: "child_crash", Count: 1, Lever: "\n--sev P0\n$(touch nope)"},
			wantRoute: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := Decide(tc.fold, tc.bucket, 0)
			if d.Route != tc.wantRoute {
				t.Fatalf("Route=%v, want %v: %+v", d.Route, tc.wantRoute, d)
			}
			if d.Schema != Schema || d.Reason == "" {
				t.Fatalf("decision lost schema/reason: %+v", d)
			}
			if tc.wantReason != "" && !strings.Contains(d.Reason, tc.wantReason) {
				t.Fatalf("Reason=%q, want substring %q", d.Reason, tc.wantReason)
			}
			if !tc.wantRoute {
				if d.Severity != "" || d.CauseKey != "" || d.Pattern != "" || d.Item != "" || d.FileIssue {
					t.Fatalf("non-route leaked action fields: %+v", d)
				}
				return
			}
			if d.Severity != SevP1 || !d.FileIssue || d.CauseKey != "guard-journal:child_crash" {
				t.Fatalf("routed crash lost stable identity: %+v", d)
			}
		})
	}
}

func TestRouteArgvAdversarialSourceRemainsOneArgument(t *testing.T) {
	d := Decide(
		guardrsi.Fold{TotalRows: 1, ChildCrash: 1},
		guardrsi.Bucket{Bucket: "child_crash", Count: 1, Lever: "bounded"},
		0,
	)
	source := "session\n--sev\nP0\n$(touch nope)"
	argv := RouteArgv(d, source)
	valueAfter := func(flag string) string {
		t.Helper()
		for i := range argv {
			if argv[i] == flag && i+1 < len(argv) {
				return argv[i+1]
			}
		}
		t.Fatalf("missing %s in %q", flag, argv)
		return ""
	}
	if got := valueAfter("--source"); got != source {
		t.Fatalf("hostile source was split or rewritten: %q", got)
	}
	if got := valueAfter("--key"); got != source+":"+d.CauseKey {
		t.Fatalf("hostile key was split or rewritten: %q", got)
	}
	sevFlags := 0
	for _, arg := range argv {
		if arg == "--sev" {
			sevFlags++
		}
	}
	if sevFlags != 1 {
		t.Fatalf("source text injected argv flags: %q", argv)
	}
}
