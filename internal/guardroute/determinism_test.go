package guardroute

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/guardrsi"
)

func TestChildCrashRoutingDeterminism(t *testing.T) {
	cases := []struct {
		name      string
		fold      guardrsi.Fold
		bucket    guardrsi.Bucket
		wantRoute bool
	}{
		{
			name:      "valid crash evidence routes",
			fold:      guardrsi.Fold{TotalRows: 5, ChildCrash: 2},
			bucket:    guardrsi.Bucket{Bucket: "child_crash", Count: 2, Lever: "harden supervision"},
			wantRoute: true,
		},
		{
			name:      "mismatched crash evidence is rejected",
			fold:      guardrsi.Fold{TotalRows: 5, ChildCrash: 1},
			bucket:    guardrsi.Bucket{Bucket: "child_crash", Count: 2, Lever: "stale"},
			wantRoute: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			first := Decide(tc.fold, tc.bucket, 0)
			second := Decide(tc.fold, tc.bucket, 0)
			if !reflect.DeepEqual(first, second) {
				t.Fatalf("identical input produced different decisions:\nfirst:  %+v\nsecond: %+v", first, second)
			}
			if first.Route != tc.wantRoute {
				t.Fatalf("Route=%v, want %v: %+v", first.Route, tc.wantRoute, first)
			}
		})
	}
}
