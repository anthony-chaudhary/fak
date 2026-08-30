package guardroute

import (
	"bytes"
	"encoding/json"
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

			firstBytes, err := json.Marshal(first)
			if err != nil {
				t.Fatalf("marshal first decision: %v", err)
			}
			secondBytes, err := json.Marshal(second)
			if err != nil {
				t.Fatalf("marshal second decision: %v", err)
			}
			if !bytes.Equal(firstBytes, secondBytes) {
				t.Fatalf("identical input produced different bytes:\nfirst:  %s\nsecond: %s", firstBytes, secondBytes)
			}
			if first.Route != tc.wantRoute {
				t.Fatalf("Route=%v, want %v: %+v", first.Route, tc.wantRoute, first)
			}
		})
	}
}
