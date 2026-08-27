package guardroute

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/guardrsi"
)

func TestDecideRefusalsNameRecovery(t *testing.T) {
	cases := []struct {
		name         string
		fold         guardrsi.Fold
		bucket       guardrsi.Bucket
		wantRecovery string
	}{
		{
			name:         "empty journal",
			wantRecovery: "run a guarded session to record at least one adjudicated row",
		},
		{
			name:         "negative total rows",
			fold:         guardrsi.Fold{TotalRows: -1},
			wantRecovery: "rebuild the fold from verified journal rows",
		},
		{
			name:         "child crash outside total rows",
			fold:         guardrsi.Fold{TotalRows: 1, ChildCrash: 2},
			bucket:       guardrsi.Bucket{Bucket: "child_crash", Count: 2},
			wantRecovery: "rebuild the fold from verified journal rows",
		},
		{
			name:         "crash bucket disagrees with fold",
			fold:         guardrsi.Fold{TotalRows: 2, ChildCrash: 1},
			bucket:       guardrsi.Bucket{Bucket: "child_crash", Count: 2},
			wantRecovery: "recompute the bucket with guardrsi.WorstBucket",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := Decide(tc.fold, tc.bucket, 0)
			if d.Route {
				t.Fatalf("refusal routed malformed evidence: %+v", d)
			}
			if !strings.Contains(d.Reason, tc.wantRecovery) {
				t.Fatalf("Reason=%q, want recovery %q", d.Reason, tc.wantRecovery)
			}
		})
	}
}
