package fleet

import (
	"testing"
	"time"
)

// fak-test:runtime fast est=5ms lane=default
func TestMergeBoxConditionsIdempotent(t *testing.T) {
	t0 := time.Unix(1000, 0).UTC()
	t1 := time.Unix(1060, 0).UTC()
	t2 := time.Unix(1120, 0).UTC()

	conds, changed := MergeBoxConditions(nil, BoxCondition{
		Type: "GPUReset", Status: true, Reason: "Xid79", Message: "fallen off bus",
	}, t0)
	if !changed || len(conds) != 1 || !conds[0].LastTransitionTime.Equal(t0) {
		t.Fatalf("initial merge failed: %+v changed=%v", conds, changed)
	}

	// Identical patch is idempotent (changed=false, LastTransitionTime preserved).
	conds2, changed2 := MergeBoxConditions(conds, BoxCondition{
		Type: "GPUReset", Status: true, Reason: "Xid79", Message: "fallen off bus",
	}, t1)
	if changed2 || !conds2[0].LastTransitionTime.Equal(t0) {
		t.Fatalf("identical patch must be idempotent: %+v changed=%v", conds2, changed2)
	}

	// Message-only update preserves LastTransitionTime while reporting changed=true.
	conds3, changed3 := MergeBoxConditions(conds2, BoxCondition{
		Type: "GPUReset", Status: true, Reason: "Xid79", Message: "fallen off bus (x2)",
	}, t1)
	if !changed3 || !conds3[0].LastTransitionTime.Equal(t0) || conds3[0].Message != "fallen off bus (x2)" {
		t.Fatalf("message update failed: %+v changed=%v", conds3, changed3)
	}

	// Status transition updates LastTransitionTime.
	conds4, changed4 := MergeBoxConditions(conds3, BoxCondition{
		Type: "GPUReset", Status: false, Reason: "Recovered", Message: "healthy",
	}, t2)
	if !changed4 || !conds4[0].LastTransitionTime.Equal(t2) || conds4[0].Status {
		t.Fatalf("status transition failed: %+v changed=%v", conds4, changed4)
	}
}
