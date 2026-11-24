// Package deadlinerefusalregress pins the 2026-10-06 symptom with only the
// fabric's pre-existing exported API, so it builds on the parent commit and
// witnesses red-then-green: a Halo's typed deadline 503 was classified
// UPSTREAM_UNAVAILABLE -> FAILOVER_NEXT, a backend fault that spilled to cloud.
package deadlinerefusalregress

import (
	"net/http/httptest"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/deadlineadmit"
	"github.com/anthony-chaudhary/fak/pkg/routing"
)

// wantClass is the closed class's wire value; the constant itself is new, so
// naming it would not build on the parent this witness runs against.
const wantClass = routing.FailureClass("DEADLINE_INFEASIBLE")

// fak-test:runtime fast
func TestClassifyHaloDeadlineRefusalIsNotABackendFault(t *testing.T) {
	rec := httptest.NewRecorder()
	deadlineadmit.WriteRefusal(rec, deadlineadmit.Verdict{Reason: deadlineadmit.ReasonDeadlineInfeasible, Estimate: 275 * time.Second, RetryAfter: 30 * time.Second})
	got := routing.Classify(rec.Code, rec.Body.String())
	if got.Class != wantClass || !got.Class.IsKnown() {
		t.Fatalf("Classify(node deadline refusal) = %s/%s, want known %s", got.Class, got.Action, wantClass)
	}
	if got.Action != routing.ActionRetrySame {
		t.Fatalf("action = %s, want %s: a healthy node asked to be retried, not failed over", got.Action, routing.ActionRetrySame)
	}
}
