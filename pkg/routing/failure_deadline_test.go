package routing

import (
	"errors"
	"fmt"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/deadlineadmit"
)

// nodeRefusalBody is the exact body a serving node's admission gate writes, so
// the classifier is tested against the wire contract rather than a paraphrase.
func nodeRefusalBody(t *testing.T) (int, string) {
	t.Helper()
	rec := httptest.NewRecorder()
	deadlineadmit.WriteRefusal(rec, deadlineadmit.Verdict{Reason: deadlineadmit.ReasonDeadlineInfeasible, Estimate: 275 * time.Second, RetryAfter: 30 * time.Second})
	return rec.Code, rec.Body.String()
}

func TestClassifyNodeRefusalIsDeadlineInfeasible(t *testing.T) {
	status, body := nodeRefusalBody(t)
	got := Classify(status, body)
	if got.Class != FailureDeadlineInfeasible {
		t.Fatalf("Classify(node refusal).Class = %q, want %q", got.Class, FailureDeadlineInfeasible)
	}
	if got.Action != ActionRetrySame || got.Retries != 0 {
		t.Fatalf("Classify(node refusal) = %s/%d, want %s/0", got.Action, got.Retries, ActionRetrySame)
	}
}

func TestClassifyDeadlineInfeasibleVerdictContract(t *testing.T) {
	c := FailureDeadlineInfeasible
	if !c.IsKnown() || !c.RequestScoped() || !c.AdmissionRefused() {
		t.Fatalf("%s: known=%v requestScoped=%v admissionRefused=%v, want all true", c, c.IsKnown(), c.RequestScoped(), c.AdmissionRefused())
	}
	if c.Transient() {
		t.Fatalf("%s.Transient() = true: a fixed re-walk backoff would ignore the node's Retry-After", c)
	}
	for _, headroom := range []int{1, 2, 5} {
		if got := newFailure(c, "").RetryBudget(headroom); got != 0 {
			t.Fatalf("RetryBudget(%d) = %d, want 0: another key of the same node gets the same estimate", headroom, got)
		}
	}
	for _, other := range FailureClasses() {
		if other != c && other.AdmissionRefused() {
			t.Fatalf("%s.AdmissionRefused() = true, want only %s", other, c)
		}
	}
}

func TestClassifyDeadlineInfeasibleNeedsTheWireStatus(t *testing.T) {
	_, body := nodeRefusalBody(t)
	for _, status := range []int{500, 502, 504, 429} {
		if got := Classify(status, body); got.Class == FailureDeadlineInfeasible {
			t.Fatalf("Classify(%d, refusal body).Class = %q: only the 503 wire contract is an admission refusal", status, got.Class)
		}
	}
	if got := Classify(503, `{"error":{"message":"server overloaded"}}`); got.Class != FailureUnavailable {
		t.Fatalf("Classify(503, plain overload).Class = %q, want %q", got.Class, FailureUnavailable)
	}
}

func TestClassifyErrorDeadlineInfeasibleSentinel(t *testing.T) {
	err := fmt.Errorf("admit: %w", deadlineadmit.ErrDeadlineInfeasible)
	if !errors.Is(err, deadlineadmit.ErrDeadlineInfeasible) {
		t.Fatal("fixture does not wrap the sentinel")
	}
	if got := ClassifyError(err); got.Class != FailureDeadlineInfeasible {
		t.Fatalf("ClassifyError(sentinel).Class = %q, want %q", got.Class, FailureDeadlineInfeasible)
	}
	if got := ClassifyError(errors.New("context deadline exceeded")); got.Class != FailureTimeout {
		t.Fatalf("ClassifyError(deadline exceeded).Class = %q, want %q", got.Class, FailureTimeout)
	}
}
