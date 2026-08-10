package issuefanout

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

func TestLiveBodyRendersRootPointFields(t *testing.T) {
	body := LiveBody(issuepolicy.Candidate{
		RootPoint: "candidate creation", OriginSignal: "contract review",
		PreventsRecurrence: "strict mode refuses omissions",
	})
	for _, want := range []string{"## Root point\n\ncandidate creation", "## Origin signal\n\ncontract review", "## Prevents recurrence\n\nstrict mode refuses omissions"} {
		if !strings.Contains(body, want) {
			t.Fatalf("body missing %q:\n%s", want, body)
		}
	}
}
