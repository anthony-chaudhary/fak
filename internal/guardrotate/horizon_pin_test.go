package guardrotate_test

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/guardrotate"
	"github.com/anthony-chaudhary/fak/internal/resume/rehome"
)

// TestWaitResetHorizonMatchesRehome pins the launch-time and resume-time reset-imminence horizons
// equal. guardrotate.WaitResetHorizon (a time.Duration) and rehome.WaitResetHorizonSeconds (int64
// seconds) encode the SAME "about to elapse" threshold in two packages and two types, otherwise
// coupled only by doc comments. This test makes the mirror load-bearing: editing one without the
// other (or slipping in a 15*60*1000 ms typo) fails CI. It guards ONLY the shared value — the two
// paths intentionally apply the horizon with different rules (guardrotate lets an OFFERABLE
// alternate beat an imminent cool; rehome's WAIT_RESET fires before target selection), which each
// package's docs spell out. It lives in the external guardrotate_test package: rehome imports no
// internal fak package, so importing both here creates no cycle.
func TestWaitResetHorizonMatchesRehome(t *testing.T) {
	got := int64(guardrotate.WaitResetHorizon / time.Second)
	if got != rehome.WaitResetHorizonSeconds {
		t.Fatalf("reset-imminence horizon drift: guardrotate.WaitResetHorizon=%s (%ds) but rehome.WaitResetHorizonSeconds=%ds — the launch-time and resume-time imminence horizons must stay equal",
			guardrotate.WaitResetHorizon, got, rehome.WaitResetHorizonSeconds)
	}
}
