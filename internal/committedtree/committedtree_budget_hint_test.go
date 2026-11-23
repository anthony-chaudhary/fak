package committedtree

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

// The extraction budget is separate from any caller deadline, so its timeout
// must name its own knob and stay recognisable as a deadline to callers.
func TestExtractionStopErrorNamesBudgetEnv(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	ctx = context.WithValue(ctx, budgetKey{}, 2*time.Minute)
	defer cancel()
	<-ctx.Done()
	err := extractionStopError(ctx)
	for _, want := range []string{"FAK_COMMITTEDTREE_EXTRACT_BUDGET", "timed out after 2m0s"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("extractionStopError = %q, want it to contain %q", err, want)
		}
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("extractionStopError = %v, want it to unwrap to context.DeadlineExceeded", err)
	}
}
