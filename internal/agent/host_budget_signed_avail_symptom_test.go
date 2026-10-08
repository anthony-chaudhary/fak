package agent

import (
	"context"
	"errors"
	"strconv"
	"strings"
	"testing"
)

// TestHostBudgetDeclineReportsSignedShortfall is the symptom regression: when the resident
// footprint already exceeds the armed host ceiling by N bytes, the refusal must report the
// signed shortfall (-N), not a clamped "available budget is 0 bytes" that hides how far over
// the ceiling the process sits.
// fak-test:runtime fast est=50ms lane=default
func TestHostBudgetDeclineReportsSignedShortfall(t *testing.T) {
	const promptTokens, maxNew = 32, 8
	const ceiling = int64(1 << 30)
	const over = int64(3 << 20) // usage exceeds the ceiling by exactly this many bytes

	p := bareHostPlanner()
	p.SetHostMemoryBudget(ceiling, constHostUsage(ceiling+over, true).used)

	_, release, err := p.admitHostMemory(context.Background(), promptTokens, maxNew)
	if release != nil {
		defer release()
	}
	var capErr *InKernelCapacityError
	if !errors.As(err, &capErr) {
		t.Fatalf("usage above ceiling: error = %T (%v), want *InKernelCapacityError", err, err)
	}
	if capErr.Site != "host-memory-precheck" {
		t.Fatalf("site = %q, want host-memory-precheck", capErr.Site)
	}
	want := strconv.FormatInt(-over, 10)
	if msg := capErr.Error(); !strings.Contains(msg, want) {
		t.Fatalf("Error() = %q, want the signed shortfall %s", msg, want)
	}
}
