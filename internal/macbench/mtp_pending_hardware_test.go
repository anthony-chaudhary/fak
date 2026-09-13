package macbench

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// TestDefaultMTPAdaptersReportTypedPendingHardware proves that every default canonical
// arm adapter reports a TYPED blocked-on-hardware error, not merely free text. Each arm
// must fail with an error that both names the missing dependency and satisfies
// errors.Is(err, ErrMTPPendingHardware), so callers can programmatically distinguish
// "blocked on hardware" from a genuine failure without fabricating observed evidence.
func TestDefaultMTPAdaptersReportTypedPendingHardware(t *testing.T) {
	opts := DefaultMTPRunnerOptions()
	if err := ValidateMTPRunnerEnvelope(opts); err != nil {
		t.Fatalf("DefaultMTPRunnerOptions failed envelope validation: %v", err)
	}

	adapters := DefaultMTPAdapters()
	if len(adapters) != len(canonicalMTPArms) {
		t.Fatalf("expected %d adapters, got %d", len(canonicalMTPArms), len(adapters))
	}

	for _, armName := range canonicalMTPArms {
		armName := armName
		t.Run(armName, func(t *testing.T) {
			adapter, ok := adapters[armName]
			if !ok || adapter == nil {
				t.Fatalf("missing or nil adapter for canonical arm %q", armName)
			}

			req := expectedMTPArmRequest(opts, armName)
			_, err := adapter(context.Background(), req)
			if err == nil {
				t.Fatalf("arm %q: expected a pending-hardware error, got nil", armName)
			}
			if !errors.Is(err, ErrMTPPendingHardware) {
				t.Fatalf("arm %q: expected errors.Is(err, ErrMTPPendingHardware) to be true, got: %v", armName, err)
			}
			if !strings.Contains(err.Error(), "PENDING_HARDWARE") {
				t.Errorf("arm %q: expected error text to retain PENDING_HARDWARE, got: %v", armName, err)
			}
		})
	}
}
