package gitdaily

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gitgate"
)

func TestRefusalMessagesNameRepairWithoutRawErrors(t *testing.T) {
	tests := []struct {
		name     string
		result   Result
		code     RefusalCode
		contains string
	}{
		{"busy", Result{Skipped: SkipTickBusy}, RefusalTickBusy, "already running"},
		{"serializer", Result{TickLockErr: `open C:\secret\repo\.git\lock: denied`}, RefusalTickLock, "check permissions"},
		{"lock cleanup", Result{Locks: LockSweep{LeaseErr: `remove C:\secret\lock: denied`}}, RefusalLockCleanup, "inspect the lock sweep"},
		{"maintenance", Result{Maint: gitgate.MaintResult{Incident: true}}, RefusalMaintenance, "maintenance safety gate"},
		{"ledger", Result{LedgerErr: `open C:\secret\ledger: denied`}, RefusalLedgerWrite, "record its witness"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.result.Refusal()
			if got == nil || got.Code != tt.code || !strings.Contains(got.Message, tt.contains) {
				t.Fatalf("Refusal() = %#v", got)
			}
			if strings.Contains(got.Message, `C:\secret`) {
				t.Fatalf("message leaked raw path: %q", got.Message)
			}
		})
	}
}

func TestRefusalTreatsDedupeAndSuccessAsNonErrors(t *testing.T) {
	for _, result := range []Result{{}, {Skipped: SkipAlreadyRanToday}} {
		if got := result.Refusal(); got != nil {
			t.Fatalf("Refusal() = %#v, want nil", got)
		}
	}
}

func TestRunRejectsMissingRootsBeforeEffects(t *testing.T) {
	calls := 0
	run := func(context.Context, string, ...string) (string, int, error) { calls++; return "", 0, nil }
	got := Run(context.Background(), run, Options{Apply: true})
	if got.Refusal() == nil || got.Refusal().Code != RefusalInvalidOptions {
		t.Fatalf("Refusal() = %#v", got.Refusal())
	}
	if calls != 0 {
		t.Fatalf("runner calls = %d, want zero", calls)
	}
}
