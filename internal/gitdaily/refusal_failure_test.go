package gitdaily

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gitgate"
)

// TestRefusalErrorChannelsRequireRecovery pins every structural failure channel.
// Each assertion names the operator action, rather than merely checking that a
// refusal exists, so a new opaque error cannot silently join the surface.
func TestRefusalErrorChannelsRequireRecovery(t *testing.T) {
	tests := []struct {
		name     string
		result   Result
		code     RefusalCode
		retry    bool
		recovery string
	}{
		{"missing repo root", Result{ConfigErr: "gitdaily: missing RepoRoot"}, RefusalInvalidOptions, false, "git rev-parse --show-toplevel"},
		{"missing common dir", Result{ConfigErr: "gitdaily: missing GitCommonDir"}, RefusalInvalidOptions, false, "git rev-parse --git-common-dir"},
		{"missing both roots", Result{ConfigErr: "gitdaily: missing RepoRoot and GitCommonDir"}, RefusalInvalidOptions, false, "GitCommonDir"},
		{"tick lock open", Result{TickLockErr: "access denied"}, RefusalTickLock, true, "check permissions"},
		{"tick already active", Result{Skipped: SkipTickBusy}, RefusalTickBusy, true, "fak git-daily status"},
		{"lease lock cleanup", Result{Locks: LockSweep{LeaseErr: "access denied"}}, RefusalLockCleanup, false, "fak git-daily --json"},
		{"index lock cleanup", Result{Locks: LockSweep{IndexErr: "access denied"}}, RefusalLockCleanup, false, "fak git-daily --json"},
		{"tree lock cleanup", Result{Locks: LockSweep{Actions: []string{"FAILED access denied"}}}, RefusalLockCleanup, false, "fak git-daily --json"},
		{"maintenance safety gate", Result{Maint: gitgate.MaintResult{Incident: true}}, RefusalMaintenance, false, "fak git-daily --json"},
		{"ledger write", Result{LedgerErr: "access denied"}, RefusalLedgerWrite, false, "fak git-daily --force"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.result.Refusal()
			if got == nil {
				t.Fatal("Refusal() = nil")
			}
			if got.Code != tt.code || got.Retry != tt.retry {
				t.Fatalf("Refusal() = %#v, want code %s retry=%t", got, tt.code, tt.retry)
			}
			if !strings.Contains(got.Message, tt.recovery) {
				t.Fatalf("message %q does not name recovery %q", got.Message, tt.recovery)
			}
			if strings.Contains(got.Message, "access denied") {
				t.Fatalf("message leaked raw error: %q", got.Message)
			}
		})
	}
}
