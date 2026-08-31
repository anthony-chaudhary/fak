package coordination

import (
	"errors"
	"strings"
	"testing"
)

func validBootstrapRepairRequest() BootstrapRepairRequest {
	return BootstrapRepairRequest{
		Issue:                 9905,
		BaseSHA:               strings.Repeat("a", 40),
		CandidatePath:         `C:\scratch\fak-worker`,
		Lease:                 "coordination/codex-9905",
		Files:                 []string{"internal/workerworktree/prepare.go"},
		PrepareFailureDigest:  "sha256:prepare",
		DispatchFailureDigest: "sha256:dispatch",
		LeaseActive:           true,
	}
}

func TestBootstrapRepairGateAdmitsExactlyOneGuardedRepair(t *testing.T) {
	gate := new(BootstrapRepairGate)
	req := validBootstrapRepairRequest()
	got, err := gate.Admit(req)
	if err != nil {
		t.Fatalf("Admit() error = %v", err)
	}
	if got.Issue != req.Issue || got.CandidatePath != req.CandidatePath || got.Lease != req.Lease {
		t.Fatalf("Admit() = %#v", got)
	}
	req.Files[0] = "product/runtime.go"
	if got.Files[0] != "internal/workerworktree/prepare.go" {
		t.Fatal("admission retained caller-owned file slice")
	}
	if _, err := gate.Admit(validBootstrapRepairRequest()); !errors.Is(err, ErrBootstrapRepairRefused) {
		t.Fatalf("second Admit() error = %v, want refusal", err)
	}
}

func TestBootstrapRepairGateRefusesUnsafeCandidates(t *testing.T) {
	tests := []struct {
		name string
		edit func(*BootstrapRepairRequest)
		want string
	}{
		{"dirty candidate", func(r *BootstrapRepairRequest) { r.CandidateDirty = true }, "dirty"},
		{"live owner", func(r *BootstrapRepairRequest) { r.CandidateLiveOwned = true }, "live owner"},
		{"inactive lease", func(r *BootstrapRepairRequest) { r.LeaseActive = false }, "not active"},
		{"path overlap", func(r *BootstrapRepairRequest) { r.PathsOverlap = true }, "overlap"},
		{"missing prepare failure", func(r *BootstrapRepairRequest) { r.PrepareFailureDigest = "" }, "both canonical failure"},
		{"missing dispatch failure", func(r *BootstrapRepairRequest) { r.DispatchFailureDigest = "" }, "both canonical failure"},
		{"healthy prepare", func(r *BootstrapRepairRequest) { r.CanonicalPrepareOK = true }, "healthy"},
		{"healthy dispatch", func(r *BootstrapRepairRequest) { r.CanonicalDispatchOK = true }, "healthy"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := validBootstrapRepairRequest()
			tt.edit(&req)
			_, err := new(BootstrapRepairGate).Admit(req)
			if !errors.Is(err, ErrBootstrapRepairRefused) || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Admit() error = %v, want refusal containing %q", err, tt.want)
			}
		})
	}
}

func TestBootstrapRepairGateRetiresOnlyAfterCanonicalRecovery(t *testing.T) {
	gate := new(BootstrapRepairGate)
	if err := gate.Retire(true, false); !errors.Is(err, ErrBootstrapRepairRefused) {
		t.Fatalf("Retire(true, false) error = %v, want refusal", err)
	}
	if err := gate.Retire(true, true); err != nil {
		t.Fatalf("Retire(true, true) error = %v", err)
	}
	if _, err := gate.Admit(validBootstrapRepairRequest()); !errors.Is(err, ErrBootstrapRepairRefused) {
		t.Fatalf("Admit() after retirement error = %v, want refusal", err)
	}
}
