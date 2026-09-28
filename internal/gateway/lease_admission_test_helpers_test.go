package gateway

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// installEmptyTestLeaseAuthority supplies the host-owned half of workspace
// admission for tests whose subject is downstream of lease ownership. Tests of
// the lease boundary itself intentionally install their own authority instead.
func installEmptyTestLeaseAuthority(t *testing.T) {
	t.Helper()
	packageDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	repoRoot := filepath.Clean(filepath.Join(packageDir, "..", ".."))
	SetLeasePlaneProviders(nil, func(context.Context) (LeasePresenceView, error) {
		return LeasePresenceView{ClassifiedLeases: json.RawMessage(`[]`)}, nil
	})
	SetWorkspaceLeaseAdmissionProvider(func(context.Context) (WorkspaceLeaseAdmissionView, error) {
		return WorkspaceLeaseAdmissionView{WorkspaceRoot: repoRoot}, nil
	})
	t.Cleanup(func() {
		SetLeasePlaneProviders(nil, nil)
		SetWorkspaceLeaseAdmissionProvider(nil)
	})
}
