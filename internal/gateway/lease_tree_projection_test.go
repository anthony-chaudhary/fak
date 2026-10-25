package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

// fak-test:runtime fast est=1s
func TestLeaseTreeProjectionAdmitsDisjointExistingTargetThenProtectsCreatedTree(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "docs", "existing.txt"), []byte("existing"), 0o600); err != nil {
		t.Fatal(err)
	}
	installProjectedLeaseAuthority(t, root, []WorkspaceAdmissionLease{{
		SessionID: "peer",
		TreeGlobs: []string{
			"internal/agent/inkernel_mtp_diagnostic_test.go",
			"internal/compute/nvfp4_native_test.go",
			"internal/model/dflash2_round_observation_test.go",
			"internal/model/v41_attention_state_rowview_test.go",
		},
	}})

	srv := &Server{}
	verdict, refused := srv.adjudicateLeaseAdmission(context.Background(), "allow_write", `{"file_path":"docs/existing.txt"}`, false)
	if refused {
		t.Fatalf("disjoint existing target refused: %+v", verdict)
	}

	if err := os.MkdirAll(filepath.Join(root, "internal", "agent"), 0o755); err != nil {
		t.Fatal(err)
	}
	verdict, refused = srv.adjudicateLeaseAdmission(context.Background(), "allow_write", `{"file_path":"internal/agent/inkernel_mtp_diagnostic_test.go"}`, false)
	if !refused || verdict.Reason != "LEASE_HELD" {
		t.Fatalf("created leased tree verdict = %+v, refused=%t; want LEASE_HELD", verdict, refused)
	}

	verdict, refused = srv.adjudicateLeaseAdmission(context.Background(), "allow_write", `{"file_path":"uncreated/nested/probe.txt"}`, false)
	if !refused || verdict.Reason != "DEFAULT_DENY" {
		t.Fatalf("missing proposed parent verdict = %+v, refused=%t; want DEFAULT_DENY", verdict, refused)
	}
}

// fak-test:runtime fast est=1s
func TestLeaseTreeProjectionDoesNotOmitMalformedOrUnknownAuthority(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "docs", "existing.txt"), nil, 0o600); err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		name   string
		leases []WorkspaceAdmissionLease
	}{
		{
			name: "malformed glob after projectable lease",
			leases: []WorkspaceAdmissionLease{
				{SessionID: "peer-a", TreeGlobs: []string{"internal/model/future/**"}},
				{SessionID: "peer-b", TreeGlobs: []string{"internal/[unterminated"}},
			},
		},
		{name: "unknown empty tree", leases: []WorkspaceAdmissionLease{{SessionID: "peer"}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			installProjectedLeaseAuthority(t, root, tc.leases)
			verdict, refused := (&Server{}).adjudicateLeaseAdmission(context.Background(), "allow_write", `{"file_path":"docs/existing.txt"}`, false)
			if !refused || verdict.Reason != "DEFAULT_DENY" {
				t.Fatalf("verdict = %+v, refused=%t; want fail-closed DEFAULT_DENY", verdict, refused)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestLeaseTreeProjectionRejectsEscapesAndDanglingSymlink(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()
	for _, tree := range []string{"../outside/**", "future/*/../../docs/**", "*/../docs/**"} {
		if _, ok := adjudicateCanonicalTreeGlob(root, tree); ok {
			t.Fatalf("lexical escape %q canonicalized", tree)
		}
	}
	absInside := filepath.ToSlash(filepath.Join(root, "future", "nested", "**"))
	if got, ok := adjudicateCanonicalTreeGlob(root, absInside); !ok || got != "future/nested/**" {
		t.Fatalf("absolute in-root lease = %q, ok=%t; want future/nested/**", got, ok)
	}

	outsideAlias := filepath.Join(root, "outside-alias")
	if err := makeLeaseProjectionDirectoryAlias(outside, outsideAlias); err != nil {
		t.Fatalf("create outside alias: %v", err)
	}
	if _, ok := adjudicateCanonicalTreeGlob(root, "outside-alias/**"); ok {
		t.Fatal("canonical escape canonicalized")
	}

	absOutside := filepath.ToSlash(filepath.Join(outside, "**"))
	if _, ok := adjudicateCanonicalTreeGlob(root, absOutside); ok {
		t.Fatal("absolute outside-root tree canonicalized")
	}

	filePrefix := filepath.Join(root, "not-a-directory")
	if err := os.WriteFile(filePrefix, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, ok := adjudicateCanonicalTreeGlob(root, "not-a-directory/child/**"); ok {
		t.Fatal("missing suffix beneath an existing non-directory canonicalized")
	}

	danglingTarget := filepath.Join(root, "dangling-target")
	if err := os.Mkdir(danglingTarget, 0o755); err != nil {
		t.Fatal(err)
	}
	dangling := filepath.Join(root, "dangling")
	if err := makeLeaseProjectionDirectoryAlias(danglingTarget, dangling); err != nil {
		t.Fatalf("create dangling alias precursor: %v", err)
	}
	if err := os.Remove(danglingTarget); err != nil {
		t.Fatalf("remove alias target: %v", err)
	}
	if _, ok := adjudicateCanonicalTreeGlob(root, "dangling/missing/**"); ok {
		t.Fatal("dangling symlink was treated as an absent literal suffix")
	}
}

// fak-test:runtime fast est=1s
func TestLeaseTreeProjectionCanonicalizesExistingSymlinkAlias(t *testing.T) {
	root := t.TempDir()
	realDir := filepath.Join(root, "internal", "gateway")
	if err := os.MkdirAll(realDir, 0o755); err != nil {
		t.Fatal(err)
	}
	alias := filepath.Join(root, "gateway-alias")
	if err := makeLeaseProjectionDirectoryAlias(realDir, alias); err != nil {
		t.Fatalf("create directory alias: %v", err)
	}
	if got, ok := adjudicateCanonicalTreeGlob(root, "gateway-alias/missing/future/**"); !ok || got != "internal/gateway/missing/future/**" {
		t.Fatalf("missing lease tail through alias = %q, ok=%t", got, ok)
	}
	rootAlias := filepath.Join(t.TempDir(), "root-alias")
	if err := makeLeaseProjectionDirectoryAlias(root, rootAlias); err != nil {
		t.Fatalf("create workspace root alias: %v", err)
	}
	absoluteAliasTree := filepath.ToSlash(filepath.Join(rootAlias, "missing", "future", "**"))
	if got, ok := adjudicateCanonicalTreeGlob(root, absoluteAliasTree); !ok || got != "missing/future/**" {
		t.Fatalf("absolute missing lease tail through root alias = %q, ok=%t", got, ok)
	}
	installProjectedLeaseAuthority(t, root, []WorkspaceAdmissionLease{{
		SessionID: "peer",
		TreeGlobs: []string{"gateway-alias/**"},
	}})

	verdict, refused := (&Server{}).adjudicateLeaseAdmission(context.Background(), "allow_write", `{"file_path":"internal/gateway/probe.txt"}`, false)
	if !refused || verdict.Reason != "LEASE_HELD" {
		t.Fatalf("alias lease verdict = %+v, refused=%t; want LEASE_HELD", verdict, refused)
	}
}

func installProjectedLeaseAuthority(t *testing.T, root string, leases []WorkspaceAdmissionLease) {
	t.Helper()
	SetLeasePlaneProviders(nil, func(context.Context) (LeasePresenceView, error) {
		return LeasePresenceView{ClassifiedLeases: json.RawMessage(`[]`)}, nil
	})
	SetWorkspaceLeaseAdmissionProvider(func(context.Context) (WorkspaceLeaseAdmissionView, error) {
		return WorkspaceLeaseAdmissionView{WorkspaceRoot: root, Leases: leases}, nil
	})
	t.Cleanup(func() {
		SetLeasePlaneProviders(nil, nil)
		SetWorkspaceLeaseAdmissionProvider(nil)
	})
}

func makeLeaseProjectionDirectoryAlias(target, alias string) error {
	if err := os.Symlink(target, alias); err == nil {
		return nil
	} else if runtime.GOOS != "windows" {
		return err
	}
	out, err := exec.Command("cmd", "/c", "mklink", "/J", alias, target).CombinedOutput()
	if err != nil {
		return errors.New(err.Error() + ": " + string(out))
	}
	return nil
}
