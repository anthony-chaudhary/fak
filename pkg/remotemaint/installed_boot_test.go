package remotemaint

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func installedBootTestGrant(now time.Time) Grant {
	g := validTestGrant(now)
	g.Operation = OperationInstalledBootApply
	g.Paths = []string{"/boot/efi/kernel.efi", "/boot/kernel.img"}
	g.Units, g.PathPrefixes = nil, nil
	g.InstalledBoot = &InstalledBootScope{Phase: "apply", ArtifactSHA256: g.PayloadSHA256,
		PrivateCommit: strings.Repeat("a", 40), PublicCommit: strings.Repeat("b", 40), PublicArchiveSHA256: repeated("c"),
		ManifestSHA256: repeated("d"), StageDigest: repeated("e"), PlanDigest: repeated("f"), IdentitySHA256: repeated("a"),
		ServiceIdentitySHA256: repeated("b"), ReadPaths: append([]string(nil), g.Paths...), WritePaths: append([]string(nil), g.Paths...),
		BackupPrefix: "/var/tmp/fak-boot.backup.", RequiredNodeIDs: []string{"node-1"}, CatalogGeneration: "catalog-1", ExpectedStoreEpoch: 1,
		LocalScopes: []InstalledBootLocalScope{{Root: filepath.Join(os.TempDir(), "fak-local"), Paths: []string{filepath.Join(os.TempDir(), "fak-local", "store.json")},
			PathPrefixes: []PathPrefix{{Directory: filepath.Join(os.TempDir(), "fak-local"), FilenamePrefix: "store.tmp."}}}}}
	return g
}

func installedBootTestObservation(g Grant) *InstalledBootObservation {
	b := g.InstalledBoot
	plan := b.PlanDigest
	if plan == "" && b.Phase == "stage-plan" {
		plan = repeated("8")
	}
	return &InstalledBootObservation{Kind: map[string]string{"preflight": "preflight", "stage-plan": "staged_plan", "apply": "applied", "readback": "readback", "rollback": "rolled_back"}[b.Phase],
		ArtifactSHA256: b.ArtifactSHA256, StageDigest: b.StageDigest, PlanDigest: plan, IdentitySHA256: b.IdentitySHA256,
		ServiceIdentitySHA256: b.ServiceIdentitySHA256, RootImageSHA256: repeated("1"), ESPImageSHA256: repeated("2"),
		Files: []InstalledBootFileState{{Path: g.Paths[0], SHA256: repeated("3"), Mode: "0644"}}}
}

// fak-test:runtime fast est=0.1s lane=default
func TestInstalledBootPreparedAndReceiptBindings(t *testing.T) {
	ctx := context.Background()
	for _, phase := range []string{"preflight", "stage-plan", "apply", "readback", "rollback"} {
		t.Run(phase, func(t *testing.T) {
			g := installedBootTestGrant(time.Now().UTC())
			g.InstalledBoot.Phase = phase
			g.Operation = map[string]string{"preflight": OperationInstalledBootReadback, "stage-plan": OperationInstalledBootStagePlan,
				"apply": OperationInstalledBootApply, "readback": OperationInstalledBootReadback, "rollback": OperationInstalledBootRollback}[phase]
			if phase == "preflight" || phase == "readback" {
				g.InstalledBoot.WritePaths, g.InstalledBoot.BackupPrefix = nil, ""
			}
			if phase == "preflight" {
				g.InstalledBoot.StageDigest, g.InstalledBoot.PlanDigest, g.InstalledBoot.IdentitySHA256, g.InstalledBoot.ServiceIdentitySHA256 = "", "", "", ""
			}
			if phase == "stage-plan" {
				g.InstalledBoot.PlanDigest = ""
			}
			a := &fakeAuthority{grant: g, receiptMutator: func(r *Receipt) {
				r.InstalledBoot = installedBootTestObservation(g)
				if phase == "preflight" || phase == "readback" {
					r.Outcome = OutcomeUnchanged
				}
				if phase == "rollback" {
					r.Outcome = OutcomeRolledBack
				}
			}}
			p, err := Prepare(ctx, a, Request{ProfileID: g.ProfileID, OperationDigest: g.OperationDigest})
			if err != nil {
				t.Fatal(err)
			}
			copy := p.Grant()
			copy.InstalledBoot.ReadPaths[0] = "/changed"
			copy.InstalledBoot.RequiredNodeIDs[0] = "changed"
			copy.InstalledBoot.LocalScopes[0].Paths[0] = "changed"
			copy.InstalledBoot.LocalScopes[0].PathPrefixes[0].FilenamePrefix = "changed"
			if GrantDigest(p.Grant()) != GrantDigest(g) {
				t.Fatal("prepared scope aliases a returned grant")
			}
			r, err := p.Execute(ctx)
			if err != nil || r.InstalledBoot == nil || len(a.validateCalls) != 2 || len(a.executeCalls) != 1 {
				t.Fatalf("receipt=%+v err=%v", r, err)
			}
			if phase == "stage-plan" && !isDigest(r.InstalledBoot.PlanDigest) {
				t.Fatal("stage did not return actual plan digest")
			}
			if _, err := p.Execute(ctx); !errors.Is(err, ErrChanged) {
				t.Fatalf("replay=%v", err)
			}
		})
	}
	g := installedBootTestGrant(time.Now().UTC())
	for name, mutate := range map[string]func(*Receipt){
		"missing":    func(r *Receipt) { r.InstalledBoot = nil },
		"hash":       func(r *Receipt) { r.InstalledBoot.PlanDigest = repeated("9") },
		"kind":       func(r *Receipt) { r.InstalledBoot.Kind = "staged_plan" },
		"cross-kind": func(r *Receipt) { r.Readback = &SystemdProfileReadback{} },
		"path":       func(r *Receipt) { r.InstalledBoot.Files[0].Path = "/secret" },
	} {
		t.Run(name, func(t *testing.T) {
			a := &fakeAuthority{grant: g, receiptMutator: func(r *Receipt) { r.InstalledBoot = installedBootTestObservation(g); mutate(r) }}
			p, err := Prepare(ctx, a, Request{ProfileID: g.ProfileID, OperationDigest: g.OperationDigest})
			if err != nil {
				t.Fatal(err)
			}
			r, err := p.Execute(ctx)
			if !errors.Is(err, ErrUnknownEffect) || r.Verified || r.InstalledBoot != nil {
				t.Fatalf("unbound receipt=%+v err=%v", r, err)
			}
		})
	}
	for _, op := range []string{OperationSystemdProfileInstallReload, OperationSystemdProfileReadback, OperationMaintainLeaseAcquire, OperationMaintainLeaseRelease, "installed-boot-reboot"} {
		candidate := cloneGrant(g)
		candidate.Operation = op
		if err := validateGrant(candidate, time.Now().UTC()); err == nil {
			t.Fatalf("boot scope aliased %s", op)
		}
	}
	shared := installedBootTestObservation(g)
	a := &fakeAuthority{grant: g, receiptMutator: func(r *Receipt) { r.InstalledBoot = shared }}
	p, err := Prepare(ctx, a, Request{ProfileID: g.ProfileID, OperationDigest: g.OperationDigest})
	if err != nil {
		t.Fatal(err)
	}
	r, err := p.Execute(ctx)
	if err != nil {
		t.Fatal(err)
	}
	shared.PlanDigest = repeated("9")
	shared.Files[0].SHA256 = repeated("9")
	if r.InstalledBoot.PlanDigest != g.InstalledBoot.PlanDigest || r.InstalledBoot.Files[0].SHA256 != repeated("3") {
		t.Fatal("returned receipt aliases authority-owned observations")
	}
}

// fak-test:runtime fast est=0.1s lane=default
func TestInstalledBootFiniteGrantScope(t *testing.T) {
	now := time.Now().UTC()
	for _, phase := range []string{"preflight", "stage-plan", "apply", "readback", "rollback"} {
		t.Run(phase, func(t *testing.T) {
			g := installedBootTestGrant(now)
			g.InstalledBoot.Phase = phase
			g.Operation = map[string]string{"preflight": OperationInstalledBootReadback, "stage-plan": OperationInstalledBootStagePlan,
				"apply": OperationInstalledBootApply, "readback": OperationInstalledBootReadback, "rollback": OperationInstalledBootRollback}[phase]
			if phase == "preflight" || phase == "readback" {
				g.InstalledBoot.WritePaths, g.InstalledBoot.BackupPrefix = nil, ""
			}
			if phase == "preflight" {
				g.InstalledBoot.StageDigest, g.InstalledBoot.PlanDigest, g.InstalledBoot.IdentitySHA256, g.InstalledBoot.ServiceIdentitySHA256 = "", "", "", ""
			}
			if phase == "stage-plan" {
				g.InstalledBoot.PlanDigest = ""
			}
			if err := validateInstalledBootGrant(g, now); err != nil {
				t.Fatal(err)
			}
		})
	}
	for name, mutate := range map[string]func(*Grant){
		"old-dropin-alias": func(g *Grant) { g.Operation = OperationSystemdProfileInstallReload },
		"changed-phase":    func(g *Grant) { g.InstalledBoot.Phase = "rollback" },
		"missing-plan":     func(g *Grant) { g.InstalledBoot.PlanDigest = "" },
		"missing-manifest": func(g *Grant) { g.InstalledBoot.ManifestSHA256 = "" },
		"missing-commit":   func(g *Grant) { g.InstalledBoot.PrivateCommit = "main" },
		"missing-pin":      func(g *Grant) { g.LeasePins = nil },
		"cross-kind":       func(g *Grant) { g.Reservation = &TrustedReservation{} },
		"systemd-scope":    func(g *Grant) { g.Units = []string{"fak.service"} },
		"changed-source":   func(g *Grant) { g.InstalledBoot.ArtifactSHA256 = repeated("9") },
		"broad-path":       func(g *Grant) { g.InstalledBoot.WritePaths = []string{"/boot/*"} },
		"wrong-node":       func(g *Grant) { g.InstalledBoot.RequiredNodeIDs = []string{"node-2"} },
		"stale-pin":        func(g *Grant) { g.LeasePins[0].ExpiresAt = now },
	} {
		t.Run(name, func(t *testing.T) {
			g := installedBootTestGrant(now)
			mutate(&g)
			if err := validateInstalledBootGrant(g, now); err == nil {
				t.Fatal("accepted invalid boot scope")
			}
		})
	}
	// An absent extension preserves existing canonical JSON and its old digests.
	raw, err := json.Marshal(validTestGrant(now))
	if err != nil || strings.Contains(string(raw), "installed_boot") {
		t.Fatalf("legacy JSON=%s err=%v", raw, err)
	}
}
