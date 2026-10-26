package guardedremotemaint

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/remotemaint"
)

type reservationFlowAuthority struct {
	mu          sync.Mutex
	reservation remotemaint.Grant
	install     remotemaint.Grant
	readback    remotemaint.Grant
	release     remotemaint.Grant
	egress      remotemaint.Grant
	observation remotemaint.SystemdProfileReadback
	ownedPins   []remotemaint.LeasePin
	executions  []string
}

func (a *reservationFlowAuthority) Resolve(_ context.Context, req remotemaint.Request) (remotemaint.Grant, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	switch req.ProfileID {
	case a.reservation.ProfileID:
		return a.reservation, nil
	case a.install.ProfileID:
		g := a.install
		g.LeasePins = append([]remotemaint.LeasePin(nil), a.ownedPins...)
		return g, nil
	case a.readback.ProfileID:
		g := a.readback
		g.LeasePins = append([]remotemaint.LeasePin(nil), a.ownedPins...)
		return g, nil
	case a.release.ProfileID:
		g := a.release
		g.LeasePins = append([]remotemaint.LeasePin(nil), a.ownedPins...)
		return g, nil
	case a.egress.ProfileID:
		g := a.egress
		g.LeasePins = append([]remotemaint.LeasePin(nil), a.ownedPins...)
		return g, nil
	default:
		return remotemaint.Grant{}, remotemaint.ErrScope
	}
}

func (a *reservationFlowAuthority) Validate(context.Context, remotemaint.Grant) error { return nil }

func (a *reservationFlowAuthority) Execute(_ context.Context, g remotemaint.Grant) (remotemaint.Receipt, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.executions = append(a.executions, g.Operation)
	now := time.Now().UTC()
	r := facadeReceipt(g, now)
	if g.Operation == remotemaint.OperationMaintainLeaseAcquire {
		if err := os.WriteFile(filepath.Join(g.Reservation.WorkspaceRoot, ".fak", "nvlease", "leases.json"), []byte(`{"reserved":true}`), 0o600); err != nil {
			return remotemaint.Receipt{}, err
		}
		a.ownedPins = []remotemaint.LeasePin{{NodeID: g.Reservation.NodeIDs[0], ID: "native-flow-lease", Epoch: g.Reservation.ExpectedStoreEpoch + 1, ExpiresAt: now.Add(time.Duration(g.Reservation.TTLSeconds) * time.Second)}}
		r.ReservationLeases = append([]remotemaint.LeasePin(nil), a.ownedPins...)
	} else if g.Operation == remotemaint.OperationSystemdProfileReadback {
		r.Outcome = remotemaint.OutcomeUnchanged
		observation := a.observation
		observation.Files = append([]remotemaint.SystemdFileState(nil), a.observation.Files...)
		r.Readback = &observation
	} else if g.Operation == remotemaint.OperationMaintainLeaseRelease {
		if err := os.Remove(filepath.Join(g.Reservation.WorkspaceRoot, ".fak", "nvlease", "leases.json")); err != nil {
			return remotemaint.Receipt{}, err
		}
		r.ReservationLeases = append([]remotemaint.LeasePin(nil), g.LeasePins...)
	}
	return r, nil
}

// fak-test:runtime medium est=3s lane=default
func TestReservationControlReceiptDoesNotTaintFollowingFiniteInstall(t *testing.T) {
	root := t.TempDir()
	authority := filepath.Join(root, ".fak", "nvlease")
	if err := os.MkdirAll(authority, 0o700); err != nil {
		t.Fatal(err)
	}
	if raw, err := exec.Command("git", "init", "--quiet", root).CombinedOutput(); err != nil {
		t.Fatalf("git fixture: %s: %v", raw, err)
	}
	if err := os.WriteFile(filepath.Join(root, "dos-fixture.json"), []byte("[]"), 0o600); err != nil {
		t.Fatal(err)
	}
	bin := t.TempDir()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	rawExecutable, err := os.ReadFile(executable)
	if err != nil {
		t.Fatal(err)
	}
	dos := filepath.Join(bin, "dos")
	if strings.HasSuffix(executable, ".exe") {
		dos += ".exe"
	}
	if err := os.WriteFile(dos, rawExecutable, 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))

	now := time.Now().UTC()
	reservation := remotemaint.Grant{
		ProfileID: "reservation-control-flow", OperationDigest: strings.Repeat("1", 64), ContextID: strings.Repeat("2", 64),
		Operation: remotemaint.OperationMaintainLeaseAcquire, Holder: "flow-holder", ExpiresAt: now.Add(time.Minute),
		Reservation: &remotemaint.TrustedReservation{WorkspaceRoot: root, NodeIDs: []string{"node-one"}, Kind: "maintain", TTLSeconds: 7200, CatalogGeneration: "catalog-flow"},
		Paths:       []string{authority, filepath.Join(authority, "leases.json"), filepath.Join(authority, "leases.json.lock"), filepath.Join(authority, "lease-history.jsonl")},
	}
	sort.Strings(reservation.Paths)
	install := facadeGrant(now)
	install.ProfileID, install.OperationDigest, install.ContextID, install.Holder = "install-after-reservation", strings.Repeat("3", 64), strings.Repeat("4", 64), "flow-holder"
	readback := facadeGrant(now)
	readback.ProfileID, readback.OperationDigest, readback.ContextID, readback.Holder = "readback-after-install", strings.Repeat("5", 64), strings.Repeat("6", 64), "flow-holder"
	readback.Operation, readback.PathPrefixes = remotemaint.OperationSystemdProfileReadback, nil
	observation := remotemaint.SystemdProfileReadback{Hostname: "fixture-host", UnitGraph: "fak.service\nActiveState=active\n"}
	for _, path := range readback.Paths {
		observation.Files = append(observation.Files, remotemaint.SystemdFileState{Path: path, SHA256: strings.Repeat("a", 64), Mode: "0644"})
	}
	readback.PriorStateSHA256 = remotemaint.ReadbackDigest(observation)
	release := reservation
	release.Reservation = &remotemaint.TrustedReservation{
		WorkspaceRoot: reservation.Reservation.WorkspaceRoot, NodeIDs: append([]string(nil), reservation.Reservation.NodeIDs...),
		Kind: reservation.Reservation.Kind, CatalogGeneration: reservation.Reservation.CatalogGeneration,
	}
	release.ProfileID, release.OperationDigest, release.ContextID = "release-after-readback", strings.Repeat("7", 64), strings.Repeat("8", 64)
	release.Operation, release.Reservation.TTLSeconds, release.Reservation.ExpectedStoreEpoch = remotemaint.OperationMaintainLeaseRelease, 0, 1
	egress := facadeGrant(now)
	egress.ProfileID, egress.OperationDigest, egress.ContextID, egress.Holder = "external-after-readback", strings.Repeat("9", 64), strings.Repeat("a", 64), "flow-holder"
	a := &reservationFlowAuthority{reservation: reservation, install: install, readback: readback, release: release, egress: egress, observation: observation}
	s, err := NewAgentSession(HostConfig{Authority: a, AgentID: "reservation-flow-agent"})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	reserved, err := s.Execute(context.Background(), remotemaint.Request{ProfileID: reservation.ProfileID, OperationDigest: reservation.OperationDigest})
	if err != nil || !reserved.Authorized || reserved.Effect != "completed" || reserved.Receipt == nil || !reserved.Receipt.Verified || len(reserved.Receipt.ReservationLeases) != 1 {
		t.Fatalf("reservation=%+v err=%v", reserved, err)
	}
	installed, err := s.Execute(context.Background(), remotemaint.Request{ProfileID: install.ProfileID, OperationDigest: install.OperationDigest})
	if err != nil || !installed.Authorized || installed.Effect != "completed" || installed.Receipt == nil || !installed.Receipt.Verified {
		t.Fatalf("install after native control receipt=%+v err=%v", installed, err)
	}
	observed, err := s.Execute(context.Background(), remotemaint.Request{ProfileID: readback.ProfileID, OperationDigest: readback.OperationDigest})
	if err != nil || !observed.Authorized || observed.Effect != "completed" || observed.Receipt == nil || !observed.Receipt.Verified || observed.Receipt.Readback == nil {
		t.Fatalf("readback after install=%+v err=%v", observed, err)
	}
	released, err := s.Execute(context.Background(), remotemaint.Request{ProfileID: release.ProfileID, OperationDigest: release.OperationDigest})
	if err != nil || !released.Authorized || released.Effect != "completed" || released.Receipt == nil || !released.Receipt.Verified || !reflect.DeepEqual(released.Receipt.ReservationLeases, reserved.Receipt.ReservationLeases) {
		t.Fatalf("release after readback=%+v err=%v", released, err)
	}
	denied, err := s.Execute(context.Background(), remotemaint.Request{ProfileID: egress.ProfileID, OperationDigest: egress.OperationDigest})
	if err == nil || denied.Authorized || denied.Effect != "not_attempted" || denied.Reason != "TRUST_VIOLATION" || denied.DecidedBy != "ifc-sink" {
		t.Fatalf("external egress after readback=%+v err=%v", denied, err)
	}
	if reserved.TraceID == "" || installed.TraceID != reserved.TraceID || observed.TraceID != reserved.TraceID || released.TraceID != reserved.TraceID || denied.TraceID != reserved.TraceID || installed.Receipt.Target != install.Target ||
		!reflect.DeepEqual(a.executions, []string{remotemaint.OperationMaintainLeaseAcquire, remotemaint.OperationSystemdProfileInstallReload, remotemaint.OperationSystemdProfileReadback, remotemaint.OperationMaintainLeaseRelease}) {
		t.Fatalf("trace reservation=%q install=%q receipt=%+v executions=%v", reserved.TraceID, installed.TraceID, installed.Receipt, a.executions)
	}
}
