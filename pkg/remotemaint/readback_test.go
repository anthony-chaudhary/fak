package remotemaint

import (
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=0.1s lane=default
func TestReadbackReceiptBindsExactPassiveObservations(t *testing.T) {
	now := time.Now().UTC()
	g := validTestGrant(now)
	g.Operation = OperationSystemdProfileReadback
	g.PathPrefixes = nil
	b := SystemdProfileReadback{Hostname: "fixture-host", UnitGraph: "fak.service active\n", Files: make([]SystemdFileState, len(g.Paths))}
	for i, p := range g.Paths {
		b.Files[i] = SystemdFileState{Path: p, SHA256: strings.Repeat("a", 64), Mode: "0644"}
	}
	g.PriorStateSHA256 = ReadbackDigest(b)
	if err := validateGrant(g, now); err != nil {
		t.Fatal(err)
	}
	r := receiptFor(g, now)
	r.Outcome = OutcomeUnchanged
	r.BackupDisposition = BackupNotRequired
	r.Readback = &b
	if err := validateReceipt(r, g, now); err != nil {
		t.Fatal(err)
	}
	for _, edit := range []func(*Receipt){
		func(r *Receipt) { r.Outcome = OutcomeApplied },
		func(r *Receipt) { r.Readback.Files[0].Path = "/etc/systemd/system/other.service" },
		func(r *Receipt) { r.Readback.UnitGraph = "changed graph" },
		func(r *Receipt) { r.Readback.UnitGraph = strings.Repeat("x", (64<<10)+1) },
		func(r *Receipt) { r.Readback.Files[0].Mode = "8888" },
		func(r *Receipt) { r.Readback = nil },
	} {
		candidate := r
		copy := b
		copy.Files = append([]SystemdFileState(nil), b.Files...)
		candidate.Readback = &copy
		edit(&candidate)
		if err := validateReceipt(candidate, g, now); err == nil {
			t.Fatal("unbound passive observation admitted")
		}
	}
	install := validTestGrant(now)
	installReceipt := receiptFor(install, now)
	installReceipt.Readback = &b
	if err := validateReceipt(installReceipt, install, now); err == nil {
		t.Fatal("install accepted readback block")
	}
	g.PathPrefixes = []PathPrefix{{Directory: "/var/tmp", FilenamePrefix: "stage."}}
	if err := validateGrant(g, now); err == nil {
		t.Fatal("passive readback accepted temporary write prefix")
	}
}
