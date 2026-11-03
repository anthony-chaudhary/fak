package main

import (
	"context"
	"encoding/base64"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/guardedremotemaint"
	"github.com/anthony-chaudhary/fak/pkg/remotemaint"
)

// Cross-package loopback witness for debt lane guardedremotemaint: compiling
// pkg/guardedremotemaint into the production command package proves presence,
// not invokability. This drives NewAgentSession.Execute through the embedded
// gateway's real HTTP+MCP layers with an in-memory finite authority, so the
// facade is witnessed end-to-end from cmd/fak without any network or disk.

type loopbackMaintEntry struct {
	grant   remotemaint.Grant
	receipt remotemaint.Receipt
}

type loopbackMaintAuthority struct {
	mu         sync.Mutex
	entries    []loopbackMaintEntry
	executions []string
}

func newLoopbackMaintEntry(now time.Time, profile, digest string, paths ...string) loopbackMaintEntry {
	g := remotemaint.Grant{
		ProfileID: profile, OperationDigest: digest, ContextID: strings.Repeat("2", 64),
		Target: "service@maintenance.example.invalid", HostKey: "SHA256:" + base64.RawStdEncoding.EncodeToString(make([]byte, 32)),
		Operation: remotemaint.OperationSystemdProfileInstallReload, PayloadSHA256: strings.Repeat("3", 64),
		Holder: "loopback-agent", BootID: "boot-facade", PriorStateSHA256: strings.Repeat("4", 64),
		Paths:        paths,
		PathPrefixes: []remotemaint.PathPrefix{{Directory: "/var/tmp", FilenamePrefix: "maintenance-stage."}},
		Units:        []string{"fak.service"},
		LeasePins:    []remotemaint.LeasePin{{NodeID: "node-1", ID: "lease-1", Epoch: 1, ExpiresAt: now.Add(10 * time.Minute).UTC()}},
		ExpiresAt:    now.Add(5 * time.Minute).UTC(),
	}
	return loopbackMaintEntry{grant: g, receipt: remotemaint.Receipt{
		Schema: remotemaint.ReceiptSchema, GrantDigest: remotemaint.GrantDigest(g), ProfileID: g.ProfileID,
		OperationDigest: g.OperationDigest, ContextID: g.ContextID, Target: g.Target, HostKey: g.HostKey,
		Operation: g.Operation, BootID: g.BootID, StartedAt: now.Add(-time.Millisecond).UTC(),
		EndedAt: now.UTC(), Outcome: remotemaint.OutcomeApplied, Verified: true,
		BackupDisposition: remotemaint.BackupNotRequired,
	}}
}

func newLoopbackMaintAuthority() *loopbackMaintAuthority {
	now := time.Now().UTC()
	return &loopbackMaintAuthority{entries: []loopbackMaintEntry{
		// Authorized maintenance scope: the fixture shape pkg/guardedremotemaint's own tests use.
		newLoopbackMaintEntry(now, "facade-profile", strings.Repeat("1", 64), "/etc/systemd/system/fak.service"),
		// Same shape, but the exact path hits the facade's blocked-credential floor, so the
		// kernel adjudicates DENY before the authority may execute anything.
		newLoopbackMaintEntry(now, "blocked-profile", strings.Repeat("5", 64), "/root/.ssh/id_rsa"),
	}}
}

func (a *loopbackMaintAuthority) Resolve(_ context.Context, req remotemaint.Request) (remotemaint.Grant, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, entry := range a.entries {
		if req.ProfileID == entry.grant.ProfileID && req.OperationDigest == entry.grant.OperationDigest {
			return entry.grant, nil
		}
	}
	return remotemaint.Grant{}, remotemaint.ErrScope
}

func (a *loopbackMaintAuthority) Validate(context.Context, remotemaint.Grant) error { return nil }

func (a *loopbackMaintAuthority) Execute(_ context.Context, g remotemaint.Grant) (remotemaint.Receipt, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.executions = append(a.executions, g.Operation)
	for _, entry := range a.entries {
		if entry.grant.OperationDigest == g.OperationDigest {
			return entry.receipt, nil
		}
	}
	return remotemaint.Receipt{}, errors.New("loopback authority: unknown grant")
}

func (a *loopbackMaintAuthority) executionCount() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return len(a.executions)
}

func TestGuardedRemoteMaintenanceFacadeLoopback(t *testing.T) {
	auth := newLoopbackMaintAuthority()
	s, err := guardedremotemaint.NewAgentSession(guardedremotemaint.HostConfig{Authority: auth, AgentID: "loopback-agent"})
	if err != nil {
		t.Fatalf("NewAgentSession: %v", err)
	}
	defer s.Close()

	out, err := s.Execute(context.Background(), remotemaint.Request{ProfileID: "facade-profile", OperationDigest: strings.Repeat("1", 64)})
	if err != nil {
		t.Fatalf("authorized Execute: err=%v outcome=%+v", err, out)
	}
	if !out.Authorized || out.Verdict != "ALLOW" || out.Effect != "completed" || out.Receipt == nil || !out.Receipt.Verified || out.TraceID == "" {
		t.Fatalf("authorized Execute: %+v", out)
	}

	// Adjudication-level denial: the kernel refuses the blocked-credential scope
	// before execution, so the facade reports a clean not-attempted denial.
	denied, deniedErr := s.Execute(context.Background(), remotemaint.Request{ProfileID: "blocked-profile", OperationDigest: strings.Repeat("5", 64)})
	if !errors.Is(deniedErr, remotemaint.ErrAuthority) || denied.Authorized || denied.Effect != "not_attempted" {
		t.Fatalf("blocked-scope Execute: err=%v outcome=%+v", deniedErr, denied)
	}
	// Reason/DecidedBy are pinned because they are code constants of the
	// credential-block rung (internal/adjudicator/decide_rungs.go) and were
	// observed identical across three consecutive runs.
	if denied.Reason != "POLICY_BLOCK" || denied.DecidedBy != "monitor/credential-block" {
		t.Fatalf("blocked-scope verdict drifted: Reason=%q DecidedBy=%q", denied.Reason, denied.DecidedBy)
	}
	if denied.TraceID != out.TraceID {
		t.Fatalf("one session must keep one trace namespace: allow=%q deny=%q", out.TraceID, denied.TraceID)
	}

	// Authority scope mismatch: the host authority cannot resolve the request, so
	// the pre-execution failure surfaces as an unbound-effect error rather than a
	// structured denial. The facade must still refuse authorization and record
	// no effect, even though it cannot prove the absence of one.
	scopeMismatch, scopeErr := s.Execute(context.Background(), remotemaint.Request{ProfileID: "unknown-profile", OperationDigest: strings.Repeat("9", 64)})
	if !errors.Is(scopeErr, remotemaint.ErrUnknownEffect) || scopeMismatch.Authorized || scopeMismatch.Effect != "attempted_unknown" {
		t.Fatalf("scope-mismatch Execute: err=%v outcome=%+v", scopeErr, scopeMismatch)
	}

	if n := auth.executionCount(); n != 1 {
		t.Fatalf("authority executions=%d, want 1 (both refusals must never reach Execute)", n)
	}
}
