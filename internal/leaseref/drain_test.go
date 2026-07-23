package leaseref

import (
	"context"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// drainPushArgvs returns every `git push ...` argv the fake recorded, so a drain test can
// prove EXACTLY which delete-push(es) ran — and, for a dry-run, that none did.
func drainPushArgvs(f *fakeGit) [][]string {
	var out [][]string
	for _, c := range f.calls {
		if len(c) > 0 && c[0] == "push" {
			out = append(out, c)
		}
	}
	return out
}

// TestReportDescriptorDrainTargetsExpiredExcludesLive is the core report-first property: with
// one live (long TTL) and one expired descriptor, the plan names exactly the expired id and
// its delete refspec, counts the live one as excluded — and, decisively, DELETES NOTHING
// (both descriptors survive the call). Producing the plan is a pure read.
func TestReportDescriptorDrainTargetsExpiredExcludesLive(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	now := time.Now()
	if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: "liveD", Host: "n1", PCBState: "RUNNING", UpdatedAt: now.Unix(), TTLSecs: 3600}); err != nil {
		t.Fatalf("Publish live: %v", err)
	}
	if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: "deadD", Host: "n2", PCBState: "DRAINING", UpdatedAt: 100, TTLSecs: 10}); err != nil {
		t.Fatalf("Publish dead: %v", err)
	}

	plan, err := s.ReportDescriptorDrain(ctx(), "origin", now)
	if err != nil {
		t.Fatalf("ReportDescriptorDrain: %v", err)
	}
	if plan.Remote != "origin" {
		t.Fatalf("plan.Remote = %q, want origin", plan.Remote)
	}
	if len(plan.ExpiredIDs) != 1 || plan.ExpiredIDs[0] != "deadD" {
		t.Fatalf("plan.ExpiredIDs = %v, want [deadD]", plan.ExpiredIDs)
	}
	if len(plan.DeleteRefspecs) != 1 || plan.DeleteRefspecs[0] != ":refs/fak/locks/session-deadD" {
		t.Fatalf("plan.DeleteRefspecs = %v, want [:refs/fak/locks/session-deadD]", plan.DeleteRefspecs)
	}
	if plan.LiveExcluded != 1 {
		t.Fatalf("plan.LiveExcluded = %d, want 1 (the live descriptor)", plan.LiveExcluded)
	}

	// The load-bearing report-first guarantee: planning is READ-ONLY. Neither the expired
	// nor the live descriptor may be removed by producing the plan.
	if _, ok, _ := s.GetSession(ctx(), "deadD"); !ok {
		t.Fatalf("ReportDescriptorDrain deleted the expired descriptor — it must be read-only")
	}
	if _, ok, _ := s.GetSession(ctx(), "liveD"); !ok {
		t.Fatalf("ReportDescriptorDrain deleted the live descriptor — it must be read-only")
	}
}

// TestReportDescriptorDrainEmptyNamespace: with no descriptors published, the plan is empty
// with no error — the same "absence is a valid empty view" rule the readers hold.
func TestReportDescriptorDrainEmptyNamespace(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	plan, err := s.ReportDescriptorDrain(ctx(), "origin", time.Now())
	if err != nil {
		t.Fatalf("ReportDescriptorDrain empty: %v", err)
	}
	if len(plan.ExpiredIDs) != 0 || len(plan.DeleteRefspecs) != 0 || plan.LiveExcluded != 0 {
		t.Fatalf("plan over empty namespace = %+v, want all zero", plan)
	}
}

// TestReportDescriptorDrainRejectsInvalidRemote: a remote that cannot be one safe git argv
// token is refused up front (before any ref read), so a reported plan is always one a real
// push could run verbatim — argv hygiene identical to Sync's.
func TestReportDescriptorDrainRejectsInvalidRemote(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	for _, bad := range []string{"", "-origin", "a b"} {
		if _, err := s.ReportDescriptorDrain(ctx(), bad, time.Now()); err == nil {
			t.Fatalf("ReportDescriptorDrain(%q) = nil error, want a rejected-remote error", bad)
		}
	}
}

// TestReportDescriptorDrainIgnoresLockLeases is the namespace-split guarantee on the plan side:
// an expired LOCK LEASE sharing the same id stem as an expired descriptor must never appear in
// the drain plan — only session-<id> refs are drain targets, exactly as ReapSessions deletes
// only descriptors.
func TestReportDescriptorDrainIgnoresLockLeases(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	now := time.Now()
	if _, err := s.Acquire(ctx(), Record{ID: "shared", TreeGlobs: []string{"a"}, Holder: "h", AcquiredAt: 100, TTLSeconds: 10}); err != nil {
		t.Fatalf("Acquire expired lock lease: %v", err)
	}
	if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: "shared", Host: "n", PCBState: "RUNNING", UpdatedAt: 100, TTLSecs: 10}); err != nil {
		t.Fatalf("Publish expired descriptor: %v", err)
	}
	plan, err := s.ReportDescriptorDrain(ctx(), "origin", now)
	if err != nil {
		t.Fatalf("ReportDescriptorDrain: %v", err)
	}
	if len(plan.ExpiredIDs) != 1 || plan.ExpiredIDs[0] != "shared" {
		t.Fatalf("plan.ExpiredIDs = %v, want [shared] (the descriptor only, never the lock lease)", plan.ExpiredIDs)
	}
	if len(plan.DeleteRefspecs) != 1 || plan.DeleteRefspecs[0] != ":refs/fak/locks/session-shared" {
		t.Fatalf("plan.DeleteRefspecs = %v, want the session- ref only", plan.DeleteRefspecs)
	}
}

// TestConvergeDescriptorDrainDryRunPushesNothing is the dry-run-default guarantee that makes
// the drainer safe to schedule beside every other retention collector: apply=false computes
// the SAME plan ReportDescriptorDrain does and delete-pushes NOTHING — no `git push` argv is
// ever issued, both the live and the expired local refs survive, and the result reports the
// target set without acting on it.
func TestConvergeDescriptorDrainDryRunPushesNothing(t *testing.T) {
	g := newFakeGit()
	s := NewWithStdinRunner(g.run, g.runStdin, "")
	now := time.Now()
	mustPublish(t, s, SessionDescriptor{ID: "liveD", Host: "n1", PCBState: "RUNNING", UpdatedAt: now.Unix(), TTLSecs: 3600})
	mustPublish(t, s, SessionDescriptor{ID: "deadD", Host: "n2", PCBState: "DRAINING", UpdatedAt: 100, TTLSecs: 10})

	res, err := s.ConvergeDescriptorDrain(ctx(), "origin", now, false)
	if err != nil {
		t.Fatalf("dry-run drain: %v", err)
	}
	if res.Applied || res.Pushed != 0 || res.ReapedLocal != 0 {
		t.Fatalf("dry-run must not act, got %+v", res)
	}
	if len(res.ExpiredIDs) != 1 || res.ExpiredIDs[0] != "deadD" {
		t.Fatalf("dry-run plan ExpiredIDs = %v, want [deadD]", res.ExpiredIDs)
	}
	if pushes := drainPushArgvs(g); len(pushes) != 0 {
		t.Fatalf("dry-run issued a push: %v", pushes)
	}
	if _, ok, _ := s.GetSession(ctx(), "deadD"); !ok {
		t.Fatalf("dry-run reaped the expired descriptor locally — it must be read-only")
	}
	if _, ok, _ := s.GetSession(ctx(), "liveD"); !ok {
		t.Fatalf("dry-run reaped the live descriptor locally")
	}
}

// TestConvergeDescriptorDrainAppliesTargetedDeletePush is the convergence property: with
// apply=true the drain issues EXACTLY the one-sided delete-push :refs/fak/locks/session-deadD
// (the targeted alternative to a blanket prune) for the proven-expired id only, never the live
// one, then reaps that same id LOCALLY so a later glob sync push cannot resurrect it. The
// counts report one pushed, one reaped.
func TestConvergeDescriptorDrainAppliesTargetedDeletePush(t *testing.T) {
	g := newFakeGit()
	s := NewWithStdinRunner(g.run, g.runStdin, "")
	now := time.Now()
	mustPublish(t, s, SessionDescriptor{ID: "liveD", Host: "n1", PCBState: "RUNNING", UpdatedAt: now.Unix(), TTLSecs: 3600})
	mustPublish(t, s, SessionDescriptor{ID: "deadD", Host: "n2", PCBState: "DRAINING", UpdatedAt: 100, TTLSecs: 10})

	res, err := s.ConvergeDescriptorDrain(ctx(), "origin", now, true)
	if err != nil {
		t.Fatalf("apply drain: %v", err)
	}
	if !res.Applied || res.Pushed != 1 || res.ReapedLocal != 1 {
		t.Fatalf("apply outcome = %+v, want Applied Pushed=1 ReapedLocal=1", res)
	}
	pushes := drainPushArgvs(g)
	if len(pushes) != 1 {
		t.Fatalf("want exactly one delete-push, got %v", pushes)
	}
	if got := strings.Join(pushes[0], " "); got != "push origin :refs/fak/locks/session-deadD" {
		t.Fatalf("delete-push argv = %q, want the targeted expired-only refspec", got)
	}
	if _, ok, _ := s.GetSession(ctx(), "deadD"); ok {
		t.Fatalf("apply did not reap the expired descriptor locally")
	}
	if _, ok, _ := s.GetSession(ctx(), "liveD"); !ok {
		t.Fatalf("apply wrongly reaped the live descriptor")
	}
}

// TestConvergeDescriptorDrainPushFailureLeavesLocalForRetry proves the best-effort contract:
// when the origin delete-push FAILS, the drain reaps NOTHING locally and returns the error, so
// the still-on-origin descriptor stays in LiveSessions for a later run to retry rather than
// being stranded on origin with its local copy gone.
func TestConvergeDescriptorDrainPushFailureLeavesLocalForRetry(t *testing.T) {
	g := newFakeGit()
	// Wrap the fake so `push` exits non-zero (origin unreachable / rejected) while every other
	// verb still models real git — the store cannot tell the difference from a real push failure.
	pushFails := func(ctx context.Context, dir string, args ...string) (string, int, error) {
		if len(args) > 0 && args[0] == "push" {
			g.calls = append(g.calls, args)
			return "", 1, nil
		}
		return g.run(ctx, dir, args...)
	}
	s := NewWithStdinRunner(pushFails, g.runStdin, "")
	now := time.Now()
	mustPublish(t, s, SessionDescriptor{ID: "deadD", Host: "n2", PCBState: "DRAINING", UpdatedAt: 100, TTLSecs: 10})

	res, err := s.ConvergeDescriptorDrain(ctx(), "origin", now, true)
	if err == nil {
		t.Fatal("want an error when the delete-push fails")
	}
	if !res.Applied || res.Pushed != 0 || res.ReapedLocal != 0 {
		t.Fatalf("failed drain must push/reap nothing, got %+v", res)
	}
	if _, ok, _ := s.GetSession(ctx(), "deadD"); !ok {
		t.Fatalf("a failed push must NOT reap the local ref — it is the retry target")
	}
}

// TestConvergeDescriptorDrainDeletesExpiredOnOriginRealGit is the end-to-end origin-mutation
// witness on a real two-clone temp remote: an expired session descriptor pushed to origin is
// removed from origin AND reaped locally by an apply drain, the LIVE descriptor is untouched on
// both ends, a dry-run first proves it deletes nothing, and a second apply is a clean idempotent
// no-op. No real fleet remote is ever touched — the remote is a throwaway bare repo.
func TestConvergeDescriptorDrainDeletesExpiredOnOriginRealGit(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
	root := t.TempDir()
	remote := filepath.Join(root, "remote.git")
	local := filepath.Join(root, "local")
	runGit := func(dir string, args ...string) string {
		t.Helper()
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v in %s: %v\n%s", args, dir, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	runGit(root, "init", "--bare", remote)
	runGit(root, "clone", remote, local)
	runGit(local, "-c", "user.name=fak-test", "-c", "user.email=fak-test@example.invalid", "commit", "--allow-empty", "-m", "seed")
	runGit(local, "push", "origin", "HEAD:main")

	s := NewInDir(local)
	now := time.Now()
	mustPublish(t, s, SessionDescriptor{ID: "liveD", Host: "n1", PCBState: "RUNNING", UpdatedAt: now.Unix(), TTLSecs: 3600})
	mustPublish(t, s, SessionDescriptor{ID: "deadD", Host: "n2", PCBState: "DRAINING", UpdatedAt: 100, TTLSecs: 10})
	// Publish both descriptors to origin (the glob sync push), so the drain has an origin target.
	if _, err := s.Sync(ctx(), "origin", true, false); err != nil {
		t.Fatalf("seed push to origin: %v", err)
	}

	originHas := func(id string) bool {
		out := runGit(remote, "for-each-ref", "--format=%(refname)", refPrefix)
		for _, ln := range strings.Split(out, "\n") {
			if strings.TrimSpace(ln) == refPrefix+sessionPrefix+id {
				return true
			}
		}
		return false
	}
	if !originHas("liveD") || !originHas("deadD") {
		t.Fatalf("seed failed: origin live=%v dead=%v", originHas("liveD"), originHas("deadD"))
	}

	// DRY-RUN first: proves the default deletes nothing on origin.
	if dry, err := s.ConvergeDescriptorDrain(ctx(), "origin", now, false); err != nil {
		t.Fatalf("dry-run drain: %v", err)
	} else if dry.Applied || dry.Pushed != 0 {
		t.Fatalf("dry-run mutated origin: %+v", dry)
	}
	if !originHas("deadD") || !originHas("liveD") {
		t.Fatalf("dry-run deleted an origin ref (must be read-only)")
	}

	// APPLY: the expired descriptor leaves origin and the local store; the live one stays on both.
	got, err := s.ConvergeDescriptorDrain(ctx(), "origin", now, true)
	if err != nil {
		t.Fatalf("apply drain: %v", err)
	}
	if !got.Applied || got.Pushed != 1 || got.ReapedLocal != 1 {
		t.Fatalf("apply outcome = %+v, want Applied Pushed=1 ReapedLocal=1", got)
	}
	if originHas("deadD") {
		t.Fatalf("apply did not delete the expired descriptor on origin")
	}
	if !originHas("liveD") {
		t.Fatalf("apply wrongly deleted the LIVE descriptor on origin")
	}
	if _, ok, _ := s.GetSession(ctx(), "deadD"); ok {
		t.Fatalf("apply did not reap the expired descriptor locally")
	}
	if _, ok, _ := s.GetSession(ctx(), "liveD"); !ok {
		t.Fatalf("apply wrongly reaped the LIVE descriptor locally")
	}

	// IDEMPOTENT: a second apply finds nothing expired (local reaped) — a clean no-op.
	again, err := s.ConvergeDescriptorDrain(ctx(), "origin", now, true)
	if err != nil {
		t.Fatalf("second apply: %v", err)
	}
	if again.Applied || again.Pushed != 0 || len(again.ExpiredIDs) != 0 {
		t.Fatalf("second apply not idempotent: %+v", again)
	}
}

// mustPublish publishes a descriptor or fails the test — the shared seed helper for the drain
// convergence tests.
func mustPublish(t *testing.T, s *Store, d SessionDescriptor) {
	t.Helper()
	if _, err := s.PublishSession(ctx(), d); err != nil {
		t.Fatalf("PublishSession %s: %v", d.ID, err)
	}
}
