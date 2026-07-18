package main

// resume_watchdog_affinity_test.go — #4140 / #5189: the relaunch cache-affinity carry.
// Pins that a watchdog relaunch (a) threads the transcript-derived cache route onto the
// resumed child's env (RelaunchCacheAffinityEnv), and (b) records the route in the
// durable transcript-UUID-keyed affinity store next to the relaunch-reset store, so the
// warm provider cache route #4140 promised actually survives the OS relaunch.

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
	"github.com/anthony-chaudhary/fak/internal/resume"
)

// The broker attempt the live spawn path is granted from carries the derived cache
// route on the child env — and it SURVIVES the launch broker's secret-floor env
// sanitizer (newLaunchBrokerAttempt sanitizes; this asserts the post-sanitize map).
func TestResumeBrokerAttemptCarriesRelaunchCacheAffinityEnv(t *testing.T) {
	const sid = "4b66d1ba-b3dc-4f74-a8f5-d78ef67933c4"
	p := resume.WatchdogPlanRow{Session: sid, Account: ".claude-a", Project: "P"}
	a := rwResumeBrokerAttempt("fak", "claude", p, ".claude-a", nil)
	want := resume.RelaunchCacheAffinityKey(sid)
	if want == "" {
		t.Fatal("derived affinity key is empty for a non-blank session")
	}
	if got := a.Env[resume.RelaunchCacheAffinityEnv]; got != want {
		t.Fatalf("attempt env %s = %q, want the derived key %q", resume.RelaunchCacheAffinityEnv, got, want)
	}
	// A blank session derives no route and must set nothing.
	blank := rwResumeBrokerAttempt("fak", "claude", resume.WatchdogPlanRow{Account: ".claude-a"}, ".claude-a", nil)
	if got, ok := blank.Env[resume.RelaunchCacheAffinityEnv]; ok {
		t.Fatalf("blank-session attempt env carries %s=%q, want unset", resume.RelaunchCacheAffinityEnv, got)
	}
}

// #4140 end-to-end at the launch site: one LIVE relaunch hands the spawn a grant whose
// env carries the derived cache route, and appends a TS-stamped RelaunchAffinityRow to
// the durable store that the read/fold path (rwLoadRelaunchAffinity ->
// resume.FoldRelaunchAffinity) recovers — the cache-route sibling of
// TestRelaunchResetWiredAtLaunchSite.
func TestRelaunchAffinityWiredAtLaunchSite(t *testing.T) {
	rwHoldTestEnv(t)
	regDir := t.TempDir()
	logDir := t.TempDir()
	sid := "sid-relaunch-affinity-123"
	plan := `{"plan":[{"session":"` + sid + `","account":".claude-a","project":"P","disp":"STOPPED_MIDTOOL"}]}`
	if err := os.WriteFile(filepath.Join(regDir, "resume_plan.json"), []byte(plan), 0o644); err != nil {
		t.Fatal(err)
	}

	oldBroker := launchSpawnBroker
	oldSpawn := rwSpawnResumeLaunch
	var grantEnv map[string]string
	launchSpawnBroker = func(a launchBrokerAttempt) launchBrokerGrant { return allowLaunchBrokerGrant(a, "unit-test-allow") }
	rwSpawnResumeLaunch = func(claudeExe string, p resume.WatchdogPlanRow, resumeCfg, logDir string, grant launchBrokerGrant) (int, error) {
		grantEnv = grant.Env
		return 12345, nil
	}
	t.Cleanup(func() { launchSpawnBroker = oldBroker; rwSpawnResumeLaunch = oldSpawn })

	var out, errb bytes.Buffer
	rc := runResumeWatchdog(&out, &errb, []string{
		"--live", "--no-refresh", "--reg-dir", regDir, "--log-dir", logDir, "--spacing-sec", "0",
	})
	if rc != 0 {
		t.Fatalf("watchdog rc=%d stderr=%s stdout=%s", rc, errb.String(), out.String())
	}
	// Sanity: the row actually launched (else there is no carry to witness).
	if led, _ := os.ReadFile(filepath.Join(regDir, "resume_ledger.jsonl")); !strings.Contains(string(led), `"phase":"launched"`) {
		t.Fatalf("expected a launched ledger row:\n%s", led)
	}

	want := resume.RelaunchCacheAffinityKey(sid)
	// (a) The spawn's grant env — the authoritative child env on the live path — carries
	// the derived route.
	if got := grantEnv[resume.RelaunchCacheAffinityEnv]; got != want {
		t.Fatalf("spawned grant env %s = %q, want the derived key %q (env: %v)",
			resume.RelaunchCacheAffinityEnv, got, want, grantEnv)
	}
	// (b) The durable store next to the reset store holds the route, and the fold
	// recovers it last-row-wins.
	routes := rwLoadRelaunchAffinity(regDir)
	if routes[sid] != want {
		t.Fatalf("fold recovered route %q for %s, want %q (routes: %v)", routes[sid], sid, want, routes)
	}
	// (c) The shell stamped TS at the write site (the pure constructor leaves it "").
	raw, err := os.ReadFile(rwRelaunchAffinityLedger(regDir))
	if err != nil {
		t.Fatalf("affinity store unreadable: %v", err)
	}
	rows := jsonlledger.Parse[resume.RelaunchAffinityRow](string(raw), nil)
	if len(rows) == 0 {
		t.Fatalf("affinity store has no rows:\n%s", raw)
	}
	for _, row := range rows {
		if row.Session == sid && row.TS == "" {
			t.Fatalf("affinity row TS was not stamped at the write site: %+v", row)
		}
	}
}
