package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/resume"
)

// #3582 — shell half of the post-reset backlog SLO gate: the roster read that arms it, and
// the dedup discipline that keeps a standing page from becoming per-tick spam.

func writeRoster(t *testing.T, regDir, body string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(regDir, "sessions.json"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestThrottledAccountsCountsUsageHeldSeats(t *testing.T) {
	cases := []struct {
		name      string
		body      string
		wantCount int
		wantOK    bool
	}{
		{"none throttled", `{"accounts":[{"account":"a"},{"account":"b"}]}`, 0, true},
		{"one throttled", `{"accounts":[{"account":"a","throttled":true},{"account":"b"}]}`, 1, true},
		{"usage block_kind counts too", `{"accounts":[{"account":"a","block_kind":"usage"}]}`, 1, true},
		{"auth block is not a throttle", `{"accounts":[{"account":"a","block_kind":"auth"}]}`, 0, true},
		// Fail closed: neither of these proves an unthrottled fleet.
		{"empty roster is unknown", `{"accounts":[]}`, 0, false},
		{"malformed roster is unknown", `{oops`, 0, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			writeRoster(t, dir, tc.body)
			got, ok := rwThrottledAccounts(dir)
			if got != tc.wantCount || ok != tc.wantOK {
				t.Errorf("rwThrottledAccounts = (%d, %v), want (%d, %v)", got, ok, tc.wantCount, tc.wantOK)
			}
		})
	}
}

// A missing roster file is unknown — the gate must stay silent rather than page on absence.
func TestThrottledAccountsMissingRosterIsUnknown(t *testing.T) {
	if got, ok := rwThrottledAccounts(t.TempDir()); ok {
		t.Errorf("missing roster read as known (%d); want unknown so the gate fails closed", got)
	}
}

// Acceptance 3: a gate that stays tripped notifies ONCE and thereafter refreshes an
// occurrence-counted record under the same signature — no per-tick spam.
func TestEmitBacklogPageNotifiesOnceThenRefreshes(t *testing.T) {
	regDir, logDir := t.TempDir(), t.TempDir()
	page := &resume.WatchdogPage{
		Reason:    resume.WatchdogPageBacklogPersists,
		Signature: resume.WatchdogPageBacklogPersists + ":threshold=20",
		Depth:     25, Threshold: 20, Ticks: 3,
		Detail: "RESUME BACKLOG PERSISTS AFTER RESET: depth 25",
	}
	notes := 0
	note := func(string, ...any) { notes++ }

	if fresh := rwEmitBacklogPage(regDir, logDir, page, note); !fresh {
		t.Fatal("first occurrence must raise a NEW page")
	}
	// Later ticks: deeper backlog, same gate — must not re-notify.
	for i := 0; i < 4; i++ {
		page.Depth += 3
		if fresh := rwEmitBacklogPage(regDir, logDir, page, note); fresh {
			t.Fatalf("tick %d raised a second page; want a refresh of the deduped one", i+2)
		}
	}

	raw, err := os.ReadFile(filepath.Join(regDir, rwBacklogPageStore))
	if err != nil {
		t.Fatalf("dedup store not written: %v", err)
	}
	var store map[string]rwBacklogPageRecord
	if err := json.Unmarshal(raw, &store); err != nil {
		t.Fatal(err)
	}
	if len(store) != 1 {
		t.Fatalf("store holds %d records, want exactly 1 deduped signature: %v", len(store), store)
	}
	rec := store[page.Signature]
	if rec.Count != 5 {
		t.Errorf("occurrence count = %d, want 5 (one per tick)", rec.Count)
	}
	if rec.FirstSeen == "" || rec.LastSeen == "" || rec.FirstSeen == rec.LastSeen && rec.Count > 1 {
		t.Errorf("first/last seen not tracked across ticks: %+v", rec)
	}
	if rec.LastDepth != page.Depth {
		t.Errorf("last_depth = %d, want the refreshed %d", rec.LastDepth, page.Depth)
	}

	// Exactly one operator notification landed, despite five firings.
	toasts, err := os.ReadFile(filepath.Join(logDir, "notifications.log"))
	if err != nil {
		t.Fatalf("no notification recorded: %v", err)
	}
	if n := strings.Count(string(toasts), resume.WatchdogPageBacklogPersists); n > 1 {
		t.Errorf("%d notifications for one standing page; want 1", n)
	}
	if !strings.Contains(string(toasts), "Resume backlog persists") {
		t.Errorf("notification missing the page headline: %s", toasts)
	}
}

// A nil page (gate not tripped) is a no-op: no store, no notification.
func TestEmitBacklogPageNilIsNoop(t *testing.T) {
	regDir, logDir := t.TempDir(), t.TempDir()
	if rwEmitBacklogPage(regDir, logDir, nil, func(string, ...any) {}) {
		t.Error("nil page raised a page")
	}
	if _, err := os.Stat(filepath.Join(regDir, rwBacklogPageStore)); !os.IsNotExist(err) {
		t.Error("nil page wrote a dedup store")
	}
	if _, err := os.Stat(filepath.Join(logDir, "notifications.log")); !os.IsNotExist(err) {
		t.Error("nil page raised a notification")
	}
}
