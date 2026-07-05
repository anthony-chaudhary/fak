package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// mkFailoverHome writes a config home under root with an optional login (.claude.json oauthAccount)
// and an optional .credentials.json carrying a live OAuth access token expiring at expiresAtMs
// (0 => no .credentials.json at all). It returns the discovered Home so tests exercise the same
// Identity/LoginStatus path production does.
func mkFailoverHome(t *testing.T, root, dir, email, uuid, accessToken string, expiresAtMs int64) accounts.Home {
	t.Helper()
	full := filepath.Join(root, dir)
	if err := os.MkdirAll(filepath.Join(full, "projects"), 0o755); err != nil {
		t.Fatal(err)
	}
	if email != "" || uuid != "" {
		body := fmt.Sprintf(`{"oauthAccount":{"emailAddress":%q,"accountUuid":%q}}`, email, uuid)
		if err := os.WriteFile(filepath.Join(full, ".claude.json"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if expiresAtMs != 0 {
		body := fmt.Sprintf(`{"claudeAiOauth":{"accessToken":%q,"expiresAt":%d}}`, accessToken, expiresAtMs)
		if err := os.WriteFile(filepath.Join(full, ".credentials.json"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return accounts.Home{Name: dir, Dir: full, Identity: accounts.DeriveIdentity(full)}
}

func TestReadLiveAccessToken(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	root := t.TempDir()

	live := mkFailoverHome(t, root, ".claude-live", "a@x.test", "u-a", "sk-ant-oat01-live", now.Add(time.Hour).UnixMilli())
	if tok, ok := readLiveAccessToken(live.Dir, now); !ok || tok != "sk-ant-oat01-live" {
		t.Fatalf("live token: got (%q,%v), want (sk-ant-oat01-live,true)", tok, ok)
	}

	expired := mkFailoverHome(t, root, ".claude-expired", "b@x.test", "u-b", "sk-ant-oat01-old", now.Add(-time.Minute).UnixMilli())
	if _, ok := readLiveAccessToken(expired.Dir, now); ok {
		t.Fatal("an expired credential must not be usable")
	}

	nocreds := mkFailoverHome(t, root, ".claude-nocreds", "c@x.test", "u-c", "", 0)
	if _, ok := readLiveAccessToken(nocreds.Dir, now); ok {
		t.Fatal("a home with no .credentials.json must not yield a token")
	}
}

func TestPickFailoverAccount(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	future := now.Add(time.Hour).UnixMilli()

	t.Run("skips the walled account, picks a permitted live sibling", func(t *testing.T) {
		root := t.TempDir()
		walled := mkFailoverHome(t, root, ".claude-walled", "walled@x.test", "u-walled", "sk-ant-oat01-walled", future)
		good := mkFailoverHome(t, root, ".claude-good", "good@x.test", "u-good", "sk-ant-oat01-good", future)
		homes := []accounts.Home{walled, good}

		walledSet := map[string]bool{walled.Identity.AccountKey(): true}
		dir, tok, ok := pickFailoverAccount(homes, walledSet, now)
		if !ok {
			t.Fatal("expected a permitted sibling to be picked")
		}
		if dir != good.Dir || tok != "sk-ant-oat01-good" {
			t.Fatalf("picked (%q,%q), want the good sibling (%q,sk-ant-oat01-good)", dir, tok, good.Dir)
		}
	})

	t.Run("no target when every non-walled sibling has a dead or missing token", func(t *testing.T) {
		root := t.TempDir()
		walled := mkFailoverHome(t, root, ".claude-walled", "walled@x.test", "u-walled", "sk-ant-oat01-walled", future)
		expired := mkFailoverHome(t, root, ".claude-expired", "exp@x.test", "u-exp", "sk-ant-oat01-exp", now.Add(-time.Minute).UnixMilli())
		nocreds := mkFailoverHome(t, root, ".claude-nocreds", "nc@x.test", "u-nc", "", 0)
		homes := []accounts.Home{walled, expired, nocreds}

		walledSet := map[string]bool{walled.Identity.AccountKey(): true}
		if _, _, ok := pickFailoverAccount(homes, walledSet, now); ok {
			t.Fatal("no live permitted sibling exists — failover must report no target")
		}
	})

	t.Run("a same-account second dir is treated as walled (keyed on account, not dir)", func(t *testing.T) {
		root := t.TempDir()
		// Two dirs, same accountUuid: walling one must wall both, so neither is picked.
		a := mkFailoverHome(t, root, ".claude-a", "same@x.test", "u-same", "sk-ant-oat01-a", future)
		b := mkFailoverHome(t, root, ".claude-b", "same@x.test", "u-same", "sk-ant-oat01-b", future)
		other := mkFailoverHome(t, root, ".claude-other", "other@x.test", "u-other", "sk-ant-oat01-other", future)
		homes := []accounts.Home{a, b, other}

		walledSet := map[string]bool{a.Identity.AccountKey(): true}
		dir, _, ok := pickFailoverAccount(homes, walledSet, now)
		if !ok || dir != other.Dir {
			t.Fatalf("picked (%q,%v), want the distinct account %q (both same-account dirs are walled)", dir, ok, other.Dir)
		}
	})
}

// TestAccountFailoverStickyAdvance proves the failover closure marks the current account walled,
// adopts a permitted sibling, and advances currentConfigDir so the sticky per-request token source
// follows the swap across turns.
func TestAccountFailoverStickyAdvance(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	future := now.Add(time.Hour).UnixMilli()
	root := t.TempDir()
	walled := mkFailoverHome(t, root, ".claude-walled", "walled@x.test", "u-walled", "sk-ant-oat01-walled", future)
	good := mkFailoverHome(t, root, ".claude-good", "good@x.test", "u-good", "sk-ant-oat01-good", future)
	_ = good

	af := newAccountFailover(root, walled.Dir, func() time.Time { return now })
	if got := af.currentConfigDir(); got != walled.Dir {
		t.Fatalf("initial currentConfigDir = %q, want the pinned walled dir %q", got, walled.Dir)
	}

	tok, ok := af.failover("failover_account")
	if !ok || tok != "sk-ant-oat01-good" {
		t.Fatalf("failover returned (%q,%v), want (sk-ant-oat01-good,true)", tok, ok)
	}
	if got := af.currentConfigDir(); got != filepath.Join(root, ".claude-good") {
		t.Fatalf("after failover currentConfigDir = %q, want the adopted good dir", got)
	}
	// The walled account is now sticky-excluded: a second failover with no OTHER permitted sibling
	// must report no target rather than bounce back to the walled account.
	if _, ok := af.failover("failover_account"); ok {
		t.Fatal("a second failover with no further permitted sibling must report no target")
	}
}

// TestForceRehomeAdoptsNextAvailable proves the operator "switch seat now" verb: the current seat
// is left (recorded moved, NOT walled — it was not proven bad), the next available sibling is
// adopted so currentConfigDir advances, and the from/to display metadata names both seats.
func TestForceRehomeAdoptsNextAvailable(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	future := now.Add(time.Hour).UnixMilli()
	root := t.TempDir()
	a := mkFailoverHome(t, root, ".claude-a", "a@x.test", "u-a", "sk-ant-oat01-a", future)
	b := mkFailoverHome(t, root, ".claude-b", "b@x.test", "u-b", "sk-ant-oat01-b", future)

	af := newAccountFailover(root, a.Dir, func() time.Time { return now })
	res, err := af.forceRehome("")
	if err != nil {
		t.Fatalf("forceRehome: %v", err)
	}
	// Names come from accounts.Discover (the ".claude-" prefix stripped), not the raw dir.
	if res.From != "a" || res.FromEmail != "a@x.test" || res.To != "b" || res.ToEmail != "b@x.test" {
		t.Fatalf("rehome metadata = %+v, want a(a@x.test) -> b(b@x.test)", res)
	}
	if res.Reason != "operator_rehome" {
		t.Fatalf("empty reason defaulted to %q, want operator_rehome", res.Reason)
	}
	if got := af.currentConfigDir(); got != b.Dir {
		t.Fatalf("currentConfigDir after rehome = %q, want the adopted %q", got, b.Dir)
	}
	// The seat moved off is excluded from reselection but NOT reported walled: it was
	// deliberately left, not proven bad — the status area must not claim otherwise.
	if walled := af.walledKeys(); walled != nil {
		t.Fatalf("walledKeys after an operator rehome = %v, want nil (moved is not walled)", walled)
	}
	// A later swap (operator or automatic) must never bounce back to the abandoned seat:
	// with no third seat, both the operator verb and the 403 failover report no target.
	if _, err := af.forceRehome("again"); err == nil {
		t.Fatal("second forceRehome with no third seat must refuse")
	}
	if got := af.currentConfigDir(); got != b.Dir {
		t.Fatalf("a refused rehome moved currentConfigDir to %q, want it untouched at %q", got, b.Dir)
	}
	if _, ok := af.failover("failover_account"); ok {
		t.Fatal("automatic failover after an operator rehome must not re-adopt the moved-off seat")
	}
}

// TestForceRehomeNoSiblingLeavesStateUntouched proves a refused swap is side-effect free: the
// current seat stays in force and is not poisoned for a later automatic failover.
func TestForceRehomeNoSiblingLeavesStateUntouched(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	root := t.TempDir()
	only := mkFailoverHome(t, root, ".claude-only", "only@x.test", "u-only", "sk-ant-oat01-only", now.Add(time.Hour).UnixMilli())

	af := newAccountFailover(root, only.Dir, func() time.Time { return now })
	if _, err := af.forceRehome("operator_rehome"); err == nil {
		t.Fatal("forceRehome with a single-seat roster must refuse")
	}
	if got := af.currentConfigDir(); got != only.Dir {
		t.Fatalf("refused rehome advanced currentConfigDir to %q, want the pinned %q", got, only.Dir)
	}
	af.mu.Lock()
	movedLen := len(af.moved)
	af.mu.Unlock()
	if movedLen != 0 {
		t.Fatalf("refused rehome recorded %d moved seat(s), want 0 (state untouched)", movedLen)
	}
}

// sanity: the fixture writes a parseable credentials doc (guards against a fmt drift silently
// making every readLiveAccessToken fail, which would make the tests above vacuously pass on the
// "no token" branch).
func TestFailoverFixtureIsParseable(t *testing.T) {
	root := t.TempDir()
	h := mkFailoverHome(t, root, ".claude-x", "x@x.test", "u-x", "sk-ant-oat01-x", time.Now().Add(time.Hour).UnixMilli())
	b, err := os.ReadFile(filepath.Join(h.Dir, ".credentials.json"))
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		ClaudeAIOauth struct {
			AccessToken string `json:"accessToken"`
		} `json:"claudeAiOauth"`
	}
	if json.Unmarshal(b, &doc) != nil || doc.ClaudeAIOauth.AccessToken != "sk-ant-oat01-x" {
		t.Fatalf("fixture credentials.json did not round-trip: %s", b)
	}
}
