package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// bucketRegistry stages the exact papercut #4669 reports: two seats resolve to ONE account
// bucket because july6-netra was identity_mismatched onto july11's identity (its on-disk
// .claude.json names july11@…, so both seats share july11's UUID). anchor is a live seat on a
// DIFFERENT account, set as the rehome fall-forward. It returns the temp home + registry path,
// with the roster views pinned under the temp home so a sync can never touch a live roster.
func bucketRegistry(t *testing.T) (home, regPath string) {
	t.Helper()
	t.Setenv("FAK_JOB_ROSTER", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	home = t.TempDir()
	// Both seats' on-disk identity is july11@… → same UUID → same account bucket. july6-netra is
	// the name-lie duplicate (named july6, identity july11) — exactly the seat the issue says a
	// per-seat `remove july11-netra` leaves live.
	july11 := mkHome(t, home, ".claude-july11-netra", "july11@netrasystems.ai", true)
	july6 := mkHome(t, home, ".claude-july6-netra", "july11@netrasystems.ai", true)
	anchor := mkHome(t, home, ".claude-anchor-seat", "anchor@netra.test", true)

	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"july11-netra","dir":"` + jsonPath(july11) + `"},` +
		`{"name":"july6-netra","dir":"` + jsonPath(july6) + `"},` +
		`{"name":"anchor-seat","dir":"` + jsonPath(anchor) + `","default":true}` +
		`]}`
	regPath = filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}
	return home, regPath
}

// activeByName returns whether the named seat is present AND active in the (refreshed) registry.
func activeByName(reg accounts.Registry, name string) (active, present bool) {
	for _, h := range reg.Homes {
		if h.Name == name {
			return h.Active(), true
		}
	}
	return false, false
}

// TestRemoveByAccountTombstonesEverySeat is the core #4669 acceptance: `remove --by-account`
// retires BOTH seats resolving to the account bucket in one command, prints the full set before
// acting, and leaves no active seat on the retired account (both consumer reconcile clean).
func TestRemoveByAccountTombstonesEverySeat(t *testing.T) {
	home, regPath := bucketRegistry(t)

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"remove", "--by-account", "july11@netrasystems.ai", "--rehome-to", "anchor-seat",
		"--registry", regPath, "--home", home,
	})
	if rc != 0 {
		t.Fatalf("remove --by-account rc=%d stderr=%s", rc, errb.String())
	}

	// The full set is named BEFORE acting, so the operator sees every seat one command touches.
	got := out.String()
	if !strings.Contains(got, "retiring account") ||
		!strings.Contains(got, "july11-netra") || !strings.Contains(got, "july6-netra") {
		t.Fatalf("by-account should print the full seat set before acting:\n%s", got)
	}

	reg2, err := accounts.LoadRegistry(regPath)
	if err != nil {
		t.Fatalf("registry should still validate after by-account remove: %v", err)
	}
	live := reg2.Refresh()
	for _, name := range []string{"july11-netra", "july6-netra"} {
		active, present := activeByName(live, name)
		if !present {
			t.Fatalf("seat %q vanished from the registry", name)
		}
		if active {
			t.Fatalf("seat %q should be tombstoned after --by-account, still active", name)
		}
	}
	if active, _ := activeByName(live, "anchor-seat"); !active {
		t.Fatalf("rehome anchor must stay active")
	}

	// Reconcile-clean witness: no ACTIVE seat still resolves to the retired july11 bucket.
	bucket := accounts.UUIDBucketKey("u-july11@netrasystems.ai")
	for _, h := range live.Homes {
		if h.Active() && h.Identity.AccountKey() == bucket {
			t.Fatalf("account %s is still reachable via active seat %q after retirement", bucket, h.Name)
		}
	}
}

// TestRemoveByAccountRefusesRehomeIntoRetired pins the structured refusal: rehoming into a seat
// that itself resolves to the account being retired would leave the account live, so it is
// refused (non-zero, named reason) and the registry is left untouched.
func TestRemoveByAccountRefusesRehomeIntoRetired(t *testing.T) {
	home, regPath := bucketRegistry(t)

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"remove", "--by-account", "july11@netrasystems.ai", "--rehome-to", "july6-netra",
		"--registry", regPath, "--home", home,
	})
	if rc == 0 {
		t.Fatalf("rehoming into a seat on the retired account must be refused; rc=0\nout=%s", out.String())
	}
	if !strings.Contains(errb.String(), "rehome-into-retired") {
		t.Fatalf("refusal should name the structured reason `rehome-into-retired`:\n%s", errb.String())
	}

	// Nothing was tombstoned — the refusal is total.
	reg2, err := accounts.LoadRegistry(regPath)
	if err != nil {
		t.Fatalf("registry should be unchanged and valid: %v", err)
	}
	live := reg2.Refresh()
	for _, name := range []string{"july11-netra", "july6-netra"} {
		if active, _ := activeByName(live, name); !active {
			t.Fatalf("seat %q must remain active after a refused retirement", name)
		}
	}
}

// TestRemoveNameNotesOtherLiveSeat pins the second acceptance path: a single-seat
// `remove --name` prints a `note:` naming the OTHER live seat still resolving to the same
// account, so the operator isn't relying on catching the `dup ->` line by eye.
func TestRemoveNameNotesOtherLiveSeat(t *testing.T) {
	home, regPath := bucketRegistry(t)

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"remove", "--name", "july11-netra", "--rehome-to", "anchor-seat",
		"--registry", regPath, "--home", home,
	})
	if rc != 0 {
		t.Fatalf("remove --name rc=%d stderr=%s", rc, errb.String())
	}
	got := out.String()
	if !strings.Contains(got, "also reachable via july6-netra") || !strings.Contains(got, "pass --by-account") {
		t.Fatalf("remove --name should note the other live seat on the account:\n%s", got)
	}

	// The named seat is tombstoned; the duplicate is (deliberately) still live — that is the gap
	// the note warns about.
	reg2, err := accounts.LoadRegistry(regPath)
	if err != nil {
		t.Fatal(err)
	}
	live := reg2.Refresh()
	if active, _ := activeByName(live, "july11-netra"); active {
		t.Fatalf("july11-netra should be tombstoned")
	}
	if active, _ := activeByName(live, "july6-netra"); !active {
		t.Fatalf("july6-netra should still be live after a per-seat remove (the papercut)")
	}
}
