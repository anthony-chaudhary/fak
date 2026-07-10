package accounts

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// seatWithCred writes a config dir whose .claude.json names `metaEmail`/`metaUUID` and whose live
// .credentials.json carries `token` (the probe key). Either half may be empty to model a gap.
func seatWithCred(t *testing.T, dir, metaEmail, metaUUID, token string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	cfg := map[string]any{}
	if metaEmail != "" || metaUUID != "" {
		cfg["oauthAccount"] = map[string]string{"emailAddress": metaEmail, "accountUuid": metaUUID}
	}
	b, _ := json.Marshal(cfg)
	if err := os.WriteFile(filepath.Join(dir, ".claude.json"), b, 0o600); err != nil {
		t.Fatal(err)
	}
	if token != "" {
		cred := map[string]any{"claudeAiOauth": map[string]string{"accessToken": token}}
		cb, _ := json.Marshal(cred)
		if err := os.WriteFile(filepath.Join(dir, ".credentials.json"), cb, 0o600); err != nil {
			t.Fatal(err)
		}
	}
}

// proberFor returns an IdentityProber that maps known tokens to identities, erroring on unknown.
func proberFor(m map[string]ProbedIdentity) IdentityProber {
	return func(tok string) (ProbedIdentity, error) {
		if id, ok := m[tok]; ok {
			return id, nil
		}
		return ProbedIdentity{}, errors.New("unknown token")
	}
}

// The load-bearing #3953 acceptance test: a seat registered to account A whose dir was re-logged
// into account B is caught as a HIJACK — even though the on-disk metadata was rewritten in lockstep
// with the credential (so #3215's disk-vs-credential stale check would see NO disagreement).
func TestDetectLoginHijack_CaughtEvenWhenMetadataMovedInLockstep(t *testing.T) {
	dir := filepath.Join(t.TempDir(), ".claude-seatA")
	// A /login into the wrong dir rewrote BOTH the credential and the metadata to account B.
	seatWithCred(t, dir, "b@example.com", "uuid-B", "tokenB")
	probe := proberFor(map[string]ProbedIdentity{"tokenB": {Email: "b@example.com", AccountUUID: "uuid-B"}})

	// The registry still binds this seat to account A.
	expected := Identity{Email: "a@example.com", AccountUUID: "uuid-A"}
	rep := DetectLoginHijack("seatA", dir, expected, probe)

	if !rep.Hijacked() {
		t.Fatalf("expected HijackDetected, got %q (detail=%q)", rep.Verdict, rep.Detail)
	}
	if rep.Actual.AccountUUID != "uuid-B" {
		t.Errorf("Actual = %+v, want the probed account B", rep.Actual)
	}
	// The stale check would NOT fire here (disk metadata == credential, both B), proving this
	// primitive catches a class #3215 cannot.
	res := ResolveCredentialIdentity(dir, probe)
	if res.Stale {
		t.Errorf("precondition: disk-vs-credential stale should be false (both moved to B), got Stale=true")
	}
}

// The happy path: a seat whose live credential still serves its registered account is OK.
func TestDetectLoginHijack_MatchingIsOK(t *testing.T) {
	dir := filepath.Join(t.TempDir(), ".claude-ok")
	seatWithCred(t, dir, "a@example.com", "uuid-A", "tokenA")
	probe := proberFor(map[string]ProbedIdentity{"tokenA": {Email: "a@example.com", AccountUUID: "uuid-A"}})
	rep := DetectLoginHijack("ok", dir, Identity{Email: "a@example.com", AccountUUID: "uuid-A"}, probe)
	if rep.Verdict != HijackOK {
		t.Fatalf("verdict = %q, want ok (detail=%q)", rep.Verdict, rep.Detail)
	}
}

// No registered identity → Unbound (nothing to check), never a false hijack.
func TestDetectLoginHijack_UnboundWhenNoExpected(t *testing.T) {
	dir := filepath.Join(t.TempDir(), ".claude-new")
	seatWithCred(t, dir, "a@example.com", "uuid-A", "tokenA")
	probe := proberFor(map[string]ProbedIdentity{"tokenA": {Email: "a@example.com", AccountUUID: "uuid-A"}})
	rep := DetectLoginHijack("new", dir, Identity{}, probe)
	if rep.Verdict != HijackUnbound {
		t.Fatalf("verdict = %q, want unbound", rep.Verdict)
	}
	if rep.Hijacked() {
		t.Fatalf("unbound must not report hijacked")
	}
}

// No live credential, no prober, or a probe error → Unprobed, never conflated with OK.
func TestDetectLoginHijack_UnprobedNeverOK(t *testing.T) {
	expected := Identity{Email: "a@example.com", AccountUUID: "uuid-A"}

	// (a) no credential file present.
	dirNoCred := filepath.Join(t.TempDir(), ".claude-nocred")
	seatWithCred(t, dirNoCred, "a@example.com", "uuid-A", "")
	if got := DetectLoginHijack("s", dirNoCred, expected, proberFor(nil)).Verdict; got != HijackUnprobed {
		t.Fatalf("no-credential verdict = %q, want unprobed", got)
	}
	// (b) nil prober (disk-only) never yields a hijack even with a credential present.
	dirCred := filepath.Join(t.TempDir(), ".claude-cred")
	seatWithCred(t, dirCred, "a@example.com", "uuid-A", "tokenA")
	if got := DetectLoginHijack("s", dirCred, expected, nil).Verdict; got == HijackDetected {
		t.Fatalf("nil prober must never detect a hijack, got %q", got)
	}
	// (c) probe transport error → unprobed with the error surfaced.
	rep := DetectLoginHijack("s", dirCred, expected, proberFor(map[string]ProbedIdentity{"other": {Email: "x"}}))
	if rep.Verdict != HijackUnprobed || rep.ProbeErr == "" {
		t.Fatalf("probe-error verdict = %q (err=%q), want unprobed with a probe_error", rep.Verdict, rep.ProbeErr)
	}
}

// ScanLoginHijacks flags the hijacked seat among healthy ones, skips dirless/tombstoned seats, and
// AnyHijacked gates on the result.
func TestScanLoginHijacks_FleetWide(t *testing.T) {
	base := t.TempDir()
	okDir := filepath.Join(base, ".claude-ok")
	badDir := filepath.Join(base, ".claude-bad")
	seatWithCred(t, okDir, "a@example.com", "uuid-A", "tokenA")
	seatWithCred(t, badDir, "c@example.com", "uuid-C", "tokenC") // re-logged to C
	probe := proberFor(map[string]ProbedIdentity{
		"tokenA": {Email: "a@example.com", AccountUUID: "uuid-A"},
		"tokenC": {Email: "c@example.com", AccountUUID: "uuid-C"},
	})
	reg := Registry{Homes: []Home{
		{Name: "ok", Dir: okDir, Identity: Identity{Email: "a@example.com", AccountUUID: "uuid-A"}},
		{Name: "bad", Dir: badDir, Identity: Identity{Email: "b@example.com", AccountUUID: "uuid-B"}},
		{Name: "dead", Dir: "", Identity: Identity{Email: "d@example.com"}},
		{Name: "tomb", Dir: filepath.Join(base, ".claude-tomb"), Status: StatusTombstoned, Identity: Identity{Email: "t@example.com"}},
	}}
	reports := ScanLoginHijacks(reg, probe)
	if len(reports) != 2 {
		t.Fatalf("scan should check 2 live-with-dir seats (ok, bad), got %d: %+v", len(reports), reports)
	}
	if !AnyHijacked(reports) {
		t.Fatalf("expected AnyHijacked=true (bad seat rebound to C)")
	}
	byName := map[string]HijackReport{}
	for _, r := range reports {
		byName[r.Seat] = r
	}
	if byName["ok"].Verdict != HijackOK {
		t.Errorf("ok seat verdict = %q, want ok", byName["ok"].Verdict)
	}
	if byName["bad"].Verdict != HijackDetected {
		t.Errorf("bad seat verdict = %q, want hijacked", byName["bad"].Verdict)
	}
}
