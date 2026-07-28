package policy

// orgledger_test.go is the acceptance suite for the DURABLE half of #5320's three
// security properties. orgenvelope_test.go already proves the pure verifier
// refuses a bad key, a stale window and a rolled-back version when the caller
// hands it the right HighestSeenVersion. These tests prove the property survives
// the caller FORGETTING — every case here passes HighestSeenVersion: 0 and an
// empty ExpectedIssuer, so anything that still holds is held by the ledger and
// nothing else.
//
// The negative cases are the load-bearing ones:
//   - AUTH          — a foreign-key envelope is refused AND leaves the counter
//     untouched (a forged high version must not lock the org out).
//   - FRESHNESS     — an envelope outside its stated window is refused AND
//     leaves the counter untouched.
//   - ANTI-ROLLBACK — an older but perfectly-signed envelope is refused by a
//     freshly reopened ledger, i.e. across a process restart.
//   - The ledger itself, corrupted or hand-edited downward, fails CLOSED rather
//     than degrading to version 0 and re-opening the rollback window.
//
// Keys are generated per test from fixed seeds via the suite's testKey helper —
// no secret is ever embedded in the repo, and no test reads the wall clock.

import (
	"crypto/ed25519"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// ledgerPath returns a fresh, non-existent ledger path under a per-test temp dir
// (nested one level down so the ledger's own MkdirAll is exercised).
func ledgerPath(t *testing.T) string {
	t.Helper()
	return filepath.Join(t.TempDir(), "org", "policy-ledger.json")
}

// ledgerOpts is deliberately AMNESIAC: HighestSeenVersion is zero and no issuer
// is expected. Any rollback or issuer protection observed in these tests is
// therefore supplied by the ledger, not by the caller.
func ledgerOpts(pub ed25519.PublicKey) VerifyOptions {
	return VerifyOptions{
		RootPublicKey:  pub,
		Now:            fixedNow,
		RunningVersion: "0.9.0",
	}
}

// envAt builds and signs an envelope with an explicit issuer, version and
// validity window, so each case can vary exactly one axis.
func envAt(t *testing.T, priv ed25519.PrivateKey, issuer string, version uint64, notBefore, expires int64) []byte {
	t.Helper()
	signed, err := SignEnvelope(OrgEnvelope{
		Issuer:     issuer,
		Version:    version,
		NotBefore:  notBefore,
		Expires:    expires,
		MinVersion: "0.5.0",
		Target:     "group:eng",
		Body:       baseBody(t),
	}, priv)
	if err != nil {
		t.Fatalf("sign v%d: %v", version, err)
	}
	return marshal(t, signed)
}

// liveEnv is an envelope valid across fixedNow.
func liveEnv(t *testing.T, priv ed25519.PrivateKey, issuer string, version uint64) []byte {
	t.Helper()
	return envAt(t, priv, issuer, version, fixedNow.Unix()-3600, fixedNow.Unix()+3600)
}

func mustAccept(t *testing.T, l *OrgLedger, raw []byte, opts VerifyOptions) Verified {
	t.Helper()
	v, err := l.Accept(raw, opts)
	if err != nil {
		t.Fatalf("expected acceptance, got refusal: %v", err)
	}
	return v
}

func mustAcceptErr(t *testing.T, l *OrgLedger, raw []byte, opts VerifyOptions) error {
	t.Helper()
	if _, err := l.Accept(raw, opts); err != nil {
		return err
	}
	t.Fatalf("expected the ledger to refuse, but it accepted")
	return nil
}

// assertRecorded asserts the persisted counter, re-reading through a FRESH
// ledger handle so the assertion is about the file, not in-process memory.
func assertRecorded(t *testing.T, path string, wantVersion uint64) {
	t.Helper()
	l, err := OpenOrgLedger(path)
	if err != nil {
		t.Fatalf("reopen ledger: %v", err)
	}
	st, err := l.State()
	if err != nil {
		t.Fatalf("read ledger state: %v", err)
	}
	if st.Version != wantVersion {
		t.Fatalf("persisted version = %d, want %d", st.Version, wantVersion)
	}
}

func TestOrgLedgerFirstEnrollmentRecordsAcceptance(t *testing.T) {
	// A missing ledger is the legitimate pre-enrollment zero state, not a fault:
	// the first good envelope must be accepted and durably recorded.
	pub, priv := testKey(t, 1)
	path := ledgerPath(t)
	l, err := OpenOrgLedger(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if st, err := l.State(); err != nil || st.Version != 0 || st.Issuer != "" {
		t.Fatalf("missing ledger should be the zero state, got %+v err=%v", st, err)
	}

	v := mustAccept(t, l, liveEnv(t, priv, "acme-corp", 5), ledgerOpts(pub))
	if v.Envelope.Version != 5 {
		t.Fatalf("verified version = %d, want 5", v.Envelope.Version)
	}
	if !v.Runtime.Adjudicator.Allow["search_web"] {
		t.Fatalf("inner floor did not resolve: %+v", v.Runtime.Adjudicator.Allow)
	}
	assertRecorded(t, path, 5)

	st, err := l.State()
	if err != nil {
		t.Fatalf("state: %v", err)
	}
	if st.Schema != OrgLedgerSchema {
		t.Fatalf("schema = %q, want %q", st.Schema, OrgLedgerSchema)
	}
	if st.Issuer != "acme-corp" {
		t.Fatalf("issuer = %q, want acme-corp", st.Issuer)
	}
	if st.Digest == "" || st.Sum == "" {
		t.Fatalf("record must pin the accepted bytes and carry a checksum: %+v", st)
	}
}

func TestOrgLedgerMonotonicAdvance(t *testing.T) {
	pub, priv := testKey(t, 1)
	path := ledgerPath(t)
	l, _ := OpenOrgLedger(path)

	mustAccept(t, l, liveEnv(t, priv, "acme-corp", 5), ledgerOpts(pub))
	assertRecorded(t, path, 5)
	mustAccept(t, l, liveEnv(t, priv, "acme-corp", 6), ledgerOpts(pub))
	assertRecorded(t, path, 6)
	// Re-polling the same version is legal and idempotent, not a rollback.
	mustAccept(t, l, liveEnv(t, priv, "acme-corp", 6), ledgerOpts(pub))
	assertRecorded(t, path, 6)
}

// --- AUTH ------------------------------------------------------------------

// TestOrgLedgerAuthWrongKeyRefusedAndLedgerInert is the AUTH property. An
// envelope signed by a key that is not the pinned org root is REFUSED (not
// logged-and-applied), and — the part a happy-path test cannot show — the
// refusal is INERT: it does not advance the anti-rollback counter. Without that,
// anyone able to hand `fak` bytes could forge a huge version, eat the refusal,
// and permanently lock the real org out of shipping any further policy.
func TestOrgLedgerAuthWrongKeyRefusedAndLedgerInert(t *testing.T) {
	pubA, privA := testKey(t, 1) // the enrolled org root
	_, privB := testKey(t, 2)    // an attacker's key
	path := ledgerPath(t)
	l, _ := OpenOrgLedger(path)

	mustAccept(t, l, liveEnv(t, privA, "acme-corp", 5), ledgerOpts(pubA))
	assertRecorded(t, path, 5)

	// Impersonation: right issuer string, enormous version, wrong signing key.
	forged := liveEnv(t, privB, "acme-corp", 9_000_000)
	assertReason(t, mustAcceptErr(t, l, forged, ledgerOpts(pubA)), abi.ReasonTrustViolation, "signature")

	// The forged envelope moved nothing.
	assertRecorded(t, path, 5)

	// And the org is not locked out: the next genuine update still applies.
	mustAccept(t, l, liveEnv(t, privA, "acme-corp", 6), ledgerOpts(pubA))
	assertRecorded(t, path, 6)
}

// TestOrgLedgerAuthNoRootKeyRefuses proves the ledger cannot be used to bypass
// the trust anchor: with no pinned root key there is no authenticity to
// establish, so nothing is accepted and nothing is recorded.
func TestOrgLedgerAuthNoRootKeyRefuses(t *testing.T) {
	_, priv := testKey(t, 1)
	path := ledgerPath(t)
	l, _ := OpenOrgLedger(path)

	opts := ledgerOpts(nil)
	assertReason(t, mustAcceptErr(t, l, liveEnv(t, priv, "acme-corp", 5), opts), abi.ReasonTrustViolation, "no_root_key")
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("a refused envelope must not create a ledger (stat err = %v)", err)
	}
}

// --- FRESHNESS -------------------------------------------------------------

// TestOrgLedgerFreshnessStaleRefusedAndLedgerInert is the FRESHNESS property: an
// envelope whose stated validity window has closed is refused even though its
// signature is perfect and its version is newer, and the refusal leaves the
// counter untouched.
func TestOrgLedgerFreshnessStaleRefusedAndLedgerInert(t *testing.T) {
	pub, priv := testKey(t, 1)
	path := ledgerPath(t)
	l, _ := OpenOrgLedger(path)

	mustAccept(t, l, liveEnv(t, priv, "acme-corp", 5), ledgerOpts(pub))
	assertRecorded(t, path, 5)

	// Validly signed v6, but its window closed an hour before `now`.
	stale := envAt(t, priv, "acme-corp", 6, fixedNow.Unix()-7200, fixedNow.Unix()-3600)
	assertReason(t, mustAcceptErr(t, l, stale, ledgerOpts(pub)), abi.ReasonUnwitnessed, "expired")
	assertRecorded(t, path, 5)

	// Symmetrically, a window that has not opened yet is equally unusable.
	future := envAt(t, priv, "acme-corp", 7, fixedNow.Unix()+3600, fixedNow.Unix()+7200)
	assertReason(t, mustAcceptErr(t, l, future, ledgerOpts(pub)), abi.ReasonUnwitnessed, "not_before")
	assertRecorded(t, path, 5)
}

// TestOrgLedgerFreshnessDoesNotDependOnCallerClockDrift shows the window is
// evaluated against the caller-supplied `now` and the SAME envelope flips from
// accepted to refused as the clock advances past its expiry — the envelope goes
// stale on its own terms, it is not grandfathered in by having been accepted.
func TestOrgLedgerFreshnessDoesNotDependOnCallerClockDrift(t *testing.T) {
	pub, priv := testKey(t, 1)
	l, _ := OpenOrgLedger(ledgerPath(t))

	raw := envAt(t, priv, "acme-corp", 5, fixedNow.Unix()-60, fixedNow.Unix()+60)
	mustAccept(t, l, raw, ledgerOpts(pub))

	later := ledgerOpts(pub)
	later.Now = fixedNow.Add(24 * time.Hour)
	assertReason(t, mustAcceptErr(t, l, raw, later), abi.ReasonUnwitnessed, "expired")
}

// --- ANTI-ROLLBACK ---------------------------------------------------------

// TestOrgLedgerAntiRollbackSurvivesReopen is the ANTI-ROLLBACK property and the
// reason this file exists. A validly signed, in-window, correctly-issued OLDER
// envelope must not replace the newer one already accepted — and it must be
// refused by a ledger handle opened FRESH (a process restart) whose caller has
// no memory at all (HighestSeenVersion: 0). The pure verifier alone would accept
// this replay; the persisted counter is what refuses it.
func TestOrgLedgerAntiRollbackSurvivesReopen(t *testing.T) {
	pub, priv := testKey(t, 1)
	path := ledgerPath(t)

	first, err := OpenOrgLedger(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	mustAccept(t, first, liveEnv(t, priv, "acme-corp", 9), ledgerOpts(pub))
	assertRecorded(t, path, 9)

	// Simulated restart: a brand-new handle, an amnesiac caller.
	restarted, err := OpenOrgLedger(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	replay := liveEnv(t, priv, "acme-corp", 5) // genuine signature, older version
	opts := ledgerOpts(pub)
	if opts.HighestSeenVersion != 0 {
		t.Fatalf("this test is only meaningful with an amnesiac caller")
	}
	assertReason(t, mustAcceptErr(t, restarted, replay, opts), abi.ReasonUnwitnessed, "rollback")

	// The newer floor still stands.
	assertRecorded(t, path, 9)
}

// TestOrgLedgerAntiRollbackIssuerRenameEvasion closes the obvious way around the
// counter: re-issuing the same old floor under a DIFFERENT issuer name. Once an
// issuer is on record it is pinned, so the renamed replay is refused as an
// issuer mismatch rather than treated as an unrelated, unversioned stream.
func TestOrgLedgerAntiRollbackIssuerRenameEvasion(t *testing.T) {
	pub, priv := testKey(t, 1)
	path := ledgerPath(t)
	l, _ := OpenOrgLedger(path)

	mustAccept(t, l, liveEnv(t, priv, "acme-corp", 9), ledgerOpts(pub))
	renamed := liveEnv(t, priv, "acme-corp-legacy", 5)
	assertReason(t, mustAcceptErr(t, l, renamed, ledgerOpts(pub)), abi.ReasonTrustViolation, "issuer")
	assertRecorded(t, path, 9)
}

// TestOrgLedgerAntiRollbackIgnoresLowerCallerValue proves the fold only ever
// RAISES: a caller passing a stale (lower) HighestSeenVersion cannot talk the
// verifier below the durably accepted floor.
func TestOrgLedgerAntiRollbackIgnoresLowerCallerValue(t *testing.T) {
	pub, priv := testKey(t, 1)
	path := ledgerPath(t)
	l, _ := OpenOrgLedger(path)

	mustAccept(t, l, liveEnv(t, priv, "acme-corp", 9), ledgerOpts(pub))
	opts := ledgerOpts(pub)
	opts.HighestSeenVersion = 2 // caller "remembers" an older high-water mark
	assertReason(t, mustAcceptErr(t, l, liveEnv(t, priv, "acme-corp", 5), opts), abi.ReasonUnwitnessed, "rollback")
	assertRecorded(t, path, 9)
}

// --- LEDGER INTEGRITY ------------------------------------------------------

// TestOrgLedgerTamperedStateFailsClosed is the property that makes the persisted
// counter worth having: a ledger edited DOWNWARD (the natural way to re-enable a
// rollback) is refused by its checksum instead of being believed. Without the
// check, the edited record would report version 1 and the v5 replay below would
// verify — the exact rollback this file prevents.
func TestOrgLedgerTamperedStateFailsClosed(t *testing.T) {
	pub, priv := testKey(t, 1)
	path := ledgerPath(t)
	l, _ := OpenOrgLedger(path)
	mustAccept(t, l, liveEnv(t, priv, "acme-corp", 9), ledgerOpts(pub))

	// Hand-edit the recorded version down, leaving the now-stale checksum.
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	var rec map[string]json.RawMessage
	if err := json.Unmarshal(raw, &rec); err != nil {
		t.Fatalf("unmarshal ledger: %v", err)
	}
	rec["version"] = json.RawMessage("1")
	edited, err := json.Marshal(rec)
	if err != nil {
		t.Fatalf("marshal ledger: %v", err)
	}
	if err := os.WriteFile(path, edited, 0o600); err != nil {
		t.Fatalf("write ledger: %v", err)
	}

	reopened, _ := OpenOrgLedger(path)
	if _, err := reopened.State(); err == nil {
		t.Fatalf("a hand-edited ledger must not read back as valid state")
	} else {
		assertReason(t, err, abi.ReasonTrustViolation, "ledger_tampered")
	}
	// And the downgraded counter must not let the old envelope back in.
	assertReason(t, mustAcceptErr(t, reopened, liveEnv(t, priv, "acme-corp", 5), ledgerOpts(pub)), abi.ReasonTrustViolation, "ledger_tampered")
}

// TestOrgLedgerCorruptStateIsNotAFreshLedger guards the subtler failure: a
// truncated or garbage ledger must NOT be mistaken for "never enrolled"
// (version 0), which would silently re-open the rollback window.
func TestOrgLedgerCorruptStateIsNotAFreshLedger(t *testing.T) {
	pub, priv := testKey(t, 1)
	for _, tc := range []struct {
		name       string
		bytes      string
		wantDetail string
	}{
		{"truncated", `{"schema":"fak-org-policy-ledger/v1","ver`, "ledger_corrupt"},
		{"garbage", "\x00\x01not json at all", "ledger_corrupt"},
		{"unknown-schema", `{"schema":"something-else/v9","issuer":"acme-corp","version":9,"accepted_at":1,"digest":"d","sum":"s"}`, "ledger_schema"},
		{"empty", ``, "ledger_corrupt"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := ledgerPath(t)
			if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
				t.Fatalf("mkdir: %v", err)
			}
			if err := os.WriteFile(path, []byte(tc.bytes), 0o600); err != nil {
				t.Fatalf("write: %v", err)
			}
			l, _ := OpenOrgLedger(path)
			if st, err := l.State(); err == nil {
				t.Fatalf("corrupt ledger read back as valid state %+v", st)
			} else {
				assertReason(t, err, abi.ReasonTrustViolation, tc.wantDetail)
			}
			// A corrupt ledger refuses everything, including an otherwise fine
			// envelope: it fails closed rather than fresh.
			assertReason(t, mustAcceptErr(t, l, liveEnv(t, priv, "acme-corp", 5), ledgerOpts(pub)), abi.ReasonTrustViolation, tc.wantDetail)
		})
	}
}

// TestOrgLedgerEmptyPathRefused keeps the constructor fail-loud rather than
// silently binding to the process working directory.
func TestOrgLedgerEmptyPathRefused(t *testing.T) {
	if _, err := OpenOrgLedger("   "); err == nil {
		t.Fatalf("empty ledger path must be refused")
	} else {
		assertReason(t, err, abi.ReasonMalformed, "ledger_path")
	}
}
