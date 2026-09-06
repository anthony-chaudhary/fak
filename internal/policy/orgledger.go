// orgledger.go closes the last fail-open in the `fak-org-policy/v1` verifier
// (W2 of epic #5315, issue #5320).
//
// VerifyEnvelope is deliberately a PURE function: its anti-rollback check
// compares the envelope's Version against a caller-supplied
// VerifyOptions.HighestSeenVersion. That keeps the verifier deterministic and
// table-testable — but it also means the anti-rollback property is only as real
// as the caller's memory. A caller that passes 0 (a fresh process, an amnesiac
// call site, a restart after a crash) gets NO rollback protection at all: a
// replayed older — and therefore possibly more permissive — org manifest is
// still validly signed by the org root key, still inside its own validity
// window, and would verify and lower the local floor.
//
// OrgLedger is the durable memory that makes anti-rollback hold ACROSS process
// restarts. It records the highest envelope version this local has ever
// accepted and re-supplies it to the verifier, so the check no longer depends
// on the call site remembering anything.
//
// Three rules give the store its teeth:
//
//   - IT ONLY RAISES. Accept folds the persisted version into the caller's
//     opts with max(), never min(). A caller cannot talk the verifier below
//     what this local has already durably accepted.
//
//   - A REFUSAL IS INERT. The ledger advances only after VerifyEnvelope
//     returns success. This is not merely tidy: without it, anyone who can hand
//     `fak` bytes could forge an envelope claiming version 2^63, watch it be
//     refused for a bad signature, and still poison the counter — permanently
//     locking the org out of shipping any real policy update. An unauthenticated
//     input must never move authenticated state.
//
//   - A BROKEN LEDGER FAILS CLOSED. A missing ledger is the honest "never
//     accepted anything" zero state (first enrollment). A ledger that is
//     present but corrupt, truncated, or edited is REFUSED — never degraded to
//     version 0 — because a silent reset to zero re-opens exactly the rollback
//     window this file exists to close.
//
// Integrity scope, stated honestly: the recorded Sum is a plain SHA-256
// checksum, not a MAC. It has no secret, so an attacker who already has write
// access to the ledger file can recompute it. It detects corruption,
// truncation, partial writes and naive hand-edits — it does not defend against
// a privileged local attacker, whose containment is filesystem permissions (the
// ledger is written 0600 in a 0700 directory) and, ultimately, the org root key
// that still has to sign whatever replaces the floor.
//
// Stdlib-only, matching the rest of the envelope path: crypto/sha256 for the
// checksum, encoding/json for the record, os for an atomic write-and-rename.
package policy

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// OrgLedgerSchema tags the persisted acceptance record. A record carrying any
// other schema is refused rather than guessed at.
const OrgLedgerSchema = "fak-org-policy-ledger/v1"

// maxOrgLedgerBytes bounds the on-disk record. The ledger is a handful of small
// scalars; anything larger is corruption or a substituted file, and is refused
// before it is parsed.
const maxOrgLedgerBytes = 64 << 10

// OrgAcceptance is the durable record of the newest org envelope this local has
// accepted. Digest pins WHICH envelope was accepted (so an operator can audit
// the exact bytes that moved the floor), and Sum is the integrity check over
// the rest of the record.
type OrgAcceptance struct {
	Schema     string `json:"schema"`
	Issuer     string `json:"issuer"`
	Version    uint64 `json:"version"`
	AcceptedAt int64  `json:"accepted_at"`
	Digest     string `json:"digest"`
	Sum        string `json:"sum"`
}

// ledgerSum is the integrity checksum over an acceptance record excluding Sum
// itself. Every field is LENGTH-PREFIXED before hashing so no combination of
// field contents can be re-partitioned into a different record with the same
// preimage (a plain separator would let an issuer containing the separator
// forge an equal digest).
func ledgerSum(a OrgAcceptance) string {
	h := sha256.New()
	for _, f := range []string{
		a.Schema,
		a.Issuer,
		strconv.FormatUint(a.Version, 10),
		strconv.FormatInt(a.AcceptedAt, 10),
		a.Digest,
	} {
		fmt.Fprintf(h, "%d:%s", len(f), f)
	}
	return hex.EncodeToString(h.Sum(nil))
}

// OrgLedger is the durable anti-rollback store for one enrolled org policy
// stream, backed by a single JSON file. The mutex serializes readers and
// writers within this process; cross-process serialization is the caller's
// concern (the write itself is atomic via rename, so a concurrent reader sees
// either the old record or the new one, never a torn one).
type OrgLedger struct {
	mu   sync.Mutex
	path string
}

// OpenOrgLedger binds a ledger to a path. It does NOT create the file: a
// not-yet-existing ledger is the legitimate pre-enrollment state, and the file
// appears on the first successful Accept.
func OpenOrgLedger(path string) (*OrgLedger, error) {
	if strings.TrimSpace(path) == "" {
		return nil, fail(abi.ReasonMalformed, "ledger_path", errors.New("org policy ledger path is empty"))
	}
	return &OrgLedger{path: path}, nil
}

// Path reports the backing file path.
func (l *OrgLedger) Path() string { return l.path }

// State returns the persisted acceptance record.
//
// A MISSING ledger yields the zero record with a nil error — "this local has
// never accepted an org envelope", which is the correct starting point for
// first enrollment. Every OTHER problem (unreadable, oversized, unparseable,
// wrong schema, failed checksum) is an error, so a damaged ledger can never be
// mistaken for a fresh one and silently re-open the rollback window.
func (l *OrgLedger) State() (OrgAcceptance, error) {
	b, err := os.ReadFile(l.path)
	if errors.Is(err, fs.ErrNotExist) {
		return OrgAcceptance{}, nil
	}
	if err != nil {
		return OrgAcceptance{}, fail(abi.ReasonTrustViolation, "ledger_unreadable", err)
	}
	if len(b) > maxOrgLedgerBytes {
		return OrgAcceptance{}, fail(abi.ReasonOversize, "ledger_size",
			fmt.Errorf("ledger is %d bytes, budget is %d", len(b), maxOrgLedgerBytes))
	}

	var a OrgAcceptance
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&a); err != nil {
		return OrgAcceptance{}, fail(abi.ReasonTrustViolation, "ledger_corrupt", err)
	}
	if a.Schema != OrgLedgerSchema {
		return OrgAcceptance{}, fail(abi.ReasonTrustViolation, "ledger_schema",
			fmt.Errorf("ledger schema %q is not %s", a.Schema, OrgLedgerSchema))
	}
	if a.Sum == "" || a.Sum != ledgerSum(a) {
		return OrgAcceptance{}, fail(abi.ReasonTrustViolation, "ledger_tampered",
			errors.New("ledger checksum does not match its contents"))
	}
	return a, nil
}

// Accept verifies raw org-envelope bytes against opts AND the durable
// anti-rollback state, recording the acceptance only if verification succeeds.
//
// Before verifying it folds the persisted state into opts:
//   - HighestSeenVersion is RAISED to the persisted version if the caller's is
//     lower, so an amnesiac caller still gets rollback protection;
//   - ExpectedIssuer, when the caller left it unset, is pinned to the issuer
//     already on record, so a rollback cannot evade the counter by re-issuing
//     the same old floor under a fresh issuer name.
//
// On any verification failure the ledger is left EXACTLY as it was and the
// verifier's own closed-vocabulary *EnvelopeError is returned unwrapped, so
// callers keep the distinct reason (TRUST_VIOLATION / UNWITNESSED /
// POLICY_BLOCK / MALFORMED / OVERSIZE) the envelope path already established.
func (l *OrgLedger) Accept(raw []byte, opts VerifyOptions) (Verified, error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	prior, err := l.State()
	if err != nil {
		return Verified{}, err
	}
	if prior.Version > opts.HighestSeenVersion {
		opts.HighestSeenVersion = prior.Version
	}
	if strings.TrimSpace(opts.ExpectedIssuer) == "" && prior.Issuer != "" {
		opts.ExpectedIssuer = prior.Issuer
	}

	v, err := VerifyEnvelope(raw, opts)
	if err != nil {
		// Inert rejection: unauthenticated bytes do not move the counter.
		return Verified{}, err
	}

	digest := sha256.Sum256(raw)
	next := OrgAcceptance{
		Schema:     OrgLedgerSchema,
		Issuer:     v.Envelope.Issuer,
		Version:    v.Envelope.Version,
		AcceptedAt: opts.Now.Unix(),
		Digest:     hex.EncodeToString(digest[:]),
	}
	next.Sum = ledgerSum(next)

	// Re-accepting the version already on record (a routine re-poll of the same
	// envelope) is legal but not a state change; skipping the write keeps the
	// recorded AcceptedAt pinned to the moment the floor actually moved.
	if prior.Version == next.Version && prior.Digest == next.Digest && prior.Issuer == next.Issuer {
		return v, nil
	}
	if err := l.write(next); err != nil {
		return Verified{}, err
	}
	return v, nil
}

// write persists a record atomically: encode, write a sibling temp file, fsync
// it, then rename over the target. A crash mid-write therefore leaves either the
// previous record or the new one intact — never a truncated ledger that would
// have to be refused on the next boot.
func (l *OrgLedger) write(a OrgAcceptance) error {
	b, err := json.MarshalIndent(a, "", "  ")
	if err != nil {
		return fail(abi.ReasonMalformed, "ledger_encode", err)
	}
	b = append(b, '\n')

	dir := filepath.Dir(l.path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fail(abi.ReasonTrustViolation, "ledger_write", err)
	}
	tmp, err := os.CreateTemp(dir, ".fak-org-policy-ledger-*.tmp")
	if err != nil {
		return fail(abi.ReasonTrustViolation, "ledger_write", err)
	}
	name := tmp.Name()
	// Best-effort cleanup; a no-op once the rename below has consumed the temp.
	defer func() { _ = os.Remove(name) }()

	if _, err := tmp.Write(b); err != nil {
		_ = tmp.Close()
		return fail(abi.ReasonTrustViolation, "ledger_write", err)
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return fail(abi.ReasonTrustViolation, "ledger_write", err)
	}
	if err := tmp.Close(); err != nil {
		return fail(abi.ReasonTrustViolation, "ledger_write", err)
	}
	if err := os.Chmod(name, 0o600); err != nil {
		return fail(abi.ReasonTrustViolation, "ledger_write", err)
	}
	if err := os.Rename(name, l.path); err != nil {
		return fail(abi.ReasonTrustViolation, "ledger_write", err)
	}
	return nil
}
