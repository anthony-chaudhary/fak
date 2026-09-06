// enrollment.go pins the org TRUST ANCHOR (W5 of epic #5315, issue #5323).
//
// orgenvelope.go proves the pure verifier refuses a bad key when the CALLER hands
// it the right one. orgledger.go proves anti-rollback survives the caller
// forgetting. Both leave the same question open: WHERE does the right key come
// from? Today it arrives as an operator-supplied `--org-key` on each invocation,
// which means the trust anchor is re-decided every run by whoever types the
// command. An anchor that can be re-chosen per invocation is not an anchor.
//
// Enrollment answers it once, durably, and refuses to answer it again quietly:
//
//   - OPT-IN. With no enrollment on disk the org plane is INERT — not enrolled is
//     not an error, because that would make central policy mandatory for every
//     `fak` on earth. But inert is not fail-open: the anchor handed back carries
//     no root key, so a perfectly-signed envelope is still refused with
//     TRUST_VIOLATION. A caller that ignores the "not enrolled" boolean cannot
//     accidentally start trusting central policy.
//
//   - PIN. Only the enrolled key verifies, and enrollment pins the ISSUER too, so
//     a signer that keeps its key but re-badges itself as another org is refused.
//
//   - NO SILENT RE-PIN. A second enroll pointing at a different org — or at the
//     SAME org under a different root key, which is exactly what a MitM'd enroll
//     endpoint would serve — is REFUSED. Re-pinning requires an explicit revoke.
//     The refusal is INERT: the on-disk anchor is byte-identical afterwards, so
//     an attacker cannot damage the existing pin merely by attempting to replace
//     it. Re-running the IDENTICAL enroll is idempotent and does not re-stamp the
//     original enrollment time.
//
//   - FAIL CLOSED. A store that is present but corrupt, truncated, oversized, or
//     hand-edited is REFUSED — never degraded to "not enrolled". A silent degrade
//     would let anyone with write access discard whatever tightening the org
//     floor was carrying, which is the same fail-open the ledger closes.
//
//   - DEVICE BINDING. The identity recorded at enroll time is what a signed
//     target selector is matched against, and a selector shape this grammar does
//     not cover matches NOTHING. A targeting rule fak cannot parse must never be
//     read as "applies to everyone".
//
// Integrity scope, stated as honestly as the ledger states its own: the recorded
// Sum is a plain SHA-256 checksum, not a MAC. It has no secret, so an attacker
// who already has write access to this file can recompute it. It detects
// corruption, truncation, partial writes and naive hand-edits — it does not
// defend against a privileged local attacker, whose containment is filesystem
// permissions (0600 in a 0700 directory) and, ultimately, the org root key that
// still has to sign whatever replaces the floor.
//
// Only PUBLIC key material is ever persisted here. There is no private key in
// the enrollment path at all — fak verifies org policy, it never signs it.
//
// Stdlib-only, matching the rest of the envelope path.
package policy

import (
	"bytes"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// OrgEnrollmentSchema tags the persisted enrollment record. A record carrying
// any other schema is refused rather than guessed at — a future format must be
// migrated deliberately, not silently reinterpreted under the current one.
const OrgEnrollmentSchema = "fak-org-enrollment/v1"

// OrgEnrollmentPathEnv overrides where the enrollment store lives. It carries a
// PATH and never key material, so it is safe in an environment dump.
const OrgEnrollmentPathEnv = "FAK_ORG_ENROLLMENT_PATH"

// maxOrgEnrollmentBytes bounds the on-disk record. An enrollment is a handful of
// short scalars and one 32-byte public key; anything larger is corruption or a
// substituted file, and is refused BEFORE it is parsed.
const maxOrgEnrollmentBytes = 64 << 10

// OrgEnrollment is the durable trust anchor: which org this local is enrolled
// with, which root key(s) may sign its policy, and the device identity that
// signed target selectors are resolved against.
//
// RootKeys is plural so a future key rotation can pin an overlap window without
// a schema change; EnrollOrg writes exactly one today.
type OrgEnrollment struct {
	Schema     string   `json:"schema"`
	OrgURL     string   `json:"org_url"`
	Issuer     string   `json:"issuer"`
	RootKeys   []string `json:"root_keys"` // base64 std-encoded Ed25519 PUBLIC keys
	DeviceID   string   `json:"device_id"`
	AuditURL   string   `json:"audit_url,omitempty"`
	User       string   `json:"user,omitempty"`
	Groups     []string `json:"groups,omitempty"`
	EnrolledAt int64    `json:"enrolled_at"`
	Sum        string   `json:"sum"`
}

// OrgEnrollRequest is one enrollment attempt. Now is supplied by the caller so
// the record is a pure function of its inputs and the suite never reads the wall
// clock; a zero Now falls back to time.Now for real CLI use.
type OrgEnrollRequest struct {
	OrgURL   string
	Issuer   string
	RootKey  ed25519.PublicKey
	DeviceID string
	AuditURL string
	User     string
	Groups   []string
	Now      time.Time
}

// enrollmentIdentity is the checksum over everything that makes this enrollment
// THIS enrollment — every field except the timestamp and the checksum itself.
// Two records with the same identity are the same pin, which is what makes a
// repeated identical enroll idempotent and any other second enroll a re-pin.
//
// Every field is LENGTH-PREFIXED before hashing, and every list is preceded by
// its element count, so no combination of field contents can be re-partitioned
// into a different record with the same preimage (a plain separator would let a
// value containing that separator forge an equal digest).
func enrollmentIdentity(e OrgEnrollment) string {
	h := sha256.New()
	write := func(s string) { fmt.Fprintf(h, "%d:%s", len(s), s) }
	write(e.Schema)
	write(e.OrgURL)
	write(e.Issuer)
	write(e.DeviceID)
	if e.AuditURL != "" {
		write(e.AuditURL)
	}
	write(e.User)
	// Sorted copies: the pin is a SET of keys and a SET of groups, so re-running
	// the same enroll with a differently-ordered slice is the same pin. Widening
	// either set still changes the identity and is still refused.
	write(strconv.Itoa(len(e.RootKeys)))
	for _, k := range sortedCopy(e.RootKeys) {
		write(k)
	}
	write(strconv.Itoa(len(e.Groups)))
	for _, g := range sortedCopy(e.Groups) {
		write(g)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func sortedCopy(in []string) []string {
	out := append([]string(nil), in...)
	sort.Strings(out)
	return out
}

// enrollmentSum binds the identity to the moment it was pinned, so editing the
// timestamp is as detectable as editing the key.
func enrollmentSum(e OrgEnrollment) string {
	h := sha256.New()
	id := enrollmentIdentity(e)
	fmt.Fprintf(h, "%d:%s", len(id), id)
	at := strconv.FormatInt(e.EnrolledAt, 10)
	fmt.Fprintf(h, "%d:%s", len(at), at)
	return hex.EncodeToString(h.Sum(nil))
}

// OrgEnrollmentPath reports where the enrollment store lives: the env override
// when set, else a fak-owned file under the user config dir, else under the home
// directory. The final fallback is relative so a box with neither still resolves
// to something explicit rather than an empty path a caller might treat as "off".
// OrgAuditBufferPath stores privacy-screened receipts beside the enrollment anchor.
func OrgAuditBufferPath() string {
	return orgStatePath("FAK_ORG_AUDIT_BUFFER_PATH", "org-audit-buffer.jsonl")
}

func OrgEnrollmentPath() string {
	return orgStatePath(OrgEnrollmentPathEnv, "org-enrollment.json")
}

func orgStatePath(env, filename string) string {
	if value := strings.TrimSpace(os.Getenv(env)); value != "" {
		return value
	}
	if dir, err := os.UserConfigDir(); err == nil && dir != "" {
		return filepath.Join(dir, "fak", filename)
	}
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		return filepath.Join(home, ".fak", filename)
	}
	return filepath.Join(".fak", filename)
}

// LoadOrgEnrollment reads the enrollment store.
//
// A MISSING store yields the zero record, enrolled=false, and a NIL error — the
// honest "this local is not enrolled" state, which must stay ordinary because
// the org plane is opt-in. Every OTHER problem is an error, so a damaged anchor
// can never be mistaken for an un-enrolled one.
func LoadOrgEnrollment(path string) (OrgEnrollment, bool, error) {
	if strings.TrimSpace(path) == "" {
		return OrgEnrollment{}, false, fail(abi.ReasonMalformed, "enrollment_path",
			errors.New("org enrollment path is empty"))
	}
	b, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return OrgEnrollment{}, false, nil
	}
	if err != nil {
		return OrgEnrollment{}, false, fail(abi.ReasonTrustViolation, "enrollment_unreadable", err)
	}
	if len(b) > maxOrgEnrollmentBytes {
		return OrgEnrollment{}, false, fail(abi.ReasonOversize, "enrollment_size",
			fmt.Errorf("enrollment is %d bytes, budget is %d", len(b), maxOrgEnrollmentBytes))
	}

	var e OrgEnrollment
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&e); err != nil {
		return OrgEnrollment{}, false, fail(abi.ReasonTrustViolation, "enrollment_corrupt", err)
	}
	if e.Schema != OrgEnrollmentSchema {
		return OrgEnrollment{}, false, fail(abi.ReasonTrustViolation, "enrollment_schema",
			fmt.Errorf("enrollment schema %q is not %s", e.Schema, OrgEnrollmentSchema))
	}
	if e.Sum == "" || e.Sum != enrollmentSum(e) {
		return OrgEnrollment{}, false, fail(abi.ReasonTrustViolation, "enrollment_tampered",
			errors.New("enrollment checksum does not match its contents"))
	}
	// A record that passes its checksum can still be unusable — a store written by
	// a buggy future version, say. Refuse it here rather than at verify time,
	// where the failure would look like a policy problem instead of an anchor one.
	if _, err := e.RootPublicKeys(); err != nil {
		return OrgEnrollment{}, false, err
	}
	if strings.TrimSpace(e.DeviceID) == "" || strings.TrimSpace(e.OrgURL) == "" {
		return OrgEnrollment{}, false, fail(abi.ReasonTrustViolation, "enrollment_incomplete",
			errors.New("enrollment is missing its org url or device identity"))
	}
	return e, true, nil
}

// RootPublicKeys decodes the pinned root keys. A key that is not base64, or not
// exactly an Ed25519 public key, is a refusal rather than a skipped entry: a
// silently-dropped key would shrink the trusted set without saying so.
func (e OrgEnrollment) RootPublicKeys() ([]ed25519.PublicKey, error) {
	out := make([]ed25519.PublicKey, 0, len(e.RootKeys))
	for i, s := range e.RootKeys {
		raw, err := base64.StdEncoding.DecodeString(strings.TrimSpace(s))
		if err != nil {
			return nil, fail(abi.ReasonTrustViolation, "enrollment_key",
				fmt.Errorf("pinned root key %d is not base64: %w", i, err))
		}
		if len(raw) != ed25519.PublicKeySize {
			return nil, fail(abi.ReasonTrustViolation, "enrollment_key",
				fmt.Errorf("pinned root key %d is %d bytes, want %d", i, len(raw), ed25519.PublicKeySize))
		}
		out = append(out, ed25519.PublicKey(raw))
	}
	return out, nil
}

// MatchesTarget reports whether a signed grant's target selector applies to this
// enrolled identity.
//
// The grammar is closed: "", "*", "device:<id|*>", "user:<name|*>",
// "group:<name|*>". Anything else — an unknown kind, a missing prefix, a
// different case, an empty value — matches NOTHING. A selector fak cannot parse
// is not "everyone"; it is a rule this binary is not equipped to honour, and
// treating it as universal would grant on the strength of a typo.
//
// A record with no device identity matches nothing at all, wildcard included:
// with nothing to bind a grant to, there is no "this device" for it to apply to.
func (e OrgEnrollment) MatchesTarget(target string) bool {
	if strings.TrimSpace(e.DeviceID) == "" {
		return false
	}
	t := strings.TrimSpace(target)
	if t == "" || t == "*" {
		return true
	}
	kind, value, ok := strings.Cut(t, ":")
	if !ok || kind == "" || value == "" {
		return false
	}
	switch kind {
	case "device":
		return value == "*" || value == e.DeviceID
	case "user":
		if e.User == "" {
			return false
		}
		return value == "*" || value == e.User
	case "group":
		if value == "*" {
			return len(e.Groups) > 0
		}
		for _, g := range e.Groups {
			if g == value {
				return true
			}
		}
		return false
	default:
		return false
	}
}

// EnrollOrg pins a trust anchor at path.
//
// A malformed request never reaches the disk — a store holding an unusable key
// would only fail later, at verify time, where the failure reads as a policy
// problem instead of an enrollment one.
//
// If an enrollment is already present, the ONLY accepted outcome is an identical
// re-enroll, which returns the existing record untouched (so the original
// EnrolledAt survives). Any other second enroll is refused with TRUST_VIOLATION
// and leaves the stored bytes exactly as they were.
func EnrollOrg(path string, req OrgEnrollRequest) (OrgEnrollment, error) {
	if strings.TrimSpace(path) == "" {
		return OrgEnrollment{}, fail(abi.ReasonMalformed, "enrollment_path",
			errors.New("org enrollment path is empty"))
	}
	candidate, err := enrollmentFromRequest(req)
	if err != nil {
		return OrgEnrollment{}, err
	}

	// A present-but-damaged store is NOT overwritten. Failing closed here keeps
	// the cure explicit: revoke, then enroll. Silently replacing a store we could
	// not read would let a corrupted anchor be laundered into a fresh valid one.
	existing, enrolled, err := LoadOrgEnrollment(path)
	if err != nil {
		return OrgEnrollment{}, err
	}
	if enrolled {
		if enrollmentIdentity(existing) == enrollmentIdentity(candidate) {
			return existing, nil
		}
		return OrgEnrollment{}, fail(abi.ReasonTrustViolation, "enrollment_repin",
			fmt.Errorf("already enrolled with %s (issuer %q); revoke before enrolling elsewhere",
				existing.OrgURL, existing.Issuer))
	}

	candidate.Sum = enrollmentSum(candidate)
	if err := writeOrgEnrollment(path, candidate); err != nil {
		return OrgEnrollment{}, err
	}
	return candidate, nil
}

// enrollmentFromRequest validates a request and renders it as an unsummed
// record. Every check here is a way the anchor could be unusable or ambiguous.
func enrollmentFromRequest(req OrgEnrollRequest) (OrgEnrollment, error) {
	if len(req.RootKey) == 0 {
		return OrgEnrollment{}, fail(abi.ReasonMalformed, "enrollment_key",
			errors.New("enrollment carries no org root key"))
	}
	if len(req.RootKey) != ed25519.PublicKeySize {
		return OrgEnrollment{}, fail(abi.ReasonMalformed, "enrollment_key",
			fmt.Errorf("org root key is %d bytes, want %d", len(req.RootKey), ed25519.PublicKeySize))
	}
	if isAllZero(req.RootKey) {
		// An all-zero key is what an uninitialized buffer looks like. It is a
		// syntactically valid Ed25519 public key that can never verify anything,
		// so pinning it would produce an anchor that refuses the org's own policy.
		return OrgEnrollment{}, fail(abi.ReasonMalformed, "enrollment_key",
			errors.New("org root key is all zero bytes"))
	}
	orgURL := strings.TrimSpace(req.OrgURL)
	if orgURL == "" {
		return OrgEnrollment{}, fail(abi.ReasonMalformed, "enrollment_org",
			errors.New("enrollment carries no org url"))
	}
	issuer := strings.TrimSpace(req.Issuer)
	if issuer == "" {
		return OrgEnrollment{}, fail(abi.ReasonMalformed, "enrollment_issuer",
			errors.New("enrollment carries no issuer"))
	}
	auditURL := strings.TrimSpace(req.AuditURL)
	if auditURL != "" {
		u, err := url.Parse(auditURL)
		if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
			return OrgEnrollment{}, fail(abi.ReasonMalformed, "audit_url", errors.New("audit URL must be absolute http/https"))
		}
	}
	deviceID := strings.TrimSpace(req.DeviceID)
	if deviceID == "" {
		return OrgEnrollment{}, fail(abi.ReasonMalformed, "enrollment_device",
			errors.New("enrollment carries no device identity"))
	}

	now := req.Now
	if now.IsZero() {
		now = time.Now()
	}
	groups := make([]string, 0, len(req.Groups))
	for _, g := range req.Groups {
		if g = strings.TrimSpace(g); g != "" {
			groups = append(groups, g)
		}
	}
	if len(groups) == 0 {
		groups = nil
	}
	return OrgEnrollment{
		Schema:     OrgEnrollmentSchema,
		OrgURL:     orgURL,
		Issuer:     issuer,
		RootKeys:   []string{base64.StdEncoding.EncodeToString(req.RootKey)},
		DeviceID:   deviceID,
		AuditURL:   auditURL,
		User:       strings.TrimSpace(req.User),
		Groups:     groups,
		EnrolledAt: now.Unix(),
	}, nil
}

func isAllZero(b []byte) bool {
	for _, c := range b {
		if c != 0 {
			return false
		}
	}
	return true
}

// RevokeOrgEnrollment removes the pinned anchor, returning whether there was one
// to remove. Revoking an absent enrollment is a no-op, not an error, so the
// operator's "make sure this box is not enrolled" is idempotent.
//
// A store that fails to LOAD is still removed: revoke is the documented cure for
// a damaged anchor, and refusing to clear one would leave the box permanently
// locked with no route back.
func RevokeOrgEnrollment(path string) (bool, error) {
	if strings.TrimSpace(path) == "" {
		return false, fail(abi.ReasonMalformed, "enrollment_path",
			errors.New("org enrollment path is empty"))
	}
	err := os.Remove(path)
	if errors.Is(err, fs.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, fail(abi.ReasonTrustViolation, "enrollment_revoke", err)
	}
	return true, nil
}

// OrgTrustAnchor resolves the verify-time trust anchor from the enrollment on
// disk, folding in the ambient facts VerifyEnvelope deliberately refuses to read
// for itself so an enrolled caller still gets the freshness and min_version
// gates.
//
// Three outcomes, and the middle one is the whole point of an opt-in plane:
//
//	enrolled=true,  err=nil  — anchor pinned to the enrolled key and issuer
//	enrolled=false, err=nil  — NOT enrolled: options carry no root key, so a
//	                           signed envelope is still refused. Inert, not open.
//	enrolled=false, err!=nil — the store is damaged; NO options are handed back
//	                           at all, because a partially-trusted anchor is
//	                           indistinguishable to the caller from a real one.
func OrgTrustAnchor(path string, now time.Time, runningVersion string) (VerifyOptions, bool, error) {
	e, enrolled, err := LoadOrgEnrollment(path)
	if err != nil {
		return VerifyOptions{}, false, err
	}
	opts := VerifyOptions{Now: now, RunningVersion: runningVersion}
	if !enrolled {
		// Deliberately populated with the ambient facts but NO root key. The
		// verifier checks anchor presence before it checks anything the anchor
		// would authorize, so this refuses with TRUST_VIOLATION rather than
		// leaking a different reason that might read as a transient problem.
		return opts, false, nil
	}
	keys, err := e.RootPublicKeys()
	if err != nil {
		return VerifyOptions{}, false, err
	}
	if len(keys) == 0 {
		return VerifyOptions{}, false, fail(abi.ReasonTrustViolation, "enrollment_key",
			errors.New("enrollment pins no root key"))
	}
	opts.RootPublicKey = keys[0]
	opts.ExpectedIssuer = e.Issuer
	return opts, true, nil
}

// writeOrgEnrollment persists a record atomically: encode, write a sibling temp
// file, fsync it, then rename over the target. A crash mid-write leaves either
// the previous anchor or the new one intact — never a truncated store that would
// have to be refused on the next boot.
func writeOrgEnrollment(path string, e OrgEnrollment) error {
	b, err := json.MarshalIndent(e, "", "  ")
	if err != nil {
		return fail(abi.ReasonMalformed, "enrollment_encode", err)
	}
	b = append(b, '\n')

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fail(abi.ReasonTrustViolation, "enrollment_write", err)
	}
	tmp, err := os.CreateTemp(dir, ".fak-org-enrollment-*.tmp")
	if err != nil {
		return fail(abi.ReasonTrustViolation, "enrollment_write", err)
	}
	name := tmp.Name()
	// Best-effort cleanup; a no-op once the rename below has consumed the temp.
	defer func() { _ = os.Remove(name) }()

	if _, err := tmp.Write(b); err != nil {
		_ = tmp.Close()
		return fail(abi.ReasonTrustViolation, "enrollment_write", err)
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return fail(abi.ReasonTrustViolation, "enrollment_write", err)
	}
	if err := tmp.Close(); err != nil {
		return fail(abi.ReasonTrustViolation, "enrollment_write", err)
	}
	if err := os.Chmod(name, 0o600); err != nil {
		return fail(abi.ReasonTrustViolation, "enrollment_write", err)
	}
	if err := os.Rename(name, path); err != nil {
		return fail(abi.ReasonTrustViolation, "enrollment_write", err)
	}
	return nil
}
