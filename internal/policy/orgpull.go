package policy

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// internal/policy/orgpull.go — pull the org policy envelope, cache the last
// verified one, and refuse to widen when the source goes dark for too long
// (#5321, W3 of epic #5315).
//
// The whole file exists to make ONE failure impossible: an unreachable, stale,
// or tampered org endpoint must never leave a local MORE permissive than it
// would have been with no org plane at all. Everything below is arranged so the
// only path that widens anything is a path that ended in a signature this box
// verified against its own pinned key.
//
// FOUR POSTURES, and a caller must be able to tell them apart:
//
//	inert     — not enrolled. The org plane does nothing; byte-for-byte the
//	            behavior of a box that never heard of #5315.
//	fresh     — this pull fetched and VERIFIED an envelope. Widening applies.
//	last_good — the source failed or was refused, but a previously verified
//	            envelope is still inside BOTH freshness bounds. Widening applies.
//	floor     — refuse-to-widen. Nothing from the org plane applies and the
//	            caller runs on the compiled-in FROZEN floor.
//
// Note what is NOT in that list: there is no "unknown" and no "error" posture
// that a caller could accidentally treat as permissive. A transport failure, a
// bad signature, a corrupt cache, a damaged enrollment — every one of them
// lands on `floor` or `inert`, both of which widen nothing. The Err field
// reports WHAT went wrong; it never changes WHETHER anything widened.
//
// MaxStaleness is compiled in and can only ever be shortened (see
// clampStaleness). The manifest must not be able to extend the window that
// decides whether the manifest is still trusted — otherwise one captured old
// envelope with a long window widens this box forever, which is precisely the
// attack the staleness bound exists to stop.

const (
	// OrgPolicyURLEnv names the endpoint to pull the signed org envelope from.
	// A flag beats it; both are ignored on an un-enrolled box.
	OrgPolicyURLEnv = "FAK_ORG_POLICY_URL"

	// OrgPolicyCachePathEnv overrides where the last-good cache is stored.
	OrgPolicyCachePathEnv = "FAK_ORG_POLICY_CACHE_PATH"

	// OrgLastGoodSchema tags the cache file. A record that does not carry it is
	// refused rather than migrated: silently reinterpreting bytes written by a
	// different layout is how a cache becomes a forgery surface.
	OrgLastGoodSchema = "fak-org-policy-lastgood/v1"

	// MaxOrgPolicyStaleness is the FROZEN ceiling on how long a verified
	// envelope may keep widening this box after the source stops answering.
	//
	// It is a compiled-in constant on purpose. The org manifest cannot raise
	// it, no flag raises it, and no env var raises it — a knob that let the
	// pulled document extend its own tolerated staleness would make "stale"
	// unreachable by construction.
	MaxOrgPolicyStaleness = 12 * time.Hour

	// DefaultOrgPullTimeout bounds one fetch. boundarylint MISSING_HTTP_TIMEOUT
	// is StatusEnforced in this repo, and an unbounded pull on a startup path
	// would hang a box behind a black-holing endpoint.
	DefaultOrgPullTimeout = 10 * time.Second

	// maxOrgLastGoodBytes caps the cache file. The envelope inside is stored
	// base64 (≈4/3 of DefaultMaxOrgEnvelopeBytes) and the record adds a small
	// header, so twice the envelope budget is generous without being unbounded.
	maxOrgLastGoodBytes = 2 * DefaultMaxOrgEnvelopeBytes
)

// The closed posture vocabulary. Callers switch on these; nothing else is ever
// returned.
const (
	OrgPostureInert    = "inert"
	OrgPostureFresh    = "fresh"
	OrgPostureLastGood = "last_good"
	OrgPostureFloor    = "floor"
)

// OrgLastGood is the on-disk record of the most recent envelope this box
// actually verified.
//
// ReceivedAt is OUR clock at the moment WE verified it, and it is deliberately
// not read from the envelope. The envelope's own timestamps are attacker-chosen
// in the threat model this plane defends against (a captured, validly-signed,
// but old document), so freshness has to be measured against something the
// attacker cannot restate.
//
// It is also NOT the ledger's AcceptedAt: OrgLedger.Accept skips its write on an
// idempotent re-poll, so AcceptedAt pins when the floor last MOVED. Re-fetching
// the same version is exactly the case that must refresh freshness without
// changing a knob, so this record keeps its own stamp.
type OrgLastGood struct {
	Schema     string `json:"schema"`
	Envelope   string `json:"envelope"` // base64 std of the raw verified envelope bytes
	Issuer     string `json:"issuer"`
	Version    uint64 `json:"version"`
	ReceivedAt int64  `json:"received_at"`
	ExpiresAt  int64  `json:"expires_at"`
	Sum        string `json:"sum"`
}

// OrgPullResult is what one pull attempt concluded.
//
// Runtime is non-nil ONLY when Widen is true. The two are redundant by design:
// a caller that forgets to check Widen still gets a nil Runtime on every
// refusing path, so the fail-open needs two independent mistakes rather than one.
type OrgPullResult struct {
	Posture string
	Widen   bool
	Runtime *Runtime
	Version uint64
	Issuer  string
	// Age is how long ago the applied envelope was verified. Zero on a fresh
	// pull; the cache's age on last_good; meaningless (zero) on floor/inert.
	Age time.Duration
	// Reason names, in a short stable token, why this is not `fresh`.
	Reason string
	// Err is the underlying failure, if any. It is reported and never swallowed,
	// but it does not by itself decide the posture.
	Err error
}

// OrgPuller pulls, verifies, caches, and ages out the org policy envelope.
//
// Fetch is the transport seam: the DoD requires the fetch loop to be tested with
// an in-process double and no live network, so the network is a field rather
// than a call. A nil Fetch with a URL set gets the bounded HTTP default.
type OrgPuller struct {
	// URL is the endpoint. Empty means "never fetched" — which is NOT the same
	// as "no org policy": an enrolled box with no URL still ages out its cache.
	URL string
	// CachePath is the last-good record. Empty resolves to OrgPolicyCachePath().
	CachePath string
	// EnrollmentPath is the pinned anchor. Empty resolves to OrgEnrollmentPath().
	EnrollmentPath string
	// Ledger carries the anti-rollback watermark. Nil is allowed — verification
	// still runs, it just has no persisted high-water mark to fold in.
	Ledger *OrgLedger
	// RunningVersion is this binary's dotted-numeric version, for the
	// min_version gate.
	RunningVersion string
	// Fetch retrieves the raw envelope bytes. Injected in tests.
	Fetch func(ctx context.Context) ([]byte, error)
	// Now is the clock. Nil means time.Now.
	Now func() time.Time
	// MaxStaleness may only TIGHTEN the compiled ceiling; see clampStaleness.
	MaxStaleness time.Duration
}

func (p *OrgPuller) now() time.Time {
	if p.Now != nil {
		return p.Now()
	}
	return time.Now()
}

func (p *OrgPuller) cachePath() string {
	if s := strings.TrimSpace(p.CachePath); s != "" {
		return s
	}
	return OrgPolicyCachePath()
}

func (p *OrgPuller) enrollmentPath() string {
	if s := strings.TrimSpace(p.EnrollmentPath); s != "" {
		return s
	}
	return OrgEnrollmentPath()
}

// clampStaleness is the one-way valve on the freshness window.
//
// A zero or negative request means "use the ceiling". Anything LONGER than the
// ceiling is silently reduced to it — not rejected, because a caller asking for
// a longer window is asking for something that must simply not happen, and
// refusing the whole pull over it would be a worse outcome than tightening.
// Anything shorter is honored: making yourself stricter is always allowed.
func clampStaleness(d time.Duration) time.Duration {
	if d <= 0 || d > MaxOrgPolicyStaleness {
		return MaxOrgPolicyStaleness
	}
	return d
}

// OrgPolicyCachePath mirrors OrgEnrollmentPath's resolution order so the anchor
// and the cache it authorizes live side by side.
func OrgPolicyCachePath() string {
	return orgStatePath(OrgPolicyCachePathEnv, "org-policy-lastgood.json")
}

// lastGoodSum length-prefixes every field so no re-partitioning of the contents
// produces an equal preimage — the same construction orgledger and enrollment
// use, kept identical on purpose.
func lastGoodSum(r OrgLastGood) string {
	h := sha256.New()
	write := func(s string) { fmt.Fprintf(h, "%d:%s", len(s), s) }
	write(r.Schema)
	write(r.Envelope)
	write(r.Issuer)
	write(strconv.FormatUint(r.Version, 10))
	write(strconv.FormatInt(r.ReceivedAt, 10))
	write(strconv.FormatInt(r.ExpiresAt, 10))
	return hex.EncodeToString(h.Sum(nil))
}

// LoadOrgLastGood reads the cache.
//
// A MISSING cache is the zero record, ok=false, and a NIL error: never having
// pulled is an ordinary state. Every other problem is an error, because a cache
// this box cannot read must not be able to masquerade as a cache this box never
// had — the two lead to the same posture here (floor), but a caller that logs
// only errors would otherwise never learn its cache was corrupt.
func LoadOrgLastGood(path string) (OrgLastGood, bool, error) {
	if strings.TrimSpace(path) == "" {
		return OrgLastGood{}, false, fail(abi.ReasonMalformed, "lastgood_path",
			errors.New("org policy cache path is empty"))
	}
	b, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return OrgLastGood{}, false, nil
	}
	if err != nil {
		return OrgLastGood{}, false, fail(abi.ReasonTrustViolation, "lastgood_unreadable", err)
	}
	if len(b) > maxOrgLastGoodBytes {
		return OrgLastGood{}, false, fail(abi.ReasonOversize, "lastgood_size",
			fmt.Errorf("org policy cache is %d bytes, budget is %d", len(b), maxOrgLastGoodBytes))
	}

	var r OrgLastGood
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&r); err != nil {
		return OrgLastGood{}, false, fail(abi.ReasonTrustViolation, "lastgood_corrupt", err)
	}
	if r.Schema != OrgLastGoodSchema {
		return OrgLastGood{}, false, fail(abi.ReasonTrustViolation, "lastgood_schema",
			fmt.Errorf("org policy cache schema %q, want %q", r.Schema, OrgLastGoodSchema))
	}
	if r.Sum == "" || r.Sum != lastGoodSum(r) {
		return OrgLastGood{}, false, fail(abi.ReasonTrustViolation, "lastgood_sum",
			errors.New("org policy cache checksum does not match its contents"))
	}
	if strings.TrimSpace(r.Envelope) == "" {
		return OrgLastGood{}, false, fail(abi.ReasonMalformed, "lastgood_empty",
			errors.New("org policy cache carries no envelope"))
	}
	return r, true, nil
}

// EnvelopeBytes decodes the cached envelope back to the exact bytes that were
// verified when it was written.
func (r OrgLastGood) EnvelopeBytes() ([]byte, error) {
	raw, err := base64.StdEncoding.DecodeString(strings.TrimSpace(r.Envelope))
	if err != nil {
		return nil, fail(abi.ReasonTrustViolation, "lastgood_envelope", err)
	}
	return raw, nil
}

// WriteOrgLastGood persists the cache atomically: temp file, fsync, close,
// chmod, rename, inside a 0700 directory. A crash mid-write leaves the previous
// record or the new one — never a truncated file the next boot has to refuse.
func WriteOrgLastGood(path string, r OrgLastGood) error {
	if r.Sum == "" {
		r.Sum = lastGoodSum(r)
	}
	return writeOrgLastGood(path, r)
}

func writeOrgLastGood(path string, r OrgLastGood) error {
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return fail(abi.ReasonMalformed, "lastgood_encode", err)
	}
	b = append(b, '\n')

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fail(abi.ReasonTrustViolation, "lastgood_write", err)
	}
	tmp, err := os.CreateTemp(dir, ".fak-org-policy-lastgood-*.tmp")
	if err != nil {
		return fail(abi.ReasonTrustViolation, "lastgood_write", err)
	}
	name := tmp.Name()
	defer func() { _ = os.Remove(name) }()

	if _, err := tmp.Write(b); err != nil {
		_ = tmp.Close()
		return fail(abi.ReasonTrustViolation, "lastgood_write", err)
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return fail(abi.ReasonTrustViolation, "lastgood_write", err)
	}
	if err := tmp.Close(); err != nil {
		return fail(abi.ReasonTrustViolation, "lastgood_write", err)
	}
	if err := os.Chmod(name, 0o600); err != nil {
		return fail(abi.ReasonTrustViolation, "lastgood_write", err)
	}
	if err := os.Rename(name, path); err != nil {
		return fail(abi.ReasonTrustViolation, "lastgood_write", err)
	}
	return nil
}

// httpFetch builds the bounded default transport.
//
// The body is capped by io.LimitReader at one byte OVER the envelope budget:
// VerifyOptions.MaxBytes is only checked AFTER the read, so without this an
// endpoint streaming gigabytes is fully buffered before anything refuses it.
// Reading budget+1 lets the size check see an oversize body rather than a
// silently truncated one that would then fail as "malformed".
func httpFetch(url string, timeout time.Duration) func(context.Context) ([]byte, error) {
	if timeout <= 0 {
		timeout = DefaultOrgPullTimeout
	}
	client := &http.Client{Timeout: timeout}
	return func(ctx context.Context) ([]byte, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return nil, err
		}
		resp, err := client.Do(req)
		if err != nil {
			return nil, err
		}
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("org policy endpoint returned %s", resp.Status)
		}
		return io.ReadAll(io.LimitReader(resp.Body, int64(DefaultMaxOrgEnvelopeBytes)+1))
	}
}

// Pull runs one fetch-verify-cache-or-age-out cycle.
//
// It never returns an error separate from the result: every failure is folded
// into a posture, because a caller that handles `err != nil` by "carry on with
// what we had" is exactly the fail-open this plane forbids. The failure is
// still reported, in Result.Err and Result.Reason.
func (p *OrgPuller) Pull(ctx context.Context) OrgPullResult {
	now := p.now()

	// 1. The anchor. An un-enrolled box is INERT: it does not fetch, does not
	//    cache, and does not widen. A DAMAGED anchor is not "un-enrolled" — it
	//    is a refusal, and it lands on floor, not inert, so a caller cannot read
	//    a broken store as an opted-out one.
	opts, enrolled, err := OrgTrustAnchor(p.enrollmentPath(), now, p.RunningVersion)
	if err != nil {
		return OrgPullResult{Posture: OrgPostureFloor, Reason: "enrollment_unreadable", Err: err}
	}
	if !enrolled {
		return OrgPullResult{Posture: OrgPostureInert, Reason: "not_enrolled"}
	}
	opts.MaxBytes = DefaultMaxOrgEnvelopeBytes

	// 2. Try the live source. A missing URL is not an error — it just means this
	//    attempt cannot produce a fresh envelope, so we go straight to ageing
	//    out whatever is cached.
	fetchReason := "no_url"
	var fetchErr error
	if url := strings.TrimSpace(p.URL); url != "" || p.Fetch != nil {
		fetch := p.Fetch
		if fetch == nil {
			fetch = httpFetch(url, DefaultOrgPullTimeout)
		}
		raw, ferr := fetch(ctx)
		switch {
		case ferr != nil:
			fetchReason, fetchErr = "unreachable", ferr
		case len(raw) == 0:
			fetchReason, fetchErr = "empty_response", errors.New("org policy endpoint returned no bytes")
		default:
			v, aerr := p.accept(raw, opts)
			if aerr != nil {
				// A REFUSED envelope is not a transport hiccup — the source
				// answered and what it said did not verify. We still fall
				// through to the cache rather than widening on it, and we
				// deliberately do NOT overwrite the cache with it.
				fetchReason, fetchErr = "refused", aerr
				break
			}
			// Verified. Stamp freshness with OUR clock and persist.
			rt := v.Runtime
			res := OrgPullResult{
				Posture: OrgPostureFresh,
				Widen:   true,
				Runtime: &rt,
				Version: v.Envelope.Version,
				Issuer:  v.Envelope.Issuer,
			}
			if werr := p.saveLastGood(raw, v, now); werr != nil {
				// The envelope IS verified, so this pull may widen. But a cache
				// we could not write means the NEXT offline pull has nothing to
				// age out — say so rather than reporting a clean success.
				res.Reason, res.Err = "cache_write_failed", werr
			}
			return res
		}
	}

	// 3. No fresh envelope. Age out the cache.
	return p.fromLastGood(now, opts, fetchReason, fetchErr)
}

// accept routes verification through the ledger when there is one, so the
// anti-rollback watermark is folded in and persisted. Without a ledger it still
// verifies — it just has no high-water mark to remember.
func (p *OrgPuller) accept(raw []byte, opts VerifyOptions) (Verified, error) {
	if p.Ledger != nil {
		return p.Ledger.Accept(raw, opts)
	}
	return VerifyEnvelope(raw, opts)
}

func (p *OrgPuller) saveLastGood(raw []byte, v Verified, now time.Time) error {
	r := OrgLastGood{
		Schema:     OrgLastGoodSchema,
		Envelope:   base64.StdEncoding.EncodeToString(raw),
		Issuer:     v.Envelope.Issuer,
		Version:    v.Envelope.Version,
		ReceivedAt: now.Unix(),
		ExpiresAt:  v.Envelope.Expires,
	}
	r.Sum = lastGoodSum(r)
	return writeOrgLastGood(p.cachePath(), r)
}

// fromLastGood decides whether a previously verified envelope may still widen.
//
// Both bounds must hold, and they are different questions:
//
//	now - ReceivedAt <= MaxStaleness   — have WE heard from the org recently?
//	now < ExpiresAt                    — is the DOCUMENT still valid?
//
// The first is the one an attacker cannot restate by replaying an old document,
// which is why it is measured on our own clock and capped by a compiled-in
// constant.
//
// The cache is also RE-VERIFIED, not trusted. It was verified once, but the
// anchor may have changed since: a revoke, or a re-enroll onto a different org,
// must invalidate everything the previous anchor authorized. Re-running
// VerifyEnvelope against the CURRENT anchor makes that automatic instead of
// something a future caller has to remember to do.
func (p *OrgPuller) fromLastGood(now time.Time, opts VerifyOptions, reason string, cause error) OrgPullResult {
	floor := func(r string, err error) OrgPullResult {
		// Report the transport failure that got us here when the cache itself
		// had nothing more specific to say — otherwise "unreachable" would be
		// lost behind "no_cache".
		if err == nil {
			err = cause
		}
		return OrgPullResult{Posture: OrgPostureFloor, Reason: r, Err: err}
	}

	rec, ok, err := LoadOrgLastGood(p.cachePath())
	if err != nil {
		return floor("cache_unreadable", err)
	}
	if !ok {
		// Never fetched, and nothing cached. There is nothing to fall back to,
		// so the compiled-in floor is the whole policy.
		return floor("no_cache", nil)
	}

	age := now.Sub(time.Unix(rec.ReceivedAt, 0))
	if age > clampStaleness(p.MaxStaleness) {
		// THE line this issue exists for: past the bound, no grant survives.
		return floor("stale", nil)
	}
	if age < 0 {
		// A cache stamped in the future means the clock moved backwards or the
		// file was edited. Either way its age is unmeasurable, and an
		// unmeasurable age cannot satisfy a freshness bound.
		return floor("cache_future_dated", nil)
	}
	if rec.ExpiresAt != 0 && !now.Before(time.Unix(rec.ExpiresAt, 0)) {
		return floor("expired", nil)
	}

	raw, derr := rec.EnvelopeBytes()
	if derr != nil {
		return floor("cache_undecodable", derr)
	}
	// Re-verify against the CURRENT anchor. Deliberately VerifyEnvelope and not
	// Ledger.Accept: serving the cache must not advance any persisted
	// watermark, and re-accepting the same bytes on every offline poll would
	// churn the ledger for no new information.
	v, verr := VerifyEnvelope(raw, opts)
	if verr != nil {
		return floor("cache_no_longer_verifies", verr)
	}

	rt := v.Runtime
	return OrgPullResult{
		Posture: OrgPostureLastGood,
		Widen:   true,
		Runtime: &rt,
		Version: v.Envelope.Version,
		Issuer:  v.Envelope.Issuer,
		Age:     age,
		Reason:  reason,
		Err:     cause,
	}
}
