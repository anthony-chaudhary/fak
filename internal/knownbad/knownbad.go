// Package knownbad is the pure fold core of the fleet-wide known-bad signature
// ledger — the load-bearing spine of the blast-radius containment epic (#2712).
//
// The problem it fixes: today every containment surface is siloed
// (guardrsi.LivelockDetector is per-trace and in-memory, attemptbudget is
// per-issue-id, blockerpost is a human Slack post), so when agent-1 discovers a
// shared bug, agents 2..N have no durable, cross-trace signal to read before they
// burn a full cycle rediscovering it. This package is the substrate that turns
// "N agents each rediscover the bug" into "1 agent records it, the rest read it".
//
// The design mirror is deliberate: signature derivation is the same shape as
// guardrsi.ArgsDigest/failureHash (a sha256 over a canonical key, "sha256:"+hex),
// and tree-glob containment is the same directory-prefix rule the dos lease /
// gitgate collective-commit path uses, so a known-bad tree and a lease tree
// compare apples to apples.
//
// Everything here is a pure fold: the signature is content-free, liveness takes
// `now` as data (never reads a clock), and Match is a stateless projection over a
// slice of records. The impure shell (ledger read/write, clock, flags) lives in
// cmd/fak/knownbad.go.
//
// Out of scope for this spine (each is a separate epic child): cross-trace
// auto-promotion (W2), the blast-radius agent set (W3), dispatcher hold wiring
// (W4), fixer election (W5), and the operator card (W7). The witness-gated
// auto-release (W6, #2718) folds through this core: WithResolve stamps a
// superseding resolved row, and Match / FindLatestLive read a signature's state as
// its LATEST row (append-to-supersede) — so a resolved (or expired) latest row
// retracts the signature even though its earlier open rows still sit on the ledger.
//
// GC IS IN SCOPE AND HAS LANDED (#3471) — this paragraph used to say the opposite,
// and the append-to-supersede design makes that a load-bearing correction: because
// resolve/revoke/claim each ADD a row and DefaultRecordTTLSeconds only bounds MATCH
// liveness, superseded and expired rows accrete on disk forever unless something
// prunes them. Compact/CompactStats below are that fold (kept minimal current state:
// every live signature, plus a bounded DefaultCompactKeepTerminal tail of
// resolved/revoked history), driven by the `fak knownbad compact` verb in the shell;
// the read side is bounded separately by the dispatch route's stat-keyed ledger cache
// so the hot path no longer re-parses the whole file per tick. Read this before
// re-filing "the ledger grows without bound" — the bound exists.
package knownbad

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"path"
	"sort"
	"strings"
)

// Schema is the JSONL record tag every known-bad row carries. A reader filters a
// shared, multi-writer ledger to rows bearing exactly this schema so foreign or
// future rows are ignored rather than misread.
const Schema = "fak.known-bad.v1"

// StatusOpen is the only live status this spine writes. A consumer child (W6
// auto-release) closes a signature by appending a superseding row; the plain
// liveness check here only treats "open" (case-insensitive) as active.
const StatusOpen = "open"

// StatusResolved is the terminal status W6 (auto-release, #2718) stamps onto a
// superseding row to close a signature on a WITNESSED fix. A resolved row is never
// Live (Live is true only for "open"), so the dispatcher scope-hold (W4), the
// fixer-election lease consumer (W5), and the match primitive all drop it the
// moment the row is appended — "releasing the hold is a ledger status flip the
// dispatcher already re-reads each tick". The witness kind (green tests, dos
// verify) is carried in Witness; an empty Witness means the resolve was not
// evidence-backed and must be refused by the shell.
const StatusResolved = "resolved"

// StatusRevoked is the terminal status the UNWITNESSED release valve (W8, #2720)
// stamps onto a superseding row to retract a signature WITHOUT a witnessed fix: a
// mis-recorded, stale, or wrong signature an operator kills by hand. It is the
// deliberate counterpart to StatusResolved — resolve closes a signature on proven
// evidence (a green/verify), revoke closes it on operator judgement (the failure
// was never real, or the tree was wrong) with the reason stamped in RevokeReason.
// Like a resolved row it is never Live (Live is true only for "open"), so the
// dispatcher scope-hold (W4), the fixer-election lease consumer (W5), Match, and
// the operator card (W7) all drop it the moment the row is appended — the SAME
// supersede path a resolve takes, minus the witness gate. Distinct from an EXPIRED
// row (a live "open" row whose TTL has lapsed): both stop matching, but a revoke is
// an explicit human retraction the ledger records, an expiry is the passive
// self-healing a bounded TTL gives every signature.
const StatusRevoked = "revoked"

// DefaultLedgerRel is the repo-relative, fleet-visible ledger the cmd shell
// appends to by default — the same docs/nightrun/*.jsonl idiom the other fak
// ledgers use.
const DefaultLedgerRel = "docs/nightrun/known-bad.jsonl"

// DefaultRecordTTLSeconds is the bounded default lifetime a `record` stamps when the
// discoverer does not pass an explicit --ttl: 45 minutes. It exists so a signature
// SELF-HEALS if its fix lands quietly (or the discovery was spurious) and nobody runs
// `resolve`/`revoke` — the ledger stops matching on its own rather than parking the
// fleet forever on a stale row. It is a floor, not a cap: an explicit `--ttl 0`
// still means "no expiry" for a signature an operator is sure is durable, and any
// positive `--ttl` overrides the default. The window is a few dispatch cycles — long
// enough that a real shared bug is not un-held before a fixer is elected (W5), short
// enough that a forgotten signature does not outlive the failure it names.
const DefaultRecordTTLSeconds int64 = 45 * 60

// DefaultCompactKeepTerminal bounds how many terminally-retracted signatures a
// `fak knownbad compact` keeps for audit: the most-recently-retracted resolved/revoked
// rows. LIVE signatures are ALWAYS kept (dropping one would silently un-hold the fleet);
// this only bounds the dead-history tail so the ledger — append-to-supersede, so a
// resolve/revoke adds a row rather than removing one — does not grow without bound. A
// negative value keeps every terminal row (unbounded audit).
const DefaultCompactKeepTerminal = 50

// Record is one appended known-bad ledger row: a durable, cross-trace statement
// that a shared failure exists over a tree. The required fields are the ones the
// epic names; Note and FailureHash are optional context (omitted when empty) —
// FailureHash is the bridge W2 uses to correlate a ledger row back to a
// guardrsi trace.
type Record struct {
	Schema           string   `json:"schema"`
	Signature        string   `json:"signature"`
	ReasonClass      string   `json:"reason_class"`
	TreeGlobs        []string `json:"tree_globs"`
	DiscoveredBy     string   `json:"discovered_by,omitempty"`
	DiscoveredAtUnix int64    `json:"discovered_at_unix"`
	TTLSeconds       int64    `json:"ttl_seconds"`
	Status           string   `json:"status"`
	Note             string   `json:"note,omitempty"`
	FailureHash      string   `json:"failure_hash,omitempty"`
	// DerivedFrom is the parent signature (or candidate id) this row was derived
	// from — the attempt-genealogy edge (#4100). A rejected attempt links to the
	// parent it mutated, turning the flat ledger into a traversable attempt tree so
	// a later loop can read "approach X was tried and lost" instead of re-deriving
	// it. It is a BIRTH attribute (set once at record time, like FailureHash), never
	// a lifecycle transition. Empty on a root/hand-recorded row; omitempty keeps a
	// pre-#4100 row byte-identical, so the genealogy is fully backward-compatible.
	DerivedFrom string `json:"derived_from,omitempty"`
	// ClaimedBy names the single elected fixer that holds the exclusive lease
	// electing it as this signature's sole owner-of-the-fix (W5, #2717). It is
	// bookkeeping the scope-hold (W4) and operator (W7) surfaces read to point a
	// parked agent at the owner — the exactly-one invariant itself is enforced by
	// the exclusive lease at refs/fak/locks/<LeaseID(signature)>, never by this
	// stamp. Empty on an unclaimed row; omitempty keeps a pre-W5 row byte-identical.
	ClaimedBy string `json:"claimed_by,omitempty"`
	// ClaimedAtUnix is the unix-seconds instant the claim row was appended (0 when
	// unclaimed) — the companion to ClaimedBy.
	ClaimedAtUnix int64 `json:"claimed_at_unix,omitempty"`
	// ResolvedBy names the agent (or operator) that closed this signature on a
	// WITNESSED fix (W6, #2718). It is bookkeeping the operator card (W7) reads to
	// point a human at who released the fleet — the witness gate itself is enforced
	// by the cmd shell (it runs the green/verify BEFORE appending this row), never by
	// this stamp. Empty on an open/claimed row; omitempty keeps a pre-W6 row
	// byte-identical.
	ResolvedBy string `json:"resolved_by,omitempty"`
	// ResolvedAtUnix is the unix-seconds instant the resolve row was appended (0 when
	// unresolved) — the companion to ResolvedBy.
	ResolvedAtUnix int64 `json:"resolved_at_unix,omitempty"`
	// Witness names the independent evidence the fix landed: "tests" for a green
	// `fak affected`/`go test` over the broken tree, "verify" for a `dos verify`
	// binding the fixer's commit to the signature's tree. Empty on a non-resolved
	// row; the shell refuses to append a resolved row with an empty Witness (that
	// would be a self-report release, the exact failure W6 exists to forbid).
	Witness string `json:"witness,omitempty"`
	// RevokedBy names the operator (or agent) that retracted this signature WITHOUT a
	// witnessed fix (W8, #2720) — the unwitnessed release valve for a mis-recorded or
	// stale signature. It is the deliberate escape hatch resolve does NOT provide (a
	// resolve demands evidence), so unlike ResolvedBy it carries no witness — instead
	// RevokeReason states WHY the human killed the row. Empty on a non-revoked row;
	// omitempty keeps a pre-W8 row byte-identical.
	RevokedBy string `json:"revoked_by,omitempty"`
	// RevokedAtUnix is the unix-seconds instant the revoke row was appended (0 when not
	// revoked) — the companion to RevokedBy.
	RevokedAtUnix int64 `json:"revoked_at_unix,omitempty"`
	// OccurrenceCount is how many crash events this row COALESCES (#3586): one root
	// cause that killed N workers is one row carrying N, not N rows. It is stamped by
	// CoalesceCrashes/WithOccurrences and climbs while the signature's window stays
	// live. Zero on a hand-recorded row; omitempty keeps a pre-#3586 row byte-identical.
	OccurrenceCount int64 `json:"occurrence_count,omitempty"`
	// LastSeenAtUnix is the newest observation instant folded into OccurrenceCount —
	// the companion that makes a coalesced row readable as a span ("first seen at
	// DiscoveredAtUnix, still crashing at LastSeenAtUnix") rather than a bare count. It
	// only ever moves forward. Zero on a non-coalesced row.
	LastSeenAtUnix int64 `json:"last_seen_at_unix,omitempty"`
	// RevokeReason is the free-text justification the operator stamps when revoking:
	// "signature was spurious", "tree was wrong", "fix landed without a green".
	// Unlike Witness (a closed vocabulary of evidence kinds) this is human prose — a
	// revoke is a JUDGEMENT, not a proof, so it records the why for the audit trail
	// rather than a machine-checkable witness. Empty on a non-revoked row.
	RevokeReason string `json:"revoke_reason,omitempty"`
}

// Query is a match request: the tree globs a worker (or the dispatcher) is about
// to touch and wants to check against the live known-bad signatures.
type Query struct {
	TreeGlobs []string
}

// isWindowsDrive reports whether s has a Windows drive-letter prefix (e.g.
// "C:/", "C:\", or bare "C:").
func isWindowsDrive(s string) bool {
	if len(s) < 2 {
		return false
	}
	c := s[0]
	if !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
		return false
	}
	if s[1] != ':' {
		return false
	}
	return len(s) == 2 || s[2] == '/' || s[2] == '\\'
}

// NormalizeTree canonicalizes one tree glob to a repo-relative directory/file
// prefix so a lease-style "dir/**" and a bare file path under it compare as the
// same containment: backslash->slash, strip any trailing "/**" or "/*" (or bare
// "**"/"*") glob stars, path.Clean. It returns "" for anything that cannot be a
// repo-relative tree — empty, absolute, an escape above the root (".", "..",
// "../x"), or a NUL — so a bad glob is dropped rather than silently matching
// everything.
func NormalizeTree(raw string) string {
	s := strings.TrimSpace(strings.ReplaceAll(raw, "\\", "/"))
	if s == "" || strings.HasPrefix(s, "/") || strings.ContainsRune(s, 0) || isWindowsDrive(s) {
		return ""
	}
	s = strings.TrimRight(s, "/")
	// Peel trailing recursive/simple stars, e.g. "a/**", "a/*/**", "a/*".
	for {
		switch {
		case strings.HasSuffix(s, "/**"):
			s = strings.TrimSuffix(s, "/**")
		case strings.HasSuffix(s, "/*"):
			s = strings.TrimSuffix(s, "/*")
		case s == "**" || s == "*":
			// A bare star is "everything" — not a repo-relative tree; drop it.
			return ""
		default:
			goto cleaned
		}
	}
cleaned:
	s = strings.TrimRight(s, "/")
	if s == "" {
		return ""
	}
	p := path.Clean(s)
	if p == "." || p == ".." || strings.HasPrefix(p, "../") {
		return ""
	}
	return p
}

// normalizeAll normalizes, drops empties, dedups, and sorts a set of globs. The
// sort+dedup is what makes the signature order-independent: two agents supplying
// the same trees in a different order produce the same canonical key.
func normalizeAll(globs []string) []string {
	seen := make(map[string]struct{}, len(globs))
	out := make([]string, 0, len(globs))
	for _, g := range globs {
		n := NormalizeTree(g)
		if n == "" {
			continue
		}
		if _, dup := seen[n]; dup {
			continue
		}
		seen[n] = struct{}{}
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// NormalizeAll is the exported form of the glob canonicalizer the record shape
// stores and the signature folds over, so the cmd shell and any consumer child
// normalize a tree set through this one function.
func NormalizeAll(globs []string) []string { return normalizeAll(globs) }

// treeContains reports whether normalized prefix `tree` contains path `p`: equal,
// or p sits beneath tree/ — the same directory-prefix rule gitgate.TreeContains
// uses for lease trees.
func treeContains(tree, p string) bool {
	return p == tree || strings.HasPrefix(p, tree+"/")
}

// TreesIntersect reports whether any glob in a overlaps any glob in b after
// normalization — overlap meaning one normalized prefix contains the other. This
// is the "does the requested tree touch a known-bad tree" primitive both Match
// and the dispatcher hold check fold over.
func TreesIntersect(a, b []string) bool {
	na := normalizeAll(a)
	nb := normalizeAll(b)
	for _, x := range na {
		for _, y := range nb {
			if treeContains(x, y) || treeContains(y, x) {
				return true
			}
		}
	}
	return false
}

// Signature is a stable, content-free id for a shared failure cause: sha256 over
// a canonical key of (reason_class, sorted+deduped normalized tree globs,
// optional guardrsi failure hash), rendered "sha256:"+hex like
// guardrsi.failureHash. Two agents hitting the same shared cause — even with the
// globs supplied in a different order or with redundant "/**" suffixes — produce
// the same signature; a different reason class or a disjoint tree produces a
// different one.
func Signature(reasonClass string, treeGlobs []string, failureHash string) string {
	key := strings.TrimSpace(reasonClass) + "\x00" +
		strings.Join(normalizeAll(treeGlobs), "\x00") + "\x00" +
		strings.TrimSpace(failureHash)
	sum := sha256.Sum256([]byte(key))
	return "sha256:" + hex.EncodeToString(sum[:])
}

// leaseIDTag prefixes a fixer-election lease id so its ref basename under
// refs/fak/locks/ is greppable and cannot collide with another lease kind, and so
// the id is a valid ref segment even if a signature's body ever led with a dash.
const leaseIDTag = "knownbad-"

// LeaseID derives the exclusive-lease ref id that elects the single fixer for a
// signature (W5, #2717): the mutex is a lease at refs/fak/locks/<LeaseID>, so two
// agents deriving the id from the SAME signature contend for the SAME ref and the
// arbiter admits exactly one holder. The "sha256:" scheme prefix and any byte that
// is not ref-safe (leaseref.validID: [A-Za-z0-9._-], no colon) are dropped, yielding
// one safe segment "knownbad-<hex>". A signature with no usable content returns "",
// which the shell refuses rather than acquiring a degenerate lease.
func LeaseID(signature string) string {
	s := strings.TrimSpace(signature)
	if i := strings.IndexByte(s, ':'); i >= 0 {
		s = s[i+1:] // drop the "sha256:" scheme — a colon is ref-illegal
	}
	var b strings.Builder
	for _, c := range []byte(s) {
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9',
			c == '-', c == '_', c == '.':
			b.WriteByte(c)
		}
	}
	if b.Len() == 0 {
		return ""
	}
	return leaseIDTag + b.String()
}

// NewRecord builds a well-formed open record: normalized+deduped tree globs, the
// derived signature, and the schema/status stamped. discoveredAtUnix and
// ttlSeconds are supplied as data (clock-injected by the shell) so the
// constructor stays pure. A caller must check the returned record's TreeGlobs is
// non-empty — an all-invalid glob set yields a record that can never match.
func NewRecord(reasonClass string, treeGlobs []string, note, discoveredBy, failureHash string, discoveredAtUnix, ttlSeconds int64) Record {
	norm := normalizeAll(treeGlobs)
	if ttlSeconds < 0 {
		ttlSeconds = 0
	}
	return Record{
		Schema:           Schema,
		Signature:        Signature(reasonClass, treeGlobs, failureHash),
		ReasonClass:      strings.TrimSpace(reasonClass),
		TreeGlobs:        norm,
		DiscoveredBy:     strings.TrimSpace(discoveredBy),
		DiscoveredAtUnix: discoveredAtUnix,
		TTLSeconds:       ttlSeconds,
		Status:           StatusOpen,
		Note:             strings.TrimSpace(note),
		FailureHash:      strings.TrimSpace(failureHash),
	}
}

// Live reports whether a record is still an active known-bad signal at nowUnix:
// status "open" (case-insensitive) AND either no TTL (<=0, never expires) or not
// yet expired (discovered_at + ttl > now). Liveness is a pure function of the row
// plus the supplied clock — never a wall-clock read.
//
// Both terminal statuses (resolved on a witness, revoked by an operator) and a
// lapsed TTL make this false, so all three retraction paths — witnessed resolve
// (W6), unwitnessed revoke (W8), and passive expiry (W8) — flow through the same
// supersede-aware Match/FindLatestLive fold: a signature stops matching the moment
// its LATEST row is not Live, whether that row is resolved, revoked, or an open row
// whose bounded TTL has lapsed.
func (r Record) Live(nowUnix int64) bool {
	if !strings.EqualFold(strings.TrimSpace(r.Status), StatusOpen) {
		return false
	}
	if r.TTLSeconds <= 0 {
		return true
	}
	return r.DiscoveredAtUnix+r.TTLSeconds > nowUnix
}

// Match folds the ledger to the live records whose tree globs intersect the
// query's, preserving first-seen ledger order. Pure: nowUnix is supplied as data,
// so the same (records, query, now) always yields the same matches. An empty result
// means the requested tree is clear of every live known-bad signature.
//
// SUPERSEDE-AWARE (the invariant W6 depends on): a signature's state is its LATEST
// row (append-to-supersede), not each row independently. So a signature whose most
// recent row is resolved/expired does NOT match even though its earlier open rows
// are still on the ledger — otherwise a resolve could never clear the hold, since
// the original open row would keep matching forever. Collapsing to the latest row
// per signature before the liveness/intersection test is what makes WithResolve's
// "supersedes an earlier row" actually retract the hold.
func Match(records []Record, q Query, nowUnix int64) []Record {
	// Latest row per signature (last write wins), and the first-seen order so the
	// result is deterministic and stable against the earlier per-row behavior.
	latest := make(map[string]Record, len(records))
	order := make([]string, 0, len(records))
	for _, rec := range records {
		sig := strings.TrimSpace(rec.Signature)
		if _, seen := latest[sig]; !seen {
			order = append(order, sig)
		}
		latest[sig] = rec
	}
	var out []Record
	for _, sig := range order {
		rec := latest[sig]
		if !rec.Live(nowUnix) {
			continue
		}
		if TreesIntersect(rec.TreeGlobs, q.TreeGlobs) {
			out = append(out, rec)
		}
	}
	return out
}

// LiveRecords collapses the ledger to the current LIVE record of EVERY signature —
// the same supersede-aware fold Match applies (latest row per signature, in
// first-seen order), but WITHOUT a tree query: it answers "which known-bad
// signatures are live right now" rather than "which intersect this tree". The
// operator blast card (W7, #2719) folds over exactly this set — one card per live
// signature — so a resolved/expired signature whose earlier open rows still sit on
// the ledger is dropped, same as Match drops it. Pure: nowUnix is data.
func LiveRecords(records []Record, nowUnix int64) []Record {
	latest := make(map[string]Record, len(records))
	order := make([]string, 0, len(records))
	for _, rec := range records {
		sig := strings.TrimSpace(rec.Signature)
		if _, seen := latest[sig]; !seen {
			order = append(order, sig)
		}
		latest[sig] = rec
	}
	var out []Record
	for _, sig := range order {
		rec := latest[sig]
		if rec.Live(nowUnix) {
			out = append(out, rec)
		}
	}
	return out
}

// Claimed reports whether a row records an elected fixer (a non-empty ClaimedBy).
func (r Record) Claimed() bool { return strings.TrimSpace(r.ClaimedBy) != "" }

// Resolved reports whether a row closes a signature on a witnessed fix (status
// "resolved", case-insensitive). A resolved row is the terminal state W6 stamps
// onto a superseding row; Live is false for it, so the dispatcher hold and the
// match primitive both drop it.
func (r Record) Resolved() bool {
	return strings.EqualFold(strings.TrimSpace(r.Status), StatusResolved)
}

// WithResolve returns a copy of r stamped as resolved with the resolver, the
// resolve instant, and the independent witness kind that gated the release. The
// signature/tree/reason are left untouched: a resolve SUPERSEDES an earlier row
// for the same signature via the append-to-supersede idiom, it does not mutate
// the failure it points at. The caller MUST have run the witness (green tests /
// dos verify) BEFORE appending the returned row — this method records the
// evidence-backed release, it does not check the evidence.
func (r Record) WithResolve(resolvedBy string, resolvedAtUnix int64, witness string) Record {
	out := r
	out.Status = StatusResolved
	out.ResolvedBy = strings.TrimSpace(resolvedBy)
	out.ResolvedAtUnix = resolvedAtUnix
	out.Witness = strings.TrimSpace(witness)
	// The claim bookkeeping is preserved so the operator card (W7) can still name
	// the fixer that owned the fix, even after the release.
	return out
}

// Revoked reports whether a row retracts a signature by operator judgement (status
// "revoked", case-insensitive) rather than a witnessed fix. Like a resolved row it is
// terminal — Live is false for it, so the dispatcher hold, Match, and the operator
// card all drop it.
func (r Record) Revoked() bool {
	return strings.EqualFold(strings.TrimSpace(r.Status), StatusRevoked)
}

// WithRevoke returns a copy of r stamped as revoked with the operator, the revoke
// instant, and the human justification. It is the UNWITNESSED counterpart to
// WithResolve: a revoke closes a signature on judgement (the failure was spurious,
// the tree was wrong, the fix landed without a green) NOT on evidence, so it stamps a
// prose reason instead of a witness kind. Like a resolve it SUPERSEDES an earlier row
// for the same signature via append-to-supersede, leaving the signature/tree/reason
// untouched. There is no gate to run first — the whole point of revoke is that it does
// not require one — so this is the terminal stamp a human's retraction records.
func (r Record) WithRevoke(revokedBy string, revokedAtUnix int64, revokeReason string) Record {
	out := r
	out.Status = StatusRevoked
	out.RevokedBy = strings.TrimSpace(revokedBy)
	out.RevokedAtUnix = revokedAtUnix
	out.RevokeReason = strings.TrimSpace(revokeReason)
	// Claim bookkeeping is preserved (same as WithResolve) so the operator card can
	// still name who had owned the now-retracted fix.
	return out
}

// WithClaim returns a copy of r stamped with the elected fixer and the claim
// instant, leaving the signature/tree/reason untouched: a claim SUPERSEDES an
// earlier row for the same signature via the append-to-supersede idiom, it does not
// mutate the failure it points at. Recording the claimant is bookkeeping the
// scope-hold and operator surfaces read — the exactly-one invariant is enforced by
// the exclusive lease, not this stamp — so a caller must have WON the lease before
// appending the returned row.
func (r Record) WithClaim(claimant string, claimedAtUnix int64) Record {
	out := r
	out.ClaimedBy = strings.TrimSpace(claimant)
	out.ClaimedAtUnix = claimedAtUnix
	return out
}

// WithDerivedFrom returns a copy of r stamped with the parent signature (or
// candidate id) it was derived from — the attempt-genealogy edge (#4100). Unlike
// the lifecycle With* stamps (claim/resolve/revoke), which supersede an earlier
// row on a state transition, this records a BIRTH attribute set at record time, so
// it is a builder that threads the optional edge onto NewRecord's result rather
// than churning the constructor's many call sites. An empty parent leaves the row
// flat and byte-identical to a pre-#4100 row (omitempty drops the key), so the
// genealogy is opt-in and fully backward-compatible.
func (r Record) WithDerivedFrom(parent string) Record {
	out := r
	out.DerivedFrom = strings.TrimSpace(parent)
	return out
}

// Derived reports whether a row links to a parent it was derived from (a non-empty
// DerivedFrom) — the predicate a lineage read folds over to distinguish a root
// signature from a rejected-attempt child (#4100).
func (r Record) Derived() bool { return strings.TrimSpace(r.DerivedFrom) != "" }

// FindLatestLive returns the current LIVE record carrying signature and whether one
// was found. Later rows supersede earlier ones (the append-to-supersede idiom), so
// the LATEST row is the signature's current state — the tree to lease over and
// whether a claim already stands. The latest row must itself be live: a signature
// whose most recent row is resolved/expired has no live failure to elect a fixer for
// (or to resolve again), EVEN THOUGH its earlier open rows are still on the ledger.
// Checking "the latest row, is it live" rather than "the last row that is live" is
// what makes a resolve/expiry actually retract the signature — the same supersede
// discipline Match applies.
func FindLatestLive(records []Record, signature string, nowUnix int64) (Record, bool) {
	sig := strings.TrimSpace(signature)
	var latest Record
	seen := false
	for _, r := range records {
		if strings.TrimSpace(r.Signature) == sig {
			latest = r
			seen = true
		}
	}
	if !seen || !latest.Live(nowUnix) {
		return Record{}, false
	}
	return latest, true
}

// LatestState folds the ledger to a signature's LATEST row and classifies WHY it is
// not actionable when it is not live — the distinction the shell needs to choose
// between a plain "never recorded" usage error and a STRUCTURED refuse for acting on a
// signature that WAS recorded but has since been retracted (resolved, revoked, or
// expired). It returns the latest row, whether the signature was seen at all, and a
// terse state string: "live" (open + unexpired), "resolved", "revoked", "expired" (an
// open row whose TTL lapsed), or "" when the signature was never recorded. Same
// append-to-supersede discipline as FindLatestLive: only the LATEST row decides.
func LatestState(records []Record, signature string, nowUnix int64) (rec Record, seen bool, state string) {
	sig := strings.TrimSpace(signature)
	for _, r := range records {
		if strings.TrimSpace(r.Signature) == sig {
			rec = r
			seen = true
		}
	}
	if !seen {
		return Record{}, false, ""
	}
	switch {
	case rec.Live(nowUnix):
		return rec, true, "live"
	case rec.Resolved():
		return rec, true, "resolved"
	case rec.Revoked():
		return rec, true, "revoked"
	default:
		// An open row whose bounded TTL has lapsed — the passive expiry path. Any
		// other non-live status a future child adds also lands here as "expired"
		// (retracted-by-time) unless it declares its own predicate above.
		return rec, true, "expired"
	}
}

// CompactStats reports what a Compact fold dropped, so the shell can print an honest
// before/after and a test can assert the reduction. The arithmetic balances:
// InputRows - KeptRows == SupersededDropped + ExpiredDropped + TerminalDropped, and
// KeptRows == LiveKept + TerminalKept.
type CompactStats struct {
	InputRows         int // rows read from the ledger
	KeptRows          int // rows written back (one latest row per kept signature)
	Signatures        int // distinct signatures seen
	LiveKept          int // live signatures kept (always ALL of them)
	TerminalKept      int // resolved/revoked signatures kept (the bounded tail)
	SupersededDropped int // non-latest rows folded away by the append-to-supersede collapse
	ExpiredDropped    int // signatures whose latest row lapsed its TTL (passive self-heal)
	TerminalDropped   int // resolved/revoked signatures beyond the kept tail
}

// Compact folds a multi-writer append ledger down to its minimal current state — the GC
// the append-to-supersede design otherwise lacks. Because resolve/revoke/claim each ADD
// a row (never remove one) and the bounded TTL only gates MATCH liveness, expired and
// superseded rows accrete on disk forever; Compact prunes them.
//
// It keeps exactly the LATEST row per signature (earlier rows are dead once a later row
// exists), then classifies each signature by that latest row:
//
//   - LIVE (open, unexpired): always kept — dropping one would silently release a hold
//     the fleet is still honoring.
//   - RESOLVED / REVOKED (an explicit witnessed-or-operator retraction with audit value):
//     kept up to a bounded tail of the `keepTerminal` most-recently-retracted, so recent
//     history survives without the ledger growing without bound.
//   - EXPIRED (an open row whose bounded TTL lapsed — passive self-healing, no audit
//     value): dropped.
//
// keepTerminal < 0 keeps every terminal row (unbounded audit); >= 0 caps the tail (0
// drops all terminal history, live-only). Kept rows are emitted in original ledger append
// order, so the rewrite is stable and a re-compact of an already-compact ledger is a
// no-op. Pure: nowUnix is data, no I/O.
func Compact(records []Record, nowUnix int64, keepTerminal int) ([]Record, CompactStats) {
	stats := CompactStats{InputRows: len(records)}

	// Latest row index per signature, in first-seen order (append-to-supersede: the last
	// row for a signature is its current state).
	latestIdx := make(map[string]int, len(records))
	order := make([]string, 0, len(records))
	for i, rec := range records {
		sig := strings.TrimSpace(rec.Signature)
		if _, seen := latestIdx[sig]; !seen {
			order = append(order, sig)
		}
		latestIdx[sig] = i
	}
	stats.Signatures = len(order)

	// Classify each signature by its latest row. Terminal (resolved/revoked) signatures
	// carry their latest-row index so the tail can keep only the most-recently-retracted.
	keep := make(map[string]bool, len(order))
	type termSig struct {
		sig string
		idx int
	}
	var terminals []termSig
	for _, sig := range order {
		rec := records[latestIdx[sig]]
		switch {
		case rec.Live(nowUnix):
			keep[sig] = true
			stats.LiveKept++
		case rec.Resolved() || rec.Revoked():
			terminals = append(terminals, termSig{sig: sig, idx: latestIdx[sig]})
		default:
			// An open row whose bounded TTL has lapsed (or any other non-live, non-terminal
			// status): passive self-healing with no audit value — dropped.
			stats.ExpiredDropped++
		}
	}

	// Bound the terminal tail to the keepTerminal most-recently-retracted (highest
	// latest-row index). keepTerminal < 0 keeps all of them.
	if keepTerminal < 0 || len(terminals) <= keepTerminal {
		for _, t := range terminals {
			keep[t.sig] = true
		}
		stats.TerminalKept = len(terminals)
	} else {
		sort.SliceStable(terminals, func(i, j int) bool { return terminals[i].idx > terminals[j].idx })
		for _, t := range terminals[:keepTerminal] {
			keep[t.sig] = true
		}
		stats.TerminalKept = keepTerminal
		stats.TerminalDropped = len(terminals) - keepTerminal
	}

	// Emit each kept signature's LATEST row, in original ledger append order, so the
	// rewrite is stable (byte-for-byte re-compactable).
	out := make([]Record, 0, len(keep))
	for i, rec := range records {
		sig := strings.TrimSpace(rec.Signature)
		if latestIdx[sig] != i {
			continue // a superseded row — folded away by the collapse
		}
		if !keep[sig] {
			continue // expired, or a terminal row beyond the kept tail
		}
		out = append(out, rec)
	}
	stats.KeptRows = len(out)
	stats.SupersededDropped = stats.InputRows - stats.Signatures
	return out, stats
}

// ParseLedger folds JSONL bytes into records. It is deliberately robust for a
// shared, multi-writer append ledger: blank lines are skipped, a line that is not
// valid JSON or does not carry this package's Schema is skipped (a torn append
// from a crashed peer, or a foreign row in a co-located ledger, must not blind
// every reader). It never errors — the worst a bad line can do is not be seen.
func ParseLedger(data []byte) []Record {
	var out []Record
	for _, line := range strings.Split(string(data), "\n") {
		s := strings.TrimSpace(line)
		if s == "" {
			continue
		}
		var rec Record
		if err := json.Unmarshal([]byte(s), &rec); err != nil {
			continue
		}
		if rec.Schema != Schema {
			continue
		}
		out = append(out, rec)
	}
	return out
}

// MarshalLine renders one record as a single compact JSONL line (no trailing
// newline). The shell appends "\n" when it writes, matching the other fak
// ledgers' append idiom.
func MarshalLine(rec Record) (string, error) {
	b, err := json.Marshal(rec)
	if err != nil {
		return "", err
	}
	return string(b), nil
}
