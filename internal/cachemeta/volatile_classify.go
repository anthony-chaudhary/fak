package cachemeta

import (
	"fmt"
	"regexp"
	"strings"
)

// volatile_classify.go — issue #3341 (epic #2783 cache-value; relates #1540 preserve
// stable prefix bytes). Study-borrow of headroom's `detect_volatile_content` /
// `_classify_token` (Apache-2.0 @ 38074888) — the borrow is the CLASS COVERAGE + NAMING
// + operator warning, clean-room in Go; not the parser style.
//
// What this closes. fak already detects SOME volatile tokens in the cache-prefix head, but
// only as a bare bool used to MOVE blocks (agent.headValueIsVolatile matches canonical
// UUID + sub-day ISO-8601 and returns one bool; the M2 anchor reorders on it). When a
// customer's cache hit-rate silently collapses, the only signal today is an internal
// counter (metrics `outcome="volatile_head"`) — it never says WHICH token class caused
// the drift, and it misses JWTs and hex hashes entirely. This file adds the missing
// diagnosis: a read-only classifier that reports NAMED per-class counts and renders one
// operator-visible warning line. It is diagnosis only — it never rewrites the prompt
// bytes (the M2 anchor still owns reordering; this composes with, and does not replace it).

// VolatileClass names a category of per-request token whose presence in a cache-prefix
// head changes the bytes between turns and busts the prefix a breakpoint is meant to
// secure. The name is the operator-facing label emitted in the warning.
type VolatileClass string

const (
	// VolatileUUID is a canonical UUID/GUID (8-4-4-4-12 hex) — the standard nonce/request-id.
	VolatileUUID VolatileClass = "uuid"
	// VolatileISO8601 is an ISO-8601 timestamp carrying a sub-day TIME component (a `T`/space
	// then HH:MM); a date-ONLY token is byte-stable across a session and is NOT this class.
	VolatileISO8601 VolatileClass = "iso8601"
	// VolatileJWT is a JSON Web Token (three base64url segments), which fak's bool check misses.
	VolatileJWT VolatileClass = "jwt"
	// VolatileHexHash is a bare hex digest at a canonical width (MD5/SHA-1/SHA-256/SHA-512),
	// which fak's bool check misses.
	VolatileHexHash VolatileClass = "hex_hash"
)

// volatileClassOrder is the stable render/iteration order. The classifier and the warning
// iterate this slice — never a map — so counts and the emitted line are deterministic.
var volatileClassOrder = []VolatileClass{VolatileUUID, VolatileISO8601, VolatileJWT, VolatileHexHash}

// Only UNAMBIGUOUS shapes are matched: a false positive merely warns (fail-safe), while a
// missed busting token is the real harm. Each pattern mirrors a headroom class.
var (
	// reUUID — canonical UUID/GUID; same shape agent.volUUID uses, kept in-package so
	// cachemeta stays a self-contained consumer (no import of internal/agent).
	reUUID = regexp.MustCompile(`[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}`)
	// reISO8601 — ISO-8601 date with a trailing HH:MM (sub-day resolution changes faster
	// than the ephemeral cache TTL). A date-only token lacks the time and is not matched.
	reISO8601 = regexp.MustCompile(`[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt ][0-9]{2}:[0-9]{2}`)
	// reJWT — header.payload.signature in base64url. The header nearly always begins with
	// the base64url of `{"` = `eyJ`; anchoring on it keeps an ordinary dotted phrase
	// (a.b.c) from registering as a false JWT.
	reJWT = regexp.MustCompile(`eyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}`)
	// reHexHash — a contiguous hex run at a canonical digest width (MD5 32, SHA-1 40,
	// SHA-256 64, SHA-512 128). Word-bounded so a UUID's short hex groups, or a longer hex
	// blob, do not register a spurious hash.
	reHexHash = regexp.MustCompile(`\b(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{40}|[0-9a-fA-F]{64}|[0-9a-fA-F]{128})\b`)
)

var volatileClassPattern = map[VolatileClass]*regexp.Regexp{
	VolatileUUID:    reUUID,
	VolatileISO8601: reISO8601,
	VolatileJWT:     reJWT,
	VolatileHexHash: reHexHash,
}

// VolatileReport is the NAMED per-class classification of the volatile tokens found in a
// cache-prefix head. It is a pure read-only diagnosis: constructing it never mutates the
// scanned bytes.
type VolatileReport struct {
	// Counts is the per-class occurrence count. A class with zero matches is ABSENT from
	// the map (never a 0 entry), so len(Counts) is the number of distinct classes present.
	Counts map[VolatileClass]int
}

// ClassifyVolatile scans a cache-prefix head (the raw `system`/`tools` bytes, or any
// candidate prefix span) and returns the named per-class volatile-token counts. It sees a
// token embedded anywhere in the head — a UUID in a tool description, a JWT in a system
// block. An empty head yields an empty report. Read-only: `head` is never modified.
func ClassifyVolatile(head []byte) VolatileReport {
	rep := VolatileReport{Counts: map[VolatileClass]int{}}
	if len(head) == 0 {
		return rep
	}
	for _, class := range volatileClassOrder {
		if n := len(volatileClassPattern[class].FindAll(head, -1)); n > 0 {
			rep.Counts[class] = n
		}
	}
	return rep
}

// Total is the number of volatile tokens across all classes.
func (r VolatileReport) Total() int {
	t := 0
	for _, class := range volatileClassOrder {
		t += r.Counts[class]
	}
	return t
}

// Unstable reports whether any volatile token was found — i.e. whether the scanned head
// is byte-unstable across turns and cannot anchor a stable cache prefix.
func (r VolatileReport) Unstable() bool { return r.Total() > 0 }

// Classes lists the volatile classes present, in the stable render order (uuid, iso8601,
// jwt, hex_hash). Empty when the head is stable.
func (r VolatileReport) Classes() []VolatileClass {
	out := make([]VolatileClass, 0, len(r.Counts))
	for _, class := range volatileClassOrder {
		if r.Counts[class] > 0 {
			out = append(out, class)
		}
	}
	return out
}

// Warning renders one operator-visible line naming the volatile classes present with their
// counts, e.g. `cache prefix unstable: uuid=1, jwt=1, hex_hash=1 — move dynamic values out
// of the system prompt`. It returns the empty string when the head is stable, so a caller
// can gate on `if w := rep.Warning(); w != "" { ... }`. Purely derived from Counts; the
// class order is deterministic.
func (r VolatileReport) Warning() string {
	if !r.Unstable() {
		return ""
	}
	parts := make([]string, 0, len(volatileClassOrder))
	for _, class := range volatileClassOrder {
		if n := r.Counts[class]; n > 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", class, n))
		}
	}
	return "cache prefix unstable: " + strings.Join(parts, ", ") +
		" — move dynamic values out of the system prompt"
}
