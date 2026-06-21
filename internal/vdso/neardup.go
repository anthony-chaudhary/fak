package vdso

// neardup.go — an OPT-IN near-duplicate dedup key plus the temporal-cache NEGATIVE-RESULT
// guard. Both are OFF by default: the sound exact key (vdso.go keyLocked via argHash) is
// unchanged, and the Part B novelty posture — the cross-agent soundness WITNESS is an
// external world-state token, explicitly NOT a semantic / similarity signal
// (PRIOR-ART-fak-partb-residue-2026-06-17 C3) — is untouched. Near-dup is a SEPARATE,
// additional hit-rate axis a host opts into; it is deliberately NOT the witness key, so
// turning it on never moves the moat onto a semantic signal.
//
// What it does when ON (SetNearDup(true)). Two reads whose args differ only in the
// FORMATTING of their string values — surrounding/internal whitespace and letter case —
// collapse to ONE tier-2 entry, so {"code":"USD"} and {"code":" usd "} hit each other.
// It is a CONSERVATIVE, deterministic, model-free normalization (no embedding, no NLI, no
// similarity threshold) layered on top of the JSON canonicalization argHash already does
// (sorted keys + normalized number spelling). Because case-folding is NOT universally
// safe (a case-SENSITIVE id field would alias two distinct entities), this is opt-in and
// intended for tools whose string args are case/whitespace-insensitive identifiers
// (currency/airport codes, names) — the host asserts that precondition by enabling it.
//
// The temporal-cache negative-result guard. A NEGATIVE result — null, an empty object or
// array, an empty collection, or an explicit not-found / error shape — is the classic
// unsound thing to share across non-identical queries in a time-varying cache: a cached
// "no flights found" for one phrasing must not suppress a now-positive answer for a
// near-dup phrasing (negatives flip positive as the world fills in). So in near-dup mode a
// negative result is NEVER stored — only positive results are near-dup-shared. (With
// near-dup OFF, the exact key still caches negatives soundly under their witness/epoch:
// an exact repeat of the same query is genuinely unchanged.)

import (
	"bytes"
	"encoding/json"
	"strings"
	"sync/atomic"
	"unicode"
)

// SetNearDup enables/disables near-duplicate key collapsing (off by default). It does not
// clear the cache: existing exact-key entries remain, and new keys are computed under the
// selected mode.
func (v *VDSO) SetNearDup(on bool) {
	var n int32
	if on {
		n = 1
	}
	atomic.StoreInt32(&v.nearDup, n)
}

// NearDupOf reports whether near-duplicate key collapsing is enabled.
func (v *VDSO) NearDupOf() bool { return atomic.LoadInt32(&v.nearDup) == 1 }

// argHashFor returns the content hash the tier-2 key binds: the near-dup-normalized hash
// when near-dup is on, else the exact canonical-JSON hash. Keeping this one indirection in
// keyLocked means the epoch-stamping and witness logic are identical in both modes.
func (v *VDSO) argHashFor(args []byte) string {
	if v.NearDupOf() {
		return nearDupArgHash(args)
	}
	return argHash(args)
}

// nearDupArgHash canonicalizes args, then normalizes the FORMATTING of every string value
// (trim, collapse internal whitespace to a single space, fold to lower case) so formatting
// variants collapse to one key. Object keys are NOT folded (a field name is structural).
// Non-JSON args fall back to the exact raw-bytes hash (no normalization we can trust).
func nearDupArgHash(b []byte) string {
	var v any
	if json.Unmarshal(b, &v) != nil {
		return argHash(b) // not JSON: exact only
	}
	normalizeStrings(&v)
	out, err := json.Marshal(v)
	if err != nil {
		return argHash(b)
	}
	return argHash(out)
}

// normalizeStrings walks a decoded JSON value and rewrites each string VALUE to its
// formatting-normal form (string keys in maps are left intact — they are structure).
func normalizeStrings(p *any) {
	switch t := (*p).(type) {
	case string:
		*p = normalizeStr(t)
	case []any:
		for i := range t {
			normalizeStrings(&t[i])
		}
	case map[string]any:
		for k, val := range t {
			nv := val
			normalizeStrings(&nv)
			t[k] = nv
		}
	}
}

// normalizeStr trims, collapses internal whitespace runs to a single space, and lower-cases.
func normalizeStr(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return s
	}
	var sb strings.Builder
	sb.Grow(len(s))
	space := false
	for _, r := range s {
		if unicode.IsSpace(r) {
			space = true
			continue
		}
		if space && sb.Len() > 0 {
			sb.WriteByte(' ')
		}
		space = false
		sb.WriteRune(unicode.ToLower(r))
	}
	return sb.String()
}

// negativeResult reports whether a result body is a NEGATIVE answer that must not be
// near-dup-shared (the temporal-cache guard): null, an empty object/array, an object whose
// every collection field is empty, or an explicit not-found / error / false-found shape.
// Conservative by construction — anything it cannot positively classify as negative is
// treated as POSITIVE and cached normally.
func negativeResult(b []byte) bool {
	trimmed := bytes.TrimSpace(b)
	if len(trimmed) == 0 {
		return true
	}
	var v any
	if json.Unmarshal(trimmed, &v) != nil {
		return false // not JSON: treat as a positive opaque body
	}
	switch t := v.(type) {
	case nil:
		return true
	case []any:
		return len(t) == 0
	case map[string]any:
		if len(t) == 0 {
			return true
		}
		// explicit not-found markers
		if f, ok := t["found"].(bool); ok && !f {
			return true
		}
		if _, ok := t["error"]; ok {
			return true
		}
		if ok, present := t["ok"].(bool); present && !ok {
			return true
		}
		// an object whose only meaningful payload is empty collections (e.g.
		// {"results":[]}, {"flights":[],"count":0}) is a negative answer.
		sawCollection := false
		for _, val := range t {
			switch c := val.(type) {
			case []any:
				sawCollection = true
				if len(c) != 0 {
					return false
				}
			case map[string]any:
				sawCollection = true
				if len(c) != 0 {
					return false
				}
			}
		}
		return sawCollection
	default:
		return false // a bare string/number/bool is a positive scalar answer
	}
}
