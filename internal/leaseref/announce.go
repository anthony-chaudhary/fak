package leaseref

// announce.go is PLANE 2 of the multi-node coordination epic (#2254 / #2300): the
// DURABLE, HUMAN-VISIBLE GH-comment announce channel for lease and intent state. When a
// node can reach neither the git remote (plane 0 — refs/fak/locks/*, this package's ref
// store) nor the dev server (plane 1), it can still post a structured one-line comment to
// a designated coordination issue. GH comments are already the fleet's always-on audit
// trail (the dogfood ACTION bridge, internal/unwitnessedclaim), and the fleet token can
// always CREATE + COMMENT — so a comment is the coordination substrate of last resort.
//
// THE DISCIPLINE (internal/unwitnessedclaim's: pure evaluation in Go, `gh` at the edge).
// This file is PURE and stdlib-only: it RENDERS an AnnounceRecord into a comment body
// carrying one fenced JSON line, and PARSES that body back into the same record. The
// `gh issue comment` that POSTS it and the `gh issue view` that READS the comments back
// are the CALLER's job (cmd/fak/leaseref_announce.go). The round-trip
// (render -> parse -> identical record) is the witness the issue named.
//
// A COMMENT IS EVIDENCE, NEVER A LOCK (the load-bearing boundary carried down from
// #2254). FoldAnnouncements' output is ADVISORY visibility context — a human-legible
// "who announced holding what, and when" view — never an admission input on its own.
// Admission stays the refs/fak/locks compare-and-swap (fence.go) and dos_arbitrate; this
// plane only makes lease intent VISIBLE when the faster, authoritative planes are
// unreachable. Every plane speaks the SAME vocabulary (lease id, holder, generation,
// tree globs, TTL, lifecycle action) so a consumer folds them into one view — see
// AnnounceFromRecord, the bridge from a fenced lock-lease Record onto this plane.

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// AnnounceSchema tags the fenced JSON line an announce comment carries. It rides INSIDE
// the JSON (a "schema" field) rather than depending on the markdown code-fence info
// string, so the parser identifies a fak announce robustly regardless of how GitHub (or
// a hand-editing human) reflows the surrounding markdown. Versioned so a future field
// addition can bump it without silently mis-parsing an old comment.
const AnnounceSchema = "fak-leaseref-announce/1"

const PublicAnnounceSchema = "fak-leaseref-public-announce/1"

// The closed set of lifecycle actions an announce carries — the same acquire|renew|
// release transitions the fenced lock lease (fence.go) already distinguishes. Kept as a
// closed vocabulary so a folded view never has to reason about an unknown transition.
const (
	AnnounceAcquire = "acquire"
	AnnounceRenew   = "renew"
	AnnounceRelease = "release"
)

// ValidAnnounceAction reports whether action is one of the closed lifecycle transitions.
// The caller-side CLI rejects an out-of-vocabulary action before rendering; the parser
// drops a comment whose action is not valid, so a corrupt/forward-incompatible line
// never blinds the fold.
func ValidAnnounceAction(action string) bool {
	switch action {
	case AnnounceAcquire, AnnounceRenew, AnnounceRelease:
		return true
	}
	return false
}

// AnnounceRecord is one lease/intent state transition as it rides plane 2. It carries the
// SAME vocabulary the other planes speak (the ref-store Record's id/holder/generation/
// tree/TTL) plus the lifecycle Action, so a consumer folds an announce and a ref lease
// into one view. Encoded as one JSON object (the fenced line the comment carries), with
// omitempty on the derived fields so a minimal release announce stays compact.
type AnnounceRecord struct {
	LeaseID           string   `json:"lease_id,omitempty"`    // the lease id (ref basename under refs/fak/locks/)
	Holder            string   `json:"holder,omitempty"`      // who holds it (machine/session identity, free-form or the <node>/<session> convention)
	Generation        int64    `json:"generation,omitempty"`  // the fencing token at announce time (0 = legacy/unfenced)
	Tree              []string `json:"tree,omitempty"`        // the repo-relative tree globs the lease covers
	TTLSeconds        int64    `json:"ttl_seconds,omitempty"` // the lease lifetime in seconds (0 = no expiry)
	Action            string   `json:"action"`                // acquire | renew | release
	LeaseFingerprint  string   `json:"lease_fingerprint,omitempty"`
	HolderFingerprint string   `json:"holder_fingerprint,omitempty"`
	TreeFingerprints  []string `json:"tree_fingerprints,omitempty"`
}

// PublicSafeAnnounce replaces raw coordination identifiers with domain-separated HMAC
// fingerprints. Nodes sharing key can compare exact values without publishing them.
func PublicSafeAnnounce(rec AnnounceRecord, key []byte) (AnnounceRecord, error) {
	if len(key) == 0 {
		return AnnounceRecord{}, fmt.Errorf("public-safe announce key is empty")
	}
	out := AnnounceRecord{LeaseFingerprint: announceFingerprint(key, "lease", rec.LeaseID), HolderFingerprint: announceFingerprint(key, "holder", rec.Holder), Generation: rec.Generation, TTLSeconds: rec.TTLSeconds, Action: rec.Action}
	for _, tree := range rec.Tree {
		out.TreeFingerprints = append(out.TreeFingerprints, announceFingerprint(key, "tree", tree))
	}
	return out, nil
}

func announceFingerprint(key []byte, domain, value string) string {
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte(domain))
	_, _ = mac.Write([]byte{0})
	_, _ = mac.Write([]byte(value))
	return fmt.Sprintf("hmac-sha256:%x", mac.Sum(nil))
}

func (r AnnounceRecord) publicSafe() bool { return r.LeaseFingerprint != "" }
func (r AnnounceRecord) identity() string {
	if r.publicSafe() {
		return r.LeaseFingerprint
	}
	return r.LeaseID
}

// AnnounceFromRecord projects a fenced lock-lease Record (fence.go's AcquireFenced /
// Renew result) into the announce vocabulary, tagging it with the lifecycle action. It
// is the one-vocabulary bridge (#2254): `fak leaseref acquire`/`renew`/`release` can
// announce the SAME lease state they wrote to refs/fak/locks/* onto plane 2, without the
// caller re-deriving the fields.
func AnnounceFromRecord(rec Record, action string) AnnounceRecord {
	return AnnounceRecord{
		LeaseID:    rec.ID,
		Holder:     rec.Holder,
		Generation: rec.Generation,
		Tree:       rec.TreeGlobs,
		TTLSeconds: rec.TTLSeconds,
		Action:     action,
	}
}

// announceWire is the on-the-wire envelope: the schema tag plus the flattened record.
// The embedded AnnounceRecord promotes its fields, so the JSON is one flat object
// {"schema":...,"lease_id":...,...}; parsing lifts the schema off and returns the bare
// AnnounceRecord, so the schema tag never leaks into the caller's record.
type announceWire struct {
	Schema string `json:"schema"`
	AnnounceRecord
}

// RenderAnnounce builds the comment body for a lease transition: a human-legible summary
// line (so the announce is legible to a person scanning the coordination issue) followed
// by exactly ONE fenced JSON line (the machine-parseable evidence ParseAnnounce reads
// back). Pure and deterministic — the JSON field order is the struct field order, so the
// same record always renders byte-identically.
func RenderAnnounce(rec AnnounceRecord) string {
	schema := AnnounceSchema
	if rec.publicSafe() {
		schema = PublicAnnounceSchema
	}
	line, err := json.Marshal(announceWire{Schema: schema, AnnounceRecord: rec})
	if err != nil {
		// AnnounceRecord is all stdlib-serializable scalars/strings, so Marshal cannot
		// fail in practice; degrade to an empty object rather than panic in a best-effort
		// announce path.
		line = []byte("{}")
	}
	tree, lease, holder := "-", rec.LeaseID, rec.Holder
	if rec.publicSafe() {
		lease, holder = rec.LeaseFingerprint, rec.HolderFingerprint
		if len(rec.TreeFingerprints) > 0 {
			tree = strings.Join(rec.TreeFingerprints, ", ")
		}
	} else if len(rec.Tree) > 0 {
		tree = strings.Join(rec.Tree, ", ")
	}
	gen := ""
	if rec.Generation > 0 {
		gen = fmt.Sprintf(" · gen %d", rec.Generation)
	}
	ttl := "no-expiry"
	if rec.TTLSeconds > 0 {
		ttl = fmt.Sprintf("%ds", rec.TTLSeconds)
	}
	return fmt.Sprintf(
		"**leaseref announce — %s** · lease `%s` · holder `%s`%s · tree `%s` · ttl %s\n\n"+
			"```json\n%s\n```\n\n"+
			"_Plane 2 (GH-comment backup, #2300): advisory evidence of lease intent, never an admission input — admission stays the refs/fak/locks compare-and-swap._",
		rec.Action, lease, holder, gen, tree, ttl, line)
}

// ParseAnnounce extracts the FIRST fak announce record from a comment body, or
// (zero, false) when the body carries none. It scans line-by-line: a candidate is a line
// that looks like a JSON object AND mentions the schema tag (a cheap pre-filter), which
// is then unmarshalled and validated (right schema, in-vocabulary action). This is
// tolerant of the surrounding markdown — it finds the fenced JSON line whether or not the
// code fence survived a human edit — while still honoring the "one fenced JSON line"
// contract. A line that fails any check is skipped, never surfaced as an error.
func ParseAnnounce(body string) (AnnounceRecord, bool) {
	recs := scanAnnouncements(body)
	if len(recs) == 0 {
		return AnnounceRecord{}, false
	}
	return recs[0], true
}

// scanAnnouncements returns every valid announce record embedded in a comment body, in
// document order. One comment normally carries exactly one transition, but scanning all
// lines keeps the fold robust to a comment that (by edit or batching) carries several.
func scanAnnouncements(body string) []AnnounceRecord {
	var out []AnnounceRecord
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "{") || !strings.Contains(line, AnnounceSchema) && !strings.Contains(line, PublicAnnounceSchema) {
			continue
		}
		var w announceWire
		if err := json.Unmarshal([]byte(line), &w); err != nil {
			continue
		}
		if (w.Schema != AnnounceSchema && w.Schema != PublicAnnounceSchema) || !ValidAnnounceAction(w.Action) {
			continue
		}
		if (w.Schema == AnnounceSchema && w.LeaseID == "") || (w.Schema == PublicAnnounceSchema && w.LeaseFingerprint == "") {
			continue
		}
		out = append(out, w.AnnounceRecord)
	}
	return out
}

// FoldAnnouncements folds a coordination issue's comment bodies (oldest-first, the order
// `gh issue view --json comments` returns) into the advisory held-set view: the LATEST
// announced record per lease id. A `release` removes the lease from the view (it is no
// longer held); an `acquire`/`renew` sets or refreshes it. The result is sorted by lease
// id for a stable view and is ADVISORY visibility only — never an admission input. A body
// with no announce is skipped.
func FoldAnnouncements(bodies []string) []AnnounceRecord {
	held := map[string]AnnounceRecord{}
	for _, body := range bodies {
		for _, rec := range scanAnnouncements(body) {
			if rec.Action == AnnounceRelease {
				delete(held, rec.identity())
				continue
			}
			held[rec.identity()] = rec
		}
	}
	out := make([]AnnounceRecord, 0, len(held))
	for _, r := range held {
		out = append(out, r)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].identity() < out[j].identity() })
	return out
}
