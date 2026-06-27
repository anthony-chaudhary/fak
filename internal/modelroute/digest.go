package modelroute

// Content-addressed routing decisions (#615, epic #595).
//
// The axis that distinguishes fak from learned-predictor routers (RouteLLM,
// Martian, NotDiamond) is that the routing DECISION is deterministic and
// AUDITABLE. This file makes the PURE half of that real and checkable NOW — no
// live dispatch required:
//
//   - Digest content-addresses a Decision: a stable sha256 over the manifest
//     version + Subject + matched rule name + chosen Plan. "the same subject
//     under the same manifest always routed the same way" becomes a WITNESS, not
//     a promise — recompute the digest and compare.
//   - AuditRecord is the dos-verifiable record of a decision that ROUND-TRIPS
//     (digest in -> same digest out), so a route can be replayed and bound to
//     evidence after the fact.
//
// SCOPE (load-bearing, mirrors the package doc): the DECISION and the reduce
// FOLD are deterministic, so their digest is stable. The member models' OUTPUTS
// are not (non-bit-exact engines) — the digest deliberately covers only the
// route, never the answer, so this can never be mistaken for end-to-end output
// reproducibility.
//
// The hash scheme reuses internal/journal's existing "sha256:"+hex content-hash
// form so a route digest reads the same as every other audit identity in fak,
// WITHOUT importing journal (the tier-1 leaf stays stdlib-only): journal hashes
// opaque bytes; here we hash a canonical JSON encoding of the decision fields.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

// digestPayload is the canonical, field-stable pre-image of a routing decision.
// Only the fields that DEFINE the route are hashed — the manifest version (so a
// schema bump re-addresses), the echoed Subject, the matched rule name (empty ==
// fail-closed default), and the chosen Plan. The Matched bool is implied by
// RuleName ("" iff default) so it is not separately hashed. The struct has fixed
// field order and json.Marshal is deterministic for it (no maps at the top
// level; Subject.Labels is the only map and Go's encoder sorts map keys), so the
// pre-image bytes are reproducible across runs and processes.
type digestPayload struct {
	Version  string  `json:"version"`
	Subject  Subject `json:"subject"`
	RuleName string  `json:"rule"`
	Plan     Plan    `json:"plan"`
}

// Digest content-addresses a Decision under a manifest version: a stable
// "sha256:"+hex hash over (version, Subject, matched rule, Plan). Two calls with
// the same inputs always yield the same digest — within a run, across runs, and
// across processes — because the pre-image is a canonical JSON encoding of fixed
// fields (Go's json encoder sorts the one map, Subject.Labels). This is the
// witness that the route is content-addressed, not a promise that it is.
//
// version is the manifest's declared Version; pass Manifest.Version (callers that
// hold the Decision but not the Manifest can pass modelroute.Version for the
// current schema). It binds the digest to the policy schema so a schema bump
// re-addresses every decision rather than silently colliding.
func (d Decision) Digest(version string) string {
	p := digestPayload{
		Version:  version,
		Subject:  d.Subject,
		RuleName: d.RuleName,
		Plan:     d.Plan,
	}
	// json.Marshal of a struct with no top-level map, and a sorted-key map for
	// Labels, is deterministic — the canonical pre-image. The error path cannot
	// trigger for these stdlib-marshalable types, but we surface it defensively.
	b, err := json.Marshal(p)
	if err != nil {
		// Unreachable for digestPayload; a non-nil here would mean a corrupt
		// Subject/Plan, which Validate already rejects upstream.
		return ""
	}
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:])
}

// Digest content-addresses the result of routing s under this Manifest in one
// call: Route then Digest, bound to the Manifest's own Version. Same subject +
// same manifest -> same digest, every time.
func (m Manifest) Digest(s Subject) string {
	return m.Route(s).Digest(m.Version)
}

// AuditRecord is the dos-verifiable record of a routing decision: the Decision it
// describes, the manifest version it was taken under, and the content-address
// Digest binding the two. It round-trips through JSON and re-derives its own
// digest, so a stored route can be replayed and checked against evidence — the
// determinism/verifiability axis that sets fak apart from learned-predictor
// routers. It carries ONLY the decision, never any member model output (those are
// non-deterministic; see the package SCOPE caveat).
type AuditRecord struct {
	Version  string   `json:"version"`
	Decision Decision `json:"decision"`
	Digest   string   `json:"digest"`
}

// NewAuditRecord builds the audit record for routing s under m: it routes, stamps
// the manifest version (current Version when the manifest omits it), and binds the
// content-address digest. The returned record is self-checkable via Verify.
func (m Manifest) NewAuditRecord(s Subject) AuditRecord {
	ver := m.Version
	if ver == "" {
		ver = Version
	}
	d := m.Route(s)
	return AuditRecord{
		Version:  ver,
		Decision: d,
		Digest:   d.Digest(ver),
	}
}

// Verify recomputes the record's digest from its decision + version and reports
// whether the stamped Digest still binds — the dos-checkable round-trip. A record
// whose Digest field was tampered with, or whose Decision was edited after the
// fact, fails: the recomputed digest no longer matches. This is what lets a route
// be REPLAYED and bound to evidence rather than trusted.
func (r AuditRecord) Verify() error {
	want := r.Decision.Digest(r.Version)
	if r.Digest != want {
		return fmt.Errorf("modelroute: audit record digest mismatch: stamped %s, recomputed %s", r.Digest, want)
	}
	return nil
}

// JSON renders the audit record as canonical indented JSON, newline-terminated so
// it appends cleanly to a journal/ledger line.
func (r AuditRecord) JSON() []byte {
	b, _ := json.MarshalIndent(r, "", "  ")
	return append(b, '\n')
}

// ParseAuditRecord decodes an audit record and verifies its digest round-trips —
// a stored record that no longer self-checks fails loudly at the boundary, so a
// replayed route can never silently disagree with the evidence it claims to bind.
func ParseAuditRecord(b []byte) (AuditRecord, error) {
	var r AuditRecord
	if err := json.Unmarshal(b, &r); err != nil {
		return AuditRecord{}, fmt.Errorf("modelroute: parse audit record: %w", err)
	}
	if err := r.Verify(); err != nil {
		return AuditRecord{}, err
	}
	return r, nil
}
