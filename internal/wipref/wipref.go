// Package wipref is the PURE core of `fak wip`: the working-tree checkpoint
// ledger that lives under refs/fak/wip/<session> — a sibling of the lease refs
// (internal/leaseref, refs/fak/locks/*). It carries zero git I/O: the cmd shell
// (cmd/fak/wip.go) runs git and hands this package the ref listing; this package
// owns the ref-name grammar, the stamp encode/decode, and the status fold, so the
// I/O-free decisions are unit-testable without a repo.
//
// A checkpoint is one commit object minted from a temp-index tree (the tracked
// working-tree delta), anchored at refs/fak/wip/<session> so the object stays
// gc-reachable until the ref is deleted. The commit MESSAGE carries a one-line
// JSON stamp — {session_id, start_sha, leaves, buildable, checkpointed_at} — so a
// single ref is both the retention anchor and the metadata carrier.
package wipref

import (
	"encoding/json"
	"sort"
	"strings"
)

// RefNamespace is the ref directory every working-tree checkpoint lives under,
// one safe segment per session. It deliberately mirrors leaseref's refs/fak/locks/
// so the two side-ref families read as siblings.
const RefNamespace = "refs/fak/wip/"

// stampMarker prefixes the single commit-message line that carries the JSON stamp.
// A commit-tree message may hold anything; the marker lets DecodeStamp find OUR
// line and ignore any human prose git or a hook might append.
const stampMarker = "fak-wip: "

// Stamp is the metadata a checkpoint records in its commit message. It is the
// portable identity of the checkpoint, independent of the object's git internals.
//
// Leaves and Scope answer two DIFFERENT questions and must not be conflated.
// Leaves is descriptive — the directories the CAPTURE happened to sweep up, which
// on a shared working tree includes every concurrently-dirty peer. Scope is a
// CLAIM: the paths the capturing session declared it owns. A capture is tree-wide
// by design (that width is what makes it lossless for crash recovery), so the ref's
// session key names the capturer, never the author; Scope is the only field that
// carries authorship, and it is why a stamped checkpoint can be landed safely by a
// LATER process — a fleet host recovering a crashed session cannot otherwise know
// what the dead session owned. An empty Scope means "nothing declared", not
// "everything claimed" (#5539).
type Stamp struct {
	SessionID      string   `json:"session_id"`
	StartSHA       string   `json:"start_sha"`
	Leaves         []string `json:"leaves,omitempty"`
	Scope          []string `json:"scope,omitempty"`
	Buildable      bool     `json:"buildable"`
	CheckpointedAt int64    `json:"checkpointed_at"`
}

// ValidSession reports whether id is a single safe ref segment: no '/', no
// whitespace, none of git's refname metacharacters. Same rule as leaseref.validID
// — a checkpoint ref must be exactly one path component under RefNamespace.
func ValidSession(id string) bool {
	if id == "" {
		return false
	}
	for _, c := range id {
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9':
		case c == '-' || c == '_' || c == '.':
		default:
			return false
		}
	}
	return true
}

// SessionRef is the fully-qualified ref a session's checkpoint lives at.
func SessionRef(id string) string { return RefNamespace + id }

// SessionFromRef recovers the session id from a fully-qualified checkpoint ref.
// A ref outside RefNamespace is returned unchanged (the fold uses it as a label of
// last resort when a stamp is missing or unparseable).
func SessionFromRef(ref string) string { return strings.TrimPrefix(ref, RefNamespace) }

// EncodeStamp renders s as the single commit-message line a checkpoint carries:
// the marker followed by compact JSON. Round-trips through DecodeStamp.
func EncodeStamp(s Stamp) (string, error) {
	b, err := json.Marshal(s)
	if err != nil {
		return "", err
	}
	return stampMarker + string(b), nil
}

// DecodeStamp extracts the Stamp from a commit message, scanning for the marker
// line so it tolerates surrounding prose. ok=false means no parseable stamp line
// was present — the caller falls back to the ref name for identity.
func DecodeStamp(msg string) (Stamp, bool) {
	for _, line := range strings.Split(msg, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, stampMarker) {
			continue
		}
		var s Stamp
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, stampMarker)), &s); err == nil {
			return s, true
		}
	}
	return Stamp{}, false
}

// RefRecord is one live checkpoint ref plus the object it points at and its decoded
// stamp. The cmd shell builds these from `git for-each-ref` + `git log`.
type RefRecord struct {
	Ref    string
	Object string
	Stamp  Stamp
}

// SessionStatus is one checkpoint's public row in a StatusReport.
type SessionStatus struct {
	Session    string   `json:"session"`
	Object     string   `json:"object"`
	StartSHA   string   `json:"start_sha"`
	Leaves     []string `json:"leaves"`
	Buildable  bool     `json:"buildable"`
	AgeSeconds int64    `json:"age_seconds"`
}

// StatusReport is the deterministic fold of every live checkpoint, sorted by
// session so a JSON snapshot is byte-stable across runs (timestamps aside).
type StatusReport struct {
	Count    int             `json:"count"`
	Sessions []SessionStatus `json:"sessions"`
}

// Fold projects the live ref records into a sorted StatusReport, computing each
// checkpoint's age against nowUnix. It never inspects git — it is a pure function
// of (records, now), which is what makes the status path unit-testable. A record
// whose stamp lost its session id is labelled from its ref name; a negative age
// (clock skew / a future stamp) is clamped to 0.
func Fold(recs []RefRecord, nowUnix int64) StatusReport {
	out := make([]SessionStatus, 0, len(recs))
	for _, r := range recs {
		sess := r.Stamp.SessionID
		if sess == "" {
			sess = SessionFromRef(r.Ref)
		}
		age := nowUnix - r.Stamp.CheckpointedAt
		if age < 0 {
			age = 0
		}
		leaves := r.Stamp.Leaves
		if leaves == nil {
			leaves = []string{}
		}
		out = append(out, SessionStatus{
			Session:    sess,
			Object:     r.Object,
			StartSHA:   r.Stamp.StartSHA,
			Leaves:     leaves,
			Buildable:  r.Stamp.Buildable,
			AgeSeconds: age,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Session < out[j].Session })
	return StatusReport{Count: len(out), Sessions: out}
}
