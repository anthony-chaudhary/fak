// Package snapshot is the uniform DUMP/RESTORE seam over fak's primitives. The kernel
// is a ladder of nested loops — a syscall inside a turn inside a session inside a fleet
// inside an RSI loop (docs/explainers/engineering-is-building-loops.md) — and the goal
// this package serves is that ANY level of that ladder can be frozen to bytes and
// thawed back: a turn, a tool, a session, a fleet, an RSI loop.
//
// # The one envelope
//
// Every primitive dumps to the SAME shape — a Snapshot: a kind tag, an id, a versioned
// wrapper, the primitive-specific Body as raw JSON, and a sha256 over that body so a
// truncated or tampered dump fails closed on restore. Because the envelope carries the
// body as `json.RawMessage`, Marshal works over ANY value and Parse/Into restores into
// any typed target — so a new primitive becomes dumpable without changing this package:
// call Marshal(kind, id, yourState) and you have a portable, integrity-checked dump.
//
// # The registry (the ladder, named)
//
// Kinds are registered with their ladder LEVEL and a one-line description, so a tool can
// enumerate "what can I dump?" (Kinds) and a restore can validate that an envelope's
// kind is one it understands (Known). The canonical fak ladder — turn, tool, session,
// fleet, rsi — is registered at init; an application registers its own kinds the same
// way. Registering a kind is METADATA (it names a level on the ladder); the generic
// Marshal/Parse seam is what actually carries the bytes, so the registry never lies
// about a codec that does not exist.
//
// # Typed codecs (codecs.go)
//
// For the primitives whose live state is a fak value, this package ships typed
// convenience codecs built on the generic seam: DumpTrace/RestoreTrace over a
// trajectory's Turn rows (the turn level), DumpFleet/RestoreFleet over a
// session.Table's drive snapshot (the fleet level, restored verbatim via the §5
// session.Table.Restore). The SESSION level has a richer, multi-part image of its own —
// internal/sessionimage — because a session carries content-addressed bytes a single
// JSON body should not inline; this package's "session" kind points at that format.
//
// snapshot is a data-plane integrator: it imports the session + trajectory data planes
// and the stdlib only, adds nothing to the frozen ABI, and is off the request path.
package snapshot

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
)

// EnvelopeVersion is the on-disk format tag stamped into every Snapshot. Parse refuses
// an envelope whose version it does not recognize (fail closed — a wrong-version dump is
// worse than none), mirroring recall.ManifestVersion and sessionimage.Version.
const EnvelopeVersion = "fak.snapshot.v1"

// Snapshot is the uniform envelope for one dumped primitive. Body is the
// primitive-specific state as raw JSON; BodyDigest is the sha256 over those exact bytes,
// so Parse can re-verify integrity before anyone trusts the body. The remaining fields
// are the self-describing header: which kind, which id, when, and under which build.
type Snapshot struct {
	Envelope    string            `json:"envelope"`
	Kind        string            `json:"kind"`
	ID          string            `json:"id"`
	CreatedUnix int64             `json:"created_unix"`
	AppVersion  string            `json:"app_version"`
	Meta        map[string]string `json:"meta,omitempty"`
	Body        json.RawMessage   `json:"body"`
	BodyDigest  string            `json:"body_digest"`
}

// Marshal wraps a primitive's state into an integrity-stamped Snapshot. body is any
// JSON-marshalable value (a Turn slice, a fleet drive snapshot, an RSI row, your own
// struct) — that genericity is the whole point: a new primitive is dumpable with no
// change here. now is an injected unix clock for determinism (0 uses wall time). kind
// and id must be non-empty.
func Marshal(kind, id string, body any, meta map[string]string, now int64) (Snapshot, error) {
	if kind == "" {
		return Snapshot{}, fmt.Errorf("snapshot: kind is required")
	}
	if id == "" {
		return Snapshot{}, fmt.Errorf("snapshot: id is required")
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return Snapshot{}, fmt.Errorf("snapshot: marshal body: %w", err)
	}
	if now == 0 {
		now = time.Now().Unix()
	}
	dg, err := bodyDigest(raw)
	if err != nil {
		return Snapshot{}, err
	}
	return Snapshot{
		Envelope:    EnvelopeVersion,
		Kind:        kind,
		ID:          id,
		CreatedUnix: now,
		AppVersion:  appversion.Current(),
		Meta:        meta,
		Body:        raw,
		BodyDigest:  dg,
	}, nil
}

// Encode serializes the envelope to indented JSON — the on-disk / on-wire form.
func (s Snapshot) Encode() ([]byte, error) { return json.MarshalIndent(s, "", "  ") }

// Parse decodes an envelope and VERIFIES it: the version must match and the body must
// hash to its recorded digest, or Parse fails closed. A caller can trust a returned
// Snapshot's Body has not been truncated or tampered since Marshal.
func Parse(b []byte) (Snapshot, error) {
	var s Snapshot
	if err := json.Unmarshal(b, &s); err != nil {
		return Snapshot{}, fmt.Errorf("snapshot: decode envelope: %w", err)
	}
	if s.Envelope != EnvelopeVersion {
		return Snapshot{}, fmt.Errorf("snapshot: envelope version %q != %q", s.Envelope, EnvelopeVersion)
	}
	d, err := bodyDigest(s.Body)
	if err != nil {
		return Snapshot{}, fmt.Errorf("snapshot: body is not valid JSON: %w", err)
	}
	if d != s.BodyDigest {
		return Snapshot{}, fmt.Errorf("snapshot: body digest mismatch (got %s want %s) — integrity check failed", short(d), short(s.BodyDigest))
	}
	return s, nil
}

// Into unmarshals the (already integrity-verified) Body into a typed value.
func (s Snapshot) Into(v any) error {
	if err := json.Unmarshal(s.Body, v); err != nil {
		return fmt.Errorf("snapshot: unmarshal body into %T: %w", v, err)
	}
	return nil
}

// WriteFile marshals a primitive's body into an envelope and writes it to path.
func WriteFile(path, kind, id string, body any, meta map[string]string, now int64) error {
	s, err := Marshal(kind, id, body, meta, now)
	if err != nil {
		return err
	}
	b, err := s.Encode()
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

// ReadFile reads and verifies an envelope from path.
func ReadFile(path string) (Snapshot, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Snapshot{}, err
	}
	return Parse(b)
}

// bodyDigest is the canonical content address (sha256 hex) of a JSON body: it hashes the
// COMPACTED bytes, so the digest is independent of indentation. This matters because
// Encode pretty-prints the envelope (re-indenting the embedded RawMessage body), so the
// body bytes Parse sees differ in whitespace from the bytes Marshal hashed — compacting
// both to a canonical form makes the integrity check stable across a pretty-print
// round-trip while still catching any real content change.
func bodyDigest(raw []byte) (string, error) {
	var buf bytes.Buffer
	if err := json.Compact(&buf, raw); err != nil {
		return "", err
	}
	sum := sha256.Sum256(buf.Bytes())
	return hex.EncodeToString(sum[:]), nil
}

func short(d string) string {
	if len(d) > 12 {
		return d[:12]
	}
	return d
}
