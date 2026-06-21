package recall

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// ErrTombstoned is returned when a page was explicitly suppressed from future
// model-visible context by the memory-control plane. It is a tombstone, not a
// physical erase: the CAS byte record remains available to audit code.
var ErrTombstoned = errors.New("recall: page tombstoned by context control")

// ErrBadContextChange is returned for malformed or unsupported context-change
// requests.
var ErrBadContextChange = errors.New("recall: bad context-change request")

// ContextAction names an agent/requester initiated mutation of future context
// assembly. The first shipped action is deliberately negative-only: it narrows
// what may enter future model context, and never expands authority or deletes
// evidence.
type ContextAction string

const (
	ContextActionTombstone ContextAction = "tombstone_page"
)

// ContextChangeRequest is the structured shape an agent can file when it notices
// that a memory should not be offered to its future context. A request is not
// treated as truth about the world; applying a tombstone only suppresses page-in
// for model-visible recall. The underlying CAS/page-table evidence remains.
type ContextChangeRequest struct {
	Action      ContextAction `json:"action"`
	Step        int           `json:"step"`
	Digest      string        `json:"digest,omitempty"`
	Reason      string        `json:"reason"`
	RequestedBy string        `json:"requested_by"`
	Witness     string        `json:"witness,omitempty"`
}

// ContextChange is an applied, durable context-control ledger row. Today all rows
// are tombstones. Keeping this as a ledger rather than mutating Page preserves the
// original core image while letting future context assembly honor the suppression.
type ContextChange struct {
	ID          string        `json:"id"`
	Action      ContextAction `json:"action"`
	Step        int           `json:"step"`
	Digest      string        `json:"digest"`
	Reason      string        `json:"reason"`
	RequestedBy string        `json:"requested_by"`
	Witness     string        `json:"witness,omitempty"`
	TrustEpoch  uint64        `json:"trust_epoch,omitempty"`
	Applied     bool          `json:"applied"`
}

// RequestContextChange applies a safe, negative-only context mutation. A model can
// request that a page be tombstoned when it detects semantic poison, stale
// preference, or otherwise unwanted memory; the kernel records the request and
// suppresses that page from future Resolve/Recall calls. This never deletes bytes.
func (s *Session) RequestContextChange(req ContextChangeRequest) (ContextChange, error) {
	action := req.Action
	if action == "" {
		action = ContextActionTombstone
	}
	if action != ContextActionTombstone {
		return ContextChange{}, fmt.Errorf("%w: unsupported action %q", ErrBadContextChange, action)
	}
	if req.Step < 0 || req.Step >= len(s.Manifest.Pages) {
		return ContextChange{}, fmt.Errorf("%w: no page %d", ErrBadContextChange, req.Step)
	}
	p := s.Manifest.Pages[req.Step]
	if req.Digest != "" && req.Digest != p.Digest {
		return ContextChange{}, fmt.Errorf("%w: digest mismatch for page %d", ErrBadContextChange, req.Step)
	}
	reason := strings.TrimSpace(req.Reason)
	if reason == "" {
		return ContextChange{}, fmt.Errorf("%w: reason is required", ErrBadContextChange)
	}
	requestedBy := strings.TrimSpace(req.RequestedBy)
	if requestedBy == "" {
		requestedBy = "agent"
	}
	if existing, ok := s.tombstoneFor(req.Step); ok {
		return existing, nil
	}
	ch := ContextChange{
		ID:          contextChangeID(s.Manifest.SessionID, p.Step, p.Digest),
		Action:      ContextActionTombstone,
		Step:        p.Step,
		Digest:      p.Digest,
		Reason:      reason,
		RequestedBy: requestedBy,
		Witness:     strings.TrimSpace(req.Witness),
		TrustEpoch:  vdso.Default.TrustEpoch(),
		Applied:     true,
	}
	s.Manifest.ContextChanges = append(s.Manifest.ContextChanges, ch)
	return ch, nil
}

// Tombstoned reports whether a page is suppressed from model-visible recall.
func (s *Session) Tombstoned(step int) bool {
	_, ok := s.tombstoneFor(step)
	return ok
}

// ContextChanges returns a copy of the applied context-control ledger.
func (s *Session) ContextChanges() []ContextChange {
	return append([]ContextChange(nil), s.Manifest.ContextChanges...)
}

// Persist writes the possibly-mutated loaded session back to a core-image
// directory. This is used by the context-control plane after tombstoning: the
// manifest changes, the CAS bytes are preserved byte-for-byte.
func (s *Session) Persist(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	mb, err := json.MarshalIndent(s.Manifest, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), mb, 0o644); err != nil {
		return err
	}
	cb, err := json.MarshalIndent(s.cas, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "cas.json"), cb, 0o644)
}

func (s *Session) tombstoneFor(step int) (ContextChange, bool) {
	if step < 0 {
		return ContextChange{}, false
	}
	for _, ch := range s.Manifest.ContextChanges {
		if ch.Applied && ch.Action == ContextActionTombstone && ch.Step == step {
			return ch, true
		}
	}
	return ContextChange{}, false
}

func contextChangeID(sessionID string, step int, digest string) string {
	sum := sha256.Sum256([]byte(sessionID + "\x00" + strconv.Itoa(step) + "\x00" + digest))
	return "ctx-" + hex.EncodeToString(sum[:])[:12]
}
