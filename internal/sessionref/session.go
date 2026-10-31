// Package sessionref is the shared local session-descriptor publishing leaf.
// It publishes side refs only; lease admission, ownership and remote transport remain
// in leaseref. Both CLI spellings and the legacy Store use these implementations.
package sessionref

import (
	"context"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/refid"
	"strings"
	"time"
)

const RefPrefix = "refs/fak/locks/"
const SessionPrefix = "session-"

// SessionDescriptor is the small projection of a LIVE GUARD SESSION published under
// refs/fak/locks/session-<id>. It is the lightweight POINTER the fleet reads — the
// heavy checkpoint stays the sessionimage bundle. Encoded as one JSON object (the blob
// the ref points at), so the bytes are diffable and git-integrity-checked, exactly like
// the lock-lease Record.
type SessionDescriptor struct {
	ID        string `json:"id"`          // the session id (the ref basename minus the session- prefix)
	Host      string `json:"host"`        // the node this session is live on (machine identity, free-form)
	PCBState  string `json:"pcb_state"`   // the session PCB run-state: RUNNING/THROTTLED/PAUSED/DRAINING/STOPPED
	UpdatedAt int64  `json:"updated_at"`  // unix seconds of the last publish (register or transition)
	TTLSecs   int64  `json:"ttl_seconds"` // lifetime in seconds; 0 means no expiry (an explicit Remove ends it)
	// AgentUUID is the STABLE Claude Code session UUID (the transcript id, e.g. the value of
	// CLAUDE_CODE_SESSION_ID) this guard session runs under. The descriptor's own ID is the
	// VOLATILE agent-claude-<pid>-<hash> trace id — its pid rotates across restarts — but a
	// wip checkpoint (wipref) stamps this STABLE UUID instead. Carrying the UUID here lets a
	// liveness reader JOIN a checkpoint's stamped session to a live guard session that the
	// volatile trace id could never match (#5343, prerequisite for the #5340 collection cut).
	// OPTIONAL and additive: empty on a legacy blob published by a prior binary, or when no
	// Claude session env is set; omitempty keeps such a record byte-identical on the wire and
	// an older reader simply ignores the unknown key.
	AgentUUID string `json:"agent_uuid,omitempty"`
}

// Expired reports whether the descriptor is past its TTL at time now, measured from its
// last UpdatedAt. A zero TTL never expires. A live session republishes on each PCB
// transition, refreshing UpdatedAt, so a still-running session keeps its ref fresh; a
// crashed node's descriptor lapses once TTL elapses past the last update and a reader
// drops it from the LIVE view — bounded staleness, not a permanent ghost.
func (d SessionDescriptor) Expired(now time.Time) bool {
	if d.TTLSecs <= 0 {
		return false
	}
	return now.Unix() >= d.UpdatedAt+d.TTLSecs
}

// Ref returns the full ref path this descriptor is stored at: refs/fak/locks/session-<id>.
func (d SessionDescriptor) Ref() string { return RefPrefix + SessionPrefix + d.ID }

// PublishSession validates and timestamps a descriptor before the shared blob writer.
func PublishSession(ctx context.Context, d SessionDescriptor, putBlobRef func(context.Context, string, any) (string, error)) (string, error) {
	if !ValidSessionID(d.ID) {
		return "", fmt.Errorf("leaseref: invalid session id %q (must be one safe ref segment)", d.ID)
	}
	if d.UpdatedAt == 0 {
		d.UpdatedAt = time.Now().Unix()
	}
	return putBlobRef(ctx, d.Ref(), d)
}

// ValidSessionID confines an ID to the original single-segment session namespace.
func ValidSessionID(id string) bool {
	if !refid.Valid(id) {
		return false
	}
	// Reject a caller-supplied id that already carries the namespace marker, so
	// PublishSession("session-x") can never collide with PublishSession("x").
	return !strings.HasPrefix(id, SessionPrefix)
}
