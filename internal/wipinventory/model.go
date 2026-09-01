package wipinventory

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
)

// WIPUnitSchema is the stable wire-format version for WIP unit histories.
const WIPUnitSchema = "fak-wip-unit/1"

var wipUnitIDPattern = regexp.MustCompile(`^wip:v1:[0-9a-f]{32}$`)

// WIPUnitID is an opaque identity for one logical unit of work. Its payload is
// deliberately random: callers must not derive it from a mutable surface ID.
type WIPUnitID string

// NewWIPUnitID returns a new opaque, versioned identity.
func NewWIPUnitID() (WIPUnitID, error) {
	var payload [16]byte
	if _, err := rand.Read(payload[:]); err != nil {
		return "", fmt.Errorf("create WIP unit ID: %w", err)
	}
	return WIPUnitID("wip:v1:" + hex.EncodeToString(payload[:])), nil
}

// ParseWIPUnitID accepts only the canonical versioned opaque form.
func ParseWIPUnitID(value string) (WIPUnitID, error) {
	id := WIPUnitID(value)
	if err := id.Validate(); err != nil {
		return "", err
	}
	return id, nil
}

func (id WIPUnitID) Validate() error {
	if !wipUnitIDPattern.MatchString(string(id)) {
		return fmt.Errorf("invalid WIP unit ID %q: want wip:v1 followed by 128 random bits", id)
	}
	return nil
}

func (id WIPUnitID) MarshalJSON() ([]byte, error) {
	if err := id.Validate(); err != nil {
		return nil, err
	}
	return json.Marshal(string(id))
}

func (id *WIPUnitID) UnmarshalJSON(data []byte) error {
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	parsed, err := ParseWIPUnitID(value)
	if err != nil {
		return err
	}
	*id = parsed
	return nil
}

// SurfaceKind names the subsystem-local identity carried by a SurfaceReference.
type SurfaceKind string

const (
	SurfaceIssue               SurfaceKind = "issue"
	SurfaceDispatchSession     SurfaceKind = "dispatch_session"
	SurfaceCheckpoint          SurfaceKind = "checkpoint"
	SurfaceLaneLease           SurfaceKind = "lane_lease"
	SurfaceManagedWorktree     SurfaceKind = "managed_worktree"
	SurfaceWitnessedRetirement SurfaceKind = "witnessed_retirement"
)

// SurfaceReference preserves a subsystem's authoritative local identifier.
// Exactly one typed member must be present and must agree with Kind.
type SurfaceReference struct {
	Kind                SurfaceKind                   `json:"kind"`
	Issue               *IssueReference               `json:"issue,omitempty"`
	DispatchSession     *DispatchSessionReference     `json:"dispatch_session,omitempty"`
	Checkpoint          *CheckpointReference          `json:"checkpoint,omitempty"`
	LaneLease           *LaneLeaseReference           `json:"lane_lease,omitempty"`
	ManagedWorktree     *ManagedWorktreeReference     `json:"managed_worktree,omitempty"`
	WitnessedRetirement *WitnessedRetirementReference `json:"witnessed_retirement,omitempty"`
}

type IssueReference struct {
	Repository string `json:"repository"`
	Number     int    `json:"number"`
}

type DispatchSessionReference struct {
	DispatchID string `json:"dispatch_id,omitempty"`
	SessionID  string `json:"session_id,omitempty"`
}

type CheckpointReference struct {
	CheckpointID string `json:"checkpoint_id"`
}

type LaneLeaseReference struct {
	Lane    string `json:"lane"`
	LeaseID string `json:"lease_id"`
}

type ManagedWorktreeReference struct {
	WorktreeID string `json:"worktree_id"`
}

type WitnessedRetirementReference struct {
	RetirementID string `json:"retirement_id"`
	Witness      string `json:"witness"`
}

func (ref SurfaceReference) Validate() error {
	present := 0
	if ref.Issue != nil {
		present++
	}
	if ref.DispatchSession != nil {
		present++
	}
	if ref.Checkpoint != nil {
		present++
	}
	if ref.LaneLease != nil {
		present++
	}
	if ref.ManagedWorktree != nil {
		present++
	}
	if ref.WitnessedRetirement != nil {
		present++
	}
	if present != 1 {
		return fmt.Errorf("surface reference %q must contain exactly one typed reference", ref.Kind)
	}
	nonempty := func(values ...string) bool {
		for _, value := range values {
			if strings.TrimSpace(value) == "" {
				return false
			}
		}
		return true
	}
	switch ref.Kind {
	case SurfaceIssue:
		if ref.Issue == nil || !nonempty(ref.Issue.Repository) || ref.Issue.Number <= 0 {
			return errors.New("invalid issue reference")
		}
	case SurfaceDispatchSession:
		if ref.DispatchSession == nil || (strings.TrimSpace(ref.DispatchSession.DispatchID) == "" && strings.TrimSpace(ref.DispatchSession.SessionID) == "") {
			return errors.New("invalid dispatch/session reference")
		}
	case SurfaceCheckpoint:
		if ref.Checkpoint == nil || !nonempty(ref.Checkpoint.CheckpointID) {
			return errors.New("invalid checkpoint reference")
		}
	case SurfaceLaneLease:
		if ref.LaneLease == nil || !nonempty(ref.LaneLease.Lane, ref.LaneLease.LeaseID) {
			return errors.New("invalid lane/lease reference")
		}
	case SurfaceManagedWorktree:
		if ref.ManagedWorktree == nil || !nonempty(ref.ManagedWorktree.WorktreeID) {
			return errors.New("invalid managed worktree reference")
		}
	case SurfaceWitnessedRetirement:
		if ref.WitnessedRetirement == nil || !nonempty(ref.WitnessedRetirement.RetirementID, ref.WitnessedRetirement.Witness) {
			return errors.New("invalid witnessed retirement reference")
		}
	default:
		return fmt.Errorf("unknown surface kind %q", ref.Kind)
	}
	return nil
}

// Provenance records who asserted a transition and under which stable mechanism.
type Provenance struct {
	Actor     string `json:"actor"`
	Mechanism string `json:"mechanism"`
}

// History is an ordered, deterministic record of WIP-unit transitions.
type History struct {
	Schema      string       `json:"schema"`
	Transitions []Transition `json:"transitions"`
}

// MarshalDeterministic validates history before emitting its stable JSON form.
// Struct fields and transition order are fixed; no unordered maps enter the wire form.
func MarshalDeterministic(history History) ([]byte, error) {
	if err := ValidateHistory(history); err != nil {
		return nil, err
	}
	return json.Marshal(history)
}
