package modelsetlock

import (
	"github.com/anthony-chaudhary/fak/internal/harnessmodelset"
	"github.com/anthony-chaudhary/fak/internal/modelinventory"
	"github.com/anthony-chaudhary/fak/internal/modelsetresolve"
)

const (
	// Schema is the only lock schema accepted by this package.
	Schema = "fak.model-set-lock/1"
	// DefaultFileName is the generated harness model-set lock name.
	DefaultFileName = "harness.model-set.lock.json"
)

// Inputs are the complete facts whose changes can make a lock stale. RuleBytes
// must be the canonical, credential-free selection-policy bytes owned by the caller.
type Inputs struct {
	Intent          harnessmodelset.Intent
	Inventory       modelinventory.Inventory
	RuleBytes       []byte
	Target          Target
	ResolverVersion string
}

// Target is the exact execution platform used during resolution.
type Target struct {
	OS           string `json:"os"`
	Architecture string `json:"architecture"`
	Accelerator  string `json:"accelerator"`
	Runtime      string `json:"runtime"`
}

// InputDigests bind a lock to every independently changeable resolution input.
type InputDigests struct {
	Intent    string `json:"intent"`
	Inventory string `json:"inventory"`
	Policy    string `json:"policy"`
	Target    string `json:"target"`
}

// Lock is one deterministic, digest-bound model-set resolution.
type Lock struct {
	Schema          string       `json:"schema"`
	ContentDigest   string       `json:"content_digest"`
	Inputs          InputDigests `json:"inputs"`
	Target          Target       `json:"target"`
	ResolverVersion string       `json:"resolver_version"`
	EvaluatedAt     string       `json:"evaluated_at"`
	Roles           []Role       `json:"roles"`
}

// Role records the complete resolution outcome for one harness responsibility.
// AlternativeIDs preserve the intent's semantic declared order.
type Role struct {
	ID             string                      `json:"id"`
	Required       bool                        `json:"required"`
	Status         modelsetresolve.RoleStatus  `json:"status"`
	AlternativeIDs []string                    `json:"alternative_ids"`
	Selected       *Selected                   `json:"selected,omitempty"`
	Rejections     []modelsetresolve.Rejection `json:"rejections"`
}

// Selected snapshots the resolver choice, its immutable artifact identity, and
// the credential-free evidence references that supported the inventory entry.
type Selected struct {
	AlternativeID string                   `json:"alternative_id"`
	CandidateID   string                   `json:"candidate_id"`
	Identity      modelinventory.Identity  `json:"identity"`
	Evidence      []modelinventory.Witness `json:"evidence"`
}
