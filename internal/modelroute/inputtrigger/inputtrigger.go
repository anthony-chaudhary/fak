// Package inputtrigger re-exports pkg/routepolicy/inputtrigger, the public home of
// the turn-input-shape taxonomy, so in-tree importers keep compiling unchanged. The
// contract (closed vocabulary, routing hint never an authorization fact,
// fail-conservative classification) is documented at the public package.
package inputtrigger

import pub "github.com/anthony-chaudhary/fak/pkg/routepolicy/inputtrigger"

type (
	Trigger      = pub.Trigger
	Role         = pub.Role
	Message      = pub.Message
	Provenance   = pub.Provenance
	Explicit     = pub.Explicit
	InputTrigger = pub.InputTrigger
)

const (
	Other            = pub.Other
	SystemOnly       = pub.SystemOnly
	UserMessage      = pub.UserMessage
	AssistantPrefill = pub.AssistantPrefill
	ToolResult       = pub.ToolResult
)

const (
	RoleSystem    = pub.RoleSystem
	RoleUser      = pub.RoleUser
	RoleAssistant = pub.RoleAssistant
	RoleTool      = pub.RoleTool
)

const (
	Schema              = pub.Schema
	MaxMetadataEntries  = pub.MaxMetadataEntries
	MaxMetadataKeyBytes = pub.MaxMetadataKeyBytes
	MaxMetadataValBytes = pub.MaxMetadataValBytes
)

const (
	ProvenanceExplicit             = pub.ProvenanceExplicit
	ProvenanceRetry                = pub.ProvenanceRetry
	ProvenanceCompatibilityMissing = pub.ProvenanceCompatibilityMissing
)

func Known(t Trigger) bool                    { return pub.Known(t) }
func ParseRole(raw string) (Role, bool)       { return pub.ParseRole(raw) }
func Classify(turn []Message) Trigger         { return pub.Classify(turn) }
func KnownProvenance(p Provenance) bool       { return pub.KnownProvenance(p) }
func Parse(data []byte) (InputTrigger, error) { return pub.Parse(data) }

func AdmitTurn(turn []Message, explicit *Explicit) (InputTrigger, error) {
	return pub.AdmitTurn(turn, explicit)
}
