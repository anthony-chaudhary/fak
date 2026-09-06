package promptcomp

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
)

// SegmentKind defines the tier of a prompt fragment relative to the prompt structure and cache boundary.
type SegmentKind uint8

const (
	// KindSpine represents immutable root attention sink / universal identity.
	KindSpine SegmentKind = iota
	// KindPolicy represents system floor, security gates, and tool safety.
	KindPolicy
	// KindContract represents role-specific instructions and behavioral expectations.
	KindContract
	// KindTools represents active tool schemas and tool calling grammar.
	KindTools
	// KindOverlay represents dynamic paged working-set rules and capability cards.
	KindOverlay
)

// KindSafety is an alias for KindPolicy for backward compatibility with earlier callers.
const KindSafety = KindPolicy

func (k SegmentKind) String() string {
	switch k {
	case KindSpine:
		return "spine"
	case KindPolicy:
		return "policy"
	case KindContract:
		return "contract"
	case KindTools:
		return "tools"
	case KindOverlay:
		return "overlay"
	default:
		return fmt.Sprintf("unknown(%d)", k)
	}
}

// IsPrefixZone returns true if the segment belongs to Zone 1 (immutable warm prefix before breakpoint).
func (k SegmentKind) IsPrefixZone() bool {
	return k == KindSpine || k == KindPolicy
}

// Env carries contextual execution environment attributes evaluated by fragment predicates.
type Env struct {
	Model         string         // Target model name or identifier (e.g. "qwen3.8")
	ModelFamily   string         // Model family identifier (compatibility with harnessinit)
	IsSmallLocal  bool           // True for 7B-14B models requiring concise contracts
	AgentTier     string         // "coordinator", "leaf", "validator"
	ContextBudget int            // Remaining context tokens available
	WireFormat    string         // "openai", "gguf", "anthropic"
	Metadata      map[string]any // Extensible metadata attributes
	Extra         map[string]any // Compatibility alias for metadata
}

// PromptPart is an atomic, content-addressed prompt building block.
type PromptPart struct {
	ID            string         // Canonical identifier (e.g. "core.identity")
	Digest        string         // Hex SHA-256 of Content (computed automatically if created via NewPromptPart)
	Content       string         // Raw UTF-8 fragment content
	Kind          SegmentKind    // Segment placement tier
	Rank          int            // Secondary tie-breaker ordering key within same Kind
	DependsOn     []string       // Required prerequisites that must be active
	ConflictsWith []string       // Mutually exclusive part IDs
	Predicate     func(Env) bool // Dynamic activation condition (nil means unconditionally active)
}

// PartOption configures optional fields on a PromptPart during construction.
type PartOption func(*PromptPart)

// WithDependsOn adds prerequisite part IDs that must be active.
func WithDependsOn(deps ...string) PartOption {
	return func(p *PromptPart) {
		p.DependsOn = append(p.DependsOn, deps...)
	}
}

// WithConflictsWith adds conflicting part IDs that must not be active concurrently.
func WithConflictsWith(conflicts ...string) PartOption {
	return func(p *PromptPart) {
		p.ConflictsWith = append(p.ConflictsWith, conflicts...)
	}
}

// WithPredicate sets the dynamic activation predicate.
func WithPredicate(pred func(Env) bool) PartOption {
	return func(p *PromptPart) {
		p.Predicate = pred
	}
}

// WithDigest explicitly sets the hex digest (useful for testing digest mismatch).
func WithDigest(digest string) PartOption {
	return func(p *PromptPart) {
		p.Digest = digest
	}
}

// ComputeDigest returns the lowercase hex SHA-256 digest of the given content string.
func ComputeDigest(content string) string {
	h := sha256.Sum256([]byte(content))
	return hex.EncodeToString(h[:])
}

// NewPromptPart creates a PromptPart with automatically computed SHA-256 Content digest and optional configurations.
func NewPromptPart(id string, content string, kind SegmentKind, rank int, opts ...PartOption) PromptPart {
	p := PromptPart{
		ID:      id,
		Content: content,
		Kind:    kind,
		Rank:    rank,
		Digest:  ComputeDigest(content),
	}
	for _, opt := range opts {
		opt(&p)
	}
	return p
}

// Standard typed errors emitted across promptcomp operations.
var (
	ErrMissingID            = errors.New("promptcomp: fragment ID is required")
	ErrEmptyID              = ErrMissingID
	ErrEmptyContent         = errors.New("promptcomp: fragment content is empty")
	ErrDuplicateID          = errors.New("promptcomp: duplicate fragment ID")
	ErrNotFound             = errors.New("promptcomp: fragment not found")
	ErrMissingDependency    = errors.New("promptcomp: required dependency is missing")
	ErrConflict             = errors.New("promptcomp: conflicting fragments detected")
	ErrConflictingFragments = ErrConflict
	ErrCyclicDependency     = errors.New("promptcomp: cyclic dependency detected")
	ErrDigestMismatch       = errors.New("promptcomp: content digest mismatch")
	ErrInvalidDigest        = ErrDigestMismatch
	ErrZoneInversion        = errors.New("promptcomp: zone 1 prefix segment placed after zone 2 segment")
)

// Validate checks internal consistency of a PromptPart.
func (p *PromptPart) Validate() error {
	p.ID = strings.TrimSpace(p.ID)
	if p.ID == "" {
		return ErrMissingID
	}
	if strings.TrimSpace(p.Content) == "" {
		return ErrEmptyContent
	}
	computed := ComputeDigest(p.Content)
	if p.Digest == "" {
		p.Digest = computed
	} else if !strings.EqualFold(p.Digest, computed) {
		return fmt.Errorf("%w: declared %s != computed %s for %s", ErrDigestMismatch, p.Digest, computed, p.ID)
	}
	return nil
}
