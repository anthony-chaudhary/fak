package policy

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// PolicySummary captures the runtime digest and generation state of a live policy floor.
type PolicySummary struct {
	ContentDigest string `json:"content_digest"`
	Generation    uint64 `json:"generation"`
}

// Policy encapsulates the live capability floor, stamped with a content digest
// and a monotonic reload generation counter.
type Policy struct {
	mu            sync.RWMutex
	contentDigest string
	generation    uint64
	manifest      Manifest
	runtime       Runtime
	adjudicator   adjudicator.Policy
	raw           []byte
}

// ComputeContentDigest returns the hex-encoded SHA-256 hash of manifestBytes.
// If manifestBytes is nil or empty, it computes the SHA-256 digest of an empty byte slice.
func ComputeContentDigest(manifestBytes []byte) string {
	sum := sha256.Sum256(manifestBytes)
	return hex.EncodeToString(sum[:])
}

// ComputeRulesetDigest computes the deterministic SHA-256 content digest of a Manifest.
func ComputeRulesetDigest(m Manifest) string {
	b, err := json.Marshal(m)
	if err != nil {
		return ComputeContentDigest(nil)
	}
	return ComputeContentDigest(b)
}

// ContentDigest returns the hex-encoded SHA-256 digest of the manifest's JSON representation.
func (m Manifest) ContentDigest() string {
	return ComputeRulesetDigest(m)
}

// New constructs a new Policy from manifest bytes, initializing Generation to 1.
func New(manifestBytes []byte) (*Policy, error) {
	return NewPolicy(manifestBytes)
}

// NewPolicy creates a new Policy initialized with manifest bytes, starting at Generation 1.
func NewPolicy(manifestBytes []byte) (*Policy, error) {
	p := &Policy{
		generation:    1,
		contentDigest: ComputeContentDigest(manifestBytes),
		raw:           cloneBytes(manifestBytes),
	}
	if len(manifestBytes) > 0 {
		m, err := ParseManifest(manifestBytes)
		if err != nil {
			return nil, err
		}
		rt, err := m.ToRuntime()
		if err != nil {
			return nil, err
		}
		p.manifest = m
		p.runtime = rt
		p.adjudicator = rt.Adjudicator
	}
	return p, nil
}

// FromBytes creates a Policy from manifest bytes without failing on parse errors,
// initialized with Generation 1 and the SHA-256 digest of the bytes.
func FromBytes(manifestBytes []byte) *Policy {
	p := &Policy{
		generation:    1,
		contentDigest: ComputeContentDigest(manifestBytes),
		raw:           cloneBytes(manifestBytes),
	}
	if len(manifestBytes) > 0 {
		if m, err := ParseManifest(manifestBytes); err == nil {
			p.manifest = m
			if rt, err := m.ToRuntime(); err == nil {
				p.runtime = rt
				p.adjudicator = rt.Adjudicator
			}
		}
	}
	return p
}

// ParsePolicy parses manifest bytes and returns an initialized Policy at Generation 1.
func ParsePolicy(manifestBytes []byte) (*Policy, error) {
	return NewPolicy(manifestBytes)
}

// LoadPolicy reads and parses a manifest file from disk into a Policy with Generation 1.
func LoadPolicy(path string) (*Policy, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, &LoadError{Op: LoadOpRead, Path: path, Err: err}
	}
	p, err := NewPolicy(b)
	if err != nil {
		return nil, &LoadError{Op: LoadOpParse, Path: path, Err: err}
	}
	return p, nil
}

// ContentDigest returns the hex-encoded SHA-256 content digest representing the live floor.
func (p *Policy) ContentDigest() string {
	if p == nil {
		return ""
	}
	p.mu.RLock()
	defer p.mu.RUnlock()
	if p.contentDigest != "" {
		return p.contentDigest
	}
	if len(p.raw) > 0 {
		return ComputeContentDigest(p.raw)
	}
	return ComputeContentDigest(nil)
}

// Generation returns the monotonic reload generation counter.
// It starts at 1 when the initial floor is applied/loaded.
func (p *Policy) Generation() uint64 {
	if p == nil {
		return 0
	}
	p.mu.RLock()
	defer p.mu.RUnlock()
	if p.generation == 0 {
		return 1
	}
	return p.generation
}

// RecordReload increments the generation counter and updates the content digest on reload.
func (p *Policy) RecordReload(manifestBytes []byte) {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.generation == 0 {
		p.generation = 1
	}
	p.generation++
	p.contentDigest = ComputeContentDigest(manifestBytes)
	p.raw = cloneBytes(manifestBytes)
	if m, err := ParseManifest(manifestBytes); err == nil {
		p.manifest = m
		if rt, err := m.ToRuntime(); err == nil {
			p.runtime = rt
			p.adjudicator = rt.Adjudicator
		}
	}
}

// Reload updates the policy with manifest bytes, returning an error if parsing or runtime compilation fails.
// Regardless of errors, the reload attempt is recorded with an incremented generation and updated digest.
func (p *Policy) Reload(manifestBytes []byte) error {
	p.RecordReload(manifestBytes)
	m, err := ParseManifest(manifestBytes)
	if err != nil {
		return err
	}
	_, err = m.ToRuntime()
	return err
}

// Adjudicator returns the underlying adjudicator.Policy.
func (p *Policy) Adjudicator() adjudicator.Policy {
	if p == nil {
		return adjudicator.Policy{}
	}
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.adjudicator
}

// Runtime returns the compiled Runtime policy.
func (p *Policy) Runtime() Runtime {
	if p == nil {
		return Runtime{}
	}
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.runtime
}

// Manifest returns the parsed Manifest.
func (p *Policy) Manifest() Manifest {
	if p == nil {
		return Manifest{}
	}
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.manifest
}

// RawBytes returns a copy of the raw manifest bytes.
func (p *Policy) RawBytes() []byte {
	if p == nil {
		return nil
	}
	p.mu.RLock()
	defer p.mu.RUnlock()
	return cloneBytes(p.raw)
}

// Summary returns a PolicySummary snapshot of this policy.
func (p *Policy) Summary() PolicySummary {
	return PolicySummary{
		ContentDigest: p.ContentDigest(),
		Generation:    p.Generation(),
	}
}

// PolicySummary returns a PolicySummary snapshot of this policy.
func (p *Policy) PolicySummary() PolicySummary {
	return p.Summary()
}

func cloneBytes(b []byte) []byte {
	if b == nil {
		return nil
	}
	cp := make([]byte, len(b))
	copy(cp, b)
	return cp
}
