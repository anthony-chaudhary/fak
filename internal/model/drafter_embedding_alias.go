package model

import (
	"errors"
	"fmt"
	"sync/atomic"
)

// drafter_embedding_alias.go — alias a speculative drafter's (MTP head) token
// embedding table onto the target model's table instead of loading a second copy.
//
// Ported from vLLM's DeepSeek-V4 MTP embedding alias fix (vllm-project/vllm,
// Apache-2.0, vllm/models/deepseek_v4/nvidia/mtp.py@3f34cac035724890e6ee5707a6649517b83d2d61,
// PR #65816). No Python source is copied. Under pipeline parallelism the MTP drafter held its own copy of the
// target model's embedding table; the fix aliases the drafter to the target's
// embedding parameters. That removes duplicate memory that scales with vocabulary
// size and guarantees the drafter scores tokens against the same embeddings the
// target produces.
//
// The fak port keeps the same contract at the component level: a drafter that
// declares an embedding table identical to the target's (same vocab, hidden, and
// dtype) receives a VIEW onto the target's backing storage — never a fresh
// allocation. A drafter whose declared table differs in any dimension is refused
// with an explicit mismatch error naming the offending dimension; there is no
// silent fallback to a second copy.

// EmbeddingTableIdentity is the declared identity of a token embedding table: the
// name it is known by, its [vocab, hidden] geometry, and its element dtype. Two
// tables are aliasable exactly when all three agree.
type EmbeddingTableIdentity struct {
	Name   string
	Vocab  int
	Hidden int
	Dtype  string
}

// EmbeddingTable is a token embedding table backed by a []float32 payload. When it
// is an alias, data shares the target table's backing array (same first-element
// address); it is never a copy.
type EmbeddingTable struct {
	identity EmbeddingTableIdentity
	data     []float32
}

// Identity returns the table's declared identity.
func (t *EmbeddingTable) Identity() EmbeddingTableIdentity {
	if t == nil {
		return EmbeddingTableIdentity{}
	}
	return t.identity
}

// Data returns the backing payload. For an aliased table this is the target's own
// storage, not a copy.
func (t *EmbeddingTable) Data() []float32 {
	if t == nil {
		return nil
	}
	return t.data
}

// Aliases reports whether t and other share the same backing storage (the same
// first-element address). Two nil or empty tables do not alias.
func (t *EmbeddingTable) Aliases(other *EmbeddingTable) bool {
	if t == nil || other == nil || len(t.data) == 0 || len(other.data) == 0 {
		return false
	}
	return &t.data[0] == &other.data[0]
}

// Mismatch sentinels. Each names the dimension that differed; callers match with
// errors.Is and read the wrapped detail for the two values.
var (
	ErrDrafterEmbeddingNilTarget      = errors.New("model: drafter embedding alias needs a target table")
	ErrDrafterEmbeddingVocabMismatch  = errors.New("model: drafter embedding vocab size differs from target")
	ErrDrafterEmbeddingHiddenMismatch = errors.New("model: drafter embedding hidden size differs from target")
	ErrDrafterEmbeddingDtypeMismatch  = errors.New("model: drafter embedding dtype differs from target")
)

// embeddingTableAllocCount counts payload allocations made by NewEmbeddingTable.
// The alias path must never increment it: an aliased drafter shares the target's
// storage rather than allocating a second copy. It is a test seam, mirroring the
// package's tensorLookupProbe hook.
var embeddingTableAllocCount atomic.Int64

// EmbeddingTableAllocCount reports how many embedding-table payloads have been
// allocated through NewEmbeddingTable. Tests use it to prove the alias path
// allocates nothing.
func EmbeddingTableAllocCount() int64 { return embeddingTableAllocCount.Load() }

// NewEmbeddingTable builds a table that owns its payload. It is the allocating
// constructor; AliasDrafterEmbedding is the non-allocating one.
func NewEmbeddingTable(identity EmbeddingTableIdentity, data []float32) (*EmbeddingTable, error) {
	if identity.Vocab <= 0 || identity.Hidden <= 0 {
		return nil, fmt.Errorf("model: embedding table %q has invalid geometry [%d, %d]", identity.Name, identity.Vocab, identity.Hidden)
	}
	if want := identity.Vocab * identity.Hidden; len(data) != want {
		return nil, fmt.Errorf("model: embedding table %q payload %d elements, want %d", identity.Name, len(data), want)
	}
	embeddingTableAllocCount.Add(1)
	return &EmbeddingTable{identity: identity, data: data}, nil
}

// AliasDrafterEmbedding resolves a drafter's declared embedding table against the
// target model's table. When the declared identity matches the target's on vocab,
// hidden, and dtype, it returns a table that shares the target's backing storage
// (no allocation). Otherwise it returns an explicit mismatch error naming the
// differing dimension — it never falls back to a second copy.
func AliasDrafterEmbedding(target *EmbeddingTable, drafter EmbeddingTableIdentity) (*EmbeddingTable, error) {
	if target == nil {
		return nil, ErrDrafterEmbeddingNilTarget
	}
	t := target.identity
	if drafter.Vocab != t.Vocab {
		return nil, fmt.Errorf("%w: drafter %q vocab=%d, target %q vocab=%d",
			ErrDrafterEmbeddingVocabMismatch, drafter.Name, drafter.Vocab, t.Name, t.Vocab)
	}
	if drafter.Hidden != t.Hidden {
		return nil, fmt.Errorf("%w: drafter %q hidden=%d, target %q hidden=%d",
			ErrDrafterEmbeddingHiddenMismatch, drafter.Name, drafter.Hidden, t.Name, t.Hidden)
	}
	if drafter.Dtype != t.Dtype {
		return nil, fmt.Errorf("%w: drafter %q dtype=%q, target %q dtype=%q",
			ErrDrafterEmbeddingDtypeMismatch, drafter.Name, drafter.Dtype, t.Name, t.Dtype)
	}
	// Alias: same backing array, the drafter's own name retained for diagnostics.
	return &EmbeddingTable{identity: drafter, data: target.data}, nil
}

// TargetEmbeddingTable returns the target model's token embedding table as an
// EmbeddingTable, reading its identity from the manifest and its payload from the
// zero-copy f32 view (m.tensor). It is the table a drafter aliases onto. A model
// whose token table is a packed (non-f32) store has no f32 alias view and is
// refused rather than silently expanded.
func (m *Model) TargetEmbeddingTable() (*EmbeddingTable, error) {
	if m == nil {
		return nil, ErrDrafterEmbeddingNilTarget
	}
	if m.Q2KEmbedding != nil {
		return nil, fmt.Errorf("model: target %s is a packed %s store with no f32 alias view",
			tiedEmbeddingName, m.Q2KEmbedding.Format())
	}
	meta, ok := m.manifest[tiedEmbeddingName]
	if !ok {
		return nil, fmt.Errorf("model: target model has no %s", tiedEmbeddingName)
	}
	if len(meta.Shape) != 2 {
		return nil, fmt.Errorf("model: target %s shape %v is not [vocab, hidden]", tiedEmbeddingName, meta.Shape)
	}
	return &EmbeddingTable{
		identity: EmbeddingTableIdentity{
			Name:   tiedEmbeddingName,
			Vocab:  meta.Shape[0],
			Hidden: meta.Shape[1],
			Dtype:  meta.Dtype,
		},
		data: m.tensor(tiedEmbeddingName),
	}, nil
}

// AliasDrafterEmbeddingTable is the model-level entry point: it resolves the
// target model's own token table and aliases the drafter's declared identity onto
// it. It is the fak analogue of the vLLM MTP fix binding the drafter's embedding
// to the owning stage's parameters.
func (m *Model) AliasDrafterEmbeddingTable(drafter EmbeddingTableIdentity) (*EmbeddingTable, error) {
	target, err := m.TargetEmbeddingTable()
	if err != nil {
		return nil, err
	}
	return AliasDrafterEmbedding(target, drafter)
}
