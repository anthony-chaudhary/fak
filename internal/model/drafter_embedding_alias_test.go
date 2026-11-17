package model

import (
	"encoding/binary"
	"errors"
	"math"
	"testing"
)

// drafterEmbeddingAliasFixture builds a target model whose token embedding table is
// a [vocab, hidden] f32 tensor, mirroring the dflash2 fixture's raw+manifest shape.
func drafterEmbeddingAliasFixture(vocab, hidden int, dtype string) *Model {
	raw := make([]byte, vocab*hidden*4)
	for i := 0; i < vocab*hidden; i++ {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(float32(i)))
	}
	return &Model{
		Cfg: Config{HiddenSize: hidden, VocabSize: vocab},
		manifest: map[string]tensorMeta{
			tiedEmbeddingName: {Dtype: dtype, Shape: []int{vocab, hidden}, Nbytes: len(raw)},
		},
		raw: raw,
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestDrafterEmbeddingAlias(t *testing.T) {
	const vocab, hidden = 6, 4
	target := drafterEmbeddingAliasFixture(vocab, hidden, "F32")

	t.Run("matching_drafter_aliases_target", func(t *testing.T) {
		before := EmbeddingTableAllocCount()
		alias, err := target.AliasDrafterEmbeddingTable(EmbeddingTableIdentity{
			Name: "mtp.embed_tokens.weight", Vocab: vocab, Hidden: hidden, Dtype: "F32",
		})
		if err != nil {
			t.Fatalf("matching drafter refused: %v", err)
		}
		// Identity: the alias shares the target's backing storage.
		targetTable, err := target.TargetEmbeddingTable()
		if err != nil {
			t.Fatal(err)
		}
		if !alias.Aliases(targetTable) {
			t.Fatal("alias does not share the target's backing storage")
		}
		if &alias.Data()[0] != &targetTable.Data()[0] {
			t.Fatal("alias first-element address differs from target")
		}
		// Mutation through the alias is visible in the target.
		alias.Data()[0] = 12345
		if targetTable.Data()[0] != 12345 {
			t.Fatalf("mutation through alias not visible in target: got %g", targetTable.Data()[0])
		}
		// No second allocation on the alias path.
		if got := EmbeddingTableAllocCount(); got != before {
			t.Fatalf("alias path allocated: count %d -> %d", before, got)
		}
	})

	t.Run("vocab_mismatch_names_vocab", func(t *testing.T) {
		_, err := target.AliasDrafterEmbeddingTable(EmbeddingTableIdentity{
			Name: "mtp.embed_tokens.weight", Vocab: vocab + 1, Hidden: hidden, Dtype: "F32",
		})
		if !errors.Is(err, ErrDrafterEmbeddingVocabMismatch) {
			t.Fatalf("want vocab mismatch, got %v", err)
		}
		if errors.Is(err, ErrDrafterEmbeddingHiddenMismatch) || errors.Is(err, ErrDrafterEmbeddingDtypeMismatch) {
			t.Fatalf("vocab mismatch misreported: %v", err)
		}
	})

	t.Run("hidden_mismatch_names_hidden", func(t *testing.T) {
		_, err := target.AliasDrafterEmbeddingTable(EmbeddingTableIdentity{
			Name: "mtp.embed_tokens.weight", Vocab: vocab, Hidden: hidden + 2, Dtype: "F32",
		})
		if !errors.Is(err, ErrDrafterEmbeddingHiddenMismatch) {
			t.Fatalf("want hidden mismatch, got %v", err)
		}
		if errors.Is(err, ErrDrafterEmbeddingVocabMismatch) || errors.Is(err, ErrDrafterEmbeddingDtypeMismatch) {
			t.Fatalf("hidden mismatch misreported: %v", err)
		}
	})

	t.Run("dtype_mismatch_names_dtype", func(t *testing.T) {
		_, err := target.AliasDrafterEmbeddingTable(EmbeddingTableIdentity{
			Name: "mtp.embed_tokens.weight", Vocab: vocab, Hidden: hidden, Dtype: "BF16",
		})
		if !errors.Is(err, ErrDrafterEmbeddingDtypeMismatch) {
			t.Fatalf("want dtype mismatch, got %v", err)
		}
		if errors.Is(err, ErrDrafterEmbeddingVocabMismatch) || errors.Is(err, ErrDrafterEmbeddingHiddenMismatch) {
			t.Fatalf("dtype mismatch misreported: %v", err)
		}
	})

	t.Run("mismatch_never_allocates_a_copy", func(t *testing.T) {
		before := EmbeddingTableAllocCount()
		for _, decl := range []EmbeddingTableIdentity{
			{Name: "d", Vocab: vocab + 1, Hidden: hidden, Dtype: "F32"},
			{Name: "d", Vocab: vocab, Hidden: hidden + 1, Dtype: "F32"},
			{Name: "d", Vocab: vocab, Hidden: hidden, Dtype: "BF16"},
		} {
			got, err := target.AliasDrafterEmbeddingTable(decl)
			if err == nil {
				t.Fatalf("mismatched drafter %+v silently aliased", decl)
			}
			if got != nil {
				t.Fatalf("mismatched drafter %+v returned a table instead of nil", decl)
			}
		}
		if after := EmbeddingTableAllocCount(); after != before {
			t.Fatalf("mismatch path allocated a second copy: count %d -> %d", before, after)
		}
	})

	t.Run("nil_target_refused", func(t *testing.T) {
		var m *Model
		if _, err := m.AliasDrafterEmbeddingTable(EmbeddingTableIdentity{Vocab: 1, Hidden: 1, Dtype: "F32"}); !errors.Is(err, ErrDrafterEmbeddingNilTarget) {
			t.Fatalf("want nil-target refusal, got %v", err)
		}
		if _, err := AliasDrafterEmbedding(nil, EmbeddingTableIdentity{Vocab: 1, Hidden: 1, Dtype: "F32"}); !errors.Is(err, ErrDrafterEmbeddingNilTarget) {
			t.Fatalf("want nil-target refusal, got %v", err)
		}
	})

	t.Run("packed_target_refused_not_expanded", func(t *testing.T) {
		packed := drafterEmbeddingAliasFixture(vocab, hidden, "F32")
		packed.Q2KEmbedding = &Q2KEmbedding{vocab: vocab, hidden: hidden, format: packedEmbeddingQ2K}
		if _, err := packed.AliasDrafterEmbeddingTable(EmbeddingTableIdentity{Vocab: vocab, Hidden: hidden, Dtype: "F32"}); err == nil {
			t.Fatal("packed target silently produced an f32 alias view")
		}
	})
}
