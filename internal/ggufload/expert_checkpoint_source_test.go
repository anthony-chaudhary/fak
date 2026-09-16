package ggufload

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// expert_checkpoint_source_test.go — the internal/ggufload witnesses for R5 (#5616, epic #5606,
// docs/MOE-ACTIVATED-OFFLOAD-PLAN.md).
//
// The rung's claim on this side of the seam is that a checkpoint's fused routed-expert slabs can be
// DESCRIBED without being READ. Two properties carry it, and the tests are ordered by how badly
// their failure would hurt:
//
//   - Describing costs nothing. If FusedExpertTensors read payload it would have materialized the
//     expert bulk to avoid materializing the expert bulk, and the whole rung would be a rename.
//   - The description is byte-exact. A descriptor is arithmetic (offset + e*stride) with no
//     checksum behind it, so an off-by-one in Rows/Cols/Experts would not fail — it would feed a
//     GEMM misaligned quant blocks and return plausible garbage. So the descriptors are checked
//     against the SHIPPED eager splitter's output, expert by expert, byte for byte.
//
// The rest pin the decline rules (a tensor this tier cannot serve must keep the path it has today)
// and the lifetime contract (the entry point that closes the checkpoint must refuse to build a tier
// over it, rather than return a model that dies on its first routed expert).

// glmMoeDsaCheckpointFixture writes a complete glm_moe_dsa GGUF whose batched routed experts carry
// `typ`, and returns its path together with the expert count and the projection suffixes. The
// reduction dims are 256 so the expert rows are whole super-blocks, mirroring the real GLM-5.2
// constraint the raw split gates on.
func glmMoeDsaCheckpointFixture(t *testing.T, typ TensorType) (path string, experts int) {
	t.Helper()
	const (
		H, V                = 256, 8
		qLora, kvLora       = 32, 32
		qkNope, qkRope, vHd = 16, 16, 16
		nH                  = 2
		idxHeads, idxDim    = 2, 16
		E, I, sharedI       = 3, 256, 256
	)
	blob := glmMoeDsaFullGGUFTyped(H, V, qLora, kvLora, qkNope, qkRope, vHd, nH, idxHeads, idxDim, E, I, sharedI, typ)
	path = filepath.Join(t.TempDir(), "glm_experts.gguf")
	if err := os.WriteFile(path, blob, 0o644); err != nil {
		t.Fatal(err)
	}
	return path, E
}

// countingReaderAt records every payload read made through it, so a test can assert that a code
// path performed none at all.
type countingReaderAt struct {
	r     io.ReaderAt
	reads atomic.Int64
	bytes atomic.Int64
}

func (c *countingReaderAt) ReadAt(p []byte, off int64) (int, error) {
	c.reads.Add(1)
	n, err := c.r.ReadAt(p, off)
	c.bytes.Add(int64(n))
	return n, err
}

// TestFusedExpertDescriptorsCostNoPayloadIO is the rung's load-bearing property on this side. The
// tensor directory is parsed at open and already carries every field the tier's stride math needs,
// so describing the expert slabs must not touch the payload — the slab stays on disk until a router
// picks an expert out of it. A descriptor builder that read even one slab to "check" it would have
// paid the whole cost the rung exists to remove.
func TestFusedExpertDescriptorsCostNoPayloadIO(t *testing.T) {
	path, E := glmMoeDsaCheckpointFixture(t, TensorQ6_K)

	f, gg, size, err := openAndRead(path)
	if err != nil {
		t.Fatalf("openAndRead: %v", err)
	}
	defer f.Close()
	counting := &countingReaderAt{r: f}
	ws, err := NewWeightSource(gg, counting, size)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}

	shards, err := ws.FusedExpertTensors()
	if err != nil {
		t.Fatalf("FusedExpertTensors: %v", err)
	}
	if n := counting.reads.Load(); n != 0 {
		t.Fatalf("describing the expert slabs issued %d reads (%d bytes); the description must come from the "+
			"tensor directory alone", n, counting.bytes.Load())
	}
	if len(shards) != 1 {
		t.Fatalf("a single-file checkpoint produced %d shard groups, want 1", len(shards))
	}
	if got := len(shards[0].Fused); got != 3 {
		t.Fatalf("described %d fused slabs, want 3 (gate/up/down for the one MoE layer)", got)
	}
	if shards[0].Size != size {
		t.Fatalf("shard size %d, want the file's %d", shards[0].Size, size)
	}
	for _, d := range shards[0].Fused {
		if d.Experts != E {
			t.Fatalf("slab %s describes %d experts, want %d", d.Name, d.Experts, E)
		}
		if d.Quant != model.ExpertCheckpointQ6K {
			t.Fatalf("slab %s described as %s staging, want Q6_K", d.Name, d.Quant)
		}
		if d.Offset <= 0 {
			t.Fatalf("slab %s has file offset %d; the descriptor must locate the payload absolutely", d.Name, d.Offset)
		}
	}

	// And a tier built over those descriptors is still cold: constructing it indexes, it does not read.
	tier, err := ws.ExpertCheckpointTier(0)
	if err != nil {
		t.Fatalf("ExpertCheckpointTier: %v", err)
	}
	if tier == nil {
		t.Fatal("no tier over a checkpoint whose experts are all describable")
	}
	if st := tier.Stats(); st.Tensors != E*3 || st.Reads != 0 {
		t.Fatalf("tier indexed %d experts with %d reads, want %d and 0", st.Tensors, st.Reads, E*3)
	}
	if n := counting.reads.Load(); n != 0 {
		t.Fatalf("building the tier issued %d payload reads; indexing must stay cold", n)
	}
}

// TestFusedExpertDescriptorsLocateTheEagerSplitByteForByte is the correctness witness. Each
// descriptor is arithmetic over a file offset with nothing to catch a mistake at decode, so it is
// checked against the shipped eager path: the bytes at Offset + e*stride must be EXACTLY the bytes
// splitGLMMoeDsaExpertsRawQuant made resident for expert e. Any disagreement in Rows, Cols, Experts
// or the block geometry shows up here as a byte diff instead of as garbage logits later.
func TestFusedExpertDescriptorsLocateTheEagerSplitByteForByte(t *testing.T) {
	for _, tc := range []struct {
		name string
		typ  TensorType
	}{{"Q6_K", TensorQ6_K}, {"Q5_K", TensorQ5_K}} {
		t.Run(tc.name, func(t *testing.T) {
			path, E := glmMoeDsaCheckpointFixture(t, tc.typ)

			// The eager arm: every routed expert materialized into the resident k-quant store.
			eager, err := LoadModelQ4KProfile(path, nil)
			if err != nil {
				t.Fatalf("eager load: %v", err)
			}
			if got := eager.KQuantCount(); got != E*3 {
				t.Fatalf("eager load made %d experts resident, want %d; the comparison arm is wrong", got, E*3)
			}

			ws, err := OpenWeights(path)
			if err != nil {
				t.Fatalf("OpenWeights: %v", err)
			}
			defer ws.Close()
			shards, err := ws.FusedExpertTensors()
			if err != nil {
				t.Fatalf("FusedExpertTensors: %v", err)
			}

			blockWeights, blockBytes, ok := residentExpertBlockGeometry(tc.typ)
			if !ok {
				t.Fatalf("no block geometry for %s", tc.name)
			}
			f, err := os.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()

			checked := 0
			for _, sh := range shards {
				for _, d := range sh.Fused {
					stride := int64(d.Rows*d.Cols/blockWeights) * int64(blockBytes)
					for e := 0; e < d.Experts; e++ {
						canon := fmt.Sprintf("model.layers.%d.mlp.experts.%d.%s.weight", d.Layer, e, d.Proj)
						want, ok := eager.KQuantRaw(canon)
						if !ok {
							t.Fatalf("the eager load has no resident %s to compare against", canon)
						}
						if int64(len(want)) != stride {
							t.Fatalf("%s: descriptor stride %d disagrees with the resident %d bytes; the "+
								"descriptor's shape or block geometry is wrong", canon, stride, len(want))
						}
						got := make([]byte, stride)
						if _, err := f.ReadAt(got, d.Offset+int64(e)*stride); err != nil {
							t.Fatalf("%s: read at descriptor offset %d: %v", canon, d.Offset+int64(e)*stride, err)
						}
						if !bytes.Equal(got, want) {
							t.Fatalf("%s: the bytes at the descriptor's offset are not the bytes the eager "+
								"split made resident; a fault here would feed the GEMM misaligned blocks", canon)
						}
						checked++
					}
				}
			}
			if checked != E*3 {
				t.Fatalf("compared %d experts, want %d", checked, E*3)
			}
		})
	}
}

// TestFusedExpertDescriptorsDeclineWhatTheTierCannotServe pins the decline direction. A descriptor
// the tier cannot serve exactly is worse than no descriptor at all — the missing one costs host
// RAM, the wrong one costs correctness — so every case the tier has no staging for must fall
// through to the unchanged eager path, silently and completely.
func TestFusedExpertDescriptorsDeclineWhatTheTierCannotServe(t *testing.T) {
	// Residentable expert quants the checkpoint tier has no staging for yet. They must keep the
	// eager raw-resident path they take today, not be described and then fail at decode.
	for _, tc := range []struct {
		name string
		typ  TensorType
	}{{"Q8_0", TensorQ8_0}, {"IQ4_XS", TensorIQ4_XS}, {"IQ3_XXS", TensorIQ3_XXS}} {
		t.Run(tc.name, func(t *testing.T) {
			path, E := glmMoeDsaCheckpointFixture(t, tc.typ)
			ws, err := OpenWeights(path)
			if err != nil {
				t.Fatalf("OpenWeights: %v", err)
			}
			defer ws.Close()
			shards, err := ws.FusedExpertTensors()
			if err != nil {
				t.Fatalf("FusedExpertTensors: %v", err)
			}
			if len(shards) != 0 {
				t.Fatalf("described %d shard groups of %s experts; the tier cannot stage that representation",
					len(shards), tc.name)
			}
			// The decline is inert: the eager path still holds them resident, exactly as before.
			m, err := LoadModelQ4KProfile(path, nil)
			if err != nil {
				t.Fatalf("eager load: %v", err)
			}
			if got := m.KQuantCount(); got != E*3 {
				t.Fatalf("%s experts resident = %d, want %d; declining a descriptor must change nothing",
					tc.name, got, E*3)
			}
		})
	}

	// A dense checkpoint has no batched expert tensors at all, and must not be probed for them.
	t.Run("dense checkpoint", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "dense.gguf")
		if err := os.WriteFile(path, tinyCanonicalModelGGUF(t), 0o644); err != nil {
			t.Fatal(err)
		}
		ws, err := OpenWeights(path)
		if err != nil {
			t.Fatalf("OpenWeights: %v", err)
		}
		defer ws.Close()
		shards, err := ws.FusedExpertTensors()
		if err != nil {
			t.Fatalf("FusedExpertTensors over a dense checkpoint: %v", err)
		}
		if len(shards) != 0 {
			t.Fatalf("a dense checkpoint described %d expert shard groups", len(shards))
		}
		tier, err := ws.ExpertCheckpointTier(0)
		if err != nil || tier != nil {
			t.Fatalf("ExpertCheckpointTier over a dense checkpoint = (%v, %v), want (nil, nil)", tier, err)
		}
	})
}

// TestExpertCheckpointSourceDescribesQ2KSlab is the Q2_K admission witness (issue #13122). The
// checkpoint tier is the only path that serves the published DeepSeek-V4.1 Q2_K artifact's routed
// experts without materializing the slab, and it can only do that if FusedExpertTensors DESCRIBES
// the Q2_K blocks at all: before checkpointExpertQuant admitted Q2_K, this call returned zero shard
// groups for a Q2_K checkpoint (the rung silently declined it), so a Q2_K caller kept the eager
// materialization the rung exists to avoid. The descriptor must also agree with the tier's own
// regime — Q2_K is a stageable quant with the same 256-weight super-block geometry as the other
// k-quants, so every fused slab that can be described must be describable here, cold.
func TestExpertCheckpointSourceDescribesQ2KSlab(t *testing.T) {
	path, E := glmMoeDsaCheckpointFixture(t, TensorQ2_K)

	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	shards, err := ws.FusedExpertTensors()
	if err != nil {
		t.Fatalf("FusedExpertTensors: %v", err)
	}
	if len(shards) != 1 {
		t.Fatalf("a Q2_K checkpoint described %d shard groups, want 1; before Q2_K admission this was 0",
			len(shards))
	}
	if got := len(shards[0].Fused); got != 3 {
		t.Fatalf("described %d fused Q2_K slabs, want 3 (gate/up/down for the one MoE layer)", got)
	}
	for _, d := range shards[0].Fused {
		if d.Quant != model.ExpertCheckpointQ2K {
			t.Fatalf("slab %s described as %s staging, want Q2_K", d.Name, d.Quant)
		}
		if d.Experts != E {
			t.Fatalf("slab %s describes %d experts, want %d", d.Name, d.Experts, E)
		}
		if d.Offset <= 0 {
			t.Fatalf("slab %s has file offset %d; the descriptor must locate the payload absolutely",
				d.Name, d.Offset)
		}
	}

	// And a tier built over those Q2_K descriptors indexes every expert while staying cold: it must
	// not read the slab to describe it, or the rung has merely renamed the eager materialization.
	tier, err := ws.ExpertCheckpointTier(0)
	if err != nil {
		t.Fatalf("ExpertCheckpointTier: %v", err)
	}
	if tier == nil {
		t.Fatal("no tier over a checkpoint whose Q2_K experts are all describable")
	}
	if st := tier.Stats(); st.Tensors != E*3 || st.Reads != 0 {
		t.Fatalf("Q2_K tier indexed %d experts with %d reads, want %d and 0", st.Tensors, st.Reads, E*3)
	}
}

// TestExpertCheckpointSourceQ2KFaultMatchesTheEagerSlabSlice is the Q2_K arithmetic witness on this
// side of the seam. A descriptor is offset + e*stride with no checksum behind it, and a Q2_K slab
// makes that arithmetic Q2_K-specific: the stride is Rows*Cols/256 super-blocks at 72 bytes each
// (blockQ2KBytes), a geometry the other k-quants do not share. A descriptor that used Q4_K's 144-byte
// block here would not fail at index time — it would fault a misaligned block into the decode path
// and return plausible garbage. So the bytes the descriptor points at are checked against the bytes
// the SHIPPED eager splitter made resident for the same expert: if the Q2_K geometry is wrong, this
// is a byte diff instead of a silent misdecode. The stride is also pinned to the resident length so
// the descriptor and the resident store cannot disagree about how much an expert is.
func TestExpertCheckpointSourceQ2KFaultMatchesTheEagerSlabSlice(t *testing.T) {
	path, E := glmMoeDsaCheckpointFixture(t, TensorQ2_K)

	// The eager arm: the shipped raw split materializes every routed Q2_K expert into the resident
	// k-quant store, and those bytes are the parity reference.
	eager, err := LoadModelQ4KProfile(path, nil)
	if err != nil {
		t.Fatalf("eager load: %v", err)
	}
	if got := eager.KQuantCount(); got != E*3 {
		t.Fatalf("eager load made %d Q2_K experts resident, want %d; the comparison arm is wrong", got, E*3)
	}

	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()
	shards, err := ws.FusedExpertTensors()
	if err != nil {
		t.Fatalf("FusedExpertTensors: %v", err)
	}
	if len(shards) != 1 || len(shards[0].Fused) != 3 {
		t.Fatalf("Q2_K checkpoint described %d shard groups / %d slabs, want 1 / 3",
			len(shards), fusedCount(shards))
	}

	// Locate the gate_proj descriptor (the fixture's three slabs are gate/up/down for one layer) and
	// read its LAST expert's stride bytes straight off disk. Reading the last expert proves the
	// offset advances by exactly one stride per expert, which a first-expert-only check would miss.
	var gate model.FusedExpertTensor
	found := false
	for _, d := range shards[0].Fused {
		if d.Proj == "gate_proj" {
			gate, found = d, true
			break
		}
	}
	if !found {
		t.Fatal("the Q2_K checkpoint described no gate_proj slab")
	}

	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	// blockQ2KBytes is the ggufload-side Q2_K super-block width (72 B); qkK is the 256-weight
	// super-block. Deriving the stride from those two names — not a literal — is what ties this test
	// to the SAME geometry the loader uses, so a future block-size change moves both together.
	stride := int64(gate.Rows*gate.Cols/qkK) * int64(blockQ2KBytes)
	e := E - 1
	canon := fmt.Sprintf("model.layers.%d.mlp.experts.%d.%s.weight", gate.Layer, e, gate.Proj)
	want, ok := eager.KQuantRaw(canon)
	if !ok {
		t.Fatalf("the eager load has no resident %s to compare against", canon)
	}
	if stride != int64(len(want)) {
		t.Fatalf("%s: descriptor stride %d disagrees with the resident %d bytes; the Q2_K block "+
			"geometry is wrong", canon, stride, len(want))
	}
	if len(want) == 0 {
		t.Fatalf("%s: the eager split made an empty expert resident", canon)
	}
	got := make([]byte, stride)
	if _, err := f.ReadAt(got, gate.Offset+int64(e)*stride); err != nil {
		t.Fatalf("%s: read at descriptor offset %d: %v", canon, gate.Offset+int64(e)*stride, err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("%s: the bytes at the Q2_K descriptor's offset+stride are not the bytes the eager "+
			"split made resident; a fault here would feed the GEMM misaligned Q2_K blocks", canon)
	}

	// The tier built over the same descriptors is cold and fully indexed; the byte comparison above
	// is the parity witness on this side (the model package owns the fault itself).
	tier, err := ws.ExpertCheckpointTier(0)
	if err != nil {
		t.Fatalf("ExpertCheckpointTier: %v", err)
	}
	if tier == nil {
		t.Fatal("no tier over a Q2_K checkpoint whose experts are describable")
	}
	if st := tier.Stats(); st.Tensors != E*3 {
		t.Fatalf("Q2_K tier indexed %d experts, want %d", st.Tensors, E*3)
	}
}

// TestExpertCheckpointSourceAdmitsQ3K is the Q3_K admission witness on the loader side of the seam,
// replacing the fak#13122 settlement witness. Q3_K is residentable raw (residentExpertBlockGeometry
// returns 256-weight/110-byte blocks) and internal/compute now carries a verbatim Q3_K host tensor
// kind (compute.NewQ3K, fak#13149), so the tier stages a Q3_K expert exactly like Q2_K. A
// descriptor is therefore promised AND deliverable; checkpointExpertQuant must ADMIT Q3_K, and the
// mixed Q2_K/Q3_K V4.1 artifact depends on it.
func TestExpertCheckpointSourceAdmitsQ3K(t *testing.T) {
	path, E := glmMoeDsaCheckpointFixture(t, TensorQ3_K)

	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()
	shards, err := ws.FusedExpertTensors()
	if err != nil {
		t.Fatalf("FusedExpertTensors: %v", err)
	}
	if got := fusedCount(shards); got != 3 {
		t.Fatalf("described %d fused Q3_K slabs, want 3 (gate/up/down); Q3_K must be admitted", got)
	}

	// Nothing is unstageable over a pure Q3_K expert checkpoint, so the partial-decline guard stays
	// silent and the streamed tier builds over every slab.
	unstageable, err := ws.UnstageableRoutedExpertSlabs()
	if err != nil {
		t.Fatalf("UnstageableRoutedExpertSlabs: %v", err)
	}
	if len(unstageable) != 0 {
		t.Fatalf("Q3_K slabs reported unstageable after admission: %v", unstageable)
	}
	tier, err := ws.ExpertCheckpointTier(0)
	if err != nil {
		t.Fatalf("ExpertCheckpointTier: %v", err)
	}
	if tier == nil {
		t.Fatal("no tier over a Q3_K checkpoint whose experts are now describable")
	}
	if st := tier.Stats(); st.Tensors != E*3 {
		t.Fatalf("Q3_K tier indexed %d experts, want %d", st.Tensors, E*3)
	}

	// The eager path still makes every Q3_K expert resident (unchanged); admission only adds the
	// streamed alternative, it does not remove the resident one.
	m, err := LoadModelQ4KProfile(path, nil)
	if err != nil {
		t.Fatalf("eager load: %v", err)
	}
	if got := m.KQuantCount(); got != E*3 {
		t.Fatalf("Q3_K experts resident = %d, want %d", got, E*3)
	}
}

// fusedCount totals the fused slabs across shard groups, for a diagnostic that names the shape it
// actually saw rather than only the group count.
func fusedCount(shards []FusedExpertShard) int {
	n := 0
	for _, sh := range shards {
		n += len(sh.Fused)
	}
	return n
}

// TestStreamedExpertsLoadLeavesTheSlabOnDisk is the end-to-end shape of the rung on the loader:
// WithStreamedExperts must produce a model that carries NO routed expert in host RAM and reaches
// all of them through the tier. Both halves matter — a load that attached a tier AND materialized
// the experts would report a tier while still paying the bytes, and would look identical to this
// test's happy path on the tier stats alone.
func TestStreamedExpertsLoadLeavesTheSlabOnDisk(t *testing.T) {
	path, E := glmMoeDsaCheckpointFixture(t, TensorQ6_K)

	// The eager arm, for the contrast the assertions are stated against.
	eager, err := LoadModelQ4KProfile(path, nil)
	if err != nil {
		t.Fatalf("eager load: %v", err)
	}
	if got := eager.KQuantCount(); got != E*3 {
		t.Fatalf("eager load made %d experts resident, want %d", got, E*3)
	}
	if st := eager.ExpertCheckpointStats(); st.Enabled {
		t.Fatalf("an eager load attached a checkpoint tier: %+v", st)
	}

	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()
	prof := NewLoadProfiler()
	var progress bytes.Buffer
	prof.Progress = &progress
	m, err := ws.QuantModelQ4KProfileOptions(prof, WithStreamedExperts(0))
	if err != nil {
		t.Fatalf("streamed load: %v", err)
	}

	if got := m.KQuantCount(); got != 0 {
		t.Fatalf("a streamed load still made %d routed experts resident; the slab was materialized anyway", got)
	}
	st := m.ExpertCheckpointStats()
	if !st.Enabled || st.Tensors != E*3 {
		t.Fatalf("streamed model reports %+v, want an enabled tier over %d experts", st, E*3)
	}
	if st.BudgetBytes != 0 || st.ResidentBytes != 0 {
		t.Fatalf("a 0-byte host budget retained %d bytes (budget %d); the default must be stream-through",
			st.ResidentBytes, st.BudgetBytes)
	}
	if st.Reads != 0 {
		t.Fatalf("the load itself faulted %d experts; nothing must be read until a router picks one", st.Reads)
	}

	// The load-path breakdown must not claim these bytes: they were neither held resident nor
	// round-tripped through f32, so tallying them either way would misreport what the load cost.
	for _, r := range prof.loadPathRows() {
		if r.Expert && r.QuantType == "Q6_K" {
			t.Fatalf("the streamed expert slab was tallied on the load-path breakdown as %+v", r)
		}
	}
	if !strings.Contains(progress.String(), "routed projections streamed from the checkpoint") {
		t.Fatalf("the load reported no streamed-expert line; the placement decision must be visible: %q", progress.String())
	}
}

// TestStreamedExpertsRefusedWhereTheCheckpointCannotOutliveTheModel is the lifetime gate. The tier
// reads through the WeightSource's own shard readers, so a path-taking entry point that closes the
// source on return would hand back a model whose every routed expert is unreadable — and over a
// streamed checkpoint there is no resident copy to fall back to, so it would be a dead model, not a
// slow one. Refusing at the request is the only honest answer.
func TestStreamedExpertsRefusedWhereTheCheckpointCannotOutliveTheModel(t *testing.T) {
	path, _ := glmMoeDsaCheckpointFixture(t, TensorQ6_K)

	m, err := LoadModelQ4KProfileOptions(path, nil, WithStreamedExperts(0))
	if err == nil {
		t.Fatal("the path-taking loader accepted WithStreamedExperts; it closes the checkpoint before returning")
	}
	if m != nil {
		t.Fatal("a refused load still returned a model")
	}
	if !strings.Contains(err.Error(), "outlives the model") {
		t.Fatalf("refusal %q does not name the lifetime reason, so a caller cannot act on it", err)
	}

	// A streamed tier serves every expert; an expert-parallel band says this rank must never touch
	// the others. Silently honouring one would serve a rank experts it was sharded out of.
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()
	if _, err := ws.QuantModelQ4KProfileOptions(nil, WithStreamedExperts(0), WithExpertShard(0, 1)); err == nil {
		t.Fatal("streamed experts and an expert-parallel shard were accepted together")
	}

	// And a checkpoint with nothing streamable refuses rather than silently loading eagerly under a
	// flag the caller passed precisely to avoid that.
	q8Path, _ := glmMoeDsaCheckpointFixture(t, TensorQ8_0)
	q8, err := OpenWeights(q8Path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer q8.Close()
	if _, err := q8.QuantModelQ4KProfileOptions(nil, WithStreamedExperts(0)); err == nil {
		t.Fatal("a checkpoint with no streamable expert slab loaded streamed anyway")
	}
}
