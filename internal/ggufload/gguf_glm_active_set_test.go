package ggufload

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// TestRoutedExpertActiveSet_HeaderArithmetic builds a minimal glm_moe_dsa GGUF with a known
// batched routed-expert band and asserts RoutedExpertActiveSet derives the Lane F (#3074) active
// set — routed-expert resident band, per-expert bytes, and K×per-expert active bytes/token — from
// the header alone, reading no tensor payload. The fixture's one F32 [E,I,H] routed blob is
// E*I*H*4 bytes across E experts, and glmMoeDsaExpertGGUF sets expert_used_count (K) = 2.
func TestRoutedExpertActiveSet_HeaderArithmetic(t *testing.T) {
	const E, I, H = 4, 3, 2 // 4 experts, expert FFN len 3, hidden 2
	gate := make([]float32, E*I*H)
	for i := range gate {
		gate[i] = float32(i)
	}
	raw := glmMoeDsaExpertGGUF(E, I, H, gate)
	p := filepath.Join(t.TempDir(), "glm.gguf")
	if err := os.WriteFile(p, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	ws, err := OpenWeights(p)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	as, ok, err := ws.RoutedExpertActiveSet()
	if err != nil || !ok {
		t.Fatalf("RoutedExpertActiveSet ok=%v err=%v (want ok=true)", ok, err)
	}
	wantBand := int64(E * I * H * 4) // one F32 batched routed tensor payload, all experts
	wantPer := wantBand / int64(E)
	if as.NumExperts != E {
		t.Errorf("NumExperts=%d, want %d", as.NumExperts, E)
	}
	if as.ExpertsUsed != 2 {
		t.Errorf("ExpertsUsed(K)=%d, want 2", as.ExpertsUsed)
	}
	if as.RoutedResident != wantBand {
		t.Errorf("RoutedResident=%d, want %d", as.RoutedResident, wantBand)
	}
	if as.PerExpert != wantPer {
		t.Errorf("PerExpert=%d, want %d", as.PerExpert, wantPer)
	}
	if as.ActivePerToken != wantPer*2 {
		t.Errorf("ActivePerToken=%d, want %d (K=2 × per-expert)", as.ActivePerToken, wantPer*2)
	}
	// Params + non-expert fields: this fixture is one F32 routed blob and nothing else, so the
	// non-expert remainder is zero and active-*/token collapse to the routed stream.
	wantParams := int64(E * I * H) // one F32 tensor, elem count = E*I*H
	wantPerParams := wantParams / int64(E)
	if as.RoutedParams != wantParams {
		t.Errorf("RoutedParams=%d, want %d", as.RoutedParams, wantParams)
	}
	if as.PerExpertParams != wantPerParams {
		t.Errorf("PerExpertParams=%d, want %d", as.PerExpertParams, wantPerParams)
	}
	if as.NonExpertResident != 0 || as.NonExpertParams != 0 {
		t.Errorf("non-expert remainder = %d bytes / %d params, want 0/0 (fixture has only the routed blob)", as.NonExpertResident, as.NonExpertParams)
	}
	if as.ActiveBytesPerToken != wantPer*2 {
		t.Errorf("ActiveBytesPerToken=%d, want %d (routed stream + 0 non-expert)", as.ActiveBytesPerToken, wantPer*2)
	}
	if as.ActiveParamsPerToken != wantPerParams*2 {
		t.Errorf("ActiveParamsPerToken=%d, want %d (K=2 × per-expert params + 0)", as.ActiveParamsPerToken, wantPerParams*2)
	}
}

// TestRoutedExpertActiveSet_NonExpertRemainder builds a glm_moe_dsa GGUF that carries BOTH a
// batched routed-expert band AND a non-expert (embedding) tensor, and asserts the header-derived
// active set folds the non-expert stream into active-bytes/token and active-params/token — the
// full per-token divisors, not just the routed band. It mirrors the real GLM-5.2 shape where the
// ~19 GiB attention/dense/shared/embedding resident rides alongside the routed-expert band.
func TestRoutedExpertActiveSet_NonExpertRemainder(t *testing.T) {
	const E, I, H, K = 2, 2, 64, 2 // routed [E,I,H] = 256 elems (1 K-quant super-block); K=2
	const nonElems = 256           // one non-expert (embedding) tensor
	expertType, nonType := TensorQ4_K, TensorQ8_0
	raw := glmQuantArmGGUF(t, E, I, H, K, expertType, nonType, nonElems)
	p := filepath.Join(t.TempDir(), "glm.gguf")
	if err := os.WriteFile(p, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	ws, err := OpenWeights(p)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	as, ok, err := ws.RoutedExpertActiveSet()
	if err != nil || !ok {
		t.Fatalf("RoutedExpertActiveSet ok=%v err=%v (want ok=true)", ok, err)
	}
	// Independently size both tensors straight from the header primitives.
	expertBytes := mustPayloadBytes(t, "blk.0.ffn_gate_exps.weight", []uint64{H, I, E}, expertType)
	nonBytes := mustPayloadBytes(t, "token_embd.weight", []uint64{nonElems}, nonType)
	wantPer := int64(expertBytes) / int64(E)
	wantPerParams := int64(E*I*H) / int64(E)
	wantNonParams := int64(nonElems)

	if as.NonExpertResident != int64(nonBytes) {
		t.Errorf("NonExpertResident=%d, want %d", as.NonExpertResident, nonBytes)
	}
	if as.NonExpertParams != wantNonParams {
		t.Errorf("NonExpertParams=%d, want %d", as.NonExpertParams, wantNonParams)
	}
	if want := wantPer*K + int64(nonBytes); as.ActiveBytesPerToken != want {
		t.Errorf("ActiveBytesPerToken=%d, want %d (K×per-expert bytes + non-expert resident)", as.ActiveBytesPerToken, want)
	}
	if want := wantPerParams*K + wantNonParams; as.ActiveParamsPerToken != want {
		t.Errorf("ActiveParamsPerToken=%d, want %d (K×per-expert params + non-expert params)", as.ActiveParamsPerToken, want)
	}
	// active-bytes/token must strictly exceed the routed-only stream — the non-expert stream is real.
	if as.ActiveBytesPerToken <= as.ActivePerToken {
		t.Errorf("ActiveBytesPerToken=%d must exceed routed-only ActivePerToken=%d", as.ActiveBytesPerToken, as.ActivePerToken)
	}
}

// TestRoutedExpertActiveSet_EmbeddingGatherCorrection asserts the #3074 embedding-gather
// correction: at decode the input token-embedding is a get_rows GATHER (one row/token), not a full
// sweep of the [vocab×hidden] table, so ActiveBytesPerToken (which folds the whole token_embd table
// into the non-expert stream) is an UPPER BOUND. ActiveBytesPerTokenSwept subtracts that table when
// embeddings are UNTIED (a distinct output.weight carries the swept output projection) and leaves it
// when TIED (the same table IS the swept output projection). This is the header-tight per-token
// sweep that the ceiling doc's ~32 GiB upper bound had only hand-waved as "a little lower".
func TestRoutedExpertActiveSet_EmbeddingGatherCorrection(t *testing.T) {
	const E, I, H, K = 2, 2, 64, 2
	const embElems, outElems = 256, 256
	expertType, embType, outType := TensorQ4_K, TensorQ8_0, TensorQ8_0
	embBytes := int64(mustPayloadBytes(t, "token_embd.weight", []uint64{embElems}, embType))

	open := func(raw []byte) (RoutedExpertActiveSet, bool) {
		p := filepath.Join(t.TempDir(), "glm.gguf")
		if err := os.WriteFile(p, raw, 0o644); err != nil {
			t.Fatal(err)
		}
		ws, err := OpenWeights(p)
		if err != nil {
			t.Fatalf("OpenWeights: %v", err)
		}
		defer ws.Close()
		cfg, err := ws.File.Config()
		if err != nil {
			t.Fatalf("Config: %v", err)
		}
		as, ok, err := ws.RoutedExpertActiveSet()
		if err != nil {
			t.Fatalf("RoutedExpertActiveSet err=%v", err)
		}
		// cross-check the tie flag the correction gates on, so the test fails loudly if the
		// TieWordEmbeddings derivation (gguf_config.go) ever changes under it.
		wantTied := !bytes.Contains(raw, []byte("output.weight"))
		if cfg.TieWordEmbeddings != wantTied {
			t.Errorf("TieWordEmbeddings=%v, want %v", cfg.TieWordEmbeddings, wantTied)
		}
		return as, ok
	}

	// UNTIED: a distinct output.weight exists ⇒ the input token_embd table is a gather, subtracted.
	untied, ok := open(glmEmbedGatherGGUF(t, E, I, H, K, embElems, outElems, expertType, embType, outType, true))
	if !ok {
		t.Fatal("untied: RoutedExpertActiveSet ok=false, want true")
	}
	if untied.InputEmbedResident != embBytes {
		t.Errorf("untied InputEmbedResident=%d, want %d", untied.InputEmbedResident, embBytes)
	}
	if untied.InputEmbedGather != embBytes {
		t.Errorf("untied InputEmbedGather=%d, want %d (untied ⇒ subtract the input table)", untied.InputEmbedGather, embBytes)
	}
	if want := untied.ActiveBytesPerToken - embBytes; untied.ActiveBytesPerTokenSwept != want {
		t.Errorf("untied ActiveBytesPerTokenSwept=%d, want %d (upper bound − gather)", untied.ActiveBytesPerTokenSwept, want)
	}
	if untied.ActiveBytesPerTokenSwept >= untied.ActiveBytesPerToken {
		t.Errorf("untied swept=%d must be strictly below the upper bound=%d", untied.ActiveBytesPerTokenSwept, untied.ActiveBytesPerToken)
	}

	// TIED: no output.weight ⇒ token_embd IS the swept output projection, so no gather saving.
	tied, ok := open(glmEmbedGatherGGUF(t, E, I, H, K, embElems, 0, expertType, embType, outType, false))
	if !ok {
		t.Fatal("tied: RoutedExpertActiveSet ok=false, want true")
	}
	if tied.InputEmbedResident != embBytes {
		t.Errorf("tied InputEmbedResident=%d, want %d", tied.InputEmbedResident, embBytes)
	}
	if tied.InputEmbedGather != 0 {
		t.Errorf("tied InputEmbedGather=%d, want 0 (tied ⇒ table is swept as output, no saving)", tied.InputEmbedGather)
	}
	if tied.ActiveBytesPerTokenSwept != tied.ActiveBytesPerToken {
		t.Errorf("tied ActiveBytesPerTokenSwept=%d, want %d (== upper bound)", tied.ActiveBytesPerTokenSwept, tied.ActiveBytesPerToken)
	}
}

// glmEmbedGatherGGUF writes a glm_moe_dsa GGUF with one batched routed-expert tensor, a
// token_embd.weight input-embedding table, and — when untied — a distinct output.weight output
// projection (whose presence makes TieWordEmbeddings false, arming the embedding-gather
// correction). Payloads are zero-filled; only dims+type (hence block payload length) are read.
func glmEmbedGatherGGUF(t *testing.T, E, I, H, K, embElems, outElems int, expertType, embType, outType TensorType, untied bool) []byte {
	t.Helper()
	const align = 32
	nTensors := uint64(2)
	if untied {
		nTensors = 3
	}
	expertBytes := mustPayloadBytes(t, "blk.0.ffn_gate_exps.weight", []uint64{uint64(H), uint64(I), uint64(E)}, expertType)
	embBytes := mustPayloadBytes(t, "token_embd.weight", []uint64{uint64(embElems)}, embType)
	embOffset := (expertBytes + align - 1) / align * align
	outOffset := (embOffset + embBytes + align - 1) / align * align

	var b bytes.Buffer
	writeMinimalHeader(&b, nTensors, 11)
	writeKVString(&b, "general.architecture", "glm_moe_dsa")
	writeKVUint32(&b, "general.alignment", align)
	writeKVUint32(&b, "glm_moe_dsa.embedding_length", uint32(H))
	writeKVUint32(&b, "glm_moe_dsa.block_count", 1)
	writeKVUint32(&b, "glm_moe_dsa.attention.head_count", 2)
	writeKVUint32(&b, "glm_moe_dsa.attention.head_count_kv", 1)
	writeKVUint32(&b, "glm_moe_dsa.feed_forward_length", 8)
	writeKVUint32(&b, "glm_moe_dsa.expert_count", uint32(E))
	writeKVUint32(&b, "glm_moe_dsa.expert_used_count", uint32(K))
	writeKVUint32(&b, "glm_moe_dsa.expert_feed_forward_length", uint32(I))
	writeKVFloat32(&b, "glm_moe_dsa.attention.layer_norm_rms_epsilon", 1e-5)
	writeTensorInfoForTest(&b, "blk.0.ffn_gate_exps.weight", []uint64{uint64(H), uint64(I), uint64(E)}, expertType, 0)
	writeTensorInfoForTest(&b, "token_embd.weight", []uint64{uint64(embElems)}, embType, embOffset)
	end := embOffset + embBytes
	if untied {
		outBytes := mustPayloadBytes(t, "output.weight", []uint64{uint64(outElems)}, outType)
		writeTensorInfoForTest(&b, "output.weight", []uint64{uint64(outElems)}, outType, outOffset)
		end = outOffset + outBytes
	}
	padToAlignment(&b, align)
	start := b.Len()
	padToLen(&b, start+int(end)) // zero-fill the whole data section
	return b.Bytes()
}

// mustPayloadBytes sizes a synthetic tensor's on-disk block payload from the header primitives,
// failing the test on an unsupported type/dim combination.
func mustPayloadBytes(t *testing.T, name string, dims []uint64, typ TensorType) uint64 {
	t.Helper()
	n, err := tensorPayloadBytes(TensorInfo{Name: name, Dims: dims, Type: typ})
	if err != nil {
		t.Fatalf("tensorPayloadBytes(%s): %v", name, err)
	}
	return n
}

// glmQuantArmGGUF writes a minimal glm_moe_dsa GGUF carrying one batched routed-expert tensor
// (blk.0.ffn_gate_exps.weight, dims [E,I,H], quant expertType) and one non-expert tensor
// (token_embd.weight, nonElems elements, quant nonType), with expert_used_count = K. Payloads are
// zero-filled: the header-only active-set/estimate arithmetic reads dims+type and never the tensor
// bytes, so the codes need not be valid quant data — only the block COUNT (hence payload length)
// matters. It is the shared fixture behind the active-set and quant-sweep golden tests.
func glmQuantArmGGUF(t *testing.T, E, I, H, K int, expertType, nonType TensorType, nonElems int) []byte {
	t.Helper()
	const align = 32
	expertBytes := mustPayloadBytes(t, "blk.0.ffn_gate_exps.weight", []uint64{uint64(H), uint64(I), uint64(E)}, expertType)
	nonBytes := mustPayloadBytes(t, "token_embd.weight", []uint64{uint64(nonElems)}, nonType)
	nonOffset := (expertBytes + align - 1) / align * align // second tensor's aligned data offset

	var b bytes.Buffer
	writeMinimalHeader(&b, 2, 11)
	writeKVString(&b, "general.architecture", "glm_moe_dsa")
	writeKVUint32(&b, "general.alignment", align)
	writeKVUint32(&b, "glm_moe_dsa.embedding_length", uint32(H))
	writeKVUint32(&b, "glm_moe_dsa.block_count", 1)
	writeKVUint32(&b, "glm_moe_dsa.attention.head_count", 2)
	writeKVUint32(&b, "glm_moe_dsa.attention.head_count_kv", 1)
	writeKVUint32(&b, "glm_moe_dsa.feed_forward_length", 8)
	writeKVUint32(&b, "glm_moe_dsa.expert_count", uint32(E))
	writeKVUint32(&b, "glm_moe_dsa.expert_used_count", uint32(K))
	writeKVUint32(&b, "glm_moe_dsa.expert_feed_forward_length", uint32(I))
	writeKVFloat32(&b, "glm_moe_dsa.attention.layer_norm_rms_epsilon", 1e-5)
	writeTensorInfoForTest(&b, "blk.0.ffn_gate_exps.weight", []uint64{uint64(H), uint64(I), uint64(E)}, expertType, 0)
	writeTensorInfoForTest(&b, "token_embd.weight", []uint64{uint64(nonElems)}, nonType, nonOffset)

	padToAlignment(&b, align)
	start := b.Len()
	padToLen(&b, start+int(nonOffset+nonBytes)) // zero-fill the whole data section (both payloads)
	return b.Bytes()
}
