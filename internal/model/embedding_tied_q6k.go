package model

import "fmt"

// embedding_tied_q6k.go — one resident copy for a tied Q6_K token table (fak#13567).
//
// A tied checkpoint (no output.weight) uses model.embed_tokens.weight both as the input
// row-gather table and as the LM head. The default quant load expanded that table to f32
// for gathers and kept a second native-Q8 copy for the head: 248320x2560 Q6_K (0.49 GiB on
// disk) became 2.37 GiB f32 + 0.67 GiB Q8. SetTiedQ6KEmbedding instead stores the GGUF Q6_K
// bytes once, page-aligned, in kqw under the embedding name. The same backing serves:
//   - row gather: Q2KEmbedding with format packedEmbeddingQ6K aliases kqw's raw bytes, so
//     embedRowsInto dequantizes only the requested rows (q6kDequantSuperBlock);
//   - head projection: kqHeadName resolves to the embedding name, so headResident routes to
//     headKQuant -> kQuantMatRowsIntoDispatch (Metal Q6_K GEMV, CPU Q6_K int8 fallback), and
//     the Q6_K Metal band (q6kRuntimeNames) aliases the same pages with no device copy.

const tiedEmbeddingName = "model.embed_tokens.weight"

// SetTiedQ6KEmbedding attaches a tied Q6_K token table as the single resident copy for both
// row gather and the LM head. shape is the model [vocab, hidden] convention; raw is the GGUF
// Q6_K payload (copied once into page-aligned storage so Metal can alias it). It refuses an
// untied config, a second packed embedding, and a head that already resolved elsewhere.
func (b *QuantBuilder) SetTiedQ6KEmbedding(shape []int, raw []byte) error {
	if err := b.refuseV41(); err != nil {
		return err
	}
	if b.built {
		return fmt.Errorf("model: QuantBuilder already built")
	}
	if !b.tied {
		return fmt.Errorf("model: tied Q6_K embedding requires tie_word_embeddings")
	}
	b.started = true
	return b.m.attachTiedQ6KEmbedding(shape, raw)
}

// attachTiedQ6KEmbedding validates and installs the shared tied Q6_K table on m.
func (m *Model) attachTiedQ6KEmbedding(shape []int, raw []byte) error {
	if !m.Cfg.TieWordEmbeddings {
		return fmt.Errorf("model: tied Q6_K embedding requires tie_word_embeddings")
	}
	if len(shape) != 2 || shape[0] <= 0 || shape[1] <= 0 {
		return fmt.Errorf("model: tied Q6_K embedding shape %v is not [vocab, hidden]", shape)
	}
	vocab, hidden := shape[0], shape[1]
	if hidden%qkK != 0 {
		return fmt.Errorf("model: tied Q6_K embedding hidden %d is not divisible by %d", hidden, qkK)
	}
	if want := int64(vocab) * int64(hidden/qkK) * q6kBlockBytes; int64(len(raw)) != want {
		return fmt.Errorf("model: tied Q6_K embedding payload %d bytes, want %d", len(raw), want)
	}
	if m.Q2KEmbedding != nil {
		return fmt.Errorf("model: Q2KEmbedding already set")
	}
	if m.kqw[tiedEmbeddingName] != nil || m.kqw["lm_head.weight"] != nil {
		return fmt.Errorf("model: tied Q6_K embedding conflicts with an existing resident head")
	}
	qt := quantizeKQuantFromRaw(raw, vocab, hidden, kindQ6K)
	if m.kqw == nil {
		m.kqw = map[string]*kQuantTensor{}
	}
	m.kqw[tiedEmbeddingName] = qt
	m.Q2KEmbedding = &Q2KEmbedding{raw: qt.raw, vocab: vocab, hidden: hidden, format: packedEmbeddingQ6K}
	return nil
}

// tiedQ6KHead returns the kqw tensor that backs BOTH the packed token table and the tied
// head, or nil when the model does not share one Q6_K table between them.
func (m *Model) tiedQ6KHead() *kQuantTensor {
	if m == nil || m.kqw == nil || m.Q2KEmbedding == nil || m.Q2KEmbedding.format != packedEmbeddingQ6K {
		return nil
	}
	qt := m.kqw[tiedEmbeddingName]
	if qt == nil || qt.kind != kindQ6K || len(qt.raw) == 0 || len(m.Q2KEmbedding.raw) == 0 || &qt.raw[0] != &m.Q2KEmbedding.raw[0] {
		return nil
	}
	return qt
}

// TiedQ6KEmbeddingShared reports whether the token table and LM head are one shared resident
// Q6_K copy (no f32 gather table and no duplicate Q8 head). Load witnesses assert it.
func (m *Model) TiedQ6KEmbeddingShared() bool { return m.tiedQ6KHead() != nil }

// LMHeadRoute names the resident store and engine the LM head projection uses. Metal routes
// are reported only when the Q6_K band is actually device-resident (metalQ6KHeadResident);
// otherwise the same store reports its CPU kernel. The order mirrors Session.headResident.
//
//	metal-q6k  resident Q6_K head (tied or untied) served by the Metal Q6_K GEMV
//	cpu-q6k    resident Q6_K head on the CPU Q6_K int8 GEMV
//	cpu-kquant other resident k-quant head on the CPU
//	cpu-q4k / cpu-q2 / cpu-q4 / cpu-q8 / cpu-gptq / cpu-f32
func (m *Model) LMHeadRoute() string {
	if m == nil {
		return ""
	}
	if m.q4khead != nil || m.q4kw[m.q4kHeadName()] != nil {
		return "cpu-q4k"
	}
	if name := m.kqHeadName(); name != "" {
		if m.kqw[name].kind != kindQ6K {
			return "cpu-kquant"
		}
		if m.metalQ6KHeadResident(name) {
			return "metal-q6k"
		}
		return "cpu-q6k"
	}
	// Health probes call this concurrently with decode, which may lazily fill q8w/q8head; read
	// the Q8 store under q8Mu and never the lazily pinned q8head field. residentHeadName and
	// headName both probe q8w, and cannot lock themselves (Quantize calls them under q8Mu.Lock).
	q8Mu.RLock()
	residentHead := m.residentHeadName()
	_, hasQ8 := m.q8w[m.headName()]
	q8Mu.RUnlock()
	if m.q2w[residentHead] != nil {
		return "cpu-q2"
	}
	if m.q4head != nil {
		return "cpu-q4"
	}
	if hasQ8 {
		return "cpu-q8"
	}
	if m.hasGPTQ("lm_head.weight") {
		return "cpu-gptq"
	}
	return "cpu-f32"
}
