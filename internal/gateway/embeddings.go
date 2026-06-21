package gateway

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math"
	"net/http"
	"strconv"
	"strings"
)

// ---------------------------------------------------------------------------
// OpenAI-compatible /v1/embeddings.
//
// The gateway's lane is the WIRE, not the model: it implements the
// OpenAI-compatible embeddings SURFACE (request/response shape, batch input,
// float|base64 encoding, token accounting) with a deterministic, self-contained
// backend so an OpenAI client round-trips on-host with no GPU, no weights, and no
// network. The backend is an honest feature-hashing embedding (the "hashing
// trick"): a fixed-dimension, L2-normalized projection of the input tokens. It is
// NOT a learned model — identical text yields an identical vector and texts that
// share tokens score higher cosine similarity, which is exactly what a drop-in
// embeddings endpoint needs for deterministic tests, semantic-cache keys, and
// nearest-neighbour smoke checks. Serving a LEARNED embedding model is an
// engine/model-lane concern; swapping this backend for one is a later seam.
// ---------------------------------------------------------------------------

const (
	// embedDefaultDim is the served vector width when the client does not ask for a
	// specific `dimensions`. 256 is wide enough for stable cosine separation on
	// short inputs while staying cheap to compute and serialize.
	embedDefaultDim = 256
	// embedMaxDim bounds a client-requested `dimensions` so a single request cannot
	// force an arbitrarily large allocation (a wire client is untrusted input).
	embedMaxDim = 3072
	// embedMaxItems bounds the number of inputs in one batched request, mirroring the
	// real OpenAI per-request input cap so a single call cannot fan out unboundedly.
	embedMaxItems = 2048
)

// EmbeddingsRequest is the inbound POST /v1/embeddings body. Input is raw because
// the OpenAI wire accepts FOUR shapes — a bare string, an array of strings, an
// array of token ids ([]int, one pre-tokenized input), or an array of token-id
// arrays ([][]int, a pre-tokenized batch) — and decoding straight into any one of
// them would reject the others. normalizeTextInput folds all four to a string
// slice. Unknown OpenAI fields (user, etc.) are accepted and ignored for drop-in
// compatibility; there is, by construction, no Ref field to smuggle.
type EmbeddingsRequest struct {
	Model          string          `json:"model"`
	Input          json.RawMessage `json:"input"`
	EncodingFormat string          `json:"encoding_format,omitempty"`
	Dimensions     int             `json:"dimensions,omitempty"`
	User           string          `json:"user,omitempty"`
}

// EmbeddingsResponse is the OpenAI-compatible list envelope. One EmbeddingData per
// input item, in request order, so a batched call returns per-item results a
// stock OpenAI client iterates by `index`.
type EmbeddingsResponse struct {
	Object string          `json:"object"` // always "list"
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  EmbeddingUsage  `json:"usage"`
}

// EmbeddingData is one input's embedding. Embedding is `any` because the wire
// carries EITHER a JSON array of numbers (encoding_format=float, the default) OR a
// base64 string of little-endian float32s (encoding_format=base64) — the exact
// OpenAI contract. A stock client decodes whichever it asked for.
type EmbeddingData struct {
	Object    string `json:"object"` // always "embedding"
	Index     int    `json:"index"`
	Embedding any    `json:"embedding"`
}

// EmbeddingUsage is the embeddings token accounting (no completion side).
type EmbeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// handleEmbeddings serves POST /v1/embeddings. It is fully self-contained — no
// upstream model round-trip — so it works in CI / drop-in mode identically to a
// live deployment. Batch input (an array) returns one vector per item; the
// vectors are deterministic feature-hash projections (see embedText).
func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	var req EmbeddingsRequest
	if err := decodeJSON(w, r, &req); err != nil {
		writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
		return
	}
	items, ok := normalizeTextInput(req.Input)
	if !ok || len(items) == 0 {
		writeErr(w, http.StatusBadRequest, "embeddings requires a non-empty `input` (string, array of strings, or token-id array)")
		return
	}
	if len(items) > embedMaxItems {
		writeErr(w, http.StatusBadRequest, "embeddings `input` exceeds the per-request batch limit")
		return
	}
	encodeBase64, ok := embeddingEncoding(req.EncodingFormat)
	if !ok {
		writeErr(w, http.StatusBadRequest, "unsupported encoding_format (want float or base64)")
		return
	}
	dim := embedDim(req.Dimensions)

	data := make([]EmbeddingData, len(items))
	totalTokens := 0
	for i, item := range items {
		vec, tokens := embedText(item, dim)
		totalTokens += tokens
		data[i] = EmbeddingData{Object: "embedding", Index: i, Embedding: encodeEmbedding(vec, encodeBase64)}
	}

	writeJSON(w, http.StatusOK, EmbeddingsResponse{
		Object: "list",
		Data:   data,
		Model:  s.model,
		Usage:  EmbeddingUsage{PromptTokens: totalTokens, TotalTokens: totalTokens},
	})
}

// embedDim clamps a client-requested dimensions to [1, embedMaxDim], defaulting to
// embedDefaultDim when unset (<=0). The clamp is what keeps an untrusted wire
// client from forcing an unbounded vector allocation.
func embedDim(requested int) int {
	if requested <= 0 {
		return embedDefaultDim
	}
	if requested > embedMaxDim {
		return embedMaxDim
	}
	return requested
}

// embeddingEncoding validates the OpenAI `encoding_format` field. "" and "float"
// select the float-array form; "base64" selects the packed-float32 form. Anything
// else is rejected (ok=false) rather than silently degraded.
func embeddingEncoding(format string) (base64Encoded bool, ok bool) {
	switch strings.ToLower(strings.TrimSpace(format)) {
	case "", "float":
		return false, true
	case "base64":
		return true, true
	default:
		return false, false
	}
}

// encodeEmbedding renders a vector onto the wire in the requested form: a float
// slice (default) or a base64 string of little-endian float32s (the OpenAI
// base64 contract — float32, not float64, so a stock client decodes 4 bytes per
// component).
func encodeEmbedding(vec []float64, base64Encoded bool) any {
	if !base64Encoded {
		return vec
	}
	buf := make([]byte, 4*len(vec))
	for i, v := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(float32(v)))
	}
	return base64.StdEncoding.EncodeToString(buf)
}

// embedText projects one input string into a deterministic, L2-normalized
// feature-hash vector and returns it with the token count (for usage). The
// "hashing trick": each token is hashed to a bucket and an independent sign, the
// signed unit is accumulated, and the vector is L2-normalized so cosine
// similarity is the dot product. Determinism (same text -> same vector) is what
// makes the endpoint testable and cacheable; token overlap driving similarity is
// what makes it useful as a drop-in.
func embedText(text string, dim int) (vec []float64, tokens int) {
	vec = make([]float64, dim)
	toks := tokenizeForEmbedding(text)
	for _, tok := range toks {
		bucket := int(fnv1a(tok) % uint64(dim))
		// A salted second hash gives an independent sign, decorrelating it from the
		// bucket choice (classic feature hashing uses two independent hashes).
		if fnv1aSalted(tok)&1 == 0 {
			vec[bucket] += 1
		} else {
			vec[bucket] -= 1
		}
	}
	l2Normalize(vec)
	return vec, len(toks)
}

// l2Normalize scales vec to unit length in place. An all-zero vector (empty or
// purely non-alphanumeric input) is left at zero — a defined, finite result rather
// than a divide-by-zero NaN.
func l2Normalize(vec []float64) {
	var norm float64
	for _, v := range vec {
		norm += v * v
	}
	if norm == 0 {
		return
	}
	inv := 1 / math.Sqrt(norm)
	for i := range vec {
		vec[i] *= inv
	}
}

// tokenizeForEmbedding lowercases and splits on runs of non-alphanumeric bytes —
// a deterministic, dependency-free word tokenizer. It is intentionally simpler
// than the model BPE tokenizer (a different lane): the embedding only needs a
// stable token set, not subword fidelity.
func tokenizeForEmbedding(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9')
	})
}

const (
	fnvOffset64 = 14695981039346656037
	fnvPrime64  = 1099511628211
)

// fnv1a is a dependency-free FNV-1a 64-bit hash (mirrors the package's posture of
// avoiding heavyweight deps on small, hot helpers).
func fnv1a(s string) uint64 {
	h := uint64(fnvOffset64)
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= fnvPrime64
	}
	return h
}

// fnv1aSalted hashes s under a fixed salt byte so the sign hash is independent of
// the bucket hash.
func fnv1aSalted(s string) uint64 {
	h := uint64(fnvOffset64)
	h ^= 0x73 // salt
	h *= fnvPrime64
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= fnvPrime64
	}
	return h
}

// normalizeTextInput folds the OpenAI `input` field — a bare string, an array of
// strings, a single token-id array ([]int), or a batch of token-id arrays
// ([][]int) — into a string slice the deterministic backends consume. A token-id
// array is rendered to a stable synthetic string ("tok:<id>" joined) so a
// pre-tokenized client still gets a deterministic, per-item result. Returns
// ok=false for a shape that is none of these (so the handler can 400 rather than
// embed garbage). Shared by /v1/embeddings and /v1/moderations.
func normalizeTextInput(raw json.RawMessage) (items []string, ok bool) {
	b := []byte(raw)
	i := 0
	for i < len(b) && (b[i] == ' ' || b[i] == '\t' || b[i] == '\n' || b[i] == '\r') {
		i++
	}
	if i >= len(b) {
		return nil, false
	}
	switch b[i] {
	case '"':
		var s string
		if err := json.Unmarshal(raw, &s); err == nil {
			return []string{s}, true
		}
	case '[':
		// Try the three array shapes in turn: []string, [][]int, []int.
		var strs []string
		if err := json.Unmarshal(raw, &strs); err == nil {
			return strs, true
		}
		var tokBatch [][]int
		if err := json.Unmarshal(raw, &tokBatch); err == nil {
			out := make([]string, len(tokBatch))
			for j, toks := range tokBatch {
				out[j] = joinTokenIDs(toks)
			}
			return out, true
		}
		var toks []int
		if err := json.Unmarshal(raw, &toks); err == nil {
			return []string{joinTokenIDs(toks)}, true
		}
	}
	return nil, false
}

// joinTokenIDs renders a token-id slice to a stable synthetic string so the
// feature-hash backend produces a deterministic vector for a pre-tokenized input.
func joinTokenIDs(toks []int) string {
	var b strings.Builder
	for i, t := range toks {
		if i > 0 {
			b.WriteByte(' ')
		}
		b.WriteString("tok:")
		b.WriteString(strconv.Itoa(t))
	}
	return b.String()
}
