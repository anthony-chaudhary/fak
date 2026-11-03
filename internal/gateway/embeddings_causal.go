// SPDX-License-Identifier: MIT
// llama.cpp pooled hidden-output contract: cd8427d018c07eefab6c110f7f115c31ec4c6157.
// This adapter selects the bounded FAK CPU implementation, not a remote service.
package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ratelimit"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

type causalEmbeddingProvider struct {
	model                                                  *model.Model
	tokenizer                                              *tokenizer.Tokenizer
	modelName, tokenizerID, pooling, artifactID, precision string
	hidden, vocab                                          int
	bos                                                    *int
	gate                                                   chan struct{}
}

func newCausalEmbeddingProvider(cfg Config) (*causalEmbeddingProvider, error) {
	if cfg.EmbeddingPooling == "" {
		return nil, nil
	}
	if cfg.EmbeddingPooling != "mean" && cfg.EmbeddingPooling != "last" {
		return nil, fmt.Errorf("gateway: embedding pooling must be mean or last")
	}
	if cfg.InKernelModel == nil || cfg.Tokenizer == nil || cfg.InKernelQ4K || cfg.Backend != nil || cfg.Metal || cfg.ExpertParallelRanks > 1 ||
		cfg.InKernelPlanner.RequireDeviceExecution || cfg.BaseURL != "" || len(cfg.ReplicaBaseURLs) != 0 || strings.TrimSpace(cfg.Model) == "" {
		return nil, fmt.Errorf("gateway: learned embeddings require an explicit local dense F32/Q8_0 CPU model and tokenizer, without proxy, device or EP")
	}
	if err := cfg.InKernelModel.ValidateCausalEmbedding(); err != nil {
		return nil, err
	}
	if cfg.Tokenizer.Vocab() != cfg.InKernelModel.Cfg.VocabSize {
		return nil, fmt.Errorf("gateway: embedding model/tokenizer vocabulary mismatch")
	}
	tid := cfg.Tokenizer.Identity()
	if tid == "" {
		return nil, fmt.Errorf("gateway: embedding tokenizer identity unavailable")
	}
	p := &causalEmbeddingProvider{model: cfg.InKernelModel, tokenizer: cfg.Tokenizer, modelName: cfg.Model,
		tokenizerID: tid, pooling: cfg.EmbeddingPooling, hidden: cfg.InKernelModel.Cfg.HiddenSize,
		vocab: cfg.InKernelModel.Cfg.VocabSize, artifactID: "unknown", precision: cfg.InKernelModel.CausalEmbeddingPrecision(), gate: make(chan struct{}, 1)}
	if id, ok := cfg.InKernelModel.AuthenticatedArtifactIdentity(); ok {
		p.artifactID = id.ManifestSHA256 + ":" + id.ShardSetSHA256
	}
	if b := cfg.InKernelModel.Cfg.BOSTokenID; b != nil {
		if *b < 0 || *b >= p.vocab {
			return nil, fmt.Errorf("gateway: embedding BOS outside model vocabulary")
		}
		v := *b
		p.bos = &v
	}
	return p, nil
}

// embeddingSequences distinguishes real IDs from strings. No tok:N text,
// truncation, implicit special parsing, arbitrary dimensions or batch fallback.
func (p *causalEmbeddingProvider) sequences(raw json.RawMessage) ([][]int, error) {
	if len(raw) == 0 || len(raw) > 65536 {
		return nil, fmt.Errorf("embedding input exceeds the 64 KiB admission")
	}
	raw = bytes.TrimSpace(raw)
	var texts []string
	var ids [][]int
	if len(raw) > 0 && raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return nil, err
		}
		texts = []string{s}
	} else {
		var items []json.RawMessage
		if err := json.Unmarshal(raw, &items); err != nil || len(items) == 0 {
			return nil, fmt.Errorf("embedding input must be a nonempty string or homogeneous array")
		}
		first := bytes.TrimSpace(items[0])
		if len(first) == 0 {
			return nil, fmt.Errorf("empty input element")
		}
		switch first[0] {
		case '"':
			if len(items) > 8 {
				return nil, fmt.Errorf("embedding batch exceeds eight inputs")
			}
			texts = make([]string, len(items))
			for i, item := range items {
				if err := json.Unmarshal(item, &texts[i]); err != nil {
					return nil, fmt.Errorf("mixed embedding input types")
				}
			}
		case '[':
			if len(items) > 8 {
				return nil, fmt.Errorf("embedding batch exceeds eight inputs")
			}
			ids = make([][]int, len(items))
			for i, item := range items {
				var err error
				ids[i], err = embeddingIDRow(item)
				if err != nil {
					return nil, err
				}
			}
		default:
			row, err := embeddingIDRow(raw)
			if err != nil {
				return nil, err
			}
			ids = [][]int{row}
		}
	}
	for _, s := range texts {
		if len(s) == 0 || len(s) > 4096 {
			return nil, fmt.Errorf("embedding text must be 1..4096 bytes")
		}
		row, err := p.tokenizer.EncodeSegments([]tokenizer.Segment{{Text: s, Trusted: false}})
		if err != nil {
			return nil, fmt.Errorf("embedding tokenizer: %w", err)
		}
		// Plain causal text uses the model's declared BOS once. Explicit ID inputs
		// are already complete and are never rewritten.
		if p.bos != nil && (len(row) == 0 || row[0] != *p.bos) {
			row = append([]int{*p.bos}, row...)
		}
		ids = append(ids, row)
	}
	total := 0
	for _, row := range ids {
		if len(row) == 0 || len(row) > model.CausalEmbeddingMaxTokens {
			return nil, fmt.Errorf("embedding sequence must contain 1..256 tokens")
		}
		if p.model.Cfg.MaxPositionEmbeddings > 0 && len(row) > p.model.Cfg.MaxPositionEmbeddings {
			return nil, fmt.Errorf("embedding sequence exceeds model context")
		}
		for _, id := range row {
			if id < 0 || id >= p.vocab {
				return nil, fmt.Errorf("embedding token ID outside vocabulary")
			}
		}
		total += len(row)
	}
	if len(ids) == 0 || len(ids) > 8 || total > 1024 {
		return nil, fmt.Errorf("embedding batch exceeds bounded token budget")
	}
	return ids, nil
}

func (s *Server) handleCausalEmbeddings(w http.ResponseWriter, r *http.Request, req EmbeddingsRequest) {
	p := s.causalEmbeddings
	if req.Model != "" && req.Model != p.modelName {
		writeErr(w, http.StatusBadRequest, "embedding model differs from configured model")
		return
	}
	if req.Dimensions != 0 && req.Dimensions != p.hidden {
		writeErr(w, http.StatusBadRequest, "learned embeddings require the exact model hidden dimension")
		return
	}
	b64, ok := embeddingEncoding(req.EncodingFormat)
	if !ok {
		writeErr(w, http.StatusBadRequest, "unsupported encoding_format (want float or base64)")
		return
	}
	if r.Context().Err() != nil {
		writeErr(w, http.StatusRequestTimeout, "embedding request canceled")
		return
	}
	// A busy provider refuses rather than accumulating unbounded model work.
	select {
	case p.gate <- struct{}{}:
		defer func() { <-p.gate }()
	default:
		writeErr(w, http.StatusTooManyRequests, "embedding provider busy")
		return
	}
	if p.tokenizer.Identity() != p.tokenizerID || p.model.Cfg.HiddenSize != p.hidden || p.model.Cfg.VocabSize != p.vocab || p.model.CausalEmbeddingPrecision() != p.precision {
		writeErr(w, http.StatusConflict, "embedding model/tokenizer binding changed")
		return
	}
	rows, err := p.sequences(req.Input)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	// Use the existing session/spend/memory gate without chat-reset/history rewriting:
	// embedding token IDs are the complete immutable input, not a transcript.
	ctx, trace, turn, admitted, canceled := s.beginServedRequest(w, r)
	defer turn.complete()
	if canceled {
		writeErr(w, http.StatusRequestTimeout, "embedding request canceled")
		return
	}
	if !admitted {
		writeSessionRefusal(w, turn.state)
		return
	}
	totalInput := 0
	for _, row := range rows {
		totalInput += len(row)
	}
	lease, err := s.beginCausalEmbeddingAdmission(ctx, turn, totalInput)
	if err != nil {
		s.writeUpstreamErr(w, err)
		return
	}
	defer lease.Release()
	attempted := 0
	began := time.Now()
	var modelErr error
	if c := s.ResearchArmCoordinator(); c != nil {
		arm, err := c.Admit(ctx, r, r.URL.Path, trace)
		if err != nil {
			lease.SettleUsage(agent.Usage{})
			writeErr(w, http.StatusTooManyRequests, err.Error())
			return
		}
		if arm != nil {
			defer func() { arm.Done(attempted, modelErr) }()
		}
	}
	defer func() {
		usage := agent.Usage{PromptTokens: attempted, TotalTokens: attempted}
		lease.SettleUsage(usage)
		if attempted > 0 {
			s.debitServedSessionTurn(ctx, turn, usage, time.Since(began), nil)
		}
	}()
	data := make([]EmbeddingData, len(rows))
	total := 0
	for i, row := range rows {
		if err := ctx.Err(); err != nil {
			modelErr = err
			writeErr(w, http.StatusRequestTimeout, "embedding request canceled")
			return
		}
		// A failed entered row may have consumed compute; conservatively charge its exact input.
		attempted += len(row)
		vec, err := p.model.CausalEmbedding(ctx, row, p.pooling)
		if err != nil {
			modelErr = err
			if err == context.Canceled || err == context.DeadlineExceeded {
				writeErr(w, http.StatusRequestTimeout, "embedding request canceled")
			} else {
				writeErr(w, http.StatusUnprocessableEntity, err.Error())
			}
			return
		}
		f := make([]float64, len(vec))
		for j, v := range vec {
			f[j] = float64(v)
		}
		data[i] = EmbeddingData{Object: "embedding", Index: i, Embedding: encodeEmbedding(f, b64)}
		total += len(row)
	}
	// Unknown artifact membership stays unknown; a configured model alias is not
	// an authenticated checkpoint claim. These bindings accompany every result.
	w.Header().Set("X-Fak-Embedding-Mode", "causal-"+p.pooling+"-l2")
	w.Header().Set("X-Fak-Embedding-Precision", p.precision)
	w.Header().Set("X-Fak-Tokenizer-SHA256", p.tokenizerID)
	w.Header().Set("X-Fak-Artifact-Identity", p.artifactID)
	writeJSON(w, http.StatusOK, EmbeddingsResponse{Object: "list", Data: data, Model: p.modelName, Usage: EmbeddingUsage{PromptTokens: total, TotalTokens: total}})
}

func embeddingIDRow(raw json.RawMessage) ([]int, error) {
	var items []json.RawMessage
	if err := json.Unmarshal(raw, &items); err != nil || len(items) == 0 || len(items) > model.CausalEmbeddingMaxTokens {
		return nil, fmt.Errorf("invalid or oversized token-ID row")
	}
	out := make([]int, len(items))
	for i, item := range items {
		v, err := strconv.Atoi(string(bytes.TrimSpace(item)))
		if err != nil || v < 0 {
			return nil, fmt.Errorf("token IDs must be nonnegative JSON integers")
		}
		out[i] = v
	}
	return out, nil
}

// beginCausalEmbeddingAdmission reuses the existing three-stage reservation
// ownership, but admits exact token IDs and zero generated tokens. It does not
// fabricate a chat transcript to feed a character-count estimate.
func (s *Server) beginCausalEmbeddingAdmission(ctx context.Context, turn servedSessionTurn, tokens int) (*AdmissionLease, error) {
	s.admissionMu.RLock()
	c, g, b := s.admissionCtl, s.tokenRateGate, s.principalTokenRates
	cap, fraction := s.warmupCapacity, s.warmupReserveFraction
	s.admissionMu.RUnlock()
	if err := fixedWarmupCapacityAdmissionError(cap, c, fraction); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	var lease *AdmissionLease
	if c != nil {
		req := SeqRequest{TraceID: turn.traceID, Priority: turn.state.Priority, Tokens: tokens, PromptTokens: tokens}
		if deadline, ok := ctx.Deadline(); ok {
			req.CreatedAt = time.Now()
			req.DecodeTTL = deadline.Sub(req.CreatedAt)
		}
		var err error
		lease, err = c.Acquire(ctx, req)
		if err != nil {
			return nil, err
		}
	}
	estimate := ratelimit.NewTokenUsage(int64(tokens), 0, 0)
	var res *TokenReservation
	if b != nil {
		var err error
		res, err = b.Admit(principalFromContext(ctx), estimate)
		if err != nil {
			lease.Release()
			return nil, err
		}
	}
	if g != nil {
		shared, err := g.Admit(estimate)
		if err != nil {
			res.cancel()
			lease.Release()
			return nil, err
		}
		shared.linked = res
		res = shared
	}
	if res != nil {
		if lease == nil {
			lease = &AdmissionLease{}
		}
		lease.tokenRes = res
	}
	return lease, nil
}
