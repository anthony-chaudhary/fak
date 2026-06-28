package main

import (
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelladder"
)

// arms.go runs the two serving strategies LIVE on a real in-kernel model over a
// Workload, mirroring the proven methodology of cmd/demorace / cmd/sessionbench. The
// only delta from demorace is that the tool result ingested after each turn is the
// workload's per-(agent,turn) size, not a single constant R — so the context changes
// heterogeneously, agent by agent, turn by turn.

type event map[string]any
type emitter func(event)

func ms(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e6 }

// lcgIDs makes a deterministic pseudo-token stream (same generator demorace uses) so
// both arms process byte-identical token work — the delta is purely how much the
// system makes the model re-read, never the tokens themselves.
func lcgIDs(n, vocab int, seed uint64) []int {
	if n <= 0 {
		return nil
	}
	ids := make([]int, n)
	state := 2463534242 + seed
	for i := 0; i < n; i++ {
		state = (state*1103515245 + 12345) & 0x7fffffff
		ids[i] = int(state % uint64(vocab))
	}
	return ids
}

// armResult carries the timing split — total wall-clock plus the decode/prefill
// breakdown and the prefill tokens this arm actually ingested.
type armResult struct {
	totalMS   float64
	decodeMS  float64
	prefillMS float64 // totalMS - decodeMS (wall-clock spent ingesting prefill tokens)
	prefillTk int     // prefill tokens actually ingested by this arm
}

// liveArmFak runs the fak fused arm live: the shared prefix is prefilled ONCE, cloned
// into C agents, decoded in one batched stream, and after each turn every agent's
// (distinct-sized) tool result is ingested incrementally. Emits one "turn" event per
// turn (a turn advances all C agents, so requests jump by C).
func liveArmFak(l *modelladder.Loaded, w Workload, emit emitter) armResult {
	m, vocab := l.M, l.Vocab
	P, C, T, D := w.Scn.Prefix, w.Scn.Agents, w.Scn.Turns, w.Scn.Decode
	prefix := lcgIDs(P, vocab, 1)
	ids0 := lcgIDs(C, vocab, 991)

	start := time.Now()
	base := m.NewSession()
	base.Quant = true
	tp := time.Now()
	base.Prefill(prefix)
	prefillMS := ms(time.Since(tp))

	bs := m.NewBatchFromPrefixReserve(base.Cache, C, w.maxAgentTail()+8)
	bs.SetQuant(true)

	ids := append([]int(nil), ids0...)
	var decodeMS float64
	prefilledTk := P
	for t := 0; t < T; t++ {
		td := time.Now()
		for d := 0; d < D; d++ {
			bs.StepBatch(ids)
			for j := range ids {
				ids[j] = (ids[j]*48271 + 1) % vocab
			}
		}
		decodeMS += ms(time.Since(td))
		if t < len(w.Results[0]) {
			prompts := make([][]int, C)
			tp := time.Now()
			for c := range prompts {
				prompts[c] = lcgIDs(w.Results[c][t], vocab, uint64(50000+t*1000+c*97))
				prefilledTk += w.Results[c][t]
			}
			bs.PrefillEach(prompts)
			prefillMS += ms(time.Since(tp))
		}
		emit(event{
			"type": "turn", "arm": "fak", "turn": t,
			"requests_done": (t + 1) * C, "total_requests": C * T,
			"tokens_prefilled": prefilledTk, "tokens_decoded": C * (t + 1) * D,
			"elapsed_ms": ms(time.Since(start)),
		})
	}
	base.Close()
	return armResult{
		totalMS:   ms(time.Since(start)),
		decodeMS:  decodeMS,
		prefillMS: prefillMS,
		prefillTk: prefilledTk,
	}
}

// liveArmTuned runs the tuned warm-cache arm live — the SOTA serving baseline and the
// HEADLINE comparison. Each agent keeps a PERSISTENT per-agent KV cache: the shared prefix
// is prefilled ONCE per agent (so C times across the fleet — no cross-agent sharing, no
// batching), then each turn decodes serially and ingests ONLY that turn's (distinct-sized)
// tool result incrementally. This is the real baseline a tuned single-tenant stack gives
// you — vLLM / SGLang prefix caching, provider prompt-caching, a persistent KV per session.
// fak's win over THIS arm (cross-agent prefix reuse + batched decode on top of a warm
// cache) is the honest number. No quadratic re-prefill, so it always runs live.
func liveArmTuned(l *modelladder.Loaded, w Workload, emit emitter) armResult {
	m, vocab := l.M, l.Vocab
	P, C, T, D := w.Scn.Prefix, w.Scn.Agents, w.Scn.Turns, w.Scn.Decode
	prefix := lcgIDs(P, vocab, 1)
	ids0 := lcgIDs(C, vocab, 991)

	start := time.Now()
	var decodeMS, prefillMS float64
	prefilledTk, done := 0, 0
	for c := 0; c < C; c++ {
		s := m.NewSession()
		s.Quant = true
		tp := time.Now()
		s.Prefill(prefix) // prefix prefilled ONCE per agent (warm KV — never re-prefill the growing context)
		prefillMS += ms(time.Since(tp))
		prefilledTk += P
		tok := ids0[c]
		for t := 0; t < T; t++ {
			td := time.Now()
			for d := 0; d < D; d++ {
				s.Step(tok)
				tok = (tok*48271 + 1) % vocab
			}
			decodeMS += ms(time.Since(td))
			if t < len(w.Results[c]) {
				tp := time.Now()
				s.Prefill(lcgIDs(w.Results[c][t], vocab, uint64(50000+t*1000+c*97))) // ingest ONLY this turn's result
				prefillMS += ms(time.Since(tp))
				prefilledTk += w.Results[c][t]
			}
			done++
			emit(event{
				"type": "turn", "arm": "tuned", "turn": t, "agent": c,
				"requests_done": done, "total_requests": C * T,
				"tokens_prefilled": prefilledTk, "tokens_decoded": done * D,
				"elapsed_ms": ms(time.Since(start)),
			})
		}
		s.Close()
	}
	return armResult{
		totalMS:   ms(time.Since(start)),
		decodeMS:  decodeMS,
		prefillMS: prefillMS,
		prefillTk: prefilledTk,
	}
}

// warm runs a tiny prefill+step so the first measured token doesn't eat lazy init.
func warm(l *modelladder.Loaded) {
	ws := l.M.NewSession()
	ws.Quant = true
	ws.Prefill(lcgIDs(8, l.Vocab, 77))
	ws.Step(1)
	ws.Close()
}
