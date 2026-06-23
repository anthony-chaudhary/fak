// longcontext.go — first-class PROOF for the ULTRA-LONG-CONTEXT regime (per-agent
// context > 100k tokens), the regime the session value-stack exists for but no
// committed artifact has ever reached (sessionbench tops out near ~7k tokens because
// the naive arm's O(T^2) re-prefill is intractable to run live at 100k+).
//
// THE HONEST TRICK (the same one sessionbench uses for its arm A, taken to its
// conclusion): you do NOT need to run a 100k-token session to know how much work the
// fused kernel ELIMINATES. The work each arm performs is a CLOSED-FORM function of the
// session structure (prefix P, turns T, agents C, decode D, result R) and the model
// SHAPE (layers, hidden, heads, head-dim, ff) — pure arithmetic, no weights, no decode,
// no box. So the work-elimination ratios A/C (vs naive), B/C (vs tuned) and A/B (the
// turn-tax) at the >100k regime are EXACT arithmetic facts, immune to contention, that a
// go test re-derives byte-identically. This file is that floor.
//
// TWO FLOORS, both exact and contention-free:
//
//   - TOKEN FLOOR — the exact prefill-token count each arm processes. Identical to
//     cmd/sessionbench's prefillTokens (A = C·Σ(P+t(D+R)); B = C·(P+(T-1)R); C = P+C(T-1)R),
//     so this floor cross-validates against the live bench's own contention-free floor.
//     It is LINEAR in tokens, so it is the CONSERVATIVE lower bound: it ignores that
//     re-prefilling a 100k context costs O(L^2) in attention, not O(L).
//   - FLOP FLOOR — the O(L^2)-aware work floor. Prefill attention over an L-token context
//     is Σ_{i<L}(i+1) ≈ L^2/2 query-key pairs; the naive arm pays that quadratic EVERY
//     turn over the whole growing context, while the fused kernel pays the prefix once and
//     only incrementally ingests new spans. This floor is where the ultra-long-context win
//     actually lives. It is exact arithmetic from the model shape; the decode-batching
//     BANDWIDTH win (one weight stream serving C lanes) is deliberately NOT in it — decode
//     FLOPs are identical across all three arms (batching is a bandwidth lever, proven in
//     MODEL-BATCHING-RESULTS, not a FLOP lever), so this floor isolates the REREAD-elimination
//     win alone (the scaling-laws "reread rate" term) and never double-counts the bandwidth win.
//
// THE BASELINE-SCOPING DISCIPLINE (non-negotiable, BENCHMARK-AUTHORITY law). A/C is vs the
// NAIVE re-prefill pattern (a worst-case REFERENCE, NOT a serving baseline anyone ships);
// B/C is vs a WARM per-agent KV cache (the honest serving baseline). They are reported side
// by side and never conflated. The standing repo bound — "B/C is ~2–4× vs a tuned cache" —
// is REGIME-SPECIFIC to a SMALL prefix (P≈2k, where the prefix is a tiny fraction of total
// work); this file shows WHY, exactly: B/C = [C·prefixWork + sharedWork] / [prefixWork +
// sharedWork] rises monotonically from 1 (P→0, no prefix to share) toward C (P→∞, prefix
// dominates). At an ultra-long SHARED prefix the cross-agent win approaches the agent count;
// at a tiny prefix it is ~1. That is not a contradiction of the 2–4× bound — it is the same
// formula evaluated at a different prefix fraction, and the floor makes the boundary explicit.
//
// WHAT THIS IS NOT. It is a WORK floor, not a wall-clock. It makes NO model-quality or
// resolve-rate claim (the floor is independent of what the model decodes — it depends only
// on the session structure). The live wall-clock validation of these ratios at >100k is a
// separate, bench-node-gated measurement (it needs a model resident; see the live-anchor
// issue). Here the claim is exactly: "this is the work the fused kernel eliminates, computed
// exactly, reproducible by go test."
package turnbench

import (
	"encoding/json"
	"fmt"
	"runtime"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/appversion"
)

// UltraLongThreshold is the per-agent context length (tokens) at or above which a session
// is "ultra-long-context" — the regime this proof targets. 100k is the round figure the
// goal names; it is a label on the cell, never a gate on the arithmetic.
const UltraLongThreshold = 100_000

// LongContextCostModelVersion versions the long-context floor artifact so a regenerated
// report is comparable across runs (mirrors CostModelVersion / FanoutCostModelVersion).
const LongContextCostModelVersion = "fak.longcontext-floor.v1"

// ModelShape is the minimal transformer geometry the work floor needs — a dependency-light
// local mirror of the fields cmd/sessionbench's syntheticShape carries (no internal/model
// import, so turnbench stays light). HiddenSize is d; the attention dim is NumHeads·HeadDim
// (== d for the usual MHA layout); kv-dim is NumKVHeads·HeadDim (GQA narrows K/V).
type ModelShape struct {
	Name             string `json:"name"`
	HiddenSize       int    `json:"hidden_size"`
	NumLayers        int    `json:"num_layers"`
	NumHeads         int    `json:"num_heads"`
	NumKVHeads       int    `json:"num_kv_heads"`
	HeadDim          int    `json:"head_dim"`
	IntermediateSize int    `json:"intermediate_size"`
	VocabSize        int    `json:"vocab_size"`
}

// NamedShape returns a known model geometry by name. The shapes mirror cmd/sessionbench's
// syntheticShape EXACTLY (same checkpoints), so a floor projection lines up with a live
// sessionbench run on the same model. ok=false for an unknown name.
func NamedShape(name string) (ModelShape, bool) {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "smollm2-135m", "135m", "smollm2":
		return ModelShape{
			Name: "smollm2-135m", HiddenSize: 576, NumLayers: 30, NumHeads: 9, NumKVHeads: 3,
			HeadDim: 64, IntermediateSize: 1536, VocabSize: 49152,
		}, true
	case "qwen25-1.5b", "1.5b", "qwen2.5-1.5b":
		return ModelShape{
			Name: "qwen25-1.5b", HiddenSize: 1536, NumLayers: 28, NumHeads: 12, NumKVHeads: 2,
			HeadDim: 128, IntermediateSize: 8960, VocabSize: 151936,
		}, true
	case "qwen25-7b", "7b", "qwen2.5-7b":
		return ModelShape{
			Name: "qwen25-7b", HiddenSize: 3584, NumLayers: 28, NumHeads: 28, NumKVHeads: 4,
			HeadDim: 128, IntermediateSize: 18944, VocabSize: 152064,
		}, true
	}
	return ModelShape{}, false
}

// attnDim is the attention projection width (NumHeads·HeadDim) — the coefficient on the
// O(L^2) attention term (one MAC per query-key pair per head-dim, across all heads).
func (s ModelShape) attnDim() float64 { return float64(s.NumHeads * s.HeadDim) }

// kvDim is the K/V projection width (NumKVHeads·HeadDim) — narrower than attnDim under GQA.
func (s ModelShape) kvDim() float64 { return float64(s.NumKVHeads * s.HeadDim) }

// linPerTokPerLayer is the LINEAR (per-token, per-layer) projection work: QKV + attention
// output + the SwiGLU MLP. All three are O(1) per token (independent of context length), so
// they scale linearly with the number of tokens prefilled/decoded.
//
//	QKV:        d·attnDim (Q) + 2·d·kvDim (K,V)
//	attn-out:   attnDim·d
//	MLP SwiGLU: d·ff (gate) + d·ff (up) + ff·d (down) = 3·d·ff
func (s ModelShape) linPerTokPerLayer() float64 {
	d := float64(s.HiddenSize)
	return d*s.attnDim() + 2*d*s.kvDim() + s.attnDim()*d + 3*d*float64(s.IntermediateSize)
}

// AppendWork returns the exact forward-pass work (in MAC units) to APPEND n tokens to a
// context that already holds `prior` tokens — the universal primitive: a from-scratch
// prefill is AppendWork(0,L); incremental result ingestion is AppendWork(prior,R); decoding
// D tokens is AppendWork(prior,D) (decode does the SAME work as a teacher-forced append, it
// is only memory-bound rather than compute-bound — the FLOPs are identical).
//
// The work splits into the LINEAR projection term (linPerTok·n) and the ATTENTION term: the
// n new tokens form (n·prior + n^2/2) query-key pairs (token j at position prior+j attends to
// prior+j+1 keys; Σ over the n tokens), each pair costing one MAC per head-dim across all
// heads — and counted twice (scores, then the softmax-weighted V sum). That n·prior cross
// term is why even INCREMENTAL ingestion gets dearer as the context grows, and the n^2 term
// is the prefill quadratic. Per layer, ×NumLayers.
func (s ModelShape) AppendWork(prior, n int) float64 {
	if n <= 0 {
		return 0
	}
	pairs := float64(n)*float64(prior) + float64(n)*float64(n)/2.0
	attn := 2.0 * pairs * s.attnDim()
	lin := s.linPerTokPerLayer() * float64(n)
	return float64(s.NumLayers) * (lin + attn)
}

// PrefillWork is the from-scratch prefill cost of an L-token context: AppendWork(0,L). Its
// attention term is L^2·attnDim·NumLayers — the quadratic the naive arm pays every turn.
func (s ModelShape) PrefillWork(L int) float64 { return s.AppendWork(0, L) }

// SessionShape is the workload: C concurrent agents share a P-token prefix; each runs T
// turns; a turn decodes D assistant tokens then (between turns) ingests R result tokens. So
// a single agent's context grows P → P + (T-1)·(D+R) + D. (Mirrors cmd/sessionbench.)
type SessionShape struct {
	Prefix int `json:"prefix"` // P — shared system prompt + tool schemas
	Turns  int `json:"turns"`  // T — model/tool cycles per agent
	Agents int `json:"agents"` // C — concurrent agents
	Decode int `json:"decode"` // D — assistant tokens decoded per turn
	Result int `json:"result"` // R — tool-result tokens ingested per turn

	// IdleFraction is the average fraction of the C lanes that are IDLE per decode step in a
	// heterogeneous fleet (#520): agents blocked on a tool call, finished (EOS), or simply not
	// producing this turn. A rectangular (static) batch decodes those idle lanes anyway — wasted
	// weight work — while a ragged batch (model.StepBatchActive) skips them, so the decode work
	// scales by the ACTIVE fraction (1−IdleFraction). This floor applies that scaling to the
	// decode FLOPs of every arm equally: idle-skip is a win every serving system gets (a lane
	// not generating does no useful decode, regardless of serving strategy), so it is NOT a
	// fak-exclusive win and never inflates a cross-arm ratio on its own. What it DOES do is make
	// the floor honest for idle-heavy fleets (the all-active DecodeFLOPs overstate the work) and,
	// because decode shrinks while the arms' prefill strategies do not, raise the prefill-driven
	// ratios (A/C, B/C) — reflecting that in an idle-heavy fleet the fused kernel's
	// prefill-elimination advantage is a LARGER share of the (smaller) total work. Valid range
	// [0,1); zero (the default) is the all-active worst case the rest of this file assumes.
	IdleFraction float64 `json:"idle_fraction,omitempty"`
}

// MaxContextTokens is the peak per-agent context reached in the session: prefix + every
// turn's decode + every between-turn result (T-1 of them) + the final turn's decode.
func (sh SessionShape) MaxContextTokens() int {
	if sh.Turns <= 0 {
		return sh.Prefix
	}
	return sh.Prefix + (sh.Turns-1)*(sh.Decode+sh.Result) + sh.Decode
}

// WorkFloor is one arm's exact, contention-free work over the session — both floors.
type WorkFloor struct {
	// PrefillTokens is the exact count of prefill (reread/ingest) tokens this arm processes
	// — the LINEAR token floor, identical to cmd/sessionbench's prefillTokens. The
	// conservative lower bound (ignores the O(L^2) attention penalty).
	PrefillTokens int64 `json:"prefill_tokens"`
	// PrefillFLOPs is the exact O(L^2)-aware prefill/ingest work (MAC units): the from-scratch
	// prefix prefill plus, per arm, either the naive whole-context re-prefill (A) or the
	// incremental result ingestion (B,C).
	PrefillFLOPs float64 `json:"prefill_flops"`
	// DecodeFLOPs is the exact decode work (MAC units). IDENTICAL across all three arms — the
	// fused kernel's decode-batching win is a BANDWIDTH lever (one weight stream for C lanes),
	// not a FLOP lever, so it is excluded here and never inflates the floor.
	DecodeFLOPs float64 `json:"decode_flops"`
	// TotalFLOPs = PrefillFLOPs + DecodeFLOPs — the realistic work floor (decode included, so
	// the ratios on it are CONSERVATIVE relative to prefill-only).
	TotalFLOPs float64 `json:"total_flops"`
}

// LongContextCell is the exact work floor for one (model, session-shape) point.
type LongContextCell struct {
	Shape            SessionShape `json:"shape"`
	MaxContextTokens int          `json:"max_context_tokens"`
	UltraLong        bool         `json:"ultra_long"` // MaxContextTokens >= UltraLongThreshold

	A WorkFloor `json:"arm_A_naive_stateless"` // re-prefill the whole context every turn
	B WorkFloor `json:"arm_B_per_agent_kv"`    // warm per-agent KV: prefix ×C + incremental
	C WorkFloor `json:"arm_C_fak_fused"`       // prefix ONCE + cloned + incremental

	// Token-floor ratios (prefill tokens; LINEAR, conservative, cross-validates sessionbench).
	TokenAOverC float64 `json:"token_a_over_c"` // vs naive (worst-case REFERENCE)
	TokenBOverC float64 `json:"token_b_over_c"` // vs tuned warm KV (serving baseline)

	// FLOP-floor ratios on TOTAL work (prefill + decode; O(L^2)-aware, realistic, conservative).
	FlopAOverC float64 `json:"flop_a_over_c"` // vs naive (worst-case REFERENCE)
	FlopBOverC float64 `json:"flop_b_over_c"` // vs tuned warm KV (serving baseline)
	FlopAOverB float64 `json:"flop_a_over_b"` // the turn-tax (KV persistence vs re-prefill)

	// FLOP-floor ratios on PREFILL work only (the reread-elimination win in isolation — the
	// number that grows fastest with context length, reported apart so decode dilution is
	// transparent, never to overstate).
	PrefillFlopAOverC float64 `json:"prefill_flop_a_over_c"`
	PrefillFlopBOverC float64 `json:"prefill_flop_b_over_c"`

	// Exact is always true: every number above is closed-form arithmetic for the stated shape
	// and model geometry — no model run, no box, no contention. Note records the scope.
	Exact bool   `json:"exact"`
	Note  string `json:"note,omitempty"`
}

// ProjectLongContext computes the exact work floor for one model shape and session shape.
// Pure arithmetic — no model, no decode, no wall-clock. Deterministic and contention-free.
func ProjectLongContext(s ModelShape, sh SessionShape) LongContextCell {
	P, T, C, D, R := sh.Prefix, sh.Turns, sh.Agents, sh.Decode, sh.Result

	// --- token floor (prefill tokens only) — identical to cmd/sessionbench prefillTokens ---
	var aTok int64
	for t := 0; t < T; t++ {
		aTok += int64(P + t*(D+R)) // naive: re-prefill the whole context at the start of turn t
	}
	aTok *= int64(C)
	var bTok, cTok int64
	if T >= 1 {
		bTok = int64(C) * int64(P+(T-1)*R) // per-agent: prefix ×C + incremental result ingest
		cTok = int64(P) + int64(C)*int64((T-1)*R)
	} else {
		bTok, cTok = int64(C)*int64(P), int64(P)
	}

	// --- FLOP floor (O(L^2)-aware) ---
	// Decode work is identical across arms; the shared per-agent incremental ingest is shared
	// by B and C; only the PREFILL strategy differs (A re-prefills the whole context, B pays
	// the prefix ×C, C pays it once).
	var aPrefill, decodePerAgent, ingestPerAgent float64
	for t := 0; t < T; t++ {
		Lt := P + t*(D+R)                     // context length at the START of turn t (before this turn's decode)
		aPrefill += s.PrefillWork(Lt)         // naive: re-prefill the whole growing context
		decodePerAgent += s.AppendWork(Lt, D) // decode D tokens appended at Lt
		if t < T-1 {
			ingestPerAgent += s.AppendWork(Lt+D, R) // ingest R result tokens after the decode
		}
	}
	prefixWork := s.PrefillWork(P)
	Cf := float64(C)

	// The idle-lane-skip lever (#520): in a heterogeneous fleet only the ACTIVE fraction
	// (1−IdleFraction) of the C lanes decode each step, so the decode work scales by it. Applied
	// to every arm (idle-skip is a win any serving system gets, never a fak-exclusive ratio win);
	// see SessionShape.IdleFraction. Prefill/ingest are turn-structure costs and are NOT scaled
	// (an idle agent still has its results ingested and its next turn prefilled). Clamped to [0,1]
	// so a stray negative or >=1 stays a well-defined floor rather than a sign-flipped one.
	decodeActive := 1.0 - sh.IdleFraction
	if decodeActive < 0 {
		decodeActive = 0
	} else if decodeActive > 1 {
		decodeActive = 1
	}
	decodeFLOPs := Cf * decodePerAgent * decodeActive

	cell := LongContextCell{
		Shape:            sh,
		MaxContextTokens: sh.MaxContextTokens(),
		UltraLong:        sh.MaxContextTokens() >= UltraLongThreshold,
		A: WorkFloor{
			PrefillTokens: aTok,
			PrefillFLOPs:  Cf * aPrefill, // A folds result ingest into the next re-prefill — no separate ingest
			DecodeFLOPs:   decodeFLOPs,
		},
		B: WorkFloor{
			PrefillTokens: bTok,
			PrefillFLOPs:  Cf * (prefixWork + ingestPerAgent), // prefix per agent + incremental ingest
			DecodeFLOPs:   decodeFLOPs,
		},
		C: WorkFloor{
			PrefillTokens: cTok,
			PrefillFLOPs:  prefixWork + Cf*ingestPerAgent, // prefix ONCE total + per-agent incremental ingest
			DecodeFLOPs:   decodeFLOPs,
		},
		Exact: true,
		Note: "exact contention-free work floor for the stated (model, session) shape; a real " +
			"session with variable per-turn decode/result sizes is bounded between the floors for " +
			"its min/max turn sizes. WORK floor only — no wall-clock, no resolve-rate, no quality claim.",
	}
	cell.A.TotalFLOPs = cell.A.PrefillFLOPs + cell.A.DecodeFLOPs
	cell.B.TotalFLOPs = cell.B.PrefillFLOPs + cell.B.DecodeFLOPs
	cell.C.TotalFLOPs = cell.C.PrefillFLOPs + cell.C.DecodeFLOPs

	if cTok > 0 {
		cell.TokenAOverC = float64(aTok) / float64(cTok)
		cell.TokenBOverC = float64(bTok) / float64(cTok)
	}
	if cell.C.TotalFLOPs > 0 {
		cell.FlopAOverC = cell.A.TotalFLOPs / cell.C.TotalFLOPs
		cell.FlopBOverC = cell.B.TotalFLOPs / cell.C.TotalFLOPs
	}
	if cell.B.TotalFLOPs > 0 {
		cell.FlopAOverB = cell.A.TotalFLOPs / cell.B.TotalFLOPs
	}
	if cell.C.PrefillFLOPs > 0 {
		cell.PrefillFlopAOverC = cell.A.PrefillFLOPs / cell.C.PrefillFLOPs
		cell.PrefillFlopBOverC = cell.B.PrefillFLOPs / cell.C.PrefillFLOPs
	}
	return cell
}

// LongContextReport is the regenerable artifact: the exact work floor across a ladder of
// session shapes for one model — at $0 model spend (no decode runs). Deterministic.
type LongContextReport struct {
	Provenance Provenance `json:"provenance"`
	Cost       CostModel  `json:"cost_model"`

	Model     ModelShape        `json:"model_shape"`
	Threshold int               `json:"ultra_long_threshold"`
	Cells     []LongContextCell `json:"cells"`

	// Headline picks (the two regime points the goal names) — pointers into Cells by index,
	// -1 if the ladder contained no such cell.
	SingleUltraLongIdx int `json:"single_ultra_long_idx"` // first C==1 cell with MaxContext >= threshold
	MultiUltraLongIdx  int `json:"multi_ultra_long_idx"`  // first C>1  cell with MaxContext >= threshold

	GeneratedBy string `json:"generated_by"`
}

// JSON renders the report (stable indentation, trailing newline) for an artifact file.
func (r *LongContextReport) JSON() []byte {
	b, _ := json.MarshalIndent(r, "", "  ")
	return append(b, '\n')
}

// RunLongContextLadder computes the exact work floor for every shape in the ladder against
// one model, picks the two ultra-long headline cells, and stamps provenance. No model call.
func RunLongContextLadder(s ModelShape, shapes []SessionShape, cm CostModel) *LongContextReport {
	cm = withCostModelVersion(cm)
	rep := &LongContextReport{
		Provenance: Provenance{
			AppVersion:  appversion.Current(),
			Command:     "turnbench.RunLongContextLadder",
			GoVersion:   runtime.Version(),
			OS:          runtime.GOOS,
			GeneratedBy: "fak/internal/turnbench (ultra-long-context work floor)",
		},
		Cost:               cm,
		Model:              s,
		Threshold:          UltraLongThreshold,
		SingleUltraLongIdx: -1,
		MultiUltraLongIdx:  -1,
		GeneratedBy:        "fak/internal/turnbench (ultra-long-context work floor)",
	}
	for i, sh := range shapes {
		cell := ProjectLongContext(s, sh)
		rep.Cells = append(rep.Cells, cell)
		if cell.UltraLong {
			if sh.Agents <= 1 && rep.SingleUltraLongIdx < 0 {
				rep.SingleUltraLongIdx = i
			}
			if sh.Agents > 1 && rep.MultiUltraLongIdx < 0 {
				rep.MultiUltraLongIdx = i
			}
		}
	}
	return rep
}

// CanonicalLadder is the regime ladder the proof walks — from the existing measured shape
// (a baseline anchor that is NOT ultra-long) up through the two regimes the goal names and
// on into the agent-city frontier. It is the regenerable basis for the SCALING-LAWS table at
// the >100k regime. The shapes are deliberately the long-document / deep-session shapes where
// each agent's context crosses 100k (a large shared prefix + a handful of long turns), not a
// tiny-prefix shape (where the cross-agent win is correctly ~1×).
func CanonicalLadder() []SessionShape {
	return []SessionShape{
		// anchor: the existing committed regime (P=2048, T=50, C=5) — NOT ultra-long (~7k ctx).
		{Prefix: 2048, Turns: 50, Agents: 5, Decode: 32, Result: 64},
		// single session, ultra-long: a 100k-token working set (large doc/RAG context), 10 turns.
		{Prefix: 100_000, Turns: 10, Agents: 1, Decode: 200, Result: 500},
		// multi-agent, each ultra-long: 5 agents sharing the 100k working set, 10 turns each.
		{Prefix: 100_000, Turns: 10, Agents: 5, Decode: 200, Result: 500},
		// deeper multi-agent: 5 agents, 50 turns over the 100k working set.
		{Prefix: 100_000, Turns: 50, Agents: 5, Decode: 200, Result: 500},
		// agent-city: 40 agents sharing the 100k working set (the regime where B/C approaches C).
		{Prefix: 100_000, Turns: 10, Agents: 40, Decode: 200, Result: 500},
		// idle-heavy agent-city (#520): the SAME 40-agent city, but half the lanes idle each
		// decode step (agents blocked on tool calls). The ragged batch (model.StepBatchActive)
		// skips those lanes, so the decode floor drops to the active half — and because decode
		// shrinks while the prefix-sharing prefill does not, the cross-agent win (B/C) rises,
		// reflecting that in an idle-heavy fleet the fused kernel's prefill advantage is a larger
		// share of the (smaller) total work. The all-active cell above is the conservative worst
		// case; this cell is the same fleet under the idle-skip lever.
		{Prefix: 100_000, Turns: 10, Agents: 40, Decode: 200, Result: 500, IdleFraction: 0.5},
	}
}

// FmtRatio renders a ratio compactly for the human summary (e.g. "42.1×").
func FmtRatio(x float64) string { return fmt.Sprintf("%.1f×", x) }
