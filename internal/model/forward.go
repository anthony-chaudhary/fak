package model

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/codegraph"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

// GraphInlineInstruction is one operation in the small callable model graph IR.
// Call names a direct callee; Reference names a function used as data and is not
// rewritten as a call.
type GraphInlineInstruction struct {
	Operation string   `json:"operation"`
	Value     float32  `json:"value,omitempty"`
	Call      string   `json:"call,omitempty"`
	Reference string   `json:"reference,omitempty"`
	Arguments []string `json:"arguments,omitempty"`
	Results   []string `json:"results,omitempty"`
}

// GraphInlineFunction is a callable model-graph function.
type GraphInlineFunction struct {
	Name                string                   `json:"name"`
	Instructions        []GraphInlineInstruction `json:"instructions"`
	Arguments           []string                 `json:"arguments,omitempty"`
	Results             []string                 `json:"results,omitempty"`
	ReturnValues        []string                 `json:"return_values,omitempty"`
	Exported            bool                     `json:"exported,omitempty"`
	External            bool                     `json:"external,omitempty"`
	IndirectCallable    bool                     `json:"indirect_callable,omitempty"`
	CachedResultIndexes []int                    `json:"cached_result_indexes,omitempty"`
	AlwaysInline        bool                     `json:"always_inline,omitempty"`
	NeverInline         bool                     `json:"never_inline,omitempty"`
}

// GraphInlineProgram owns the callable functions rooted at Entry.
type GraphInlineProgram struct {
	Entry     string                `json:"entry"`
	Functions []GraphInlineFunction `json:"functions"`
}

// GraphInlineDecision records why a function was inlined or kept. Retained is
// true when its symbol must survive even after all direct calls are rewritten.
type GraphInlineDecision struct {
	Function string `json:"function"`
	Action   string `json:"action"`
	Reason   string `json:"reason"`
	Retained bool   `json:"retained,omitempty"`
}

// GraphInlineReceipt is the deterministic witness for one inlining pass.
type GraphInlineReceipt struct {
	Decisions []GraphInlineDecision `json:"decisions"`
	Digest    string                `json:"digest"`
}

// GraphDeadArgumentDecision records the ABI positions removed from a function.
// Fence is non-empty when the function signature is preserved as an ABI boundary.
type GraphDeadArgumentDecision struct {
	Function       string `json:"function"`
	RemovedArgs    []int  `json:"removed_args,omitempty"`
	RemovedResults []int  `json:"removed_results,omitempty"`
	Fence          string `json:"fence,omitempty"`
}

// GraphDeadArgumentReceipt is the deterministic witness for dead graph-call ABI elimination.
type GraphDeadArgumentReceipt struct {
	Decisions []GraphDeadArgumentDecision `json:"decisions"`
	Digest    string                      `json:"digest"`
}

// EliminateDeadGraphArguments clones program and removes unused argument and
// result positions from eligible direct callees and every direct callsite.
// Entry, exported, external, indirect-callable, and address-taken functions are
// ABI fences. CachedResultIndexes preserves result positions used by callers
// outside this program. The supported envelope is a direct-call graph whose
// call operands/results exactly match the callee signature.
func EliminateDeadGraphArguments(program GraphInlineProgram) (GraphInlineProgram, GraphDeadArgumentReceipt, error) {
	functions := make(map[string]GraphInlineFunction, len(program.Functions))
	referenced := make(map[string]bool)
	for _, original := range program.Functions {
		if original.Name == "" {
			return GraphInlineProgram{}, GraphDeadArgumentReceipt{}, fmt.Errorf("function name is empty")
		}
		if _, exists := functions[original.Name]; exists {
			return GraphInlineProgram{}, GraphDeadArgumentReceipt{}, fmt.Errorf("duplicate function %q", original.Name)
		}
		fn := cloneGraphInlineFunction(original)
		if len(fn.ReturnValues) == 0 && len(fn.Results) != 0 {
			fn.ReturnValues = append([]string(nil), fn.Results...)
		}
		if len(fn.ReturnValues) != len(fn.Results) {
			return GraphInlineProgram{}, GraphDeadArgumentReceipt{}, fmt.Errorf("function %q has %d results but %d return values", fn.Name, len(fn.Results), len(fn.ReturnValues))
		}
		for _, index := range fn.CachedResultIndexes {
			if index < 0 || index >= len(fn.Results) {
				return GraphInlineProgram{}, GraphDeadArgumentReceipt{}, fmt.Errorf("function %q cached result index %d is out of range", fn.Name, index)
			}
		}
		functions[fn.Name] = fn
	}
	if _, ok := functions[program.Entry]; !ok {
		return GraphInlineProgram{}, GraphDeadArgumentReceipt{}, fmt.Errorf("entry function %q is missing", program.Entry)
	}
	for callerName, fn := range functions {
		for _, instruction := range fn.Instructions {
			if instruction.Reference != "" {
				if _, ok := functions[instruction.Reference]; !ok {
					return GraphInlineProgram{}, GraphDeadArgumentReceipt{}, fmt.Errorf("function %q references missing function %q", callerName, instruction.Reference)
				}
				referenced[instruction.Reference] = true
			}
			if instruction.Call == "" {
				continue
			}
			callee, ok := functions[instruction.Call]
			if !ok {
				return GraphInlineProgram{}, GraphDeadArgumentReceipt{}, fmt.Errorf("function %q calls missing function %q", callerName, instruction.Call)
			}
			if len(instruction.Arguments) != len(callee.Arguments) || len(instruction.Results) != len(callee.Results) {
				return GraphInlineProgram{}, GraphDeadArgumentReceipt{}, fmt.Errorf("function %q call to %q has %d arguments/%d results; want %d/%d", callerName, callee.Name, len(instruction.Arguments), len(instruction.Results), len(callee.Arguments), len(callee.Results))
			}
		}
	}

	liveArgs := make(map[string][]bool, len(functions))
	liveResults := make(map[string][]bool, len(functions))
	fences := make(map[string]string, len(functions))
	for name, fn := range functions {
		liveArgs[name] = make([]bool, len(fn.Arguments))
		liveResults[name] = make([]bool, len(fn.Results))
		fence := graphABIFence(program.Entry, fn, referenced[name])
		fences[name] = fence
		if fence != "" {
			fillBools(liveArgs[name])
			fillBools(liveResults[name])
		}
		for _, index := range fn.CachedResultIndexes {
			// A separately cached caller addresses results by position. Keep the
			// prefix through that position so compaction cannot renumber it.
			for i := 0; i <= index; i++ {
				liveResults[name][i] = true
			}
		}
	}

	for changed := true; changed; {
		changed = false
		for name, fn := range functions {
			used := make(map[string]bool)
			for i, value := range fn.ReturnValues {
				if liveResults[name][i] {
					used[value] = true
				}
			}
			for _, instruction := range fn.Instructions {
				if instruction.Call == "" {
					for _, value := range instruction.Arguments {
						used[value] = true
					}
					continue
				}
				for i, value := range instruction.Arguments {
					if liveArgs[instruction.Call][i] {
						used[value] = true
					}
				}
			}
			for i, argument := range fn.Arguments {
				if used[argument] && !liveArgs[name][i] {
					liveArgs[name][i] = true
					changed = true
				}
			}
			for _, instruction := range fn.Instructions {
				if instruction.Call == "" {
					continue
				}
				for i, value := range instruction.Results {
					if used[value] && !liveResults[instruction.Call][i] {
						liveResults[instruction.Call][i] = true
						changed = true
					}
				}
			}
		}
	}

	names := make([]string, 0, len(functions))
	for name := range functions {
		names = append(names, name)
	}
	sort.Strings(names)
	decisions := make([]GraphDeadArgumentDecision, 0, len(names))
	for _, name := range names {
		fn := functions[name]
		decision := GraphDeadArgumentDecision{Function: name, Fence: fences[name]}
		fn.Arguments, decision.RemovedArgs = filterStrings(fn.Arguments, liveArgs[name])
		fn.Results, decision.RemovedResults = filterStrings(fn.Results, liveResults[name])
		fn.ReturnValues, _ = filterStrings(fn.ReturnValues, liveResults[name])
		functions[name] = fn
		decisions = append(decisions, decision)
	}
	for name, fn := range functions {
		for i := range fn.Instructions {
			instruction := &fn.Instructions[i]
			if instruction.Call == "" {
				continue
			}
			instruction.Arguments, _ = filterStrings(instruction.Arguments, liveArgs[instruction.Call])
			instruction.Results, _ = filterStrings(instruction.Results, liveResults[instruction.Call])
		}
		functions[name] = fn
	}

	out := GraphInlineProgram{Entry: program.Entry, Functions: make([]GraphInlineFunction, 0, len(program.Functions))}
	for _, original := range program.Functions {
		out.Functions = append(out.Functions, functions[original.Name])
	}
	receipt := GraphDeadArgumentReceipt{Decisions: decisions}
	encoded, err := json.Marshal(struct {
		Program   GraphInlineProgram          `json:"program"`
		Decisions []GraphDeadArgumentDecision `json:"decisions"`
	}{out, decisions})
	if err != nil {
		return GraphInlineProgram{}, GraphDeadArgumentReceipt{}, err
	}
	digest := sha256.Sum256(encoded)
	receipt.Digest = "sha256:" + hex.EncodeToString(digest[:])
	return out, receipt, nil
}

func cloneGraphInlineFunction(fn GraphInlineFunction) GraphInlineFunction {
	fn.Arguments = append([]string(nil), fn.Arguments...)
	fn.Results = append([]string(nil), fn.Results...)
	fn.ReturnValues = append([]string(nil), fn.ReturnValues...)
	fn.CachedResultIndexes = append([]int(nil), fn.CachedResultIndexes...)
	fn.Instructions = append([]GraphInlineInstruction(nil), fn.Instructions...)
	for i := range fn.Instructions {
		fn.Instructions[i].Arguments = append([]string(nil), fn.Instructions[i].Arguments...)
		fn.Instructions[i].Results = append([]string(nil), fn.Instructions[i].Results...)
	}
	return fn
}

func graphABIFence(entry string, fn GraphInlineFunction, referenced bool) string {
	switch {
	case fn.Name == entry:
		return "entry-abi"
	case fn.Exported:
		return "exported-abi"
	case fn.External:
		return "external-abi"
	case fn.IndirectCallable:
		return "indirect-call"
	case referenced:
		return "symbol-reference"
	default:
		return ""
	}
}

func fillBools(values []bool) {
	for i := range values {
		values[i] = true
	}
}

func filterStrings(values []string, keep []bool) ([]string, []int) {
	var out []string
	var removed []int
	for i, value := range values {
		if keep[i] {
			out = append(out, value)
		} else {
			removed = append(removed, i)
		}
	}
	return out, removed
}

// InlineGraphFunctions clones program, safely replaces eligible direct calls,
// removes dead callee symbols, and returns a deterministic decision receipt.
// Recursive SCCs fail closed; non-call references keep their target symbol.
func InlineGraphFunctions(program GraphInlineProgram, maxInstructions int) (GraphInlineProgram, GraphInlineReceipt, error) {
	if maxInstructions <= 0 {
		return GraphInlineProgram{}, GraphInlineReceipt{}, fmt.Errorf("max instructions must be positive")
	}
	functions := make(map[string]GraphInlineFunction, len(program.Functions))
	for _, fn := range program.Functions {
		if fn.Name == "" {
			return GraphInlineProgram{}, GraphInlineReceipt{}, fmt.Errorf("function name is empty")
		}
		if _, exists := functions[fn.Name]; exists {
			return GraphInlineProgram{}, GraphInlineReceipt{}, fmt.Errorf("duplicate function %q", fn.Name)
		}
		fn.Instructions = append([]GraphInlineInstruction(nil), fn.Instructions...)
		functions[fn.Name] = fn
	}
	if _, ok := functions[program.Entry]; !ok {
		return GraphInlineProgram{}, GraphInlineReceipt{}, fmt.Errorf("entry function %q is missing", program.Entry)
	}

	calls := codegraph.NewGraph()
	referenced := make(map[string]bool)
	selfCall := make(map[string]bool)
	for name, fn := range functions {
		calls.AddNode(codegraph.NodeID(name), "model-graph-function")
		for _, instruction := range fn.Instructions {
			if instruction.Call != "" {
				if _, ok := functions[instruction.Call]; !ok {
					return GraphInlineProgram{}, GraphInlineReceipt{}, fmt.Errorf("function %q calls missing function %q", name, instruction.Call)
				}
				calls.AddEdge(codegraph.NodeID(name), codegraph.NodeID(instruction.Call), "calls")
				selfCall[instruction.Call] = selfCall[instruction.Call] || instruction.Call == name
			}
			if instruction.Reference != "" {
				if _, ok := functions[instruction.Reference]; !ok {
					return GraphInlineProgram{}, GraphInlineReceipt{}, fmt.Errorf("function %q references missing function %q", name, instruction.Reference)
				}
				referenced[instruction.Reference] = true
			}
		}
	}
	recursive := make(map[string]bool)
	for _, component := range calls.StronglyConnectedComponents("calls") {
		if len(component) > 1 {
			for _, id := range component {
				recursive[string(id)] = true
			}
		} else if selfCall[string(component[0])] {
			recursive[string(component[0])] = true
		}
	}

	names := make([]string, 0, len(functions))
	for name := range functions {
		if name != program.Entry {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	decisions := make([]GraphInlineDecision, 0, len(names))
	inline := make(map[string]bool, len(names))
	for _, name := range names {
		fn := functions[name]
		decision := GraphInlineDecision{Function: name, Action: "keep", Retained: referenced[name]}
		switch {
		case recursive[name]:
			decision.Reason = "recursive-scc"
		case fn.NeverInline:
			decision.Reason = "never-inline"
		case fn.AlwaysInline:
			decision.Action, decision.Reason, inline[name] = "inline", "always-inline", true
		case instructionCost(fn.Instructions) > maxInstructions:
			decision.Reason = "over-threshold"
		default:
			decision.Action, decision.Reason, inline[name] = "inline", "within-threshold", true
		}
		decisions = append(decisions, decision)
	}

	ordered := append([]string{program.Entry}, names...)
	for changed := true; changed; {
		changed = false
		for _, name := range ordered {
			fn := functions[name]
			out := make([]GraphInlineInstruction, 0, len(fn.Instructions))
			for _, instruction := range fn.Instructions {
				callee, ok := functions[instruction.Call]
				if instruction.Call == "" || !ok || !inline[instruction.Call] {
					if instruction.Operation != "noop" {
						out = append(out, instruction)
					}
					continue
				}
				out = append(out, callee.Instructions...)
				changed = true
			}
			fn.Instructions = out
			functions[name] = fn
		}
	}

	out := GraphInlineProgram{Entry: program.Entry}
	for _, name := range append([]string{program.Entry}, names...) {
		if name != program.Entry && inline[name] && !referenced[name] {
			continue
		}
		out.Functions = append(out.Functions, functions[name])
	}
	receipt := GraphInlineReceipt{Decisions: decisions}
	encoded, err := json.Marshal(struct {
		Program   GraphInlineProgram    `json:"program"`
		Decisions []GraphInlineDecision `json:"decisions"`
	}{out, decisions})
	if err != nil {
		return GraphInlineProgram{}, GraphInlineReceipt{}, err
	}
	digest := sha256.Sum256(encoded)
	receipt.Digest = "sha256:" + hex.EncodeToString(digest[:])
	return out, receipt, nil
}

func instructionCost(instructions []GraphInlineInstruction) int {
	cost := 0
	for _, instruction := range instructions {
		if instruction.Operation != "noop" && instruction.Reference == "" {
			cost++
		}
	}
	return cost
}

// Activations is the full-prefill intermediate state the oracle test compares
// against HF. Hidden[l] is the hidden state AFTER layer l-1 (Hidden[0] is the
// embedding output), flattened row-major [seq*hidden] — matching HF's
// output_hidden_states tuple of length NumLayers+1. Logits is [seq][vocab].
type Activations struct {
	Seq    int
	Hidden [][]float32 // [NumLayers+1] each len seq*hidden
	Logits [][]float32 // [seq] each len vocab
}

// rope precomputes cos/sin for every (position, freq) once per forward.
type rope struct {
	cos, sin [][]float32 // [pos][half]
}

func newRope(cfg Config, seq int) rope {
	return newRopeForLayer(cfg, 0, seq)
}

func newRopeForLayer(cfg Config, layer, seq int) rope {
	r := rope{cos: make([][]float32, seq), sin: make([][]float32, seq)}
	inv := invFreq(cfg, layer)
	scale := cfg.ropeAttentionFactor()
	for p := 0; p < seq; p++ {
		r.cos[p], r.sin[p] = ropeRowFromInvScaled(inv, p, scale)
	}
	return r
}

// apply rotates one head vector (len head_dim) in place at position p, using HF's
// non-interleaved "rotate_half" convention:
//
//	out[j]      = x[j]*cos - x[j+half]*sin
//	out[j+half] = x[j+half]*cos + x[j]*sin
func (r rope) apply(hv []float32, p int) {
	applyRopeRow(hv, r.cos[p], r.sin[p])
}

// Forward runs a full-prefill forward pass over token ids and returns every hidden
// state + per-position logits. No KV cache (that is R2); this rung proves the math.
func (m *Model) Forward(ids []int) *Activations {
	// embedding lookup -> x[t] is the working hidden vector for position t
	// (the arch embed scale applied inside embedBand).
	return m.forwardHiddenRows(m.embedBand(ids))
}

// forwardHiddenRows runs the decoder stack from already-materialized input embeddings.
// Forward builds these rows from token ids; governed multimodal callers may splice in
// externally-produced vision embeddings after admission checks.
func (m *Model) forwardHiddenRows(x [][]float32) *Activations {
	cfg := m.Cfg
	seq := len(x)
	act := &Activations{Seq: seq, Hidden: [][]float32{flatten(x)}}
	var glmDsaSharedTopK [][]int
	gemma4 := cfg.isGemma4()
	var gemma4RopeFreqs []float64
	if gemma4 {
		gemma4RopeFreqs = m.gemma4RopeFreqs()
	}
	for l := 0; l < cfg.NumLayers; l++ {
		switch {
		case gemma4:
			// Gemma 4 builds its own per-layer RoPE inside the heterogeneous-geometry
			// path, so the shared per-layer rope table is not used here.
			m.layerGemma4(l, x, gemma4RopeFreqs)
		case cfg.usesMLAMoELayout():
			m.layerGLMDsa(l, x, newRopeForLayer(cfg, l, seq), &glmDsaSharedTopK)
		case cfg.isMiniMaxSparseAttn():
			m.layerMiniMax(l, x, newRopeForLayer(cfg, l, seq))
		default:
			m.layer(l, x, newRopeForLayer(cfg, l, seq))
		}
		act.Hidden = append(act.Hidden, flatten(x))
	}

	m.fillSeqLogits(act, x)
	return act
}

// fillSeqLogits writes act.Logits for every position of x: the final norm, the (tied) LM
// head through the resident kernel, and the architecture's logit scale (Cohere/Gemma2;
// a no-op for Llama). Forward and its parallel twins ForwardTP/ForwardEP must END here
// IDENTICALLY — at ranks=1 each twin is held bit-identical to Forward, which is the
// transitive HF-oracle gate — so the tail is one function rather than one copy per twin
// that could drift a norm, a head name or a scale apart from the others.
func (m *Model) fillSeqLogits(act *Activations, x [][]float32) {
	act.Logits = make([][]float32, act.Seq)
	for t := 0; t < act.Seq; t++ {
		act.Logits[t] = m.logitsFromHidden(x[t])
	}
}

// logitsFromHidden is the per-position forward TAIL: the final norm, the (tied) LM head
// through the resident kernel (so a checkpoint whose head is resident Q8/Q4_K/k-quant is
// read from that store, not an absent f32 copy), and the architecture's logit scale /
// soft-cap / forced-token suppression. fillSeqLogits runs it at every position; a
// single-position caller that needs only the last token's distribution — the gemma4
// Session bridge (gemma4_session.go) — runs it once. Sharing the one tail is what makes
// that bridge BIT-EXACT with Forward rather than a second transcription that could drift
// a norm, a head name or a scale apart.
func (m *Model) logitsFromHidden(x []float32) []float32 {
	cfg := m.Cfg
	mat := residentKernel{m}
	xf := m.finalNorm(x)
	logits := mat.mul(m.headName(), mat.prep(xf), cfg.VocabSize, cfg.HiddenSize)
	logitScaleInPlace(logits, cfg)
	return logits
}

// layer applies one decoder block to x in place. The default (Llama, PreNorm) is
// attention + MLP, each with a pre-norm and a residual. cfg.BlockTopology selects
// the norm placement / residual wiring (arch.go); PreNorm lowers to the verbatim
// Llama instruction stream so the oracle rungs stay bit-exact.
func (m *Model) layer(l int, x [][]float32, rp rope) {
	cfg := m.Cfg
	eps := float32(cfg.RMSNormEps)
	attnNorm := m.attentionNorms(l)
	topo := cfg.BlockTopology

	// attnSub returns the per-position attention output projections for the given
	// per-position normalized inputs (one normalized vector per position). It is the
	// whole-sequence attention body — the norm placement is owned by the topology
	// composition below, so this consumes already-normalized inputs and never norms.
	attnSub := func(xn [][]float32) [][]float32 { return m.attnSeq(l, xn, rp) }
	// mlpSub returns the per-position SwiGLU MLP outputs for normalized inputs.
	mlpSub := func(xn [][]float32) [][]float32 { return m.mlpSeq(l, xn) }

	// Qwen3.5/Qwen3-Next hybrid: a linear_attention layer swaps the attention token mixer
	// for the Gated-DeltaNet recurrent scan (qwen35.go), keeping the PreNorm + SwiGLU wiring.
	if cfg.isLinearAttnLayer(l) {
		// FAK_GDN_BATCHED routes the Gated-DeltaNet prefill through the batched-projection
		// path (issue #443); it is bit-identical to linearAttnSeq on the f32 path, certified
		// by TestQwen35LinearAttnBatchedMatchesScalar, so the opt-in is witness-gated.
		linAttnSub := func(xn [][]float32) [][]float32 {
			if gdnBatchedPrefill {
				return m.linearAttnSeqBatched(l, xn)
			}
			return m.linearAttnSeq(l, xn)
		}
		composeSeqSublayer(topo, x, attnNorm, eps, cfg, linAttnSub)
		composeSeqSublayer(topo, x, m.mlpNorms(l), eps, cfg, mlpSub)
		return
	}

	m.composeSeqBlock(l, topo, x, attnNorm, eps, cfg, attnSub, mlpSub)
}

func (m *Model) layerGLMDsa(l int, x [][]float32, rp rope, sharedTopK *[][]int) {
	cfg := m.Cfg
	eps := float32(cfg.RMSNormEps)
	attnNorm := m.attentionNorms(l)
	topo := cfg.BlockTopology
	attnSub := func(xn [][]float32) [][]float32 {
		return m.glmDsaAttnSeqShared(l, xn, sharedTopK)
	}
	mlpSub := func(xn [][]float32) [][]float32 { return m.mlpSeq(l, xn) }

	m.composeSeqBlock(l, topo, x, attnNorm, eps, cfg, attnSub, mlpSub)
}

// attnSeq computes causal GQA attention over a whole sequence of already-normalized
// inputs and returns the per-position output-projection results (pre residual).
func (m *Model) attnSeq(l int, xn [][]float32, rp rope) [][]float32 {
	cfg := m.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	if cfg.usesMLAMoELayout() {
		return m.glmDsaAttnSeqShared(l, xn, nil)
	}
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	seq := len(xn)
	attnCap := float32(cfg.AttnSoftcap)
	p := func(s string) string { return layerName(l, s) }
	mat := residentKernel{m}

	// per-position q,k,v after norm + projection (+ optional bias, + optional qk-norm).
	q := make([][]float32, seq) // [seq][nH*hd]
	k := make([][]float32, seq) // [seq][nKV*hd]
	v := make([][]float32, seq) // [seq][nKV*hd]
	// Qwen3.5/Qwen3-Next full-attention layers project a doubled q_proj and split it,
	// per head, into the query and a sigmoid output gate ([query|gate] interleaved by head);
	// the gate multiplies the attention output before o_proj. Off (default) = Llama path.
	gated := cfg.AttnOutputGate
	qWidth := nH * hd
	if gated {
		qWidth = 2 * nH * hd
	}
	var gates [][]float32
	if gated {
		gates = make([][]float32, seq)
	}
	for t := 0; t < seq; t++ {
		xp := mat.prep(xn[t])
		if gated {
			qf := mat.mul(p("self_attn.q_proj.weight"), xp, qWidth, H)
			q[t], gates[t] = splitPackedQueryGate(qf, nH, hd)
		} else {
			q[t] = mat.mul(p("self_attn.q_proj.weight"), xp, nH*hd, H)
		}
		k[t] = mat.mul(p("self_attn.k_proj.weight"), xp, nKV*hd, H)
		v[t] = mat.mul(p("self_attn.v_proj.weight"), xp, nKV*hd, H)
		m.applyProjBias(l, q[t], k[t], v[t])
		// qk-norm AFTER projection, BEFORE RoPE; no-op for Llama.
		m.applyLayerQKNorm(l, q[t], k[t])
		// RoPE per head on q and k, through the shared single-row builder.
		if !cfg.Alibi {
			ropeRowQKInto(q[t], k[t], rp.cos[t], rp.sin[t], hd, nH, nKV)
		}
	}

	// scaled-dot-product attention, causal, GQA. With sliding-window attention (W>=0)
	// query t attends only to keys in [lo, t], lo=max(0,t-W+1); W=-1 (the default) keeps
	// lo=0, i.e. the full causal range 0..t exactly. This is the cacheless full-prefill
	// path, so positions are 0..seq-1 (no eviction) and the index IS the absolute position.
	W := cfg.windowForLayer(l)
	scale := cfg.attnScale()
	attnOut := make([][]float32, seq) // [seq][nH*hd]
	for t := 0; t < seq; t++ {
		attnOut[t] = make([]float32, nH*hd)
		lo := 0
		if W >= 0 {
			if lo = t - W + 1; lo < 0 {
				lo = 0
			}
		}
		for h := 0; h < nH; h++ {
			kvh := h / grp
			qh := q[t][h*hd : (h+1)*hd]
			// scores over keys lo..t (causal, optionally windowed)
			scores := make([]float32, t+1-lo)
			for j := lo; j <= t; j++ {
				kh := k[j][kvh*hd : (kvh+1)*hd]
				scores[j-lo] = dot(qh, kh)*scale + cfg.alibiScoreBias(h, j, seq)
			}
			softcapInPlace(scores, attnCap)
			m.softmaxAttentionScores(l, h, scores)
			if m.attnObs != nil { // #852: emit the post-softmax row (copy-out, math untouched)
				emitAttnRow(m.attnObs, l, t, h, lo, scores)
			}
			// weighted sum of values
			o := attnOut[t][h*hd : (h+1)*hd]
			for j := lo; j <= t; j++ {
				vh := v[j][kvh*hd : (kvh+1)*hd]
				w := scores[j-lo]
				saxpy(o, vh, w)
			}
		}
		if gated {
			gt := gates[t]
			for i := 0; i < nH*hd; i++ {
				attnOut[t][i] *= sigmoidf(gt[i])
			}
		}
		attnOut[t] = mat.mul(p("self_attn.o_proj.weight"), mat.prep(attnOut[t]), H, nH*hd)
		m.addBiasIfPresent(attnOut[t], p("self_attn.o_proj.bias"))
	}
	return attnOut
}

// splitPackedQueryGate separates a projection laid out as interleaved
// [query|gate] heads into the two ordinary packed-head vectors consumed by the
// attention and output-gate paths.
func splitPackedQueryGate(packed []float32, heads, headDim int) ([]float32, []float32) {
	query := make([]float32, heads*headDim)
	gate := make([]float32, heads*headDim)
	for h := 0; h < heads; h++ {
		copy(vectorHead(query, h, headDim), packed[h*2*headDim:h*2*headDim+headDim])
		copy(vectorHead(gate, h, headDim), packed[h*2*headDim+headDim:h*2*headDim+2*headDim])
	}
	return query, gate
}

func (m *Model) glmDsaAttnSeqShared(l int, xn [][]float32, sharedTopK *[][]int) [][]float32 {
	cfg := m.Cfg
	seq := len(xn)
	xnFlat := flatten(xn)
	var topK [][]int
	if cfg.IndexNHeads == 0 {
		// DeepSeek dense-MLA seam: no DSA lightning indexer (deepseek2), so every query
		// attends its full causal prefix. topK[t] = all positions [0..seq); the shared kernel's
		// glmDsaSelectedCausalKeys filters each row to [0..t], yielding exactly the dense causal
		// set. The one full slice is shared read-only (the kernel never mutates a topK row). This
		// is byte-identical attention math to GLM-DSA run over an un-pruned selection.
		full := glmDsaPositions(seq)
		topK = make([][]int, seq)
		for t := range topK {
			topK[t] = full
		}
	} else if glmDsaIndexerIsShared(cfg, l) {
		if sharedTopK == nil || *sharedTopK == nil {
			panic("model: glm_moe_dsa shared indexer without previous full indexer")
		}
		topK = cloneIndexDecision(*sharedTopK)
	} else {
		if !glmDsaIndexerIsFull(cfg, l) {
			panic("model: glm_moe_dsa unknown indexer type")
		}
		var ok bool
		topK, ok = glmDsaTopKIndicesNormed(m, l, xnFlat, seq)
		if !ok {
			panic("model: glm_moe_dsa top-k failed")
		}
		if sharedTopK != nil {
			*sharedTopK = cloneIndexDecision(topK)
		}
	}
	out, ok := glmDsaAttentionOutputFromTopKNormed(m, l, xnFlat, seq, topK)
	if !ok {
		panic("model: glm_moe_dsa attention failed")
	}
	return splitFlatRows(out, seq, cfg.HiddenSize)
}

// mlpSeq computes the SwiGLU MLP over a whole sequence of normalized inputs and
// returns the per-position down-projection results (pre residual).
func (m *Model) mlpSeq(l int, xn [][]float32) [][]float32 {
	out := make([][]float32, len(xn))
	ffn := m.ffnForLayer(l)
	mat := residentKernel{m}
	for t := range xn {
		// Stamp the per-token position the routing observer (#2623) reports as tokenPos,
		// but ONLY when one is installed — the unobserved path performs zero extra work and
		// stays byte-identical. route() reads m.routePos from inside ffn.apply.
		if m.routeObs != nil {
			m.routePos = t
		}
		out[t] = ffn.apply(m, l, mat.prep(xn[t]), mat)
	}
	return out
}

// normSeq normalizes each position's vector with the supplied norm weights.
func normSeq(x [][]float32, n normWeights, eps float32, cfg Config) [][]float32 {
	out := make([][]float32, len(x))
	for t := range x {
		out[t] = normCfg(x[t], n.pre, n.preBias, eps, cfg)
	}
	return out
}

// seqSublayer is one whole-sequence residual sub-layer body: normalized per-position
// inputs in, raw per-position outputs out (pre residual/post-norm).
type seqSublayer func(xn [][]float32) [][]float32

// composeSeqBlock applies the attention and MLP sublayers with one owner for the
// topology-dependent residual wiring. ParallelResidual is special: both branches
// consume the original residual before their deltas are added together. All other
// topologies keep the sequential attention-then-MLP composition.
func (m *Model) composeSeqBlock(l int, topo BlockTopology, x [][]float32, attnNorm normWeights, eps float32, cfg Config, attnSub, mlpSub seqSublayer) {
	if topo == ParallelResidual {
		mlpNorm := m.parallelMLPNorms(l, attnNorm)
		o := attnSub(normSeq(x, attnNorm, eps, cfg))
		d := mlpSub(normSeq(x, mlpNorm, eps, cfg))
		for t := range x {
			for i := range x[t] {
				x[t][i] += o[t][i] + d[t][i]
			}
		}
		return
	}
	composeSeqSublayer(topo, x, attnNorm, eps, cfg, attnSub)
	composeSeqSublayer(topo, x, m.mlpNorms(l), eps, cfg, mlpSub)
}

// composeSeqSublayer applies ONE residual sub-layer (norm placement + body + add)
// across a whole sequence under topology t. PreNorm: x += body(norm(x)) — the
// verbatim Llama placement. PostNorm: x += norm(body(x)). SandwichNorm:
// x += post(body(pre(x))). Parallel is handled separately by layer (shared norm,
// two deltas into one residual).
func composeSeqSublayer(t BlockTopology, x [][]float32, n normWeights, eps float32, cfg Config, body seqSublayer) {
	H := len(x[0])
	switch t {
	case PostNorm:
		addPostNormed(x, body(x), n, eps, cfg, H) // sub-layer reads the RAW residual stream
	case SandwichNorm:
		addPostNormed(x, body(normSeq(x, n, eps, cfg)), n, eps, cfg, H)
	default: // PreNorm — verbatim Llama
		out := body(normSeq(x, n, eps, cfg))
		for tt := range x {
			for i := 0; i < H; i++ {
				x[tt][i] += out[tt][i]
			}
		}
	}
}

// addPostNormed post-normalizes each position of out (per cfg, using the post-norm
// weights) and adds it into the residual stream x in place — the shared tail of the
// PostNorm and SandwichNorm placements (x += post(out)).
func addPostNormed(x, out [][]float32, n normWeights, eps float32, cfg Config, H int) {
	for tt := range x {
		nout := normCfg(out[tt], n.post, n.postBias, eps, cfg)
		for i := 0; i < H; i++ {
			x[tt][i] += nout[i]
		}
	}
}

// ---- primitive ops ---------------------------------------------------------

// rmsnorm: x / sqrt(mean(x^2)+eps) * weight (Llama convention: plain weight). The scalar
// in-order sum-of-squares is load-bearing for the f32 bit-exact rungs (R2/R14) — do not
// reorder it here; the in-place quant twin rmsnormInto is the one that may use fdot.
func rmsnorm(x, w []float32, eps float32) []float32 {
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * inv * w[i]
	}
	return out
}

// rmsnormInto is the allocation-free RMSNorm used by the Q8 prefill path: it writes directly
// into dst (the caller's panel row), eliminating both the per-row heap slice rmsnorm returns
// AND the caller's copy — 2*P*NumLayers (=15360 at P=256) of each per prefill. The
// sum-of-squares uses fdot (8 accumulators, vectorized) instead of the serial reduction; the
// ~1e-6 reduction-order shift is inside the Q8 gate's tolerance (logit-cosine vs f32,
// argmax-exact vs the oracle) and never reaches the f32 bit-exact rungs, which keep rmsnorm.
func rmsnormInto(dst, x, w []float32, eps float32) {
	ss := fdot(x, x)
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	for i, v := range x {
		dst[i] = v * inv * w[i]
	}
}

// matRows: y[o] = sum_i W[o*in+i]*x[i], W row-major [out,in]. (HF Linear: y = x @ W^T.)
// Routes through fdot (the 8-accumulator inner product) so the serial reference, the
// row-parallel parMatRows, and the batched matMulBatch all share ONE reduction and stay
// mutually bit-identical — the invariant the exact rungs R2/R14 rely on.
func matRows(w, x []float32, out, in int) []float32 {
	y := make([]float32, out)
	for o := 0; o < out; o++ {
		y[o] = fdot(w[o*in:o*in+in], x)
	}
	return y
}

func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func softmaxInPlace(s []float32) {
	mx := s[0]
	for _, v := range s {
		if v > mx {
			mx = v
		}
	}
	var sum float32
	for i, v := range s {
		e := float32(math.Exp(float64(v - mx)))
		s[i] = e
		sum += e
	}
	for i := range s {
		s[i] /= sum
	}
}

func silu(z float32) float32 { return z / (1 + float32(math.Exp(float64(-z)))) }

func addBias(y, b []float32) {
	for i := range y {
		y[i] += b[i]
	}
}

func flatten(x [][]float32) []float32 {
	if len(x) == 0 {
		return nil
	}
	H := len(x[0])
	out := make([]float32, len(x)*H)
	for t := range x {
		copy(out[t*H:], x[t])
	}
	return out
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

// PromoteRegionSlots replaces eligible graph-local load/store temporaries with
// SSA values carried explicitly through structured regions. Unknown regions
// fail closed: any slot they touch remains memory-backed.
func PromoteRegionSlots(graph compute.RegionSlotGraph) (compute.RegionSlotGraph, compute.RegionSlotReceipt, error) {
	declared := make(map[string]bool)
	blocked := make(map[string]bool)
	if err := inspectRegionSlots(graph.Ops, declared, blocked, false); err != nil {
		return compute.RegionSlotGraph{}, compute.RegionSlotReceipt{}, err
	}

	names := make([]string, 0, len(declared))
	for name := range declared {
		names = append(names, name)
	}
	sort.Strings(names)

	receipt := compute.RegionSlotReceipt{Promotions: make([]compute.RegionSlotPromotion, 0, len(names))}
	promoted := make(map[string]bool, len(names))
	for _, name := range names {
		promotion := compute.RegionSlotPromotion{Slot: name, Action: "promote"}
		if blocked[name] {
			promotion.Action = "keep"
			promotion.Reason = "unknown-region-use"
		} else {
			promoted[name] = true
		}
		receipt.Promotions = append(receipt.Promotions, promotion)
	}

	state := make(map[string]string, len(promoted))
	debug := make(map[string]string, len(promoted))
	for name := range promoted {
		state[name] = "undef." + name
	}
	ops, _, _ := promoteRegionOps(graph.Ops, state, debug, promoted)
	return compute.RegionSlotGraph{Ops: ops}, receipt, nil
}

func inspectRegionSlots(ops []compute.RegionSlotOp, declared, blocked map[string]bool, unknown bool) error {
	for _, op := range ops {
		if op.Slot != "" && unknown {
			blocked[op.Slot] = true
		}
		switch op.Kind {
		case compute.RegionSlotDeclare:
			if op.Slot == "" {
				return fmt.Errorf("slot declaration is missing a name")
			}
			if declared[op.Slot] {
				return fmt.Errorf("duplicate slot %q", op.Slot)
			}
			declared[op.Slot] = true
		case compute.RegionSlotLoad, compute.RegionSlotStore:
			if op.Slot == "" {
				return fmt.Errorf("%s is missing a slot", op.Kind)
			}
		case compute.RegionSlotIf:
			if err := inspectRegionSlots(op.Then, declared, blocked, unknown); err != nil {
				return err
			}
			if err := inspectRegionSlots(op.Else, declared, blocked, unknown); err != nil {
				return err
			}
		case compute.RegionSlotLoop:
			if err := inspectRegionSlots(op.Body, declared, blocked, unknown); err != nil {
				return err
			}
		case compute.RegionSlotUnknown:
			if err := inspectRegionSlots(op.Body, declared, blocked, true); err != nil {
				return err
			}
		}
	}
	return nil
}

func promoteRegionOps(ops []compute.RegionSlotOp, state, debug map[string]string, promoted map[string]bool) ([]compute.RegionSlotOp, map[string]string, map[string]string) {
	out := make([]compute.RegionSlotOp, 0, len(ops))
	for _, op := range ops {
		switch op.Kind {
		case compute.RegionSlotDeclare:
			if promoted[op.Slot] {
				state[op.Slot] = "undef." + op.Slot
				if op.Debug != "" {
					debug[op.Slot] = op.Debug
				}
				continue
			}
		case compute.RegionSlotStore:
			if promoted[op.Slot] {
				state[op.Slot] = op.Value
				if op.Debug != "" {
					debug[op.Slot] = op.Debug
				}
				continue
			}
		case compute.RegionSlotLoad:
			if promoted[op.Slot] {
				slot := op.Slot
				op.Kind = compute.RegionSlotConst
				op.Value = state[slot]
				op.Slot = ""
				if op.Debug == "" {
					op.Debug = debug[slot]
				}
			}
		case compute.RegionSlotIf:
			thenState, thenDebug := cloneRegionState(state), cloneRegionState(debug)
			elseState, elseDebug := cloneRegionState(state), cloneRegionState(debug)
			op.Then, thenState, thenDebug = promoteRegionOps(op.Then, thenState, thenDebug, promoted)
			op.Else, elseState, elseDebug = promoteRegionOps(op.Else, elseState, elseDebug, promoted)
			for _, slot := range sortedChangedSlots(state, thenState, elseState) {
				input := state[slot]
				result := nextRegionValue(op.Name, slot)
				binding := firstRegionDebug(thenDebug[slot], elseDebug[slot], debug[slot])
				op.Then = append(op.Then, compute.RegionSlotOp{Kind: compute.RegionSlotConst, Name: "yield." + slot, Value: thenState[slot], Debug: thenDebug[slot]})
				op.Else = append(op.Else, compute.RegionSlotOp{Kind: compute.RegionSlotConst, Name: "yield." + slot, Value: elseState[slot], Debug: elseDebug[slot]})
				op.Carries = append(op.Carries, compute.RegionSlotCarry{Slot: slot, Input: input, Output: result, Debug: binding})
				state[slot], debug[slot] = result, binding
			}
		case compute.RegionSlotLoop:
			before := cloneRegionState(state)
			bodyState, bodyDebug := cloneRegionState(state), cloneRegionState(debug)
			op.Body, bodyState, bodyDebug = promoteRegionOps(op.Body, bodyState, bodyDebug, promoted)
			for _, slot := range sortedChangedSlots(before, bodyState) {
				result := nextRegionValue(op.Name, slot)
				argument := result + ".arg"
				rewriteRegionValue(op.Body, before[slot], argument)
				op.Body = append(op.Body, compute.RegionSlotOp{Kind: compute.RegionSlotConst, Name: "yield." + slot, Value: bodyState[slot], Debug: bodyDebug[slot]})
				binding := firstRegionDebug(bodyDebug[slot], debug[slot])
				op.Carries = append(op.Carries, compute.RegionSlotCarry{Slot: slot, Input: before[slot], Argument: argument, Output: result, Debug: binding})
				state[slot], debug[slot] = result, binding
			}
		}
		out = append(out, op)
	}
	return out, state, debug
}

func cloneRegionState(in map[string]string) map[string]string {
	out := make(map[string]string, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}

func sortedChangedSlots(base map[string]string, variants ...map[string]string) []string {
	changed := make([]string, 0)
	for slot, value := range base {
		for _, variant := range variants {
			if variant[slot] != value {
				changed = append(changed, slot)
				break
			}
		}
	}
	sort.Strings(changed)
	return changed
}

func nextRegionValue(region, slot string) string {
	if region == "" {
		region = "region"
	}
	return region + "." + slot
}

func firstRegionDebug(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func rewriteRegionValue(ops []compute.RegionSlotOp, from, to string) {
	for i := range ops {
		if ops[i].Value == from {
			ops[i].Value = to
		}
		for carry := range ops[i].Carries {
			if ops[i].Carries[carry].Input == from {
				ops[i].Carries[carry].Input = to
			}
		}
		rewriteRegionValue(ops[i].Then, from, to)
		rewriteRegionValue(ops[i].Else, from, to)
		rewriteRegionValue(ops[i].Body, from, to)
	}
}
