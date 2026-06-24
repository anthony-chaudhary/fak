package main

// This file is the MODULAR core of the demo: a Scenario is one agentic *shape*
// (a "module"), and a Workload is the concrete, deterministic per-(agent,turn)
// plan derived from it. Everything here is timing-free and model-free — it is the
// exact, load-independent floor the live race (arms.go) later grounds in wall-clock.

// A Tool is one tool an agent can call. Its RESULT changes the running context by a
// variable number of tokens drawn from [MinTok, MaxTok]. Those *heterogeneous*
// result sizes are the "tool-call context changing" dimension this demo exists to
// make visible: unlike cmd/demorace (a single constant R per turn), here every
// agent's every turn appends a different-sized chunk, so the context grows unevenly.
type Tool struct {
	Name   string `json:"name"`
	MinTok int    `json:"min_tok"`
	MaxTok int    `json:"max_tok"`
}

// A Scenario is one agentic shape — a module in the catalog. Each scenario sits at a
// different point in (prefix size, agent count, turn count, tool-result distribution)
// space, so the cross-agent + cross-turn REUSE ratio, and the absolute work saved,
// shift between them. That shift IS the modular story: where does fak's value come
// from, and how does it move as the agentic shape changes?
type Scenario struct {
	ID     string `json:"id"`
	Label  string `json:"label"`
	Desc   string `json:"desc"`
	Prefix int    `json:"prefix"` // P — shared system prompt + tool-schema tokens
	Agents int    `json:"agents"` // C — agents that share the one prefix
	Turns  int    `json:"turns"`  // T — turns each agent runs
	Decode int    `json:"decode"` // D — tokens the model generates per turn
	Tools  []Tool `json:"tools"`
	Seed   uint64 `json:"-"`
}

// catalog is the set of scenario modules. They are deliberately spread across the
// agentic-shape space so the reuse ratio climbs visibly from one to the next:
//
//	short prefix · many agents · few turns      → reuse is mostly cross-agent
//	long prefix · few agents · many big results → reuse is mostly the turn tax
//
// All sizes are chosen so the fak arm runs live in a minute-ish on the 135M reference
// (the naive arm is the multi-minute grind — that gap is the point; see arms.go).
func catalog() []Scenario {
	return []Scenario{
		{
			ID: "support-bot", Label: "support bot — many short agents",
			Desc:   "A fleet of support agents sharing one policy+FAQ prefix; each does a couple of small lookups. The win here is mostly CROSS-AGENT: one prefix serves all of them.",
			Prefix: 256, Agents: 4, Turns: 3, Decode: 16, Seed: 0xA1,
			Tools: []Tool{
				{Name: "kb_lookup", MinTok: 16, MaxTok: 40},
				{Name: "order_status", MinTok: 20, MaxTok: 64},
			},
		},
		{
			ID: "coding-agent", Label: "coding agent — few agents, many turns",
			Desc:   "A handful of coding agents that read files, grep, and run tests over many turns. Each turn appends a bursty, variable-sized tool result, so the context grows fast — the TURN TAX (re-prefilling the whole growing context every turn) dominates.",
			Prefix: 768, Agents: 3, Turns: 6, Decode: 24, Seed: 0xC0DE,
			Tools: []Tool{
				{Name: "read_file", MinTok: 48, MaxTok: 256},
				{Name: "grep", MinTok: 32, MaxTok: 128},
				{Name: "run_tests", MinTok: 64, MaxTok: 200},
			},
		},
		{
			ID: "deep-research", Label: "deep research — long context",
			Desc:   "Research agents with a large tool-schema prefix that fetch and summarize the web. Big, growing results push every agent into LONG CONTEXT — exactly where naive re-prefill is quadratically expensive and reuse saves the most.",
			Prefix: 1536, Agents: 4, Turns: 5, Decode: 32, Seed: 0x4EE7,
			Tools: []Tool{
				{Name: "web_search", MinTok: 64, MaxTok: 160},
				{Name: "web_fetch", MinTok: 160, MaxTok: 512},
				{Name: "summarize", MinTok: 48, MaxTok: 128},
			},
		},
		{
			ID: "mixed-fleet", Label: "mixed fleet — heterogeneous agents, one org prefix",
			Desc:   "Six agents of DIFFERENT kinds (lookups, code, research) all sharing one organization prefix. The per-agent seed picks different tools per agent, so the fleet is genuinely heterogeneous — the realistic case for a shared-context agent platform.",
			Prefix: 512, Agents: 6, Turns: 4, Decode: 20, Seed: 0xF1EE7,
			Tools: []Tool{
				{Name: "kb_lookup", MinTok: 16, MaxTok: 40},
				{Name: "read_file", MinTok: 48, MaxTok: 256},
				{Name: "web_fetch", MinTok: 128, MaxTok: 448},
				{Name: "run_tests", MinTok: 64, MaxTok: 200},
			},
		},
		{
			ID: "fleet-5x50", Label: "FLEET — 5 agents × 50 turns (the headline fleet shape)",
			Desc:   "The fleet-scale shape: 5 agents sharing one org prefix, each running 50 turns with a different-sized tool result every turn. Both reuse levers stack — one shared prefix across all five agents (cross-agent) AND incremental result ingestion instead of re-prefilling the whole growing context every turn (the turn tax). At 250 total requests this is where the naive re-prefill bill is largest and the kernel's win is biggest.",
			Prefix: 1024, Agents: 5, Turns: 50, Decode: 24, Seed: 0xF1EE7_5050,
			Tools: []Tool{
				{Name: "kb_lookup", MinTok: 16, MaxTok: 40},
				{Name: "read_file", MinTok: 48, MaxTok: 256},
				{Name: "web_fetch", MinTok: 128, MaxTok: 448},
				{Name: "run_tests", MinTok: 64, MaxTok: 200},
			},
		},
	}
}

func findScenario(id string) (Scenario, bool) {
	for _, s := range catalog() {
		if s.ID == id {
			return s, true
		}
	}
	return Scenario{}, false
}

// scale applies optional per-axis overrides (used for headless smoke tests and for a
// presenter who wants to shrink a long scenario). A zero override leaves the axis as-is.
func (s Scenario) scale(P, C, T, D int) Scenario {
	if P > 0 {
		s.Prefix = P
	}
	if C > 0 {
		s.Agents = C
	}
	if T > 0 {
		s.Turns = T
	}
	if D > 0 {
		s.Decode = D
	}
	return s
}

// Workload is the concrete, deterministic per-(agent,turn) plan derived from a
// Scenario: which tool each agent calls each turn, and how many result tokens that
// call appends to the context. Results[c] has length Turns-1 — the final turn emits
// the answer and calls no tool, so there are T-1 context-changing tool results per
// agent. The plan is a pure function of the scenario (seeded LCG), so the demo is
// fully reproducible and the timing-free accounting below is exact.
type Workload struct {
	Scn     Scenario   `json:"scenario"`
	Results [][]int    `json:"results"`    // [agent][turn] result-token counts, len Turns-1
	Tools   [][]string `json:"tool_names"` // [agent][turn] tool chosen, for the timeline viz
}

// Build expands the scenario into its deterministic Workload: a per-agent seeded LCG picks each
// turn's tool and result-token count, yielding Turns-1 context-changing results per agent.
func (s Scenario) Build() Workload {
	C, T := s.Agents, s.Turns
	rt := T - 1
	if rt < 0 {
		rt = 0
	}
	res := make([][]int, C)
	tn := make([][]string, C)
	for c := 0; c < C; c++ {
		res[c] = make([]int, rt)
		tn[c] = make([]string, rt)
		// Per-agent seed so different agents pick different tools — the heterogeneity.
		st := (0x9e3779b97f4a7c15 * uint64(c+1)) ^ s.Seed
		for t := 0; t < rt; t++ {
			st = st*6364136223846793005 + 1442695040888963407
			tool := s.Tools[int(st>>33)%len(s.Tools)]
			st = st*6364136223846793005 + 1442695040888963407
			span := tool.MaxTok - tool.MinTok + 1
			if span < 1 {
				span = 1
			}
			res[c][t] = tool.MinTok + int(st>>33)%span
			tn[c][t] = tool.Name
		}
	}
	return Workload{Scn: s, Results: res, Tools: tn}
}

// totalResultTokens is the sum of all tool-result tokens appended across the session
// (the new context every strategy must ingest at least once).
func (w Workload) totalResultTokens() int {
	sum := 0
	for _, r := range w.Results {
		for _, v := range r {
			sum += v
		}
	}
	return sum
}

// maxAgentTail is the largest per-agent KV tail (decode + results over all turns),
// used to pre-reserve the fak batch cache so the measured run never triggers a
// prefix-copy on append.
func (w Workload) maxAgentTail() int {
	T, D := w.Scn.Turns, w.Scn.Decode
	maxTail := 0
	for c := 0; c < w.Scn.Agents; c++ {
		tail := T * D
		for t := 0; t < len(w.Results[c]); t++ {
			tail += w.Results[c][t]
		}
		if tail > maxTail {
			maxTail = tail
		}
	}
	return maxTail
}

// prefillTokens returns the EXACT, timing-free prefill-token work each serving
// strategy performs over the whole session. It is fixed by the workload alone, so
// the work-elimination ratio cannot drift with machine load — this is the honest
// floor the live wall-clock race is measured against.
//
//	a — NAIVE (stateless re-prefill): re-prefill the WHOLE growing context every
//	    turn, every agent. The context at the start of turn t already contains the
//	    prefix plus everything generated and ingested in prior turns, so prior
//	    decode tokens are paid for again and again. Quadratic in T, linear in C.
//	b — PER-AGENT KV (the tuned single-engine baseline, e.g. llama.cpp per slot):
//	    each agent prefills the shared prefix ONCE (so C times across the fleet),
//	    then ingests only the new result tokens incrementally. No CROSS-agent sharing.
//	c — FAK FUSED: the prefix is prefilled ONCE and cloned into all C agents; only
//	    the new result tokens are ever ingested. One prefix for the whole fleet.
//
// Decode tokens are GENERATED (not prefilled) in every arm, so they are excluded
// here — this counts re-read prefill work only, which is where the strategies differ.
func (w Workload) prefillTokens() (a, b, c int) {
	P, C, T, D := w.Scn.Prefix, w.Scn.Agents, w.Scn.Turns, w.Scn.Decode
	for ci := 0; ci < C; ci++ {
		tail := 0
		for t := 0; t < T; t++ {
			a += P + tail // naive re-prefills prefix + everything generated/ingested so far
			if t < len(w.Results[ci]) {
				tail += D + w.Results[ci][t]
			}
		}
	}
	total := w.totalResultTokens()
	b = C*P + total
	c = P + total
	return
}
