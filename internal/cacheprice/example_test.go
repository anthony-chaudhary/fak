package cacheprice

import "fmt"

// Example_disaggregationDecision walks one prefix through the full pricing stack a disaggregated-KV
// scheduler runs: the length gate that admits it as a candidate, the per-turn route that serves it
// most cheaply, and the pool-retention verdict that decides whether it earns a resident slot. Each
// layer is a distinct question answered in the same token unit, composing bottom-up.
func Example_disaggregationDecision() {
	// A 1500-token prefix could be served from the remote / disaggregated KV tier.
	const prefix = 1500

	// Layer 1.5 — length gate. Under a linear transfer model (fixed setup 500, recompute 10/token,
	// wire 3/token), what is the shortest prefix worth fetching, and does ours clear it?
	breakEven, ever := TransferBreakEvenLength(500, 10, 3)
	fmt.Printf("length gate: worthwhile beyond %d tokens (ever=%v); %d-token prefix qualifies=%v\n",
		breakEven, ever, prefix, TransferWorthwhileAtLength(prefix, 500, 10, 3))

	// Layers 1+3 — per-turn route. A 2000-token prompt whose 1500-token prefix is available only
	// from the remote tier at a 400-token transfer toll: cheapest source and its billable cost.
	route, cost := CheapestRoute(2000, prefix, 400, false, true)
	fmt.Printf("route: %s at %d billable tokens (dividend %d)\n",
		route, cost, DisaggregationDividend(prefix, 400))

	// Layer 2 — pool retention. At 1100 net tokens/fetch over ~8 expected reuses against a
	// 3000-token keep-warm cost, does the prefix earn a slot in the bounded pool?
	r := RemoteResident{DividendPerFetch: 1100, ExpectedFetches: 8, ResidencyCost: 3000, CapacityTokens: prefix}
	fmt.Printf("pool: lifetime value %d, admit=%v\n", r.RetentionValue(), AdmitToRemote(r))

	// Output:
	// length gate: worthwhile beyond 72 tokens (ever=true); 1500-token prefix qualifies=true
	// route: remote at 900 billable tokens (dividend 1100)
	// pool: lifetime value 5800, admit=true
}
