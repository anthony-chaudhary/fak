package gatewayusageledger

import (
	"encoding/json"
	"math"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cacheprice"
)

// close reports float equality within a tolerance that is generous versus the
// arithmetic here (sums of exact binary fractions) but far tighter than any real
// difference the pricing basis could produce.
func close(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

// TestExitRowCarriesCompactionEconomicsTrailer is #2792's acceptance check: an exit row
// written for a session that compacted carries the trailer, on the wire, with all five
// figures the issue names (fires / shed / observed cache_read / induced creation / net).
// It asserts against the MARSHALED JSON, not the struct, because the acceptance is about
// what a `gateway-usage.jsonl` READER sees — a field renamed or dropped from the wire
// would pass a struct assertion and still fail the issue.
func TestExitRowCarriesCompactionEconomicsTrailer(t *testing.T) {
	c := Counters{
		CompactionFired:           3,
		CompactionShedTokens:      10000,
		CompactionCacheReadTokens: 4000, // 4000 shed tokens were warm, 6000 cold
	}
	row := NewRow("exit", "guard", "claude", "sess-1", 90*time.Second, nil, c, time.Unix(0, 0))

	b, err := json.Marshal(row)
	if err != nil {
		t.Fatalf("marshal row: %v", err)
	}
	var wire struct {
		Kind                string `json:"kind"`
		CompactionEconomics *struct {
			Fires                      uint64  `json:"fires"`
			ShedTokens                 uint64  `json:"shed_tokens"`
			ObservedCacheReadTokens    uint64  `json:"observed_cache_read_tokens"`
			InducedCacheCreationTokens uint64  `json:"induced_cache_creation_tokens"`
			ShedSavingTokenEquiv       float64 `json:"shed_saving_token_equiv"`
			BurstCostTokenEquiv        float64 `json:"burst_cost_token_equiv"`
			NetTokenEquiv              float64 `json:"net_token_equiv"`
			NetIsUpperBound            bool    `json:"net_is_upper_bound"`
		} `json:"compaction_economics"`
	}
	if err := json.Unmarshal(b, &wire); err != nil {
		t.Fatalf("unmarshal row: %v", err)
	}
	if wire.Kind != "exit" {
		t.Fatalf("kind = %q, want exit", wire.Kind)
	}
	tr := wire.CompactionEconomics
	if tr == nil {
		t.Fatal("exit row for a compacting session carries NO compaction_economics trailer (#2792)")
	}
	if tr.Fires != 3 || tr.ShedTokens != 10000 || tr.ObservedCacheReadTokens != 4000 {
		t.Fatalf("trailer raw figures wrong: %+v", tr)
	}
	// The shed is priced on the canonical PROPORTIONAL blend, not a binary flip: the warm
	// 4000 at 0.1x, the cold 6000 at full input.
	wantShed := 4000*cacheprice.ReadMultiplier + 6000.0
	if !close(tr.ShedSavingTokenEquiv, wantShed) {
		t.Fatalf("shed saving = %v, want %v (warm/cold blend, cacheprice.ShedTokenEquiv)", tr.ShedSavingTokenEquiv, wantShed)
	}
	if !close(tr.ShedSavingTokenEquiv, cacheprice.ShedTokenEquiv(10000, 4000)) {
		t.Fatal("shed saving drifted from cacheprice.ShedTokenEquiv — the trailer must not carry its own copy of the basis")
	}
	// No induced-creation witness (#2785 unlanded): the debit is absent AND the row says
	// so, so the net cannot be misread as a complete one.
	if tr.InducedCacheCreationTokens != 0 || tr.BurstCostTokenEquiv != 0 {
		t.Fatalf("unwitnessed induced creation must debit nothing: %+v", tr)
	}
	if !tr.NetIsUpperBound {
		t.Fatal("a net computed with no induced-creation witness must be flagged an upper bound, not reported as the net")
	}
	if !close(tr.NetTokenEquiv, wantShed) {
		t.Fatalf("net = %v, want %v (shed saving with a zero debit)", tr.NetTokenEquiv, wantShed)
	}
}

// TestCompactionEconomicsDebitsInducedCreation locks the netting once #2785 populates the
// counter: the burst is debited at the excess-over-read on the 5m write tier, the net drops
// by exactly that, the upper-bound caveat clears, and a burst big enough to swamp the shed
// drives the net NEGATIVE rather than being floored at zero (#1303).
func TestCompactionEconomicsDebitsInducedCreation(t *testing.T) {
	base := Counters{CompactionFired: 1, CompactionShedTokens: 10000, CompactionCacheReadTokens: 10000}
	premium := cacheprice.Write5mMultiplier - cacheprice.ReadMultiplier

	withWitness := base
	withWitness.CompactionInducedCreationTokens = 400
	tr := CompactionEconomicsOf(withWitness)
	if tr == nil {
		t.Fatal("CompactionEconomicsOf returned nil for a session that fired")
	}
	// Fully warm shed: all 10000 price at the read marginal.
	wantShed := 10000 * cacheprice.ReadMultiplier
	wantBurst := 400 * premium
	if !close(tr.ShedSavingTokenEquiv, wantShed) || !close(tr.BurstCostTokenEquiv, wantBurst) {
		t.Fatalf("terms wrong: shed %v (want %v), burst %v (want %v)", tr.ShedSavingTokenEquiv, wantShed, tr.BurstCostTokenEquiv, wantBurst)
	}
	if !close(tr.NetTokenEquiv, wantShed-wantBurst) {
		t.Fatalf("net = %v, want %v (shed saving MINUS burst cost)", tr.NetTokenEquiv, wantShed-wantBurst)
	}
	if tr.NetIsUpperBound {
		t.Fatal("a witnessed induced-creation row must NOT still be flagged an upper bound")
	}

	// A burst that swamps a fully-warm shed is a value-DESTROYING session. It must read
	// negative; flooring it at zero would hide exactly the fire the RSI loop needs to see.
	swamped := base
	swamped.CompactionInducedCreationTokens = 5000
	if net := CompactionEconomicsOf(swamped).NetTokenEquiv; net >= 0 {
		t.Fatalf("net = %v, want NEGATIVE — a burst larger than the shed saving destroys value and must not be floored", net)
	}
}

// TestQuietSessionCarriesNoTrailer pins the byte-compatibility fence: a session that never
// compacted grows no trailer, so its row stays identical to the pre-#2792 schema and no
// downstream reader sees a new key appear on rows that have nothing to say.
func TestQuietSessionCarriesNoTrailer(t *testing.T) {
	if tr := CompactionEconomicsOf(Counters{ObservedTurns: 12, InputTokens: 900}); tr != nil {
		t.Fatalf("a non-compacting session must carry no trailer, got %+v", tr)
	}
	row := NewRow("exit", "serve", "http", "sess-quiet", time.Second, nil, Counters{ObservedTurns: 12}, time.Unix(0, 0))
	b, err := json.Marshal(row)
	if err != nil {
		t.Fatalf("marshal row: %v", err)
	}
	var wire map[string]json.RawMessage
	if err := json.Unmarshal(b, &wire); err != nil {
		t.Fatalf("unmarshal row: %v", err)
	}
	if _, present := wire["compaction_economics"]; present {
		t.Fatal("quiet row must omit compaction_economics entirely (omitempty), not emit a zero object")
	}
	// A fired-but-shed-nothing session DOES have a story (the fires are the story), so it
	// must still get a trailer — the gate is "did compaction happen", not "did it shed".
	if tr := CompactionEconomicsOf(Counters{CompactionFired: 1}); tr == nil {
		t.Fatal("a session that fired but shed nothing must still carry a trailer")
	}
}

// TestTrailerDoesNotDisturbRowKey guards the dedup contract: RowKey hashes the COUNTERS,
// and the trailer is derived from those same counters, so stamping it must not perturb the
// key. If it did, a retried exit flush of one snapshot would stop collapsing at fold and
// could double-count a session's savings (#2503).
func TestTrailerDoesNotDisturbRowKey(t *testing.T) {
	c := Counters{CompactionFired: 2, CompactionShedTokens: 500, CompactionCacheReadTokens: 100}
	a := NewRow("exit", "guard", "claude", "s", time.Second, nil, c, time.Unix(0, 0))
	b := NewRow("periodic", "guard", "snapshot", "s", 5*time.Second, nil, c, time.Unix(0, 0))
	if a.RowKey == "" || a.RowKey != b.RowKey {
		t.Fatalf("one snapshot must yield one RowKey across kinds: %q vs %q", a.RowKey, b.RowKey)
	}
	if a.CompactionEconomics == nil || b.CompactionEconomics == nil {
		t.Fatal("both writers must stamp the trailer — it is derived in NewRow, not by the caller")
	}
	deduped, dropped := DedupeByKey([]Row{a, b})
	if dropped != 1 || len(deduped) != 1 {
		t.Fatalf("the two flushes of one snapshot must collapse to one row: kept %d, dropped %d", len(deduped), dropped)
	}
}

// TestTrailerSurvivesTheLedgerRoundTrip is the end-to-end witness the issue's acceptance
// names: write an exit row through the real Append path and read it back through the real
// ParseLedger, and the trailer is still on the line. This is what "gateway-usage.jsonl exit
// rows carry the compaction-economics trailer" means for a fleet reader.
func TestTrailerSurvivesTheLedgerRoundTrip(t *testing.T) {
	path := t.TempDir() + "/gateway-usage.jsonl"
	c := Counters{CompactionFired: 4, CompactionShedTokens: 8000, CompactionCacheReadTokens: 8000}
	if err := Append(path, NewRow("exit", "guard", "claude", "sess-rt", time.Minute, nil, c, time.Unix(0, 0))); err != nil {
		t.Fatalf("Append: %v", err)
	}
	rows := ReadLedgerFile(path)
	if len(rows) != 1 {
		t.Fatalf("want 1 row back, got %d", len(rows))
	}
	tr := rows[0].CompactionEconomics
	if tr == nil {
		t.Fatal("the trailer did not survive the ledger round trip")
	}
	if tr.Fires != 4 || tr.ShedTokens != 8000 || tr.ObservedCacheReadTokens != 8000 {
		t.Fatalf("trailer figures changed across the round trip: %+v", tr)
	}
	if !close(tr.NetTokenEquiv, 8000*cacheprice.ReadMultiplier) {
		t.Fatalf("net = %v, want %v (a wholly warm shed nets only the read marginal)", tr.NetTokenEquiv, 8000*cacheprice.ReadMultiplier)
	}
}
