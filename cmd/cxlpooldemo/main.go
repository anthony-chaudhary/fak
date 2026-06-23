// Command cxlpooldemo is a no-model, no-GPU proof of the value a SWITCH-POOLED,
// multi-host shared memory tier (a CXL.mem / CXL-switch pool) adds to fak's
// hardware-aware cache once a KV cache is shared across a FLEET of tenants — the
// multi-tenant counterpart of cmd/hwcachedemo's single-stream demote-not-evict proof.
//
// It runs entirely on the pure metadata/policy plane (internal/cachemeta): no tensors
// are moved and no hardware is touched. Every number is a deterministic calculation
// over representative tier/pool profiles, so the output is identical on every machine
// and in CI. An operator overrides the profiles with their fabric's measured topology;
// the logic is unchanged.
//
//	go run ./cmd/cxlpooldemo
//
// What it demonstrates, in order:
//  1. the pooling character of each tier — how many hosts can attend it, whether it is
//     coherent, its zero-copy share kind, and whether ONE copy is fabric-shareable;
//  2. the three-way fleet economics for N tenants sharing one hot prefix: per-host
//     PRIVATE (N prefills, N copies) vs a copy-only pool (1 prefill, N copies) vs a
//     COHERENT CXL pool (1 prefill, 1 copy) — the only regime that saves on both axes;
//  3. the cross-tenant trust gate: a pooled cell is aliased by another tenant ONLY when
//     it is trusted, fleet-shareable, and key-matched — a poisoned, private, or
//     wrong-model cell is refused, so the dedup is honest, not a blind alias.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

func main() {
	flag.Parse()
	if err := run(os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "cxlpooldemo:", err)
		os.Exit(1)
	}
}

func run(w io.Writer) error {
	profiles := cachemeta.DefaultTierProfiles()
	pools := cachemeta.DefaultPoolProfiles()
	printTopology(w, pools)
	printFleetEconomics(w, profiles, pools)
	printTrustGate(w)
	fmt.Fprintln(w, "All numbers above are a deterministic cost model over representative")
	fmt.Fprintln(w, "tier/pool profiles on the metadata plane — no tensors moved, no hardware")
	fmt.Fprintln(w, "touched. The placement plane decides what a pooled tier is WORTH and WHO")
	fmt.Fprintln(w, "may reuse a cell in it; an engine adapter maps the physical CXL region.")
	return nil
}

// ladder is the hot->cold display order.
var ladder = []cachemeta.ResidencyTier{
	cachemeta.TierHBM, cachemeta.TierDRAM, cachemeta.TierNUMAFar,
	cachemeta.TierCXL, cachemeta.TierDisk, cachemeta.TierRemote,
}

func printTopology(w io.Writer, pools map[cachemeta.ResidencyTier]cachemeta.PoolProfile) {
	fmt.Fprintln(w, "== Pool topology (who can attend each tier) ==")
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "tier\thosts\tcoherent\tzero-copy share\tone copy serves all?")
	for _, t := range ladder {
		p := pools[t]
		fmt.Fprintf(tw, "%s\t%d\t%v\t%s\t%s\n",
			t, p.Hosts, p.Coherent, shareLabel(p.Share), fabricVerdict(p))
	}
	tw.Flush()
	fmt.Fprintln(w, "  -> only a pooled, coherent, zero-copy tier (CXL here) lets ONE resident")
	fmt.Fprintln(w, "     copy be attended in place by every host in the pool.")
	fmt.Fprintln(w)
}

// printFleetEconomics drives one hot prefix wanted by N tenants through the three
// pooling regimes and tallies the savings on both axes.
func printFleetEconomics(w io.Writer, profiles map[cachemeta.ResidencyTier]cachemeta.TierProfile, pools map[cachemeta.ResidencyTier]cachemeta.PoolProfile) {
	const tenants = 8
	const tokens = 4000
	const bytes = 64 << 20 // 64 MB KV span for the shared prefix

	fmt.Fprintf(w, "== %d tenants share one hot %d-token system+tool prefix (%s KV) ==\n",
		tenants, tokens, humanBytes(bytes))

	regimes := []struct {
		name string
		tier cachemeta.ResidencyTier
	}{
		{"per-host private (DRAM)", cachemeta.TierDRAM},
		{"copy-only pool (remote)", cachemeta.TierRemote},
		{"coherent CXL pool", cachemeta.TierCXL},
	}
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "regime\tprefill tokens\tresident copies\tprefill saved\tmemory dedup")
	for _, rg := range regimes {
		r := cachemeta.PlanFleetReuse(cachemeta.FleetReuseRequest{
			Tenants:              tenants,
			Tokens:               tokens,
			SizeBytes:            bytes,
			PerTokenPrefillNanos: 2_000_000, // 2 ms/token — expensive to rebuild
			Profile:              profiles[rg.tier],
			Pool:                 pools[rg.tier],
		})
		fmt.Fprintf(tw, "%s\t%d\t%s\t%d\t%s\n",
			rg.name, r.PooledPrefillTokens, humanBytes(r.PooledResidentBytes),
			r.PrefillTokensSaved, humanBytes(r.BytesDeduplicated))
	}
	tw.Flush()

	// Headline: the coherent pool vs the per-host-private baseline.
	r := cachemeta.PlanFleetReuse(cachemeta.FleetReuseRequest{
		Tenants: tenants, Tokens: tokens, SizeBytes: bytes,
		PerTokenPrefillNanos: 2_000_000,
		Profile:              profiles[cachemeta.TierCXL],
		Pool:                 pools[cachemeta.TierCXL],
	})
	fmt.Fprintf(w, "  -> a coherent CXL pool turns %d prefills into %d and %s of copies into %s:\n",
		tenants, 1, humanBytes(r.PrivateResidentBytes), humanBytes(r.PooledResidentBytes))
	fmt.Fprintf(w, "     %d prefill tokens saved and %s of memory deduplicated across the fleet.\n",
		r.PrefillTokensSaved, humanBytes(r.BytesDeduplicated))
	fmt.Fprintln(w)
}

// printTrustGate shows that pooled dedup is GATED: only a trusted, fleet-shareable,
// key-matched cell is aliased across a tenant boundary.
func printTrustGate(w io.Writer) {
	fmt.Fprintln(w, "== Cross-tenant reuse gate (dedup is honest, not blind aliasing) ==")
	want := wantKey("qwen3")
	cases := []struct {
		desc  string
		cell  cachemeta.Entry
		match cachemeta.MaterializationKey
	}{
		{"same model, fleet-shared, trusted", pooledCell("qwen3", abi.TaintTrusted, abi.ScopeFleet), want},
		{"different model (KV is garbage)", pooledCell("qwen3", abi.TaintTrusted, abi.ScopeFleet), wantKey("llama4")},
		{"agent-private cell (not shareable)", pooledCell("qwen3", abi.TaintTrusted, abi.ScopeAgent), want},
		{"poisoned / quarantined cell", pooledCell("qwen3", abi.TaintQuarantined, abi.ScopeFleet), want},
	}
	for _, c := range cases {
		v := cachemeta.PoolReuseVerdict(c.cell, c.match)
		fmt.Fprintf(w, "  %-36s -> %-10s %-7s %s\n", c.desc, verdictLabel(v), reuseLabel(v), reasonOrOK(v))
	}
	fmt.Fprintln(w)
}

func pooledCell(model string, taint abi.TaintLabel, scope abi.ShareScope) cachemeta.Entry {
	e := cachemeta.FromKVPrefix(cachemeta.KVPrefix{
		TokenDigest: "sysprompt-v1",
		Length:      4000,
		ModelID:     model,
		TokenizerID: model + "-tok",
		Owner:       "tenant-a",
	},
		cachemeta.WithSerializer("ser-1"),
		cachemeta.WithPolicyVersion("pol-1"),
	)
	e.Derivation.PositionMode = cachemeta.PositionPrefixAligned
	e.Labels = map[string]string{
		"position_regime":  "rope-theta-1e6",
		"admitter_version": "adj-1",
	}
	e.Security.Taint = taint
	e.Security.Scope = scope
	e.Security.AdmittedBy = "adjudicator"
	e.Security.AdmissionVerdict = cachemeta.AdmissionAllow
	return e
}

func wantKey(model string) cachemeta.MaterializationKey {
	return cachemeta.MaterializationKey{
		ModelID:         model,
		TokenizerID:     model + "-tok",
		SerializerID:    "ser-1",
		PositionRegime:  "rope-theta-1e6",
		PolicyVersion:   "pol-1",
		AdmitterVersion: "adj-1",
	}
}

func fabricVerdict(p cachemeta.PoolProfile) string {
	switch {
	case p.FabricShareable():
		return "yes (one shared copy)"
	case p.Reachable():
		return "no (copy per host)"
	default:
		return "no (host-private)"
	}
}

func verdictLabel(v cachemeta.LookupVerdict) string {
	if v.CanServe() {
		return "REUSE"
	}
	return "REFUSE"
}

func reuseLabel(v cachemeta.LookupVerdict) string {
	return string(v.Kind)
}

func reasonOrOK(v cachemeta.LookupVerdict) string {
	if v.Reason == cachemeta.ReasonNone {
		return "key matched, trusted, shareable"
	}
	return string(v.Reason)
}

func shareLabel(k cachemeta.ShareKind) string {
	if k.ZeroCopy() {
		return string(k)
	}
	return "copy"
}

func humanBytes(b int64) string {
	switch {
	case b == 0:
		return "0"
	case b >= 1<<30:
		return fmt.Sprintf("%dGB", b>>30)
	case b >= 1<<20:
		return fmt.Sprintf("%dMB", b>>20)
	case b >= 1<<10:
		return fmt.Sprintf("%dKB", b>>10)
	default:
		return fmt.Sprintf("%dB", b)
	}
}
