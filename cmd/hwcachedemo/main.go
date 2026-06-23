// Command hwcachedemo is a no-model, no-GPU proof that fak's cache is hardware-aware:
// it shows the residency-tier ladder with each tier's physical character, which tiers
// can be shared ZERO-COPY, and — the headline — how the placement policy DEMOTES a hot
// KV prefix to CXL far memory under pressure instead of EVICTING it and paying a full
// re-prefill later.
//
// Everything here runs on the pure metadata/policy plane (internal/cachemeta): no
// tensors are moved and no hardware is touched. The decisions are deterministic (a
// logical millisecond clock is injected, never a wall clock), so the output is the
// same on every machine and in CI.
//
//	go run ./cmd/hwcachedemo
//
// What it demonstrates, in order:
//  1. the tier ladder HBM -> DRAM -> NUMA-far -> CXL -> Disk -> Remote, with latency,
//     bandwidth, capacity, attendable-in-place, and the zero-copy share kind;
//  2. a 4000-token hot prefix under escalating memory pressure relocating one tier at a
//     time (demote -> spill) rather than being dropped;
//  3. the cheap-span exception (a tiny span on a slow-only tier is cheaper to rebuild);
//  4. a head-to-head tally: tokens a blind LRU cache would have RE-PREFILLED vs the
//     tokens the tiered policy preserved by demoting.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

func main() {
	flag.Parse()
	if err := run(os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "hwcachedemo:", err)
		os.Exit(1)
	}
}

func run(w io.Writer) error {
	profiles := cachemeta.DefaultTierProfiles()
	printLadder(w, profiles)
	printZeroCopy(w, profiles)
	printPressureWalk(w, profiles)
	printCheapException(w, profiles)
	printLRUvsTiered(w, profiles)
	return nil
}

// ladder is the hot->cold order used for display.
var ladder = []cachemeta.ResidencyTier{
	cachemeta.TierHBM, cachemeta.TierDRAM, cachemeta.TierNUMAFar,
	cachemeta.TierCXL, cachemeta.TierDisk, cachemeta.TierRemote,
}

func printLadder(w io.Writer, profiles map[cachemeta.ResidencyTier]cachemeta.TierProfile) {
	fmt.Fprintln(w, "== Residency tier ladder (hot -> cold) ==")
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "tier\tlatency\tbandwidth\tcapacity\tattend-in-place\tzero-copy share")
	for _, t := range ladder {
		p := profiles[t]
		fmt.Fprintf(tw, "%s\t%dns\t%d MB/s\t%s\t%v\t%s\n",
			t, p.ReadLatencyNanos, p.BandwidthMBPerSec, humanBytes(p.CapacityBytes),
			p.AttendableInPlace(), shareLabel(p.Share))
	}
	tw.Flush()
	fmt.Fprintln(w)
}

func printZeroCopy(w io.Writer, profiles map[cachemeta.ResidencyTier]cachemeta.TierProfile) {
	fmt.Fprintln(w, "== Zero-copy sharing ==")
	for _, t := range ladder {
		p := profiles[t]
		kind := p.Share
		if kind.ZeroCopy() {
			fmt.Fprintf(w, "  %-9s shareable zero-copy via %s — reuse costs a handle, not a memcpy\n", t, kind)
		} else {
			fmt.Fprintf(w, "  %-9s copy-only — a cross-consumer reuse stages the bytes\n", t)
		}
	}
	fmt.Fprintln(w)
}

// printPressureWalk drives one hot 4000-token prefix down the ladder as each tier
// fills, printing the placement decision and the KV-transfer directive at each step.
func printPressureWalk(w io.Writer, profiles map[cachemeta.ResidencyTier]cachemeta.TierProfile) {
	fmt.Fprintln(w, "== A hot 4000-token prefix under escalating memory pressure ==")
	lc := cachemeta.NewLifecycle(cachemeta.TierHBM, 0).MarkResident(profiles, 0)
	req := cachemeta.PlacementRequest{
		Lifecycle:            lc,
		SizeBytes:            64 << 20, // 64 MB KV span
		Tokens:               4000,
		Profiles:             profiles,
		Policy:               cachemeta.LifecyclePolicy{DemoteOnExpiry: true},
		PerTokenPrefillNanos: 2_000_000, // 2 ms/token — expensive to rebuild
		NowMillis:            0,
	}
	// Fill tiers one at a time; each step the current tier is full, so the policy must
	// relocate the prefix one tier colder.
	pressure := cachemeta.TierPressure{}
	now := int64(0)
	for step := 0; step < 5; step++ {
		pressure[lc.Tier] = 1.0 // the tier the prefix sits in is now full
		req.Lifecycle, req.Pressure, req.NowMillis = lc, pressure, now
		d := cachemeta.PlanPlacement(req)
		fmt.Fprintf(w, "  step %d: %-7s %s -> %-9s [%s]  directive=%s\n",
			step, d.Action, d.FromTier, tierOrNone(d.ToTier), d.Reason, dirOrNone(d.Directive))
		if d.Action == cachemeta.ActionEvict {
			break
		}
		lc, _ = d.Apply(lc, profiles, now)
		now += 1000
	}
	fmt.Fprintln(w)
}

func printCheapException(w io.Writer, profiles map[cachemeta.ResidencyTier]cachemeta.TierProfile) {
	fmt.Fprintln(w, "== The cheap-span exception (hardware cost model, not blind retention) ==")
	lc := cachemeta.NewLifecycle(cachemeta.TierHBM, 0).MarkResident(profiles, 0)
	mk := func(pressure cachemeta.TierPressure) cachemeta.PlacementDecision {
		return cachemeta.PlanPlacement(cachemeta.PlacementRequest{
			Lifecycle:            lc,
			SizeBytes:            48 << 10, // small
			Tokens:               3,        // trivial to rebuild
			Profiles:             profiles,
			Pressure:             pressure,
			PerTokenPrefillNanos: 1000, // 1 us/token
			NowMillis:            0,
		})
	}
	fast := mk(cachemeta.TierPressure{cachemeta.TierHBM: 1.0})
	slow := mk(cachemeta.TierPressure{
		cachemeta.TierHBM: 1.0, cachemeta.TierDRAM: 1.0,
		cachemeta.TierNUMAFar: 1.0, cachemeta.TierCXL: 1.0,
	})
	fmt.Fprintf(w, "  fast tier free (DRAM): %s -> %s [%s]  (worth keeping)\n", fast.Action, tierOrNone(fast.ToTier), fast.Reason)
	fmt.Fprintf(w, "  only disk free:        %s -> %s [%s]  (read-back > rebuild, so drop it)\n", slow.Action, tierOrNone(slow.ToTier), slow.Reason)
	fmt.Fprintln(w)
}

// printLRUvsTiered tallies the headline: a workload of K turns that share one hot
// system prefix, served under a tiny HBM that can hold the prefix but fills under load.
// A blind LRU cache evicts the prefix and RE-PREFILLS it every time it is needed again;
// the tiered policy demotes it to CXL and stages it back. We count the prefill tokens
// each strategy spends.
func printLRUvsTiered(w io.Writer, profiles map[cachemeta.ResidencyTier]cachemeta.TierProfile) {
	fmt.Fprintln(w, "== Blind LRU vs hardware-aware tiering (8 turns sharing a 4000-token prefix) ==")
	const turns = 8
	const prefixTokens = 4000

	// Blind LRU: HBM is full each turn, so the shared prefix is evicted and recomputed
	// on every subsequent turn.
	lruRecomputed := 0
	for turn := 1; turn < turns; turn++ {
		lruRecomputed += prefixTokens // evicted last turn, re-prefilled this turn
	}

	// Hardware-aware: the prefix demotes to CXL (attendable in place) instead of being
	// evicted, so it is never recomputed; each reuse is a cheap stage-back, not a
	// re-prefill. We verify the policy actually chose demote (not evict).
	lc := cachemeta.NewLifecycle(cachemeta.TierHBM, 0).MarkResident(profiles, 0)
	d := cachemeta.PlanPlacement(cachemeta.PlacementRequest{
		Lifecycle:            lc,
		SizeBytes:            64 << 20,
		Tokens:               prefixTokens,
		Profiles:             profiles,
		Pressure:             cachemeta.TierPressure{cachemeta.TierHBM: 1.0},
		Policy:               cachemeta.LifecyclePolicy{DemoteOnExpiry: true},
		PerTokenPrefillNanos: 2_000_000,
		NowMillis:            0,
	})
	tieredRecomputed := 0
	if d.Action == cachemeta.ActionEvict {
		tieredRecomputed = lruRecomputed // (would only happen if no colder tier had room)
	}
	fmt.Fprintf(w, "  blind LRU:      re-prefilled %d tokens (evict+recompute every reuse)\n", lruRecomputed)
	fmt.Fprintf(w, "  tiered (fak):   re-prefilled %d tokens (demote to %s, stage back)\n", tieredRecomputed, d.ToTier)
	fmt.Fprintf(w, "  -> %d prefill tokens saved by demoting instead of evicting\n", lruRecomputed-tieredRecomputed)
	fmt.Fprintln(w)
}

func humanBytes(b int64) string {
	switch {
	case b == 0:
		return "unbounded"
	case b >= 1<<40:
		return fmt.Sprintf("%dTB", b>>40)
	case b >= 1<<30:
		return fmt.Sprintf("%dGB", b>>30)
	default:
		return fmt.Sprintf("%dB", b)
	}
}

func shareLabel(k cachemeta.ShareKind) string {
	if k.ZeroCopy() {
		return string(k)
	}
	return "copy"
}

func tierOrNone(t cachemeta.ResidencyTier) cachemeta.ResidencyTier {
	if t == "" {
		return "-"
	}
	return t
}

func dirOrNone(d cachemeta.KVTransferDirection) string {
	if d == "" {
		return "-"
	}
	return string(d)
}
