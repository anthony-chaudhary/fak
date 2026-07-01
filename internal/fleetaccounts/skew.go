package fleetaccounts

import (
	"fmt"
	"sort"
)

// Account-pool dispatch-load skew: the passive warning that fires when one rate-limit
// pool is carrying disproportionately many live workers relative to its healthy peers —
// the early signal that a fan-out is piling onto one account and will hit a hard usage
// throttle or auth wall there while other pools sit idle. This is the soft, pre-emptive
// counterpart to SeatPool.DoubleBooked (a single seat with >1 live worker, already an
// invariant violation): skew is spread ACROSS distinct pools, each still within its own
// per-seat bound, yet unevenly loaded.
//
// It is deliberately kept OUT of the SeatPool JSON envelope (that shape is byte-compatible
// with the Python `seats` output) — it is a derived, human-facing advisory over an already
// built pool.

// SkewPolicy tunes the account-pool skew detector. The zero value is not useful; call
// DefaultSkewPolicy.
type SkewPolicy struct {
	// MinLoad is the absolute live-worker count a pool must reach before it can be
	// flagged — below it, a "skew" is just fan-out noise, not a throttle risk.
	MinLoad int
	// Factor is how many times the peer-mean load a pool must exceed to be called
	// skewed (peer-mean is the mean load of the OTHER pools, floored at 1 so a pool
	// that has drawn every worker while peers sit at zero still trips).
	Factor float64
}

// DefaultSkewPolicy is the tuned default: warn once a pool holds >= 2 live workers AND
// carries at least 2x the mean load of its peers.
func DefaultSkewPolicy() SkewPolicy { return SkewPolicy{MinLoad: 2, Factor: 2.0} }

// PoolSkew is one flagged rate-limit pool carrying disproportionate dispatch load.
type PoolSkew struct {
	Pool     string  // the PoolKey (rate-limit pool the load draws on)
	Tag      string  // a representative worker tag for the pool
	Account  string  // a representative account dir for the pool
	Load     int     // live workers on this pool
	PeerMean float64 // mean live-worker load of the OTHER pools
	Ratio    float64 // Load / max(PeerMean, 1)
	Message  string  // rendered one-line advisory
}

type poolAgg struct {
	key     string
	tag     string
	account string
	load    int
}

// DetectSeatSkew folds a built SeatPool into the set of pools carrying disproportionate
// dispatch load, worst-first. Load is summed per distinct rate-limit pool (two dirs on
// one Anthropic account share one PoolKey and are aggregated). With fewer than two pools
// there are no peers to be skewed against, so it returns nil.
func DetectSeatSkew(pool SeatPool, sp SkewPolicy) []PoolSkew {
	if sp.MinLoad < 1 {
		sp.MinLoad = 1
	}
	if sp.Factor <= 0 {
		sp.Factor = DefaultSkewPolicy().Factor
	}

	byKey := map[string]*poolAgg{}
	var order []string
	for _, s := range pool.Seats {
		a, ok := byKey[s.Seat]
		if !ok {
			a = &poolAgg{key: s.Seat, tag: s.Tag, account: s.Account}
			byKey[s.Seat] = a
			order = append(order, s.Seat)
		}
		a.load += len(s.Workers)
	}
	if len(order) < 2 {
		return nil
	}

	total := 0
	for _, k := range order {
		total += byKey[k].load
	}

	var out []PoolSkew
	n := len(order)
	for _, k := range order {
		a := byKey[k]
		if a.load < sp.MinLoad {
			continue
		}
		peerMean := float64(total-a.load) / float64(n-1)
		threshold := peerMean
		if threshold < 1 {
			threshold = 1
		}
		if float64(a.load) < sp.Factor*threshold {
			continue
		}
		ratio := float64(a.load) / threshold
		w := PoolSkew{
			Pool: a.key, Tag: a.tag, Account: a.account, Load: a.load,
			PeerMean: peerMean, Ratio: ratio,
		}
		w.Message = fmt.Sprintf(
			"account %q (%s) carries %d live workers vs peer mean %.1f (%.1fx) — rebalance the fan-out before it throttles or auth-walls",
			w.Tag, w.Account, w.Load, w.PeerMean, w.Ratio)
		out = append(out, w)
	}

	// worst-first: highest load, then highest ratio, then tag for determinism.
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Load != out[j].Load {
			return out[i].Load > out[j].Load
		}
		if out[i].Ratio != out[j].Ratio {
			return out[i].Ratio > out[j].Ratio
		}
		return out[i].Tag < out[j].Tag
	})
	return out
}

// RenderSeatSkew renders the human account-pool skew advisory. It returns "" when nothing
// is skewed, so a caller can append it unconditionally without changing the balanced-pool
// output.
func RenderSeatSkew(warnings []PoolSkew) string {
	if len(warnings) == 0 {
		return ""
	}
	var b []byte
	b = append(b, "\nACCOUNT-POOL SKEW (one pool carrying disproportionate dispatch load -- rebalance before it throttles):\n"...)
	for _, w := range warnings {
		b = append(b, "  "...)
		b = append(b, w.Message...)
		b = append(b, '\n')
	}
	return string(b)
}
