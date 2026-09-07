package kernel

import (
	"context"
	"fmt"
	"math/rand"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"testing/quick"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// testVerdictAuditOnly is a registered open-range verdict kind (> 1023) used to verify
// that open-range extension rungs participate in the restrictiveness lattice fold with
// the same order-independence and algebraic properties as closed core kinds.
const testVerdictAuditOnly abi.VerdictKind = 1050

var auditOnlyMu sync.Mutex

// ensureAuditOnlyRegistered registers testVerdictAuditOnly with fold rank 5 (sitting between
// Allow=0 and Transform=20 in the restrictiveness lattice) if not already registered.
func ensureAuditOnlyRegistered() {
	auditOnlyMu.Lock()
	defer auditOnlyMu.Unlock()
	if abi.FoldRank(testVerdictAuditOnly) == 5 {
		return
	}
	defer func() {
		_ = recover() // safe if already registered in snapshot
	}()
	abi.RegisterVerdictKind(testVerdictAuditOnly, "AuditOnly", 5, abi.FallbackDeny)
}

func init() {
	ensureAuditOnlyRegistered()
}

// formatVerdictKind returns a human-readable name for any verdict kind (core or test-registered).
func formatVerdictKind(k abi.VerdictKind) string {
	if k == testVerdictAuditOnly {
		return "AuditOnly"
	}
	return kindName(k)
}

// formatChain formats a slice of verdict kinds for diagnostic messages.
func formatChain(kinds []abi.VerdictKind) string {
	names := make([]string, len(kinds))
	for i, k := range kinds {
		names[i] = formatVerdictKind(k)
	}
	return fmt.Sprintf("%v", names)
}

// factorial computes n! for small n.
func factorial(n int) int {
	if n <= 1 {
		return 1
	}
	res := 1
	for i := 2; i <= n; i++ {
		res *= i
	}
	return res
}

// permutations generates all n! permutations of a slice deterministically using Heap's algorithm.
func permutations[T any](items []T) [][]T {
	if len(items) == 0 {
		return [][]T{nil}
	}
	arr := make([]T, len(items))
	copy(arr, items)

	var result [][]T
	emit := func() {
		cp := make([]T, len(arr))
		copy(cp, arr)
		result = append(result, cp)
	}
	emit()

	c := make([]int, len(arr))
	i := 0
	for i < len(arr) {
		if c[i] < i {
			if i%2 == 0 {
				arr[0], arr[i] = arr[i], arr[0]
			} else {
				arr[c[i]], arr[i] = arr[i], arr[c[i]]
			}
			emit()
			c[i]++
			i = 0
		} else {
			c[i] = 0
			i++
		}
	}
	return result
}

// makeTestVerdict creates an abi.Verdict with appropriate payload and By attribution.
func makeTestVerdict(k abi.VerdictKind, idx int) abi.Verdict {
	v := abi.Verdict{
		Kind: k,
		By:   fmt.Sprintf("rung-%d-%s", idx, formatVerdictKind(k)),
	}
	switch k {
	case abi.VerdictDeny:
		v.Reason = abi.ReasonPolicyBlock
	case abi.VerdictTransform:
		v.Payload = abi.TransformPayload{
			NewArgs: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"redacted":true}`)},
		}
	case abi.VerdictQuarantine:
		v.Payload = abi.QuarantinePayload{PageOut: true}
	case abi.VerdictRequireWitness:
		v.Payload = abi.WitnessPayload{Claim: "git:clean"}
	}
	return v
}

// foldKinds executes Fold on a sequence of verdict kinds.
func foldKinds(ctx context.Context, kinds []abi.VerdictKind) abi.Verdict {
	chain := make([]abi.Adjudicator, len(kinds))
	for i, k := range kinds {
		chain[i] = fakeAdj{v: makeTestVerdict(k, i)}
	}
	return Fold(ctx, chain, call("test_tool", "{}"))
}

// expectedFoldKind is the reference specification for the restrictiveness lattice fold:
//   - An empty chain fails closed to VerdictDeny.
//   - The conclusive verdict with the highest FoldRank wins (forbid-over-permit).
//   - If no conclusive verdicts are present: residual Indeterminate or all-Defer
//     fails closed to VerdictDeny (under default posture).
func expectedFoldKind(kinds []abi.VerdictKind) abi.VerdictKind {
	if len(kinds) == 0 {
		return abi.VerdictDeny
	}
	bestRank := -1
	var bestKind abi.VerdictKind
	sawConclusive := false
	for _, k := range kinds {
		if k == abi.VerdictDefer || k == abi.VerdictIndeterminate {
			continue
		}
		r := abi.FoldRank(k)
		if r > bestRank {
			bestRank = r
			bestKind = k
			sawConclusive = true
		}
	}
	if sawConclusive {
		return bestKind
	}
	return abi.VerdictDeny
}

// TestFoldProperty_Permutations verifies the core safety invariant promised in
// decide.go:18 ("the kernel folds the chain by the restrictiveness lattice, so order
// does not change the verdict Kind") across all permutations of chains of length 2, 3, 4, 5.
func TestFoldProperty_Permutations(t *testing.T) {
	ensureAuditOnlyRegistered()
	ctx := context.Background()

	cases := []struct {
		name     string
		chain    []abi.VerdictKind
		wantKind abi.VerdictKind
	}{
		// Length 2: Permutations of [Allow, Deny] and diverse pairs
		{
			name:     "len2_allow_deny",
			chain:    []abi.VerdictKind{abi.VerdictAllow, abi.VerdictDeny},
			wantKind: abi.VerdictDeny,
		},
		{
			name:     "len2_allow_quarantine",
			chain:    []abi.VerdictKind{abi.VerdictAllow, abi.VerdictQuarantine},
			wantKind: abi.VerdictQuarantine,
		},
		{
			name:     "len2_auditonly_allow",
			chain:    []abi.VerdictKind{testVerdictAuditOnly, abi.VerdictAllow},
			wantKind: testVerdictAuditOnly,
		},
		{
			name:     "len2_defer_allow",
			chain:    []abi.VerdictKind{abi.VerdictDefer, abi.VerdictAllow},
			wantKind: abi.VerdictAllow,
		},
		{
			name:     "len2_indeterminate_defer",
			chain:    []abi.VerdictKind{abi.VerdictIndeterminate, abi.VerdictDefer},
			wantKind: abi.VerdictDeny,
		},

		// Length 3: Permutations of diverse triples
		{
			name:     "len3_allow_quarantine_deny",
			chain:    []abi.VerdictKind{abi.VerdictAllow, abi.VerdictQuarantine, abi.VerdictDeny},
			wantKind: abi.VerdictDeny,
		},
		{
			name:     "len3_transform_allow_quarantine",
			chain:    []abi.VerdictKind{abi.VerdictTransform, abi.VerdictAllow, abi.VerdictQuarantine},
			wantKind: abi.VerdictQuarantine,
		},
		{
			name:     "len3_auditonly_allow_quarantine",
			chain:    []abi.VerdictKind{testVerdictAuditOnly, abi.VerdictAllow, abi.VerdictQuarantine},
			wantKind: abi.VerdictQuarantine,
		},
		{
			name:     "len3_defer_defer_defer",
			chain:    []abi.VerdictKind{abi.VerdictDefer, abi.VerdictDefer, abi.VerdictDefer},
			wantKind: abi.VerdictDeny,
		},

		// Length 4: Permutations of diverse 4-element chains
		{
			name:     "len4_auditonly_allow_deny_quarantine",
			chain:    []abi.VerdictKind{testVerdictAuditOnly, abi.VerdictAllow, abi.VerdictDeny, abi.VerdictQuarantine},
			wantKind: abi.VerdictDeny,
		},
		{
			name:     "len4_transform_allow_witness_quarantine",
			chain:    []abi.VerdictKind{abi.VerdictTransform, abi.VerdictAllow, abi.VerdictRequireWitness, abi.VerdictQuarantine},
			wantKind: abi.VerdictRequireWitness,
		},
		{
			name:     "len4_defer_allow_defer_deny",
			chain:    []abi.VerdictKind{abi.VerdictDefer, abi.VerdictAllow, abi.VerdictDefer, abi.VerdictDeny},
			wantKind: abi.VerdictDeny,
		},

		// Length 2, 3, 4, 5: Multiple duplicate verdict kinds
		{
			name:     "dup_len3_allow_allow_deny",
			chain:    []abi.VerdictKind{abi.VerdictAllow, abi.VerdictAllow, abi.VerdictDeny},
			wantKind: abi.VerdictDeny,
		},
		{
			name:     "dup_len3_deny_transform_deny",
			chain:    []abi.VerdictKind{abi.VerdictDeny, abi.VerdictTransform, abi.VerdictDeny},
			wantKind: abi.VerdictDeny,
		},
		{
			name:     "dup_len4_auditonly_auditonly_allow_deny",
			chain:    []abi.VerdictKind{testVerdictAuditOnly, testVerdictAuditOnly, abi.VerdictAllow, abi.VerdictDeny},
			wantKind: abi.VerdictDeny,
		},
		{
			name: "dup_len5_all_allow",
			chain: []abi.VerdictKind{
				abi.VerdictAllow, abi.VerdictAllow, abi.VerdictAllow, abi.VerdictAllow, abi.VerdictAllow,
			},
			wantKind: abi.VerdictAllow,
		},
		{
			name: "dup_len5_all_deny",
			chain: []abi.VerdictKind{
				abi.VerdictDeny, abi.VerdictDeny, abi.VerdictDeny, abi.VerdictDeny, abi.VerdictDeny,
			},
			wantKind: abi.VerdictDeny,
		},
		{
			name: "dup_len5_transform_transform_allow_quarantine_transform",
			chain: []abi.VerdictKind{
				abi.VerdictTransform, abi.VerdictTransform, abi.VerdictAllow, abi.VerdictQuarantine, abi.VerdictTransform,
			},
			wantKind: abi.VerdictQuarantine,
		},
		{
			name: "len5_allow_transform_quarantine_witness_deny",
			chain: []abi.VerdictKind{
				abi.VerdictAllow, abi.VerdictTransform, abi.VerdictQuarantine, abi.VerdictRequireWitness, abi.VerdictDeny,
			},
			wantKind: abi.VerdictDeny,
		},
		{
			name: "dup_len5_indeterminate_allow_indeterminate_quarantine_deny",
			chain: []abi.VerdictKind{
				abi.VerdictIndeterminate, abi.VerdictAllow, abi.VerdictIndeterminate, abi.VerdictQuarantine, abi.VerdictDeny,
			},
			wantKind: abi.VerdictDeny,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			perms := permutations(tc.chain)
			wantCount := factorial(len(tc.chain))
			if len(perms) != wantCount {
				t.Fatalf("expected %d permutations, generated %d", wantCount, len(perms))
			}

			// Invariant check: EVERY permutation must produce the identical verdict Kind,
			// and that Kind must match the expected restrictiveness lattice meet.
			for idx, p := range perms {
				v := foldKinds(ctx, p)
				if v.Kind != tc.wantKind {
					t.Fatalf("perm #%d %s: got Kind=%s (%d), want %s (%d)",
						idx, formatChain(p),
						formatVerdictKind(v.Kind), v.Kind,
						formatVerdictKind(tc.wantKind), tc.wantKind,
					)
				}
			}
		})
	}
}

// TestFoldProperty_AlgebraicLattice checks the algebraic lattice laws on VerdictKind:
//   - Idempotence: Fold([v, v]).Kind == Fold([v]).Kind
//   - Commutativity: Fold([v1, v2]).Kind == Fold([v2, v1]).Kind
//   - Associativity: Fold([v1, Fold([v2, v3])]).Kind == Fold([Fold([v1, v2]), v3]).Kind
func TestFoldProperty_AlgebraicLattice(t *testing.T) {
	ensureAuditOnlyRegistered()
	ctx := context.Background()

	palette := []abi.VerdictKind{
		abi.VerdictAllow,
		abi.VerdictDeny,
		abi.VerdictTransform,
		abi.VerdictQuarantine,
		abi.VerdictRequireWitness,
		abi.VerdictDefer,
		abi.VerdictIndeterminate,
		testVerdictAuditOnly,
	}

	t.Run("Idempotence", func(t *testing.T) {
		for _, k := range palette {
			v1 := foldKinds(ctx, []abi.VerdictKind{k})
			v2 := foldKinds(ctx, []abi.VerdictKind{k, k})
			v3 := foldKinds(ctx, []abi.VerdictKind{k, k, k})
			if v1.Kind != v2.Kind || v2.Kind != v3.Kind {
				t.Fatalf("idempotence violated for %s: 1-fold=%s, 2-fold=%s, 3-fold=%s",
					formatVerdictKind(k),
					formatVerdictKind(v1.Kind),
					formatVerdictKind(v2.Kind),
					formatVerdictKind(v3.Kind),
				)
			}
		}
	})

	t.Run("Commutativity", func(t *testing.T) {
		for _, k1 := range palette {
			for _, k2 := range palette {
				v12 := foldKinds(ctx, []abi.VerdictKind{k1, k2})
				v21 := foldKinds(ctx, []abi.VerdictKind{k2, k1})
				if v12.Kind != v21.Kind {
					t.Fatalf("commutativity violated for (%s, %s): [%s, %s]=%s != [%s, %s]=%s",
						formatVerdictKind(k1), formatVerdictKind(k2),
						formatVerdictKind(k1), formatVerdictKind(k2), formatVerdictKind(v12.Kind),
						formatVerdictKind(k2), formatVerdictKind(k1), formatVerdictKind(v21.Kind),
					)
				}
			}
		}
	})

	t.Run("Associativity", func(t *testing.T) {
		// Associativity is evaluated over the conclusive verdict sublattice:
		// (Allow, AuditOnly, Transform, Quarantine, RequireWitness, Deny).
		conclusiveKinds := []abi.VerdictKind{
			abi.VerdictAllow,
			testVerdictAuditOnly,
			abi.VerdictTransform,
			abi.VerdictQuarantine,
			abi.VerdictRequireWitness,
			abi.VerdictDeny,
		}

		foldBinary := func(k1, k2 abi.VerdictKind) abi.VerdictKind {
			return foldKinds(ctx, []abi.VerdictKind{k1, k2}).Kind
		}

		for _, k1 := range conclusiveKinds {
			for _, k2 := range conclusiveKinds {
				for _, k3 := range conclusiveKinds {
					left := foldBinary(k1, foldBinary(k2, k3))
					right := foldBinary(foldBinary(k1, k2), k3)
					flat := foldKinds(ctx, []abi.VerdictKind{k1, k2, k3}).Kind

					if left != right {
						t.Fatalf("associativity violated for (%s, %s, %s): left=%s, right=%s",
							formatVerdictKind(k1), formatVerdictKind(k2), formatVerdictKind(k3),
							formatVerdictKind(left), formatVerdictKind(right),
						)
					}
					if left != flat {
						t.Fatalf("binary fold does not match flat 3-chain for (%s, %s, %s): binary=%s, flat=%s",
							formatVerdictKind(k1), formatVerdictKind(k2), formatVerdictKind(k3),
							formatVerdictKind(left), formatVerdictKind(flat),
						)
					}
				}
			}
		}
	})
}

// TestFoldProperty_EarlyBreakAndTieBreaking examines the operational difference between
// the invariant verdict Kind vs the order-dependent By attribution:
//  1. Early Break on max fold rank: adjudicators after the first VerdictDeny (rank 100)
//     are short-circuited and NEVER executed.
//  2. Tie-Breaking: when two rungs share the highest rank, the FIRST rung in execution
//     order wins the By / witness attribution, documenting why By is order-dependent
//     even while Kind is mathematically invariant.
//  3. Unique winner: when the highest rank is strictly greater than all others, By
//     is ALSO order-independent.
func TestFoldProperty_EarlyBreakAndTieBreaking(t *testing.T) {
	ctx := context.Background()
	c := call("test_tool", "{}")

	t.Run("MaxFoldRankShortCircuit", func(t *testing.T) {
		var c0, c1, c2, c3 int32
		chain := []abi.Adjudicator{
			spyAdj{v: abi.Verdict{Kind: abi.VerdictAllow, By: "rung0"}, n: &c0},
			spyAdj{v: abi.Verdict{Kind: abi.VerdictDeny, By: "rung1"}, n: &c1},
			spyAdj{v: abi.Verdict{Kind: abi.VerdictAllow, By: "rung2"}, n: &c2},
			spyAdj{v: abi.Verdict{Kind: abi.VerdictQuarantine, By: "rung3"}, n: &c3},
		}

		v := Fold(ctx, chain, c)
		if v.Kind != abi.VerdictDeny {
			t.Fatalf("Fold kind=%v, want VerdictDeny", v.Kind)
		}
		if v.By != "rung1" {
			t.Fatalf("winning By=%q, want 'rung1'", v.By)
		}

		// Rungs 0 and 1 must have run; rungs 2 and 3 must have been elided by short-circuit.
		if atomic.LoadInt32(&c0) != 1 || atomic.LoadInt32(&c1) != 1 {
			t.Fatalf("expected rungs 0 and 1 to execute once, got c0=%d, c1=%d", c0, c1)
		}
		if atomic.LoadInt32(&c2) != 0 || atomic.LoadInt32(&c3) != 0 {
			t.Fatalf("expected tail rungs 2 and 3 to be elided, got c2=%d, c3=%d", c2, c3)
		}
	})

	t.Run("TieBreaking_MaxRank_FirstWins", func(t *testing.T) {
		adjAlpha := fakeAdj{abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "gate-alpha"}}
		adjBeta := fakeAdj{abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonTrustViolation, By: "gate-beta"}}

		// Alpha first
		vAlphaBeta := Fold(ctx, []abi.Adjudicator{adjAlpha, adjBeta}, c)
		if vAlphaBeta.Kind != abi.VerdictDeny || vAlphaBeta.By != "gate-alpha" || vAlphaBeta.Reason != abi.ReasonPolicyBlock {
			t.Fatalf("alpha-first tie break: got Kind=%v By=%q Reason=%v", vAlphaBeta.Kind, vAlphaBeta.By, vAlphaBeta.Reason)
		}

		// Beta first
		vBetaAlpha := Fold(ctx, []abi.Adjudicator{adjBeta, adjAlpha}, c)
		if vBetaAlpha.Kind != abi.VerdictDeny || vBetaAlpha.By != "gate-beta" || vBetaAlpha.Reason != abi.ReasonTrustViolation {
			t.Fatalf("beta-first tie break: got Kind=%v By=%q Reason=%v", vBetaAlpha.Kind, vBetaAlpha.By, vBetaAlpha.Reason)
		}

		// Kind is invariant; By is order-dependent (first-wins)
		if vAlphaBeta.Kind != vBetaAlpha.Kind {
			t.Fatalf("verdict Kind must be invariant across tie permutations")
		}
		if vAlphaBeta.By == vBetaAlpha.By {
			t.Fatalf("verdict By is expected to reflect first-wins tie breaking")
		}
	})

	t.Run("TieBreaking_NonMaxRank_FirstWins", func(t *testing.T) {
		adjAlpha := fakeAdj{abi.Verdict{Kind: abi.VerdictQuarantine, By: "quarantine-alpha"}}
		adjBeta := fakeAdj{abi.Verdict{Kind: abi.VerdictQuarantine, By: "quarantine-beta"}}

		vAB := Fold(ctx, []abi.Adjudicator{adjAlpha, adjBeta}, c)
		vBA := Fold(ctx, []abi.Adjudicator{adjBeta, adjAlpha}, c)

		if vAB.Kind != abi.VerdictQuarantine || vBA.Kind != abi.VerdictQuarantine {
			t.Fatalf("non-max tie break: both must have Kind=VerdictQuarantine")
		}
		if vAB.By != "quarantine-alpha" || vBA.By != "quarantine-beta" {
			t.Fatalf("non-max tie break must exhibit first-wins: vAB.By=%q, vBA.By=%q", vAB.By, vBA.By)
		}
	})

	t.Run("UniqueWinner_ByIsOrderIndependent", func(t *testing.T) {
		adjDeny := fakeAdj{abi.Verdict{Kind: abi.VerdictDeny, By: "sole-gate"}}
		adjAllow := fakeAdj{abi.Verdict{Kind: abi.VerdictAllow, By: "permissive-monitor"}}

		v1 := Fold(ctx, []abi.Adjudicator{adjDeny, adjAllow}, c)
		v2 := Fold(ctx, []abi.Adjudicator{adjAllow, adjDeny}, c)

		if v1.Kind != abi.VerdictDeny || v2.Kind != abi.VerdictDeny {
			t.Fatalf("Kind must be Deny")
		}
		if v1.By != "sole-gate" || v2.By != "sole-gate" {
			t.Fatalf("with unique max rank, By is invariant: v1.By=%q, v2.By=%q", v1.By, v2.By)
		}
	})
}

// TestFoldProperty_RandomizedChains runs pseudo-random property tests over hundreds of
// generated verdict chains of varying lengths with a fixed seed for determinism.
func TestFoldProperty_RandomizedChains(t *testing.T) {
	ensureAuditOnlyRegistered()
	ctx := context.Background()

	palette := []abi.VerdictKind{
		abi.VerdictAllow,
		abi.VerdictDeny,
		abi.VerdictTransform,
		abi.VerdictQuarantine,
		abi.VerdictRequireWitness,
		abi.VerdictDefer,
		abi.VerdictIndeterminate,
		testVerdictAuditOnly,
	}

	rng := rand.New(rand.NewSource(20260906))
	const numChains = 500

	for i := 0; i < numChains; i++ {
		chainLen := rng.Intn(6) + 1 // lengths 1 through 6
		chain := make([]abi.VerdictKind, chainLen)
		for j := range chain {
			chain[j] = palette[rng.Intn(len(palette))]
		}

		expected := expectedFoldKind(chain)
		base := foldKinds(ctx, chain)
		if base.Kind != expected {
			t.Fatalf("chain %d %s: got %s, want expected %s",
				i, formatChain(chain), formatVerdictKind(base.Kind), formatVerdictKind(expected))
		}

		// Test permutations of the chain
		if chainLen <= 5 {
			// Test all n! permutations exhaustively
			for pIdx, p := range permutations(chain) {
				got := foldKinds(ctx, p)
				if got.Kind != expected {
					t.Fatalf("chain %d perm %d %s: got %s, want %s",
						i, pIdx, formatChain(p), formatVerdictKind(got.Kind), formatVerdictKind(expected))
				}
			}
		} else {
			// Test 30 random shuffles for longer chains
			perm := make([]abi.VerdictKind, len(chain))
			for s := 0; s < 30; s++ {
				copy(perm, chain)
				rng.Shuffle(len(perm), func(a, b int) {
					perm[a], perm[b] = perm[b], perm[a]
				})
				got := foldKinds(ctx, perm)
				if got.Kind != expected {
					t.Fatalf("chain %d shuffle %d %s: got %s, want %s",
						i, s, formatChain(perm), formatVerdictKind(got.Kind), formatVerdictKind(expected))
				}
			}
		}
	}
}

// quickVerdictChain represents a randomized verdict chain for testing/quick.
type quickVerdictChain []abi.VerdictKind

func (quickVerdictChain) Generate(r *rand.Rand, size int) reflect.Value {
	ensureAuditOnlyRegistered()
	palette := []abi.VerdictKind{
		abi.VerdictAllow,
		abi.VerdictDeny,
		abi.VerdictTransform,
		abi.VerdictQuarantine,
		abi.VerdictRequireWitness,
		abi.VerdictDefer,
		abi.VerdictIndeterminate,
		testVerdictAuditOnly,
	}
	n := r.Intn(5) + 1
	chain := make(quickVerdictChain, n)
	for i := range chain {
		chain[i] = palette[r.Intn(len(palette))]
	}
	return reflect.ValueOf(chain)
}

// TestFoldProperty_QuickCheck uses Go's testing/quick package to check the permutation
// invariance and lattice-meet property over arbitrary generated chains.
func TestFoldProperty_QuickCheck(t *testing.T) {
	ensureAuditOnlyRegistered()
	ctx := context.Background()

	prop := func(chain quickVerdictChain) bool {
		if len(chain) == 0 {
			return true
		}
		expected := expectedFoldKind([]abi.VerdictKind(chain))

		// Random permutation
		perm := make([]abi.VerdictKind, len(chain))
		copy(perm, chain)
		r := rand.New(rand.NewSource(int64(len(chain) * 31)))
		r.Shuffle(len(perm), func(i, j int) {
			perm[i], perm[j] = perm[j], perm[i]
		})

		vOrig := foldKinds(ctx, []abi.VerdictKind(chain))
		vPerm := foldKinds(ctx, perm)

		return vOrig.Kind == expected && vPerm.Kind == expected
	}

	cfg := &quick.Config{
		MaxCount: 500,
		Rand:     rand.New(rand.NewSource(1337)),
	}

	if err := quick.Check(prop, cfg); err != nil {
		t.Fatalf("quick.Check failed: %v", err)
	}
}
