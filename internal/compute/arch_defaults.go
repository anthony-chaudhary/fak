package compute

import (
	"fmt"
	"strings"
)

// ArchDefaultEquivalence is what the cited run proved about the output when
// the default is on versus off. Bitwise needs no further owner decision; the
// other classes change bits, so they stay recorded rather than implied.
type ArchDefaultEquivalence string

const (
	// ArchDefaultBitwise means the run showed identical logits with the default on and off.
	ArchDefaultBitwise ArchDefaultEquivalence = "bitwise"
	// ArchDefaultKLChecked means the run bounded the logit change with a KL gate.
	ArchDefaultKLChecked ArchDefaultEquivalence = "kl_checked"
	// ArchDefaultGreedyIdentical means only greedy token identity against an
	// oracle was shown; a KL receipt is still owed.
	ArchDefaultGreedyIdentical ArchDefaultEquivalence = "greedy_identical"
)

// ArchDefault is one measured default: the variable a user would set, the
// value applied when they have not, and the run that earned it.
type ArchDefault struct {
	Env         string                 `json:"env"`
	Value       string                 `json:"value"`
	Evidence    string                 `json:"evidence"`
	Equivalence ArchDefaultEquivalence `json:"equivalence"`
}

// ArchDefaultTable holds the defaults for one compute capability (120 for
// sm_120). Setting KillSwitch to "0" drops the whole table for that arch, so a
// regression on one card family is reverted without touching any other.
type ArchDefaultTable struct {
	SM         int           `json:"sm"`
	KillSwitch string        `json:"kill_switch"`
	Defaults   []ArchDefault `json:"defaults"`
}

const sm120InklingResidentReceipt = "docs/benchmarks/aegis-inkling2-resident-cuda-20261123/perf-receipt-v2.json"

// archDefaultTables is the evidence-gated per-arch table. An arch with no row
// gets no defaults: a measurement on one card says nothing about another.
var archDefaultTables = []ArchDefaultTable{
	{
		SM:         120,
		KillSwitch: "FAK_SM120_DEFAULTS",
		Defaults: []ArchDefault{
			// RTX 5080, Inkling-2-Mini UD-Q3_K_XL: 6.31 -> 82.55 decode tok/s ABBA,
			// 64/64 greedy tokens identical to the llama.cpp CPU oracle.
			{Env: "FAK_INKLING2_RESIDENT", Value: "1", Evidence: sm120InklingResidentReceipt, Equivalence: ArchDefaultGreedyIdentical},
			// Same receipt: graph replay of the resident forward, climb step 3 onward.
			{Env: "FAK_INKLING2_GRAPH", Value: "1", Evidence: sm120InklingResidentReceipt, Equivalence: ArchDefaultGreedyIdentical},
		},
	},
}

// ArchDefaultSource says where a resolved value came from.
type ArchDefaultSource string

const (
	ArchDefaultFromUser   ArchDefaultSource = "user"
	ArchDefaultFromTable  ArchDefaultSource = "arch_default"
	ArchDefaultFromKilled ArchDefaultSource = "kill_switch"
	ArchDefaultFromNone   ArchDefaultSource = "none"
)

// ArchDefaultResolution is one variable's resolved value and its provenance.
// Value is empty when the source is kill_switch or none; the caller then keeps
// its own built-in behavior.
type ArchDefaultResolution struct {
	Env         string                 `json:"env"`
	Value       string                 `json:"value"`
	Source      ArchDefaultSource      `json:"source"`
	Evidence    string                 `json:"evidence,omitempty"`
	Equivalence ArchDefaultEquivalence `json:"equivalence,omitempty"`
}

// ArchDefaultTableFor returns the table for compute capability sm.
func ArchDefaultTableFor(sm int) (ArchDefaultTable, bool) {
	for _, t := range archDefaultTables {
		if t.SM == sm {
			return t, true
		}
	}
	return ArchDefaultTable{}, false
}

// ResolveArchDefault resolves env on a device of compute capability sm. A
// variable the user set always wins, even to an empty-looking value such as
// "0"; then the arch kill switch; then the table.
func ResolveArchDefault(sm int, env string, lookup func(string) (string, bool)) ArchDefaultResolution {
	if v, ok := lookup(env); ok {
		return ArchDefaultResolution{Env: env, Value: v, Source: ArchDefaultFromUser}
	}
	t, ok := ArchDefaultTableFor(sm)
	if !ok {
		return ArchDefaultResolution{Env: env, Source: ArchDefaultFromNone}
	}
	for _, d := range t.Defaults {
		if d.Env != env {
			continue
		}
		if v, set := lookup(t.KillSwitch); set && strings.TrimSpace(v) == "0" {
			return ArchDefaultResolution{Env: env, Source: ArchDefaultFromKilled, Evidence: d.Evidence, Equivalence: d.Equivalence}
		}
		return ArchDefaultResolution{Env: env, Value: d.Value, Source: ArchDefaultFromTable, Evidence: d.Evidence, Equivalence: d.Equivalence}
	}
	return ArchDefaultResolution{Env: env, Source: ArchDefaultFromNone}
}

// validateArchDefaultTables refuses a row with no evidence, an unknown
// equivalence class, a duplicate arch or variable, or a missing kill switch.
func validateArchDefaultTables(tables []ArchDefaultTable) error {
	seenSM := map[int]bool{}
	for _, t := range tables {
		if t.SM <= 0 || seenSM[t.SM] {
			return fmt.Errorf("arch default table sm_%d: missing or duplicate arch", t.SM)
		}
		seenSM[t.SM] = true
		if t.KillSwitch == "" {
			return fmt.Errorf("arch default table sm_%d: no kill switch", t.SM)
		}
		seenEnv := map[string]bool{}
		for _, d := range t.Defaults {
			switch {
			case d.Env == "" || d.Env == t.KillSwitch || seenEnv[d.Env]:
				return fmt.Errorf("arch default sm_%d %q: missing, duplicate or kill-switch variable", t.SM, d.Env)
			case !strings.HasPrefix(d.Evidence, "docs/benchmarks/"):
				return fmt.Errorf("arch default sm_%d %s: evidence %q does not cite a benchmark run", t.SM, d.Env, d.Evidence)
			}
			switch d.Equivalence {
			case ArchDefaultBitwise, ArchDefaultKLChecked, ArchDefaultGreedyIdentical:
			default:
				return fmt.Errorf("arch default sm_%d %s: unknown equivalence %q", t.SM, d.Env, d.Equivalence)
			}
			seenEnv[d.Env] = true
		}
	}
	return nil
}
