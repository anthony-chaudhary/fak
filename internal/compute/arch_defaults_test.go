package compute

import "testing"

func lookupFrom(env map[string]string) func(string) (string, bool) {
	return func(k string) (string, bool) { v, ok := env[k]; return v, ok }
}

// fak-test:runtime fast est=5ms lane=default
func TestArchDefaultsUserEnvWinsAndCiteEvidence(t *testing.T) {
	if err := validateArchDefaultTables(archDefaultTables); err != nil {
		t.Fatal(err)
	}
	none := lookupFrom(nil)
	for _, env := range []string{"FAK_INKLING2_RESIDENT", "FAK_INKLING2_GRAPH"} {
		r := ResolveArchDefault(120, env, none)
		if r.Source != ArchDefaultFromTable || r.Value != "1" || r.Evidence != sm120InklingResidentReceipt || r.Equivalence != ArchDefaultGreedyIdentical {
			t.Fatalf("sm_120 %s: %+v", env, r)
		}
		if r := ResolveArchDefault(120, env, lookupFrom(map[string]string{env: "0"})); r.Source != ArchDefaultFromUser || r.Value != "0" {
			t.Fatalf("user setting must win: %+v", r)
		}
		// The user's explicit value wins over the kill switch too.
		both := lookupFrom(map[string]string{env: "1", "FAK_SM120_DEFAULTS": "0"})
		if r := ResolveArchDefault(120, env, both); r.Source != ArchDefaultFromUser || r.Value != "1" {
			t.Fatalf("user over kill switch: %+v", r)
		}
		if r := ResolveArchDefault(120, env, lookupFrom(map[string]string{"FAK_SM120_DEFAULTS": "0"})); r.Source != ArchDefaultFromKilled || r.Value != "" {
			t.Fatalf("kill switch: %+v", r)
		}
		if r := ResolveArchDefault(120, env, lookupFrom(map[string]string{"FAK_SM120_DEFAULTS": "1"})); r.Source != ArchDefaultFromTable {
			t.Fatalf("kill switch only on 0: %+v", r)
		}
		// Another arch inherits nothing from sm_120's measurement.
		for _, sm := range []int{0, 89, 90, 121} {
			if r := ResolveArchDefault(sm, env, none); r.Source != ArchDefaultFromNone || r.Value != "" {
				t.Fatalf("sm_%d %s: %+v", sm, env, r)
			}
		}
	}
	if r := ResolveArchDefault(120, "FAK_NOT_IN_TABLE", none); r.Source != ArchDefaultFromNone {
		t.Fatalf("unknown variable: %+v", r)
	}

	bad := map[string][]ArchDefaultTable{
		"no evidence":       {{SM: 120, KillSwitch: "K", Defaults: []ArchDefault{{Env: "A", Value: "1", Equivalence: ArchDefaultBitwise}}}},
		"no equivalence":    {{SM: 120, KillSwitch: "K", Defaults: []ArchDefault{{Env: "A", Value: "1", Evidence: "docs/benchmarks/x"}}}},
		"no kill switch":    {{SM: 120, Defaults: []ArchDefault{{Env: "A", Value: "1", Evidence: "docs/benchmarks/x", Equivalence: ArchDefaultBitwise}}}},
		"duplicate env":     {{SM: 120, KillSwitch: "K", Defaults: []ArchDefault{{Env: "A", Value: "1", Evidence: "docs/benchmarks/x", Equivalence: ArchDefaultBitwise}, {Env: "A", Value: "2", Evidence: "docs/benchmarks/y", Equivalence: ArchDefaultBitwise}}}},
		"duplicate arch":    {{SM: 120, KillSwitch: "K"}, {SM: 120, KillSwitch: "K"}},
		"kill switch entry": {{SM: 120, KillSwitch: "K", Defaults: []ArchDefault{{Env: "K", Value: "1", Evidence: "docs/benchmarks/x", Equivalence: ArchDefaultBitwise}}}},
	}
	for name, tables := range bad {
		if err := validateArchDefaultTables(tables); err == nil {
			t.Fatalf("%s: must be refused", name)
		}
	}
}
