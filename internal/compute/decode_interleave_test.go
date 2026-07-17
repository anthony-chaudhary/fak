package compute

import (
	"reflect"
	"testing"
)

// eligibleSnap is a witnessed-like host: linux/amd64, 8 NUMA nodes, 256 cores, unconstrained,
// default first-touch policy — the dual EPYC-7742 the interleave cell was measured on.
func eligibleSnap() decodeInterleaveSnapshot {
	return decodeInterleaveSnapshot{
		goos: "linux", goarch: "amd64",
		online: []int{0, 1, 2, 3, 4, 5, 6, 7}, cores: 256,
		constrained: false, policyLabel: "default",
	}
}

func TestPlanDecodeInterleave(t *testing.T) {
	cases := []struct {
		name       string
		mutate     func(s *decodeInterleaveSnapshot)
		ov         decodeInterleaveOverride
		wantApply  bool
		wantReason DecodeInterleaveReason
	}{
		// The happy path fires only on a witnessed-like host.
		{"eligible auto", nil, interleaveAuto, true, DecodeInterleaveEligible},

		// Failure-first: a strict membind owns placement — interleaving could OOM the bound
		// node, so we refuse even with 8 nodes, and even under force-on.
		{"constrained refuses", func(s *decodeInterleaveSnapshot) {
			s.constrained = true
			s.policyLabel = "bind:0"
		}, interleaveAuto, false, DecodeInterleaveConstrained},
		{"constrained refuses even on force-on", func(s *decodeInterleaveSnapshot) {
			s.constrained = true
			s.policyLabel = "bind:0"
		}, interleaveForceOn, false, DecodeInterleaveConstrained},

		// Failure-first: nothing to stripe across.
		{"single node refuses", func(s *decodeInterleaveSnapshot) {
			s.online = []int{0}
		}, interleaveAuto, false, DecodeInterleaveSingleNode},

		// Failure-first: the mbind shim exists only on linux/amd64.
		{"darwin refuses", func(s *decodeInterleaveSnapshot) {
			s.goos, s.goarch = "darwin", "arm64"
		}, interleaveAuto, false, DecodeInterleaveUnsupported},
		{"windows amd64 refuses", func(s *decodeInterleaveSnapshot) {
			s.goos = "windows"
		}, interleaveAuto, false, DecodeInterleaveUnsupported},
		{"linux arm64 refuses", func(s *decodeInterleaveSnapshot) {
			s.goarch = "arm64"
		}, interleaveAuto, false, DecodeInterleaveUnsupported},
		{"force-on cannot beat unsupported platform", func(s *decodeInterleaveSnapshot) {
			s.goos, s.goarch = "darwin", "arm64"
		}, interleaveForceOn, false, DecodeInterleaveUnsupported},

		// Auto defers to an operator policy already in place (numactl --interleave=all):
		// the regime is present, so this is a (non-applying) success, not a re-apply.
		{"already placed defers", func(s *decodeInterleaveSnapshot) {
			s.policyLabel = "interleave:0-7"
		}, interleaveAuto, false, DecodeInterleaveAlreadyPlaced},
		// ...but force-on re-asserts it regardless of the pre-existing policy.
		{"force-on ignores already placed", func(s *decodeInterleaveSnapshot) {
			s.policyLabel = "interleave:0-7"
		}, interleaveForceOn, true, DecodeInterleaveOverrideOn},

		// Auto stays conservative below the witnessed manycore floor; force-on overrides it.
		{"below manycore refuses auto", func(s *decodeInterleaveSnapshot) {
			s.online = []int{0, 1}
			s.cores = 16
		}, interleaveAuto, false, DecodeInterleaveBelowManycore},
		{"force-on fires below manycore", func(s *decodeInterleaveSnapshot) {
			s.online = []int{0, 1}
			s.cores = 16
		}, interleaveForceOn, true, DecodeInterleaveOverrideOn},

		// Overrideable: off wins over an otherwise-eligible host.
		{"force-off wins", nil, interleaveForceOff, false, DecodeInterleaveOverrideOff},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := eligibleSnap()
			if tc.mutate != nil {
				tc.mutate(&s)
			}
			got := planDecodeInterleave(s, tc.ov)
			if got.Apply != tc.wantApply || got.Reason != tc.wantReason {
				t.Fatalf("planDecodeInterleave apply=%v reason=%q, want apply=%v reason=%q",
					got.Apply, got.Reason, tc.wantApply, tc.wantReason)
			}
			if !got.Apply && got.Nodes != nil {
				t.Errorf("a refusal must carry no Nodes, got %v", got.Nodes)
			}
		})
	}
}

// TestPlanDecodeInterleaveNodesDeriveFromSnapshot proves the placement node set is read from
// the host snapshot, never a hardcoded topology: an odd sparse node list must round-trip.
func TestPlanDecodeInterleaveNodesDeriveFromSnapshot(t *testing.T) {
	s := eligibleSnap()
	s.online = []int{2, 5} // sparse, non-contiguous, not the 0..7 of the witness box
	got := planDecodeInterleave(s, interleaveAuto)
	if !got.Apply || got.Reason != DecodeInterleaveEligible {
		t.Fatalf("want eligible apply, got apply=%v reason=%q", got.Apply, got.Reason)
	}
	if !reflect.DeepEqual(got.Nodes, []int{2, 5}) {
		t.Fatalf("Nodes=%v, want [2 5] derived from the snapshot", got.Nodes)
	}
	// The plan must own its slice — mutating the returned Nodes must not alias the snapshot.
	got.Nodes[0] = 99
	if s.online[0] != 2 {
		t.Errorf("plan aliased the snapshot's online slice")
	}
}

func TestDecodeInterleaveOverrideFromEnv(t *testing.T) {
	cases := map[string]decodeInterleaveOverride{
		"":        interleaveAuto,
		"auto":    interleaveAuto,
		"garbage": interleaveAuto,
		"on":      interleaveForceOn,
		"1":       interleaveForceOn,
		"true":    interleaveForceOn,
		"YES":     interleaveForceOn,
		" On ":    interleaveForceOn,
		"off":     interleaveForceOff,
		"0":       interleaveForceOff,
		"false":   interleaveForceOff,
		"no":      interleaveForceOff,
	}
	for val, want := range cases {
		getenv := func(k string) string {
			if k == "FAK_NUMA_INTERLEAVE" {
				return val
			}
			return ""
		}
		if got := decodeInterleaveOverrideFromEnv(getenv); got != want {
			t.Errorf("FAK_NUMA_INTERLEAVE=%q ⇒ %v, want %v", val, got, want)
		}
	}
}

func TestDecodeInterleaveResultLabel(t *testing.T) {
	applied := DecodeInterleaveResult{
		Plan:          DecodeInterleavePlan{Apply: true, Reason: DecodeInterleaveEligible, Nodes: []int{0, 1, 2, 3, 4, 5, 6, 7}},
		RegionsPlaced: 339,
	}
	if got, want := applied.Label(), "interleave=applied(reason=eligible,nodes=0-7,regions=339)"; got != want {
		t.Errorf("applied label=%q, want %q", got, want)
	}
	skipped := DecodeInterleaveResult{Plan: DecodeInterleavePlan{Reason: DecodeInterleaveSingleNode}}
	if got, want := skipped.Label(), "interleave=skipped(reason=single_node)"; got != want {
		t.Errorf("skipped label=%q, want %q", got, want)
	}
}
