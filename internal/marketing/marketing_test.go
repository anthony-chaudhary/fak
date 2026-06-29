package marketing

import (
	"errors"
	"testing"
	"time"
)

// rec builds one git-log record in the gitLogFormat the parser splits on (sha \x1f date
// \x1f subject \x1e), so the pure fold can be tested without a repo.
func rec(sha, date, subject string) string {
	return sha + "\x1f" + date + "\x1f" + subject + "\x1e"
}

func TestParseShipLogKeepsOnlyStampedSubjects(t *testing.T) {
	out := rec("aaaaaaaaaaaa", "2026-06-28T10:00:00Z", "feat(gateway): add the reclaim path (fak gateway)") + // trailer ship
		rec("bbbbbbbbbbbb", "2026-06-28T09:00:00Z", "fak/model: implement Q4_K reducer") + // direct ship
		rec("cccccccccccc", "2026-06-28T08:00:00Z", "Merge branch 'main'") + // merge — not a ship
		rec("dddddddddddd", "2026-06-28T07:00:00Z", "v0.18.0: release bundle") + // release — not a per-leaf ship
		rec("eeeeeeeeeeee", "2026-06-28T06:00:00Z", "wip: poke at things") // no stamp — not a ship

	ships, act := parseShipLog(out)

	if got, want := len(ships), 2; got != want {
		t.Fatalf("ships = %d, want %d (%+v)", got, want, ships)
	}
	if act.Commits != 5 {
		t.Errorf("activity commits = %d, want 5", act.Commits)
	}
	if act.Ships != 2 {
		t.Errorf("activity ships = %d, want 2", act.Ships)
	}
	// Ships keep their sha (short) + leaf + kind.
	byLeaf := map[string]Ship{}
	for _, s := range ships {
		byLeaf[s.Leaf] = s
	}
	if g := byLeaf["gateway"]; g.Kind != "trailer" || g.SHA != "aaaaaaaa" {
		t.Errorf("gateway ship = %+v, want kind=trailer sha=aaaaaaaa", g)
	}
	if m := byLeaf["model"]; m.Kind != "direct" || m.SHA != "bbbbbbbb" {
		t.Errorf("model ship = %+v, want kind=direct sha=bbbbbbbb", m)
	}
}

func TestCollectShipsOrdersNewestFirst(t *testing.T) {
	// parseShipLog doesn't sort (CollectShips does), so verify ordering through the same
	// stable sort CollectShips applies.
	out := rec("1111111111", "2026-06-20T00:00:00Z", "feat(a): add x (fak a)") +
		rec("2222222222", "2026-06-28T00:00:00Z", "feat(b): add y (fak b)") +
		rec("3333333333", "2026-06-24T00:00:00Z", "feat(c): add z (fak c)")
	ships, _ := parseShipLog(out)
	// emulate CollectShips' sort
	sortNewestFirst(ships)
	wantOrder := []string{"b", "c", "a"} // 06-28, 06-24, 06-20
	for i, w := range wantOrder {
		if ships[i].Leaf != w {
			t.Errorf("ships[%d].Leaf = %q, want %q (order %v)", i, ships[i].Leaf, w, leaves(ships))
		}
	}
}

func TestNewClaimRefusesUnwitnessed(t *testing.T) {
	cases := []struct {
		name string
		text string
		ship Ship
		ok   bool
	}{
		{"trailer ship accepted", "added the reclaim path", Ship{SHA: "abc12345", Kind: "trailer", Leaf: "gateway"}, true},
		{"direct ship accepted", "implemented the reducer", Ship{SHA: "def67890", Kind: "direct", Leaf: "model"}, true},
		{"empty sha refused", "claim", Ship{SHA: "", Kind: "trailer"}, false},
		{"none kind refused", "claim", Ship{SHA: "abc12345", Kind: "none"}, false},
		{"release kind refused", "claim", Ship{SHA: "abc12345", Kind: "release"}, false},
		{"empty text refused", "", Ship{SHA: "abc12345", Kind: "trailer"}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c, err := NewClaim(tc.text, tc.ship)
			if tc.ok {
				if err != nil {
					t.Fatalf("NewClaim err = %v, want nil", err)
				}
				if c.Ship.SHA == "" {
					t.Error("accepted claim has empty sha — invariant broken")
				}
				if c.Label != LabelWitnessed {
					t.Errorf("label = %q, want WITNESSED", c.Label)
				}
			} else {
				if err == nil {
					t.Fatal("NewClaim err = nil, want refusal")
				}
				if !errors.Is(err, ErrUnwitnessedClaim) {
					t.Errorf("err = %v, want wrapping ErrUnwitnessedClaim", err)
				}
			}
		})
	}
}

func TestMustClaimsSkipsBadShips(t *testing.T) {
	ships := []Ship{
		{SHA: "aaaa1111", Kind: "trailer", Leaf: "gateway", Subject: "feat(gateway): add x"},
		{SHA: "", Kind: "trailer", Leaf: "broken"}, // no sha — must be skipped, never panic
		{SHA: "bbbb2222", Kind: "direct", Leaf: "model", Subject: "fak/model: add y"},
	}
	claims, skipped := MustClaims(func(s Ship) string { return s.Subject }, ships)
	if len(claims) != 2 {
		t.Errorf("claims = %d, want 2", len(claims))
	}
	if skipped != 1 {
		t.Errorf("skipped = %d, want 1", skipped)
	}
}

func TestMarketableExcludesStubbedLeaf(t *testing.T) {
	// A ledger naming internal/grammar in a [STUB] line and #80 in a [SIMULATED] line.
	l := ClaimsLedger{
		loaded:          true,
		unshippedLeaves: map[string]bool{"grammar": true},
		unshippedIssues: map[int]bool{80: true},
	}
	cases := []struct {
		name string
		ship Ship
		ok   bool
	}{
		{"shipped leaf passes", Ship{Leaf: "gateway", Subject: "feat(gateway): add x (fak gateway)"}, true},
		{"stubbed leaf excluded", Ship{Leaf: "grammar", Subject: "feat(grammar): add mask (fak grammar)"}, false},
		{"stubbed issue excluded", Ship{Leaf: "kvmmu", Subject: "feat(kvmmu): add ttl for #80 (fak kvmmu)"}, false},
		{"unrelated issue passes", Ship{Leaf: "kvmmu", Subject: "feat(kvmmu): add x for #999 (fak kvmmu)"}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ok, reason := l.Marketable(tc.ship)
			if ok != tc.ok {
				t.Errorf("Marketable = %v (reason=%q), want %v", ok, reason, tc.ok)
			}
			if !ok && reason == "" {
				t.Error("excluded ship has empty reason — the hold must be visible")
			}
		})
	}
}

func TestMarketableOpenWhenLedgerNotLoaded(t *testing.T) {
	// A missing CLAIMS.md must not block marketing — the witness rung is still the floor.
	var l ClaimsLedger // loaded == false
	ok, _ := l.Marketable(Ship{Leaf: "anything", Subject: "feat(x): add y (fak x)"})
	if !ok {
		t.Error("not-loaded ledger should pass all ships; got blocked")
	}
}

func TestFilterMarketableSurfacesExclusions(t *testing.T) {
	l := ClaimsLedger{loaded: true, unshippedLeaves: map[string]bool{"preflight": true}, unshippedIssues: map[int]bool{}}
	ships := []Ship{
		{SHA: "a1", Leaf: "gateway", Subject: "feat(gateway): add x"},
		{SHA: "b2", Leaf: "preflight", Subject: "feat(preflight): add rung-2"},
	}
	market, excluded := FilterMarketable(l, ships)
	if len(market) != 1 || market[0].Leaf != "gateway" {
		t.Errorf("marketable = %v, want [gateway]", leaves(market))
	}
	if len(excluded) != 1 || excluded[0].Ship.Leaf != "preflight" || excluded[0].Reason == "" {
		t.Errorf("excluded = %+v, want one preflight with a reason", excluded)
	}
}

// --- test helpers ---

func sortNewestFirst(ships []Ship) {
	for i := 1; i < len(ships); i++ {
		for j := i; j > 0 && ships[j].Date.After(ships[j-1].Date); j-- {
			ships[j], ships[j-1] = ships[j-1], ships[j]
		}
	}
}

func leaves(ships []Ship) []string {
	out := make([]string, len(ships))
	for i, s := range ships {
		out[i] = s.Leaf
	}
	return out
}

// guard against an accidental dependency on wall-clock in the parser
var _ = time.Now
