package accounts

import (
	"testing"
)

// active builds a live, serveable seat (status active, dir present, disk identity with
// creds) logged into account `uuid`. No disk I/O — RotationPlan reads these fields directly.
func active(name, uuid, email string) Home {
	return Home{
		Name: name, Dir: "/home/" + name, Status: StatusActive,
		Identity: Identity{Exists: true, HasCreds: true, AccountUUID: uuid, Email: email},
	}
}

func boolp(b bool) *bool { return &b }

// poolNames returns the included pool seat names in order.
func poolNames(res RotationResult) []string {
	var out []string
	for _, s := range res.Pool {
		out = append(out, s.Name)
	}
	return out
}

// excludedReason returns the status recorded for seat `name` in the excluded list (or "").
func excludedReason(res RotationResult, name string) RotationStatus {
	if s, ok := excludedSeat(res, name); ok {
		return s.Status
	}
	return ""
}

func excludedSeat(res RotationResult, name string) (RotationSeat, bool) {
	for _, s := range res.Excluded {
		if s.Name == name {
			return s, true
		}
	}
	return RotationSeat{}, false
}

func TestRotationPlanDedupAndExclusions(t *testing.T) {
	reg := Registry{Homes: []Home{
		active("alice", "u-alice", "alice@x.test"),
		// Same account bucket as alice (shares u-alice) under a different, non-matching name:
		// must collapse to a duplicate, not present a second rotatable account.
		func() Home { h := active("zdup", "u-alice", "alice@x.test"); return h }(),
		active("bob", "u-bob", "bob@x.test"),
		// Reserved: held out of routine rotation (default policy avoid_reserved=true).
		func() Home { h := active("carol", "u-carol", "carol@x.test"); h.Reserved = true; return h }(),
		// Disabled via explicit enabled:false.
		func() Home { h := active("dave", "u-dave", "dave@x.test"); h.Enabled = boolp(false); return h }(),
		// Active+enabled but no live credentials -> unservable, never a rotation target.
		{Name: "eve", Dir: "/home/eve", Status: StatusActive,
			Identity: Identity{Exists: true, HasCreds: false, AccountUUID: "u-eve", Email: "eve@x.test"}},
		// Reserved is a rotation policy, not a substitute for login status: this is still
		// unservable because it cannot launch without /login.
		func() Home {
			h := active("zreserved-dead", "u-reserved-dead", "dead@x.test")
			h.Reserved = true
			h.Identity.HasCreds = false
			return h
		}(),
		// Tombstoned -> rehomes via Serve, never rotated onto.
		{Name: "old", Status: StatusTombstoned, RehomeTo: "alice"},
	}}

	res := reg.RotationPlan()

	if got := poolNames(res); len(got) != 2 || got[0] != "alice" || got[1] != "bob" {
		t.Fatalf("pool = %v, want [alice bob] (distinct eligible buckets, sorted)", got)
	}
	// The canonical for the alice bucket is "alice"; "zdup" collapses onto it.
	for _, s := range res.Pool {
		if s.Name == "alice" && s.Account != "uuid:u-alice" {
			t.Fatalf("alice pool seat account = %q, want uuid:u-alice", s.Account)
		}
	}
	if r := excludedReason(res, "zdup"); r != RotationDuplicate {
		t.Fatalf("zdup excluded reason = %q, want duplicate", r)
	}
	for _, s := range res.Excluded {
		if s.Name == "zdup" && s.Canonical != "alice" {
			t.Fatalf("zdup canonical = %q, want alice", s.Canonical)
		}
	}
	if r := excludedReason(res, "carol"); r != RotationReserved {
		t.Fatalf("carol excluded reason = %q, want reserved", r)
	}
	if r := excludedReason(res, "dave"); r != RotationDisabled {
		t.Fatalf("dave excluded reason = %q, want disabled", r)
	}
	for _, s := range res.Pool {
		if s.Login != LoginReady || !s.CanServe {
			t.Fatalf("pool seat readiness = %+v, want login ready and can_serve true", s)
		}
	}
	if s, ok := excludedSeat(res, "dave"); !ok || s.Login != LoginDisabled {
		t.Fatalf("dave login = %q,%v, want disabled,true", s.Login, ok)
	}
	if s, ok := excludedSeat(res, "eve"); !ok || s.CanServe {
		t.Fatalf("eve can_serve = %+v,%v, want false", s, ok)
	}
	if s, ok := excludedSeat(res, "eve"); !ok || s.Status != RotationUnservable || s.Login != LoginNeedsLogin {
		t.Fatalf("eve excluded = %+v,%v, want unservable with login needs_login", s, ok)
	}
	if s, ok := excludedSeat(res, "zreserved-dead"); !ok || s.Status != RotationUnservable || s.Login != LoginNeedsLogin {
		t.Fatalf("reserved no-creds excluded = %+v,%v, want unservable with login needs_login", s, ok)
	}
	if r := excludedReason(res, "old"); r != RotationTombstoned {
		t.Fatalf("old excluded reason = %q, want tombstoned", r)
	}
	if s, ok := excludedSeat(res, "old"); !ok || s.Login != LoginTombstoned {
		t.Fatalf("old login = %q,%v, want tombstoned,true", s.Login, ok)
	}
	// Honesty: applied order is the witnessed stable round-robin, not an unwitnessed by_reset.
	if res.OrderApplied != "stable-by-name" {
		t.Fatalf("OrderApplied = %q, want stable-by-name", res.OrderApplied)
	}
}

func TestNextInRotationRoundRobinSkipsCurrentBucket(t *testing.T) {
	reg := Registry{Homes: []Home{
		active("alice", "u-alice", "alice@x.test"),
		active("bob", "u-bob", "bob@x.test"),
		active("frank", "u-frank", "frank@x.test"),
		// A duplicate of alice's bucket — rotating "after" it must still skip the alice bucket.
		func() Home { h := active("zdup", "u-alice", "alice@x.test"); return h }(),
	}}

	cases := []struct{ after, want string }{
		{"alice", "bob"},
		{"bob", "frank"},
		{"frank", "alice"}, // wrap
		{"", "alice"},      // fresh start -> first in rotation
		{"zdup", "bob"},    // a duplicate resolves to alice's bucket -> next is bob
		{"unknown", "alice"},
	}
	for _, c := range cases {
		got, ok := reg.NextInRotation(c.after)
		if !ok {
			t.Fatalf("NextInRotation(%q): ok=false, want %q", c.after, c.want)
		}
		if got.Name != c.want {
			t.Fatalf("NextInRotation(%q) = %q, want %q", c.after, got.Name, c.want)
		}
	}
}

func TestNextInRotationSingleBucketHasNowhereToGo(t *testing.T) {
	reg := Registry{Homes: []Home{active("solo", "u-solo", "solo@x.test")}}
	if _, ok := reg.NextInRotation("solo"); ok {
		t.Fatal("a single-bucket registry must have nowhere to rotate (ok=false)")
	}
	// With no current seat, the sole bucket is a valid fresh start.
	if got, ok := reg.NextInRotation(""); !ok || got.Name != "solo" {
		t.Fatalf("NextInRotation(\"\") = %q,%v; want solo,true", got.Name, ok)
	}
}

func TestNextInRotationEmptyPool(t *testing.T) {
	reg := Registry{Homes: []Home{
		{Name: "old", Status: StatusTombstoned, RehomeTo: "old"}, // not serveable, excluded
	}}
	if _, ok := reg.NextInRotation(""); ok {
		t.Fatal("an empty pool must yield ok=false")
	}
}

func TestRotationPlanHeadroomOrdering(t *testing.T) {
	// alice and bob are both eligible; without a signal the pool is stable-by-name [alice, bob].
	reg := Registry{Homes: []Home{
		active("alice", "u-alice", "alice@x.test"),
		active("bob", "u-bob", "bob@x.test"),
	}}

	// A headroom signal that walls the alice bucket and gives bob room flips the order: bob
	// (room) must sort ahead of alice (walled), and the applied order + stamped scores say so.
	hr := RotationHeadroom{"uuid:u-alice": -1, "uuid:u-bob": 1}
	res := reg.RotationPlanWithHeadroom(hr)
	if got := poolNames(res); len(got) != 2 || got[0] != "bob" || got[1] != "alice" {
		t.Fatalf("headroom pool = %v, want [bob alice] (room before walled)", got)
	}
	if res.OrderApplied != "headroom-desc" {
		t.Fatalf("OrderApplied = %q, want headroom-desc", res.OrderApplied)
	}
	for _, s := range res.Pool {
		if s.Headroom == nil {
			t.Fatalf("pool seat %q missing stamped headroom", s.Name)
		}
	}

	// A nil/empty signal must be byte-for-byte the historical stable-by-name plan — no headroom
	// stamps, no order flip.
	plain := reg.RotationPlanWithHeadroom(nil)
	if got := poolNames(plain); len(got) != 2 || got[0] != "alice" || got[1] != "bob" {
		t.Fatalf("no-signal pool = %v, want [alice bob] (stable-by-name)", got)
	}
	if plain.OrderApplied != "stable-by-name" {
		t.Fatalf("no-signal OrderApplied = %q, want stable-by-name", plain.OrderApplied)
	}
	for _, s := range plain.Pool {
		if s.Headroom != nil {
			t.Fatalf("no-signal pool seat %q should not carry a headroom stamp", s.Name)
		}
	}

	// Equal headroom falls back to the deterministic name order (name breaks the tie).
	tie := reg.RotationPlanWithHeadroom(RotationHeadroom{"uuid:u-alice": 1, "uuid:u-bob": 1})
	if got := poolNames(tie); len(got) != 2 || got[0] != "alice" || got[1] != "bob" {
		t.Fatalf("equal-headroom pool = %v, want [alice bob] (name tiebreak)", got)
	}
}

func TestNextInRotationHeadroomPrefersRoom(t *testing.T) {
	// Three buckets: alice walled, bob has room, frank unknown (0). The runtime signal orders
	// the pool [bob(1), frank(0), alice(-1)].
	reg := Registry{Homes: []Home{
		active("alice", "u-alice", "alice@x.test"),
		active("bob", "u-bob", "bob@x.test"),
		active("frank", "u-frank", "frank@x.test"),
	}}
	hr := RotationHeadroom{"uuid:u-alice": -1, "uuid:u-bob": 1} // frank absent -> 0

	// A fresh start (no anchor) lands on the account with the most room, not the first name.
	if got, ok := reg.NextInRotationWithHeadroom("", hr); !ok || got.Name != "bob" {
		t.Fatalf("NextInRotationWithHeadroom(\"\") = %q,%v; want bob,true", got.Name, ok)
	}
	// Rotating OFF the roomiest bucket picks the next-best (frank), never re-handing bob and
	// never jumping straight to the walled alice while a non-walled bucket remains.
	if got, ok := reg.NextInRotationWithHeadroom("bob", hr); !ok || got.Name != "frank" {
		t.Fatalf("NextInRotationWithHeadroom(bob) = %q,%v; want frank,true", got.Name, ok)
	}
	// Rotating off a walled bucket still prefers the roomiest available one.
	if got, ok := reg.NextInRotationWithHeadroom("alice", hr); !ok || got.Name != "bob" {
		t.Fatalf("NextInRotationWithHeadroom(alice) = %q,%v; want bob,true", got.Name, ok)
	}
}

func TestRotationPolicyDefaultsAndViewRead(t *testing.T) {
	// No views -> sane defaults: reserved held out.
	def := Registry{}.RotationPolicy()
	if !def.AvoidReserved {
		t.Fatal("default AvoidReserved should be true")
	}
	if def.Order != "" {
		t.Fatalf("default Order = %q, want empty", def.Order)
	}

	reg := Registry{Views: map[string]ViewConfig{
		"job": {Blocks: map[string]any{
			"rotation": map[string]any{
				"order":          "by_reset",
				"near_cap_util":  0.95,
				"avoid_reserved": false,
			},
		}},
	}}
	pol := reg.RotationPlan().Policy
	if pol.Order != "by_reset" {
		t.Fatalf("Order = %q, want by_reset", pol.Order)
	}
	if pol.NearCapUtil != 0.95 {
		t.Fatalf("NearCapUtil = %v, want 0.95", pol.NearCapUtil)
	}
	if pol.AvoidReserved {
		t.Fatal("AvoidReserved should be false when the view block sets it false")
	}
}

func TestRotationReservedIncludedWhenPolicyAllows(t *testing.T) {
	mk := func() Home { h := active("carol", "u-carol", "carol@x.test"); h.Reserved = true; return h }
	reg := Registry{
		Homes: []Home{active("alice", "u-alice", "alice@x.test"), mk()},
		Views: map[string]ViewConfig{
			"job": {Blocks: map[string]any{"rotation": map[string]any{"avoid_reserved": false}}},
		},
	}
	res := reg.RotationPlan()
	got := poolNames(res)
	if len(got) != 2 || got[0] != "alice" || got[1] != "carol" {
		t.Fatalf("pool = %v, want [alice carol] (reserved included when avoid_reserved=false)", got)
	}
}
