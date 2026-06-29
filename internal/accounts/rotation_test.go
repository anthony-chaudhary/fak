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
	for _, s := range res.Excluded {
		if s.Name == name {
			return s.Status
		}
	}
	return ""
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
	if r := excludedReason(res, "eve"); r != RotationUnservable {
		t.Fatalf("eve excluded reason = %q, want unservable", r)
	}
	if r := excludedReason(res, "old"); r != RotationTombstoned {
		t.Fatalf("old excluded reason = %q, want tombstoned", r)
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
