package dormancy

import (
	"encoding/json"
	"testing"
	"time"
)

// TestBucketBoundaries walks every gap→band transition, asserting the half-open,
// cold-side tie-break (a gap exactly AT a threshold takes the colder band) the package
// promises. This is the acceptance witness: the bucket is a total function of the gap.
func TestBucketBoundaries(t *testing.T) {
	cases := []struct {
		name string
		gap  time.Duration
		want Horizon
	}{
		{"negative-clock-backwards", -time.Hour, Warm},
		{"zero", 0, Warm},
		{"just-active", 30 * time.Second, Warm},
		{"warm-upper-just-under", WarmMax - time.Nanosecond, Warm},
		{"warm/cool-boundary-tips-cool", WarmMax, Cool}, // exactly 5m => Cool (cold-side)
		{"cool-mid", 30 * time.Minute, Cool},
		{"cool-upper-just-under", CoolMax - time.Nanosecond, Cool},
		{"cool/cold-boundary-tips-cold", CoolMax, Cold}, // exactly 1h => Cold
		{"cold-mid", 6 * time.Hour, Cold},
		{"cold-upper-just-under", ColdMax - time.Nanosecond, Cold},
		{"cold/frozen-boundary-tips-frozen", ColdMax, Frozen}, // exactly 24h => Frozen
		{"frozen-mid", 10 * 24 * time.Hour, Frozen},
		{"frozen-upper-just-under", FrozenMax - time.Nanosecond, Frozen},
		{"frozen/ancient-boundary-tips-ancient", FrozenMax, Ancient}, // exactly 30d => Ancient
		{"ancient-far", 365 * 24 * time.Hour, Ancient},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := Bucket(c.gap); got != c.want {
				t.Fatalf("Bucket(%v) = %v, want %v", c.gap, got, c.want)
			}
		})
	}
}

// TestBandThresholdsAnchorCacheTTLs pins the load-bearing anchor in the issue: the first
// two band thresholds ARE the provider prompt-cache TTLs internal/resume encodes (300s /
// 3600s). A future edit that drifts them off the cache TTLs turns this red.
func TestBandThresholdsAnchorCacheTTLs(t *testing.T) {
	if WarmMax != 5*time.Minute || int64(WarmMax.Seconds()) != 300 {
		t.Errorf("WarmMax = %v, want 5m (300s, resume.TTL5m)", WarmMax)
	}
	if CoolMax != time.Hour || int64(CoolMax.Seconds()) != 3600 {
		t.Errorf("CoolMax = %v, want 1h (3600s, resume.TTL1h)", CoolMax)
	}
	if ColdMax != 24*time.Hour {
		t.Errorf("ColdMax = %v, want 24h", ColdMax)
	}
	if FrozenMax != 30*24*time.Hour {
		t.Errorf("FrozenMax = %v, want 30d", FrozenMax)
	}
}

// TestHorizonOrderedAndStaleness asserts the bands are ordered warm<cool<cold<frozen<
// ancient and that AtLeast reads that order — the gate a Phase-2 rung uses to fire only
// past a band.
func TestHorizonOrderedAndStaleness(t *testing.T) {
	order := []Horizon{Warm, Cool, Cold, Frozen, Ancient}
	for i := 1; i < len(order); i++ {
		if !(order[i-1] < order[i]) {
			t.Fatalf("bands not strictly ordered at %d: %v !< %v", i, order[i-1], order[i])
		}
	}
	if !Frozen.AtLeast(Cold) {
		t.Error("Frozen.AtLeast(Cold) = false, want true")
	}
	if Cool.AtLeast(Cold) {
		t.Error("Cool.AtLeast(Cold) = true, want false")
	}
	if !Cold.AtLeast(Cold) {
		t.Error("Cold.AtLeast(Cold) = false, want true (>= is reflexive)")
	}
}

// TestHorizonStringParseRoundTrip pins the closed wire vocabulary and the fail-closed
// parse (an unknown token is rejected, never coerced to Warm).
func TestHorizonStringParseRoundTrip(t *testing.T) {
	for _, h := range []Horizon{Warm, Cool, Cold, Frozen, Ancient} {
		tok := h.String()
		got, ok := ParseHorizon(tok)
		if !ok || got != h {
			t.Fatalf("round-trip %v: token=%q parsed=(%v,%v)", h, tok, got, ok)
		}
	}
	if s := Horizon(99).String(); s != "unknown" {
		t.Errorf("Horizon(99).String() = %q, want unknown", s)
	}
	if _, ok := ParseHorizon("nonsense"); ok {
		t.Error("ParseHorizon(nonsense) ok=true, want false (fail closed)")
	}
}

// TestStampGapAndHorizonNoIO is the "bucket derivable without I/O" witness: given a
// stamp and a caller-supplied now, the gap and band come out with no clock read inside.
func TestStampGapAndHorizonNoIO(t *testing.T) {
	base := time.Date(2026, 6, 28, 12, 0, 0, 0, time.UTC)
	s := At(base)

	// 30 minutes later => Cool band (past the 5m warm TTL, under the 1h cool TTL).
	now := base.Add(30 * time.Minute)
	gap, ok := s.GapAt(now)
	if !ok || gap != 30*time.Minute {
		t.Fatalf("GapAt = (%v,%v), want (30m,true)", gap, ok)
	}
	if h := s.HorizonAt(now); h != Cool {
		t.Fatalf("HorizonAt(+30m) = %v, want Cool", h)
	}

	// A now BEFORE the stamp (backwards clock) clamps the gap to 0 and the band to Warm.
	gap, ok = s.GapAt(base.Add(-time.Hour))
	if !ok || gap != 0 {
		t.Fatalf("GapAt(before) = (%v,%v), want (0,true)", gap, ok)
	}
	if h := s.HorizonAt(base.Add(-time.Hour)); h != Warm {
		t.Errorf("HorizonAt(before) = %v, want Warm", h)
	}
}

// TestZeroStampUnknownIsAncient pins the conservative ambiguity rule: an unmeasured gap
// is the most-stale band, never the least.
func TestZeroStampUnknownIsAncient(t *testing.T) {
	var z Stamp
	if !z.IsZero() {
		t.Fatal("zero Stamp IsZero() = false")
	}
	if _, ok := z.GapAt(time.Now()); ok {
		t.Error("GapAt on a zero Stamp ok=true, want false (unknown)")
	}
	if h := z.HorizonAt(time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)); h != Ancient {
		t.Errorf("zero Stamp HorizonAt = %v, want Ancient (unknown => revalidate all)", h)
	}
	if !z.Time().IsZero() {
		t.Error("zero Stamp Time() not zero")
	}
}

// TestRefreshMonotonic asserts a stamp only ever advances: a forward now updates it, a
// backwards now is ignored, and a first refresh of a zero stamp adopts now.
func TestRefreshMonotonic(t *testing.T) {
	base := time.Date(2026, 6, 28, 12, 0, 0, 0, time.UTC)
	s := At(base)

	forward := s.Refresh(base.Add(time.Minute))
	if forward.LastActiveUnixNano != base.Add(time.Minute).UnixNano() {
		t.Error("Refresh(forward) did not advance the stamp")
	}
	back := s.Refresh(base.Add(-time.Minute))
	if back != s {
		t.Error("Refresh(backwards) moved the stamp — monotonic guard failed")
	}
	equal := s.Refresh(base)
	if equal != s {
		t.Error("Refresh(equal instant) changed the stamp — must be strictly-advancing")
	}
	var z Stamp
	first := z.Refresh(base)
	if first.IsZero() || first.LastActiveUnixNano != base.UnixNano() {
		t.Error("Refresh on a zero Stamp did not adopt now as the first mark")
	}
}

// TestConstructorsAgree pins At/FromUnixNano/FromUnix on one instant and their
// non-positive => zero (unknown) handling.
func TestConstructorsAgree(t *testing.T) {
	tm := time.Date(2026, 6, 28, 12, 0, 0, 0, time.UTC)
	if At(tm) != FromUnixNano(tm.UnixNano()) {
		t.Error("At and FromUnixNano disagree on the same instant")
	}
	sec := time.Date(2026, 6, 28, 12, 0, 0, 0, time.UTC) // whole-second instant
	if FromUnix(sec.Unix()) != At(sec) {
		t.Error("FromUnix and At disagree on a whole-second instant")
	}
	if !At(time.Time{}).IsZero() {
		t.Error("At(zero time) not zero Stamp")
	}
	if !FromUnixNano(0).IsZero() || !FromUnixNano(-1).IsZero() {
		t.Error("FromUnixNano(non-positive) not zero Stamp")
	}
	if !FromUnix(0).IsZero() || !FromUnix(-5).IsZero() {
		t.Error("FromUnix(non-positive) not zero Stamp")
	}
}

// TestStampJSONOmitzero proves a zero Stamp marshals away (so surfacing it on an
// existing struct with omitzero is wire-compatible) and a set Stamp round-trips.
func TestStampJSONOmitzero(t *testing.T) {
	type carrier struct {
		Last Stamp `json:"last,omitempty,omitzero"`
	}
	zb, err := json.Marshal(carrier{})
	if err != nil {
		t.Fatal(err)
	}
	if string(zb) != "{}" {
		t.Errorf("zero Stamp marshaled to %s, want {} (omitzero wire-compat)", zb)
	}
	tm := time.Date(2026, 6, 28, 12, 0, 0, 0, time.UTC)
	b, err := json.Marshal(carrier{Last: At(tm)})
	if err != nil {
		t.Fatal(err)
	}
	var rt carrier
	if err := json.Unmarshal(b, &rt); err != nil {
		t.Fatal(err)
	}
	if rt.Last != At(tm) {
		t.Errorf("Stamp JSON round-trip = %v, want %v", rt.Last, At(tm))
	}
}
