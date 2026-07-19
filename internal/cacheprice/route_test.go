package cacheprice

import "testing"

func TestCheapestRoute(t *testing.T) {
	cases := []struct {
		name                     string
		prompt, prefix, overhead int
		localAvail, remoteAvail  bool
		wantRoute                AdmissionRoute
		wantCost                 int
	}{
		{"no residency anywhere recomputes the whole prompt", 1000, 400, 100, false, false, RouteRecompute, 1000},
		{"local residency wins outright — full discount, no toll", 1000, 400, 100, true, true, RouteLocal, 600},
		{"local wins over remote even when remote is also available and cheap", 1000, 900, 10, true, true, RouteLocal, 100},
		{"remote only, positive dividend beats recompute", 1000, 400, 100, false, true, RouteRemote, 700},
		{"remote only, toll exceeds prefix (negative dividend) falls back to recompute", 1000, 400, 500, false, true, RouteRecompute, 1000},
		{"remote only, break-even toll prefers recompute over a fabric dependency", 1000, 400, 400, false, true, RouteRecompute, 1000},
		{"cold prompt with local available yields no discount, recomputes", 1000, 0, 100, true, false, RouteRecompute, 1000},
		{"negative toll clamps to 0 — remote is a free full discount", 1000, 400, -30, false, true, RouteRemote, 600},
		{"non-positive prompt costs nothing and recomputes", 0, 400, 100, true, true, RouteRecompute, 0},
		{"over-reported prefix caps discount at the full prompt (local)", 1000, 5000, 0, true, false, RouteLocal, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			route, cost := CheapestRoute(c.prompt, c.prefix, c.overhead, c.localAvail, c.remoteAvail)
			if route != c.wantRoute || cost != c.wantCost {
				t.Fatalf("CheapestRoute(%d, %d, %d, local=%v, remote=%v) = (%s, %d), want (%s, %d)",
					c.prompt, c.prefix, c.overhead, c.localAvail, c.remoteAvail, route, cost, c.wantRoute, c.wantCost)
			}
		})
	}
}

func TestCheapestRouteAgreesWithDividendSign(t *testing.T) {
	// When only the remote tier can serve the prefix, CheapestRoute must pick RouteRemote exactly
	// when the fabric fetch is DisaggregationWorthwhile — the two decisions are one inequality.
	const prompt, prefix = 1000, 400
	for overhead := 0; overhead <= 800; overhead += 50 {
		wantWorthwhile := DisaggregationWorthwhile(prefix, overhead)
		route, _ := CheapestRoute(prompt, prefix, overhead, false, true)
		if got := route == RouteRemote; got != wantWorthwhile {
			t.Fatalf("overhead=%d: CheapestRoute chose remote=%v, DisaggregationWorthwhile=%v — must agree",
				overhead, got, wantWorthwhile)
		}
	}
}

func TestAdmissionRouteString(t *testing.T) {
	cases := map[AdmissionRoute]string{
		RouteRecompute:     "recompute",
		RouteLocal:         "local",
		RouteRemote:        "remote",
		AdmissionRoute(99): "unknown",
	}
	for r, want := range cases {
		if got := r.String(); got != want {
			t.Fatalf("AdmissionRoute(%d).String() = %q, want %q", int(r), got, want)
		}
	}
}
