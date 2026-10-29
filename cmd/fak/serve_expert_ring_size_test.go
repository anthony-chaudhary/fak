package main

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// serve_expert_ring_size_test.go — --expert-ring-bytes sizes the activated-expert device ring.
// token must stay byte-identical to the historical placement; auto and explicit sizes are judged
// against a device bound and a host bound and refuse at load with ErrServeExpertRingTooBig.

// fak-test:runtime fast est=1ms lane=default
func TestParseServeExpertRingSize(t *testing.T) {
	for _, tc := range []struct {
		raw  string
		want serveExpertRingSize
	}{
		{"", serveExpertRingSize{Mode: serveExpertRingToken}},
		{" Token ", serveExpertRingSize{Mode: serveExpertRingToken}},
		{"auto", serveExpertRingSize{Mode: serveExpertRingAuto}},
		{"1048576", serveExpertRingSize{Mode: serveExpertRingExplicit, Bytes: 1 << 20}},
		{"64g", serveExpertRingSize{Mode: serveExpertRingExplicit, Bytes: 64 << 30}},
		{"64GiB", serveExpertRingSize{Mode: serveExpertRingExplicit, Bytes: 64 << 30}},
		{"96 GB", serveExpertRingSize{Mode: serveExpertRingExplicit, Bytes: 96 << 30}},
		{"512M", serveExpertRingSize{Mode: serveExpertRingExplicit, Bytes: 512 << 20}},
		{"1t", serveExpertRingSize{Mode: serveExpertRingExplicit, Bytes: 1 << 40}},
	} {
		got, err := parseServeExpertRingSize(tc.raw)
		if err != nil || got != tc.want {
			t.Errorf("parse(%q)=%+v err=%v, want %+v", tc.raw, got, err, tc.want)
		}
	}
	for _, raw := range []string{"abc", "0", "-5", "1.5g", "g", "9223372036854775807g"} {
		if _, err := parseServeExpertRingSize(raw); !errors.Is(err, ErrServeExpertRingSize) {
			t.Errorf("parse(%q) err=%v, want ErrServeExpertRingSize", raw, err)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestResolveServeExpertRingSizeFlagWinsOverEnv(t *testing.T) {
	env := func(v string) func(string) string {
		return func(k string) string {
			if k == FakExpertRingBytesEnv {
				return v
			}
			return ""
		}
	}
	if got, err := resolveServeExpertRingSize("", env("auto")); err != nil || got.Mode != serveExpertRingAuto {
		t.Fatalf("env auto: %+v err=%v", got, err)
	}
	if got, err := resolveServeExpertRingSize("token", env("auto")); err != nil || got.Mode != serveExpertRingToken {
		t.Fatalf("flag token over env auto: %+v err=%v", got, err)
	}
	if got, err := resolveServeExpertRingSize("", nil); err != nil || got.Mode != serveExpertRingToken {
		t.Fatalf("no flag, no env: %+v err=%v", got, err)
	}
	if _, err := resolveServeExpertRingSize("", env("lots")); !errors.Is(err, ErrServeExpertRingSize) {
		t.Fatalf("malformed env err=%v, want ErrServeExpertRingSize", err)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestApplyServeExpertRingBytesNormalizesIntoEnv(t *testing.T) {
	t.Setenv(FakExpertRingBytesEnv, "auto")
	if err := applyServeExpertRingBytes("2g"); err != nil {
		t.Fatal(err)
	}
	if got, err := resolveServeExpertRingSize("", func(k string) string {
		if k == FakExpertRingBytesEnv {
			return "2147483648"
		}
		return ""
	}); err != nil || got.Bytes != 2<<30 {
		t.Fatalf("normalized explicit size %+v err=%v", got, err)
	}
	if err := applyServeExpertRingBytes("nope"); !errors.Is(err, ErrServeExpertRingSize) {
		t.Fatalf("invalid flag err=%v, want ErrServeExpertRingSize", err)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestServeExpertRingSharedPool(t *testing.T) {
	for _, tc := range []struct {
		shares, known bool
		carveout      int64
		want          bool
	}{
		{shares: false, known: true, carveout: 512 << 20, want: false},
		{shares: true, known: true, carveout: 512 << 20, want: true},
		{shares: true, known: true, carveout: 4 << 30, want: true},
		{shares: true, known: true, carveout: 96 << 30, want: false},
		{shares: true, known: false, want: true},
	} {
		if got := serveExpertRingSharedPool(tc.shares, tc.carveout, tc.known); got != tc.want {
			t.Errorf("shares=%v carveout=%d known=%v: got %v want %v", tc.shares, tc.carveout, tc.known, got, tc.want)
		}
	}
}

const ringGiB = int64(1) << 30

func ringSizingFit() ggufload.ActivatedExpertFit {
	return ggufload.ActivatedExpertFit{
		DeviceBaseBytes: 3 * ringGiB, RoutedBandBytes: 180 * ringGiB,
		ActivatedLayerBytes: 72 << 20, ActivatedTokenBytes: 3 * ringGiB,
		HostGatheredBytes: 60 * ringGiB,
	}
}

func ringSizingNonRing(f ggufload.ActivatedExpertFit, kv int64) compute.MemoryPlan {
	plan := compute.MemoryPlan{
		{Class: compute.MemoryWeights, Bytes: f.DeviceBaseBytes, Detail: serveActivatedRingBaseDetail, Scope: compute.MemoryScopeDevice},
		{Class: compute.MemoryKVCache, Bytes: kv, Detail: "kv", Scope: compute.MemoryScopeDevice},
	}
	if d, ok := f.HostGatheredDemand(); ok {
		plan = append(plan, d)
	}
	return plan
}

// fak-test:runtime fast est=1ms lane=default
func TestSizeServeExpertRingTable(t *testing.T) {
	f := ringSizingFit()
	const token = 3 * ringGiB
	nonRing := ringSizingNonRing(f, 2*ringGiB) // device 5 GiB, host page reserve 4 GiB
	roomyHost := serveExpertRingHost{Total: 125 * ringGiB, Avail: 118 * ringGiB, Known: true, RSSAnonPeak: 2 * ringGiB, SharedPool: true}

	t.Run("token_is_unchanged", func(t *testing.T) {
		tiny := serveExpertRingHost{Total: 8 * ringGiB, Avail: 1, Known: true, SharedPool: true}
		s, err := sizeServeExpertRing(serveExpertRingSize{Mode: serveExpertRingToken}, f, token, nonRing, 1, tiny)
		if err != nil || s.RingBytes != token {
			t.Fatalf("token ring=%d err=%v, want %d and no new refusal", s.RingBytes, err, token)
		}
	})
	t.Run("auto_takes_smaller_bound", func(t *testing.T) {
		// A 250 GiB host at the 0.5 ceiling fraction keeps the pool cap (125 GiB) out of the way.
		wide := serveExpertRingHost{Total: 250 * ringGiB, Avail: 118 * ringGiB, Known: true, RSSAnonPeak: 2 * ringGiB, SharedPool: true}
		wideBound := 118*ringGiB - 2*ringGiB - int64(float64(250*ringGiB)*0.05) - ggufload.HostGatheredPageReserveCap - 5*ringGiB
		auto := serveExpertRingSize{Mode: serveExpertRingAuto, PoolFraction: 0.5}
		deviceAvail := 90 * ringGiB
		s, err := sizeServeExpertRing(auto, f, token, nonRing, deviceAvail, wide)
		if err != nil || s.RingBytes != deviceAvail-5*ringGiB {
			t.Fatalf("auto ring=%d err=%v, want device bound %d", s.RingBytes, err, deviceAvail-5*ringGiB)
		}
		s, err = sizeServeExpertRing(auto, f, token, nonRing, 200*ringGiB, wide)
		if err != nil || s.RingBytes != wideBound || !s.HostBoundKnown {
			t.Fatalf("auto ring=%d err=%v, want host bound %d", s.RingBytes, err, wideBound)
		}
	})
	t.Run("auto_clamps_to_band", func(t *testing.T) {
		huge := serveExpertRingHost{Total: 1 << 50, Avail: 1 << 49, Known: true}
		s, err := sizeServeExpertRing(serveExpertRingSize{Mode: serveExpertRingAuto}, f, token, nonRing, 1<<49, huge)
		if err != nil || s.RingBytes != f.RoutedBandBytes {
			t.Fatalf("auto ring=%d err=%v, want the routed band %d", s.RingBytes, err, f.RoutedBandBytes)
		}
	})
	t.Run("auto_unmeasurable_keeps_token", func(t *testing.T) {
		s, err := sizeServeExpertRing(serveExpertRingSize{Mode: serveExpertRingAuto}, f, token, nonRing, compute.FreeUnknown, serveExpertRingHost{})
		if err != nil || s.RingBytes != token {
			t.Fatalf("unmeasurable auto ring=%d err=%v, want token %d", s.RingBytes, err, token)
		}
	})
	t.Run("explicit_fit", func(t *testing.T) {
		s, err := sizeServeExpertRing(serveExpertRingSize{Mode: serveExpertRingExplicit, Bytes: 64 * ringGiB}, f, token, nonRing, 102*ringGiB, roomyHost)
		if err != nil || s.RingBytes != 64*ringGiB {
			t.Fatalf("explicit ring=%d err=%v, want 64 GiB", s.RingBytes, err)
		}
	})
	t.Run("explicit_refuse_device", func(t *testing.T) {
		_, err := sizeServeExpertRing(serveExpertRingSize{Mode: serveExpertRingExplicit, Bytes: 99 * ringGiB}, f, token, nonRing, 102*ringGiB, roomyHost)
		var fe *compute.FitError
		if !errors.Is(err, ErrServeExpertRingTooBig) || !errors.As(err, &fe) || fe.Scope != compute.MemoryScopeDevice || fe.Want <= fe.Avail {
			t.Fatalf("explicit over device bound err=%v, want typed device FitTooBig + ErrServeExpertRingTooBig", err)
		}
	})
	t.Run("uma_single_pool_refuse", func(t *testing.T) {
		// The device heap would admit it; MemAvailable as one pool does not.
		_, err := sizeServeExpertRing(serveExpertRingSize{Mode: serveExpertRingExplicit, Bytes: 110 * ringGiB}, f, token, nonRing, 1<<50, roomyHost)
		var fe *compute.FitError
		if !errors.Is(err, ErrServeExpertRingTooBig) || !errors.As(err, &fe) || fe.Scope != compute.MemoryScopeHost {
			t.Fatalf("UMA explicit over MemAvailable err=%v, want typed host FitTooBig + ErrServeExpertRingTooBig", err)
		}
		split := roomyHost
		split.SharedPool = false
		if _, err := sizeServeExpertRing(serveExpertRingSize{Mode: serveExpertRingExplicit, Bytes: 100 * ringGiB}, f, token, nonRing, 1<<50, split); err != nil {
			t.Fatalf("a large carve-out keeps device rows out of the host bound: %v", err)
		}
	})
	t.Run("auto_refuses_when_floor_does_not_fit", func(t *testing.T) {
		starved := serveExpertRingHost{Total: 16 * ringGiB, Avail: 8 * ringGiB, Known: true, SharedPool: true}
		_, err := sizeServeExpertRing(serveExpertRingSize{Mode: serveExpertRingAuto}, f, token, nonRing, 102*ringGiB, starved)
		if !errors.Is(err, ErrServeExpertRingTooBig) {
			t.Fatalf("auto with no host room err=%v, want ErrServeExpertRingTooBig", err)
		}
	})
	t.Run("auto_pool_fraction_default", func(t *testing.T) {
		total := 1249 * ringGiB / 10 // strix3 MemTotal, 124.9 GiB
		strix := serveExpertRingHost{Total: total, Avail: 118 * ringGiB, Known: true, RSSAnonPeak: 2 * ringGiB, SharedPool: true}
		s, err := sizeServeExpertRing(serveExpertRingSize{Mode: serveExpertRingAuto}, f, token, nonRing, 200*ringGiB, strix)
		want := int64(0.10 * float64(total))
		if err != nil || s.RingBytes != want || s.PoolCapBytes != want || s.PoolFraction != 0.10 {
			t.Fatalf("auto ring=%d cap=%d fraction=%v err=%v, want %d at 0.10", s.RingBytes, s.PoolCapBytes, s.PoolFraction, err, want)
		}
		if frag := fmt.Sprintf("pool_cap=%d (fraction=0.10)", want); !strings.Contains(s.Text(), frag) {
			t.Fatalf("log line lacks %q", frag)
		}
		size, err := resolveServeExpertRingSize("", ringEnv("auto", "0.20"))
		if err != nil || size.PoolFraction != 0.20 {
			t.Fatalf("env fraction %+v err=%v", size, err)
		}
		s, err = sizeServeExpertRing(size, f, token, nonRing, 200*ringGiB, strix)
		if want := int64(0.20 * float64(total)); err != nil || s.RingBytes != want {
			t.Fatalf("auto ring=%d err=%v, want %d at 0.20", s.RingBytes, err, want)
		}
		s, err = sizeServeExpertRing(serveExpertRingSize{Mode: serveExpertRingExplicit, Bytes: 20 * ringGiB}, f, token, nonRing, 200*ringGiB, strix)
		if err != nil || s.RingBytes != 20*ringGiB || s.PoolCapBytes != 0 {
			t.Fatalf("explicit ring=%d cap=%d err=%v, want 20 GiB uncapped", s.RingBytes, s.PoolCapBytes, err)
		}
	})
	t.Run("auto_pool_fraction_gtt_tier", func(t *testing.T) {
		total := 1249 * ringGiB / 10
		gtt := serveExpertRingHost{Total: total, Avail: 118 * ringGiB, Known: true, RSSAnonPeak: 2 * ringGiB, SharedPool: true, Carveout: 512 << 20, CarveoutKnown: true}
		s, err := sizeServeExpertRing(serveExpertRingSize{Mode: serveExpertRingAuto}, f, token, nonRing, 200*ringGiB, gtt)
		if want := int64(0.5 * float64(total)); err != nil || s.RingBytes != want || s.PoolFraction != 0.5 {
			t.Fatalf("gtt-tier auto ring=%d fraction=%v err=%v, want %d at 0.5", s.RingBytes, s.PoolFraction, err, want)
		}
		s, err = sizeServeExpertRing(serveExpertRingSize{Mode: serveExpertRingAuto, PoolFraction: 0.20}, f, token, nonRing, 200*ringGiB, gtt)
		if want := int64(0.20 * float64(total)); err != nil || s.RingBytes != want {
			t.Fatalf("gtt-tier env 0.20 ring=%d err=%v, want %d", s.RingBytes, err, want)
		}
		for name, h := range map[string]serveExpertRingHost{
			"carveout_unknown": {Total: total, Avail: 118 * ringGiB, Known: true, SharedPool: true},
			"carveout_large":   {Total: total, Avail: 118 * ringGiB, Known: true, SharedPool: false, Carveout: 64 * ringGiB, CarveoutKnown: true},
		} {
			if got := serveExpertRingDefaultPoolFraction(h); got != serveExpertRingPoolFractionDef {
				t.Errorf("%s default fraction=%v, want %v", name, got, serveExpertRingPoolFractionDef)
			}
		}
	})
	t.Run("auto_pool_cap_below_floor_takes_floor", func(t *testing.T) {
		small := serveExpertRingHost{Total: 512 << 20, Avail: 100 * ringGiB, Known: true, SharedPool: true}
		s, err := sizeServeExpertRing(serveExpertRingSize{Mode: serveExpertRingAuto}, f, token, nonRing, 200*ringGiB, small)
		if err != nil || s.PoolCapBytes >= f.ActivatedLayerBytes || s.RingBytes != f.ActivatedLayerBytes {
			t.Fatalf("auto ring=%d cap=%d err=%v, want floor %d", s.RingBytes, s.PoolCapBytes, err, f.ActivatedLayerBytes)
		}
	})
	t.Run("explicit_below_floor", func(t *testing.T) {
		if _, err := sizeServeExpertRing(serveExpertRingSize{Mode: serveExpertRingExplicit, Bytes: 1 << 20}, f, token, nonRing, 102*ringGiB, roomyHost); !errors.Is(err, ErrServeExpertRingSize) {
			t.Fatalf("explicit below one layer err=%v, want ErrServeExpertRingSize", err)
		}
	})
}

// The startup log names the bound that set an auto ring, so a receipt can tell a live
// MemAvailable or device-budget reading from the pool cap without re-deriving the inputs.
// fak-test:runtime fast est=1ms lane=default
func TestSizeServeExpertRingBinding(t *testing.T) {
	f := ringSizingFit()
	const token = 3 * ringGiB
	nonRing := ringSizingNonRing(f, 2*ringGiB) // device 5 GiB, host page reserve 4 GiB
	total := 1196 * ringGiB / 10               // strix3 MemTotal, 119.6 GiB: GTT pool cap 59.8 GiB
	gtt := func(avail int64) serveExpertRingHost {
		return serveExpertRingHost{Total: total, Avail: avail, Known: true, RSSAnonPeak: 2 * ringGiB, SharedPool: true, Carveout: 512 << 20, CarveoutKnown: true}
	}
	auto := serveExpertRingSize{Mode: serveExpertRingAuto}
	for _, tc := range []struct {
		name        string
		size        serveExpertRingSize
		fit         ggufload.ActivatedExpertFit
		deviceAvail int64
		host        serveExpertRingHost
		want        serveExpertRingBinding
	}{
		{"pool_cap", auto, f, 200 * ringGiB, gtt(118 * ringGiB), serveExpertRingBindingPool},
		{"device_bound", auto, f, 40 * ringGiB, gtt(118 * ringGiB), serveExpertRingBindingDevice},
		// The same box with 43 GiB less MemAvailable at load binds on the live host reading.
		{"host_bound", auto, f, 200 * ringGiB, gtt(75 * ringGiB), serveExpertRingBindingHost},
		{"band", auto, ggufload.ActivatedExpertFit{DeviceBaseBytes: f.DeviceBaseBytes, RoutedBandBytes: 20 * ringGiB, ActivatedLayerBytes: f.ActivatedLayerBytes, ActivatedTokenBytes: f.ActivatedTokenBytes, HostGatheredBytes: f.HostGatheredBytes}, 200 * ringGiB, gtt(118 * ringGiB), serveExpertRingBindingBand},
		{"floor", auto, f, 200 * ringGiB, serveExpertRingHost{Total: 512 << 20, Avail: 100 * ringGiB, Known: true, SharedPool: true}, serveExpertRingBindingFloor},
		{"auto_unmeasured", auto, f, compute.FreeUnknown, serveExpertRingHost{}, serveExpertRingBindingNone},
		{"token", serveExpertRingSize{Mode: serveExpertRingToken}, f, 200 * ringGiB, gtt(118 * ringGiB), serveExpertRingBindingNone},
		{"explicit", serveExpertRingSize{Mode: serveExpertRingExplicit, Bytes: 20 * ringGiB}, f, 200 * ringGiB, gtt(118 * ringGiB), serveExpertRingBindingNone},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, err := sizeServeExpertRing(tc.size, tc.fit, token, nonRing, tc.deviceAvail, tc.host)
			if err != nil {
				t.Fatalf("sizing err=%v", err)
			}
			if s.Binding != tc.want {
				t.Fatalf("binding=%q ring=%d, want %q", s.Binding, s.RingBytes, tc.want)
			}
			if frag := "binding=" + string(tc.want) + " "; !strings.Contains(s.Text(), frag) {
				t.Fatalf("log line lacks %q", frag)
			}
		})
	}
}

func ringEnv(ring, fraction string) func(string) string {
	return func(k string) string {
		switch k {
		case FakExpertRingBytesEnv:
			return ring
		case FakExpertRingPoolFractionEnv:
			return fraction
		}
		return ""
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestResolveServeExpertRingPoolFraction(t *testing.T) {
	if size, err := resolveServeExpertRingSize("", ringEnv("auto", "")); err != nil || size.PoolFraction != 0 {
		t.Fatalf("unset fraction %+v err=%v, want 0 (host tier default)", size, err)
	}
	if size, err := resolveServeExpertRingSize("", ringEnv("auto", "0.5")); err != nil || size.PoolFraction != 0.5 {
		t.Fatalf("ceiling fraction %+v err=%v", size, err)
	}
	for _, bad := range []string{"0", "-0.1", "0.51", "1", "abc", "NaN"} {
		if _, err := resolveServeExpertRingSize("", ringEnv("auto", bad)); !errors.Is(err, ErrServeExpertRingSize) {
			t.Errorf("fraction %q err=%v, want ErrServeExpertRingSize", bad, err)
		}
	}
	if size, err := resolveServeExpertRingSize("", ringEnv("token", "abc")); err != nil || size.Mode != serveExpertRingToken {
		t.Fatalf("non-auto mode must not read the fraction: %+v err=%v", size, err)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestResolveServeExpertRingSizeFlagEnvParity(t *testing.T) {
	for _, tc := range []struct {
		fraction string
		want     float64
		wantErr  bool
	}{
		{fraction: "0.20", want: 0.20},
		{fraction: "", want: 0},
		{fraction: "0.5", want: 0.5},
		{fraction: "0", wantErr: true},
		{fraction: "-0.1", wantErr: true},
		{fraction: "0.51", wantErr: true},
		{fraction: "1", wantErr: true},
		{fraction: "abc", wantErr: true},
		{fraction: "NaN", wantErr: true},
		{fraction: "+Inf", wantErr: true},
	} {
		envSize, envErr := resolveServeExpertRingSize("", ringEnv("auto", tc.fraction))
		flagSize, flagErr := resolveServeExpertRingSize(" Auto ", ringEnv("invalid", tc.fraction))
		if tc.wantErr {
			if !errors.Is(envErr, ErrServeExpertRingSize) || !errors.Is(flagErr, ErrServeExpertRingSize) || envErr.Error() != flagErr.Error() {
				t.Fatalf("auto flag/env fraction=%q: flag err=%v, env err=%v, want matching fraction refusals", tc.fraction, flagErr, envErr)
			}
			continue
		}
		want := serveExpertRingSize{Mode: serveExpertRingAuto, PoolFraction: tc.want}
		if envErr != nil || flagErr != nil || envSize != want || flagSize != want {
			t.Fatalf("auto flag/env fraction=%q: flag=%+v err=%v, env=%+v err=%v, want %+v", tc.fraction, flagSize, flagErr, envSize, envErr, want)
		}
	}
	for _, flag := range []string{"token", "2g"} {
		want, _ := parseServeExpertRingSize(flag)
		if size, err := resolveServeExpertRingSize(flag, ringEnv("auto", "abc")); err != nil || size != want {
			t.Fatalf("non-auto flag %q must win and ignore the fraction: %+v err=%v, want %+v", flag, size, err, want)
		}
	}
	if size, err := resolveServeExpertRingSize("auto", nil); err != nil || size != (serveExpertRingSize{Mode: serveExpertRingAuto}) {
		t.Fatalf("auto without an environment: %+v err=%v", size, err)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestServeActivatedExpertRingTokenPlacementIsUnchanged(t *testing.T) {
	ws := serveStreamedSynthWeightSource(t)
	be := serveCapBackend{total: 1 << 20, free: 1 << 20, known: true}
	fit := serveFitBudget{Base: 1 << 20}
	legacy, ok, err := serveActivatedExpertRingPlacement(ws, be, true, serveRingFullPlanTooBig(), 64, fit)
	if err != nil || !ok {
		t.Fatalf("legacy placement ok=%v err=%v", ok, err)
	}
	sized, ok, err := serveActivatedExpertRingPlacementSized(ws, be, true, serveRingFullPlanTooBig(), 64, fit, serveExpertRingSize{Mode: serveExpertRingToken}, serveExpertRingHost{Total: 1 << 30, Avail: 1, Known: true, SharedPool: true})
	if err != nil || !ok {
		t.Fatalf("token placement ok=%v err=%v", ok, err)
	}
	if sized.RingBytes != legacy.RingBytes || len(sized.Plan) != len(legacy.Plan) || sized.Plan.Total() != legacy.Plan.Total() {
		t.Fatalf("token placement moved: ring %d vs %d, plan %d vs %d", sized.RingBytes, legacy.RingBytes, sized.Plan.Total(), legacy.Plan.Total())
	}
	for i := range legacy.Plan {
		if sized.Plan[i] != legacy.Plan[i] {
			t.Fatalf("token plan row %d: %+v vs %+v", i, sized.Plan[i], legacy.Plan[i])
		}
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestServeActivatedExpertRingAutoReservesKVBeforeRing(t *testing.T) {
	ws := serveStreamedSynthWeightSource(t)
	be := serveCapBackend{total: 1 << 30, free: 1 << 30, known: true}
	fit := serveFitBudget{Base: 1 << 30}
	token, ok, err := serveActivatedExpertRingPlacement(ws, be, true, serveRingFullPlanTooBig(), 64, fit)
	if err != nil || !ok {
		t.Fatalf("token placement ok=%v err=%v", ok, err)
	}
	auto, ok, err := serveActivatedExpertRingPlacementSized(ws, be, true, serveRingFullPlanTooBig(), 64, fit, serveExpertRingSize{Mode: serveExpertRingAuto}, serveExpertRingHost{})
	if err != nil || !ok {
		t.Fatalf("auto placement ok=%v err=%v", ok, err)
	}
	if auto.RingBytes < token.RingBytes || auto.RingBytes > auto.Fit.RoutedBandBytes || auto.Sizing.Mode != serveExpertRingAuto {
		t.Fatalf("auto ring=%d token=%d band=%d mode=%s", auto.RingBytes, token.RingBytes, auto.Fit.RoutedBandBytes, auto.Sizing.Mode)
	}
	nonRing := func(p compute.MemoryPlan) compute.MemoryPlan { return serveActivatedExpertRingSplit(p) }
	a, b := nonRing(auto.Plan), nonRing(token.Plan)
	if len(a) != len(b) {
		t.Fatalf("auto changed the non-ring rows: %+v vs %+v", a, b)
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("auto moved non-ring row %d (KV must be reserved first): %+v vs %+v", i, a[i], b[i])
		}
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestServeExpertRingContextWeightsForAllModes(t *testing.T) {
	ws := serveStreamedSynthWeightSource(t)
	be := serveCapBackend{total: 1 << 20, free: 1 << 20, known: true}
	fit := serveFitBudget{Base: 1 << 20}
	full := compute.MemoryPlan{{Class: compute.MemoryWeights, Bytes: 1 << 40, Detail: "full", Scope: compute.MemoryScopeDevice}}
	want, ok := serveExpertRingContextWeights(ws, be, true, serveExpertRingSize{Mode: serveExpertRingAuto}, full, fit)
	if !ok || want.DeviceTotal() >= full.DeviceTotal() {
		t.Fatalf("auto mode kept full weights: ok=%v total=%d", ok, want.DeviceTotal())
	}
	for _, mode := range []serveExpertRingMode{"", serveExpertRingToken, serveExpertRingAuto, serveExpertRingExplicit} {
		got, ok := serveExpertRingContextWeights(ws, be, true, serveExpertRingSize{Mode: mode}, full, fit)
		if !ok || !reflect.DeepEqual(got, want) {
			t.Errorf("mode %q context weights=%+v ok=%v, want exact bounded token plan %+v", mode, got, ok, want)
		}
	}
	if got, ok := serveExpertRingContextWeights(ws, be, false, serveExpertRingSize{Mode: serveExpertRingToken}, full, fit); ok || !reflect.DeepEqual(got, full) {
		t.Fatal("a serve that is not device-only changed its context sizing")
	}
	if got, ok := serveExpertRingContextWeights(ws, be, true, serveExpertRingSize{Mode: serveExpertRingToken}, full, serveFitBudget{}); ok || !reflect.DeepEqual(got, full) {
		t.Fatal("unknown device capacity changed the resident context weights")
	}
	unstageable, err := ggufload.NewWeightSource(ws.File, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got, ok := serveExpertRingContextWeights(unstageable, be, true, serveExpertRingSize{Mode: serveExpertRingToken}, full, fit); ok || !reflect.DeepEqual(got, full) {
		t.Fatal("an unstageable checkpoint changed its context weights")
	}
	fitting := compute.MemoryPlan{{Class: compute.MemoryWeights, Bytes: 4096, Scope: compute.MemoryScopeDevice}}
	if got, ok := serveExpertRingContextWeights(ws, be, true, serveExpertRingSize{Mode: serveExpertRingToken}, fitting, fit); ok || !reflect.DeepEqual(got, fitting) {
		t.Fatal("fitting resident arm changed its context weights")
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestServeActivatedExpertRingAutoTrimsToTheLoadGuard(t *testing.T) {
	ws := serveStreamedSynthWeightSource(t)
	fit := serveFitBudget{Base: 1 << 40}
	unified := func(host int64) serveCapBackend {
		return serveCapBackend{total: 1 << 40, free: 1 << 40, known: true, hostTotal: host, hostFree: host, hostKnown: true, tier: "integrated:test"}
	}
	token, ok, err := serveActivatedExpertRingPlacement(ws, unified(1<<40), true, serveRingFullPlanTooBig(), 64, fit)
	if err != nil || !ok {
		t.Fatalf("token placement ok=%v err=%v", ok, err)
	}
	nonRing := token.Plan.Total() - token.RingBytes
	floor, band := token.Fit.ActivatedLayerBytes, token.Fit.RoutedBandBytes
	target := floor + (band-floor)/2
	host := int64(float64(nonRing+target)/(1-serveGGUFDeviceHeadroom)) + 1
	auto, ok, err := serveActivatedExpertRingPlacementSized(ws, unified(host), true, serveRingFullPlanTooBig(), 64, fit, serveExpertRingSize{Mode: serveExpertRingAuto}, serveExpertRingHost{})
	if err != nil || !ok {
		t.Fatalf("auto placement against a small physical pool ok=%v err=%v", ok, err)
	}
	pool := compute.BudgetAfterHeadroom(host, serveGGUFDeviceHeadroom)
	if auto.Sizing.Binding != serveExpertRingBindingGuard || !strings.Contains(auto.Sizing.Text(), "binding=guard ") {
		t.Fatalf("guard-trimmed auto ring binding=%q, want %q in the sizing and its log line", auto.Sizing.Binding, serveExpertRingBindingGuard)
	}
	if auto.Sizing.GuardTrimBytes <= 0 || auto.RingBytes < floor || auto.RingBytes >= band || auto.Plan.Total() > pool {
		t.Fatalf("auto ring=%d trim=%d floor=%d band=%d plan=%d pool=%d, want a ring trimmed to the physical pool",
			auto.RingBytes, auto.Sizing.GuardTrimBytes, floor, band, auto.Plan.Total(), pool)
	}
	tooSmall := compute.BudgetAfterHeadroom(nonRing, 0) // not even the floor fits
	_, ok, err = serveActivatedExpertRingPlacementSized(ws, unified(tooSmall), true, serveRingFullPlanTooBig(), 64, fit, serveExpertRingSize{Mode: serveExpertRingAuto}, serveExpertRingHost{})
	if ok || !errors.Is(err, ErrServeExpertRingTooBig) {
		t.Fatalf("auto placement with no room for the floor ok=%v err=%v, want ErrServeExpertRingTooBig", ok, err)
	}
}

// serveContextHeaderReader makes any accidental payload read fail immediately.
// The fixture below describes realistic byte counts without allocating those bytes.
type serveContextHeaderReader struct{ reads int }

func (r *serveContextHeaderReader) ReadAt([]byte, int64) (int, error) {
	r.reads++
	panic("header-only context admission read a tensor payload")
}

// fak-test:runtime fast est=20ms lane=default
func TestServeTokenExpertRingContextAdmission(t *testing.T) {
	t.Setenv("FAK_HYBRID_KV", "0")
	t.Setenv("FAK_UP_KV_PRECISION", "f32")
	const (
		gib          = int64(1 << 30)
		layers       = 40
		hidden       = 5120
		intermediate = 2304
		experts      = 384
		active       = 6
		denseBytes   = 3 * gib
		engramRows   = uint64(384000000)
		declared     = 1048576
		requested    = 4096
	)
	// Published Flash geometry with a synthetic 3 GiB dense remainder, Q2_K
	// gate/up and Q3_K down slabs, and two packed Engram tables. This is an
	// accounting fixture, not a runnable model or a physical allocation receipt.
	f := &ggufload.File{Metadata: map[string]ggufload.Value{
		"general.architecture":                        {Type: ggufload.TypeString, Value: "deepseek41"},
		"deepseek41.embedding_length":                 {Type: ggufload.TypeUint64, Value: uint64(hidden)},
		"deepseek41.block_count":                      {Type: ggufload.TypeUint64, Value: uint64(layers)},
		"deepseek41.context_length":                   {Type: ggufload.TypeUint64, Value: uint64(declared)},
		"deepseek41.feed_forward_length":              {Type: ggufload.TypeUint64, Value: uint64(intermediate)},
		"deepseek41.attention.head_count":             {Type: ggufload.TypeUint64, Value: uint64(64)},
		"deepseek41.attention.head_count_kv":          {Type: ggufload.TypeUint64, Value: uint64(1)},
		"deepseek41.attention.layer_norm_rms_epsilon": {Type: ggufload.TypeFloat32, Value: float32(1e-6)},
		"deepseek41.attention.q_lora_rank":            {Type: ggufload.TypeUint64, Value: uint64(1280)},
		"deepseek41.attention.kv_lora_rank":           {Type: ggufload.TypeUint64, Value: uint64(512)},
		"deepseek41.attention.qk_nope_head_dim":       {Type: ggufload.TypeUint64, Value: uint64(448)},
		"deepseek41.attention.qk_rope_head_dim":       {Type: ggufload.TypeUint64, Value: uint64(64)},
		"deepseek41.attention.value_length_mla":       {Type: ggufload.TypeUint64, Value: uint64(512)},
		"deepseek41.attention.output_group_count":     {Type: ggufload.TypeUint64, Value: uint64(8)},
		"deepseek41.attention.output_lora_rank":       {Type: ggufload.TypeUint64, Value: uint64(1024)},
		"deepseek41.expert_count":                     {Type: ggufload.TypeUint64, Value: uint64(experts)},
		"deepseek41.expert_used_count":                {Type: ggufload.TypeUint64, Value: uint64(active)},
		"deepseek41.expert_feed_forward_length":       {Type: ggufload.TypeUint64, Value: uint64(intermediate)},
	}}
	var offset int64
	add := func(name string, dims []uint64, typ ggufload.TensorType, bytes int64) {
		f.Tensors = append(f.Tensors, ggufload.TensorInfo{Name: name, Dims: dims, Type: typ, FileOffset: offset})
		offset += bytes
	}
	add("dense_fixture.weight", []uint64{uint64(denseBytes / 4)}, ggufload.TensorF32, denseBytes)
	const blocksPerSlab = int64(hidden) * intermediate * experts / 256
	for l := 0; l < layers; l++ {
		add(fmt.Sprintf("blk.%d.ffn_gate_exps.weight", l), []uint64{hidden, intermediate, experts}, ggufload.TensorQ2_K, blocksPerSlab*84)
		add(fmt.Sprintf("blk.%d.ffn_up_exps.weight", l), []uint64{hidden, intermediate, experts}, ggufload.TensorQ2_K, blocksPerSlab*84)
		add(fmt.Sprintf("blk.%d.ffn_down_exps.weight", l), []uint64{intermediate, hidden, experts}, ggufload.TensorQ3_K, blocksPerSlab*110)
	}
	for _, l := range []int{1, 14} {
		add(fmt.Sprintf("blk.%d.engram_embd.weight", l), []uint64{256, engramRows}, ggufload.TensorQ2_K, int64(engramRows)*84)
	}
	reader := &serveContextHeaderReader{}
	ws, err := ggufload.NewWeightSource(f, reader, offset)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := ws.Close(); err != nil {
			t.Error(err)
		}
	})
	cfg, err := ws.File.Config()
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.IsDeepSeekV41() || cfg.MaxPositionEmbeddings != declared || cfg.HeadDim != 512 {
		t.Fatalf("fixture lost V4.1 geometry: %+v", cfg)
	}
	full, err := serveGGUFWeightMemoryPlanForArm(ws, serveLoadArmResidentQ4K)
	if err != nil {
		t.Fatal(err)
	}
	const routedBytes = blocksPerSlab * (84 + 84 + 110) * layers
	const gatheredBytes = int64(engramRows) * 84 * 2
	if full.Total() != denseBytes+routedBytes+gatheredBytes || full.Total() <= 128*gib {
		t.Fatalf("full header charge=%d, want %d above the device budget", full.Total(), denseBytes+routedBytes+gatheredBytes)
	}
	// A 128 GiB integrated pool with 96 GiB free. The reported capacities are
	// deterministic CPU test doubles; no accelerator capability is exercised.
	be := serveCapBackend{total: 128 * gib, free: 128 * gib, known: true, hostTotal: 128 * gib, hostFree: 96 * gib, hostKnown: true, tier: "integrated:context-test"}
	fit := serveDeviceFitBudget(be)
	baseline, _, err := resolveServeNativeContext(ws, be, full, fit, requested)
	if err != nil || baseline.ResolvedTokens != compute.MinAutoContextTokens {
		t.Fatalf("full-charge control=%+v err=%v, want unchanged 512-token clamp", baseline, err)
	}
	mode, err := resolveServeExpertRingSize("", nil)
	if err != nil || mode.Mode != serveExpertRingToken {
		t.Fatalf("default ring mode=%+v err=%v", mode, err)
	}
	weights, selected := serveExpertRingContextWeights(ws, be, true, mode, full, fit)
	resolution, contextPlan, err := resolveServeNativeContext(ws, be, weights, fit, requested)
	if err != nil {
		t.Fatal(err)
	}
	if resolution.ResolvedTokens != requested || !selected {
		t.Fatalf("default token-ring context resolved=%d selected=%v, want explicit4096 using its bounded load charge (full=%d, effective=%d)", resolution.ResolvedTokens, selected, full.Total(), weights.Total())
	}
	if resolution.RequestedTokens != requested || resolution.ModelDeclaredTokens != declared || resolution.Source != "explicit" {
		t.Fatalf("resolution changed declaration or request: %+v", resolution)
	}
	const wantRing = routedBytes / experts * active
	wantWeights := compute.MemoryPlan{
		{Class: compute.MemoryWeights, Bytes: denseBytes, Detail: serveActivatedRingBaseDetail, Scope: compute.MemoryScopeDevice},
		{Class: compute.MemoryWeights, Bytes: wantRing, Detail: serveActivatedRingDetail, Scope: compute.MemoryScopeDevice},
		{Class: compute.MemoryWeights, Bytes: 4 * gib, Detail: ggufload.HostGatheredDetail, Scope: compute.MemoryScopeHost},
	}
	if !reflect.DeepEqual(weights, wantWeights) {
		t.Fatalf("effective weights=%+v, want independent dense+token+page sum %+v", weights, wantWeights)
	}
	_, fullErr := refuseIfTooBigOnDevice(full, nil, be, &fit)
	if !serveFullPlanFitTooBig(fullErr) {
		t.Fatalf("full resident control did not refuse: %v", fullErr)
	}
	ring, ok, err := serveActivatedExpertRingPlacement(ws, be, true, fullErr, requested, fit)
	if err != nil || !ok {
		t.Fatalf("bounded load plan failed: ok=%v err=%v", ok, err)
	}
	var loadWeights, loadContext compute.MemoryPlan
	transientRows := 0
	for _, row := range ring.Plan {
		if row.Detail == "gguf-host-expert-band" {
			t.Fatal("disk-only expert remainder became a host allocation")
		}
		if row.Class == compute.MemoryWeights {
			loadWeights = append(loadWeights, row)
			continue
		}
		if row.Detail == "v41-forward-transient-scratch" {
			transientRows++
			if row.Bytes <= 0 {
				t.Fatal("V4.1 transient charge vanished")
			}
			continue
		}
		loadContext = append(loadContext, row)
	}
	if !reflect.DeepEqual(loadWeights, weights) || !reflect.DeepEqual(loadContext, contextPlan) || transientRows != 1 {
		t.Fatalf("load/resolve mismatch: weights=%+v context=%+v transientRows=%d; want weights=%+v context=%+v plus one transient", loadWeights, loadContext, transientRows, weights, contextPlan)
	}
	// Keep a real shared-pool refusal: device room alone cannot erase the dense,
	// ring and Engram-page charges when physical host RAM is below their sum.
	small := be
	small.hostTotal, small.hostFree = 8*gib, 8*gib
	refused, ok, err := serveActivatedExpertRingPlacement(ws, small, true, fullErr, requested, fit)
	var tooBig *compute.FitError
	if ok || !errors.As(err, &tooBig) || tooBig.Verdict != compute.FitTooBig || tooBig.Want <= tooBig.Avail || refused.Plan != nil {
		t.Fatalf("genuinely too-small shared pool accepted: ok=%v plan=%+v err=%v", ok, refused.Plan, err)
	}
	if reader.reads != 0 {
		t.Fatalf("header admission read %d payloads", reader.reads)
	}
}
