package main

import (
	"strings"
	"testing"
)

// serve_expert_ring_default_test.go — an unset --expert-ring-bytes defaults to auto on a high-RAM
// host and to token elsewhere; an explicit token stays the OFF arm; the discrete-host pool cap is
// 0.40 while the integrated tiers keep 0.10 / 0.50.

// fak-test:runtime fast est=1ms lane=default
func TestServeExpertRingHostDefault(t *testing.T) {
	noEnv := func(string) string { return "" }
	unset, err := resolveServeExpertRingSize("", noEnv)
	if err != nil || unset.Mode != serveExpertRingToken || !unset.HostDefault {
		t.Fatalf("unset size %+v err=%v, want host-defaulted token", unset, err)
	}
	for _, tc := range []struct {
		name  string
		size  serveExpertRingSize
		total int64
		known bool
		want  serveExpertRingMode
	}{
		{"high_ram_unset_auto", unset, 128 * ringGiB, true, serveExpertRingAuto},
		{"threshold_unset_auto", unset, serveExpertRingHighRAMHostMin, true, serveExpertRingAuto},
		{"small_host_unset_token", unset, 32 * ringGiB, true, serveExpertRingToken},
		{"unknown_host_unset_token", unset, 0, false, serveExpertRingToken},
	} {
		if got := tc.size.forHostTotal(tc.total, tc.known); got.Mode != tc.want {
			t.Errorf("%s: mode=%s, want %s", tc.name, got.Mode, tc.want)
		}
	}
	// OFF arm: an explicit token, by flag or environment, is never promoted.
	for name, arm := range map[string]func() (serveExpertRingSize, error){
		"flag_token": func() (serveExpertRingSize, error) { return resolveServeExpertRingSize("token", noEnv) },
		"env_token":  func() (serveExpertRingSize, error) { return resolveServeExpertRingSize("", ringEnv("token", "")) },
	} {
		size, err := arm()
		if err != nil || size.HostDefault {
			t.Fatalf("%s: %+v err=%v, want operator-named token", name, size, err)
		}
		if got := size.forHostTotal(512*ringGiB, true); got.Mode != serveExpertRingToken {
			t.Errorf("%s on a 512 GiB host: mode=%s, want token", name, got.Mode)
		}
	}
	// The pool fraction environment still applies to a host-defaulted auto ring.
	if size, err := resolveServeExpertRingSize("", ringEnv("", "0.1")); err != nil || size.PoolFraction != 0.1 || !size.HostDefault {
		t.Fatalf("host default with fraction env: %+v err=%v", size, err)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestServeExpertRingHostDefaultSizing(t *testing.T) {
	f := ringSizingFit()
	const token = 3 * ringGiB
	nonRing := ringSizingNonRing(f, 2*ringGiB)
	unset, err := resolveServeExpertRingSize("", func(string) string { return "" })
	if err != nil {
		t.Fatal(err)
	}
	total := 128 * ringGiB
	discrete := serveExpertRingHost{Total: total, Avail: 120 * ringGiB, Known: true, RSSAnonPeak: 2 * ringGiB}
	s, err := sizeServeExpertRing(unset.forHostTotal(discrete.Total, discrete.Known), f, token, nonRing, 200*ringGiB, discrete)
	want := int64(serveExpertRingPoolFractionDiscrete * float64(total))
	if err != nil || s.Mode != serveExpertRingAuto || s.RingBytes != want || s.PoolFraction != serveExpertRingPoolFractionDiscrete || s.Binding != serveExpertRingBindingPool {
		t.Fatalf("high-RAM discrete default: mode=%s ring=%d fraction=%v binding=%s err=%v, want auto %d at 0.40", s.Mode, s.RingBytes, s.PoolFraction, s.Binding, err, want)
	}
	if !strings.Contains(s.Text(), "default=auto (high-RAM host)") {
		t.Fatalf("log line does not name the host default: %s", s.Text())
	}
	// A 24 GiB device still binds the ring on a discrete host: the fraction is only a cap.
	s, err = sizeServeExpertRing(unset.forHostTotal(discrete.Total, discrete.Known), f, token, nonRing, 24*ringGiB, discrete)
	if err != nil || s.Binding != serveExpertRingBindingDevice || s.RingBytes != 24*ringGiB-s.NonRingDevice {
		t.Fatalf("device-bound discrete default: ring=%d binding=%s err=%v", s.RingBytes, s.Binding, err)
	}
	// Small host: token ring, byte-identical, no default note.
	small := serveExpertRingHost{Total: 32 * ringGiB, Avail: 28 * ringGiB, Known: true}
	s, err = sizeServeExpertRing(unset.forHostTotal(small.Total, small.Known), f, token, nonRing, 200*ringGiB, small)
	if err != nil || s.Mode != serveExpertRingToken || s.RingBytes != token || strings.Contains(s.Text(), "default=") {
		t.Fatalf("small host default: mode=%s ring=%d err=%v", s.Mode, s.RingBytes, err)
	}
	// Explicit auto names no host default.
	s, err = sizeServeExpertRing(serveExpertRingSize{Mode: serveExpertRingAuto}, f, token, nonRing, 200*ringGiB, discrete)
	if err != nil || s.HostDefaultAuto || strings.Contains(s.Text(), "default=") {
		t.Fatalf("explicit auto marked as host default: %s err=%v", s.Text(), err)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestServeExpertRingPoolFractionPerTier(t *testing.T) {
	total := 128 * ringGiB
	for _, tc := range []struct {
		name string
		host serveExpertRingHost
		want float64
	}{
		{"discrete", serveExpertRingHost{Total: total, Known: true}, serveExpertRingPoolFractionDiscrete},
		{"integrated_gtt_small_carveout", serveExpertRingHost{Total: total, Known: true, SharedPool: true, Carveout: 512 << 20, CarveoutKnown: true}, serveExpertRingPoolFractionGTT},
		{"integrated_carveout_unknown", serveExpertRingHost{Total: total, Known: true, SharedPool: true}, serveExpertRingPoolFractionDef},
		{"integrated_large_carveout", serveExpertRingHost{Total: total, Known: true, Carveout: 64 * ringGiB, CarveoutKnown: true}, serveExpertRingPoolFractionDef},
	} {
		if got := serveExpertRingDefaultPoolFraction(tc.host); got != tc.want {
			t.Errorf("%s: fraction=%v, want %v", tc.name, got, tc.want)
		}
	}
	if serveExpertRingPoolFractionDiscrete != 0.40 || serveExpertRingPoolFractionDef != 0.10 || serveExpertRingPoolFractionMax != 0.5 {
		t.Fatal("pool fraction constants drifted from the documented 0.40 / 0.10 / 0.5")
	}
}
