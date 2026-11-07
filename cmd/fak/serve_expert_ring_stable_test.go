package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// h200RingFit is the DS4.1 Q2_K census the rtp-test H200 ring plan line reported: band 183.25 GiB,
// floor 73.3 MiB, token 2.863 GiB, dense base 3.0 GiB, host-gathered Engram 60.08 GiB.
func h200RingFit() ggufload.ActivatedExpertFit {
	return ggufload.ActivatedExpertFit{
		DeviceBaseBytes: 3 * ringGiB, RoutedBandBytes: 183*ringGiB + ringGiB/4,
		ActivatedLayerBytes: 73 << 20, ActivatedTokenBytes: 2863 << 20,
		HostGatheredBytes: 60 * ringGiB,
	}
}

// fak-test:runtime fast est=1ms lane=default
// Q1 witness: on the measured H200 budget (device_avail 118.83 GiB, 2 TiB host) nothing caps the ring
// at 32 GiB — the default (host-defaulted auto) ring is device-bound at ~112 GiB, and the stable set
// then keeps ~84 GiB of the band across sweeps.
func TestH200AutoRingIsDeviceBoundNot32GiB(t *testing.T) {
	f := h200RingFit()
	nonRing := ringSizingNonRing(f, 4*ringGiB)
	host := serveExpertRingHost{Total: 2015 * ringGiB, Avail: 1977 * ringGiB, Known: true, RSSAnonPeak: 1300 << 20}
	size := serveExpertRingSize{Mode: serveExpertRingToken, HostDefault: true}.forHostTotal(host.Total, true)
	s, err := sizeServeExpertRing(size, f, f.ActivatedTokenBytes, nonRing, 118831*ringGiB/1000, host)
	if err != nil {
		t.Fatal(err)
	}
	if s.Mode != serveExpertRingAuto || s.Binding != serveExpertRingBindingDevice || s.RingBytes < 100*ringGiB {
		t.Fatalf("auto ring on H200 = %s binding=%s mode=%s, want device-bound >= 100 GiB", bytesText(uint64(s.RingBytes)), s.Binding, s.Mode)
	}
	stable := serveExpertRingStableBytes(s, func(string) string { return "" })
	if stable < 3*s.RingBytes/4-ringGiB || stable >= s.RingBytes {
		t.Fatalf("stable set %s of ring %s, want ~3/4 of the ring", bytesText(uint64(stable)), bytesText(uint64(s.RingBytes)))
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestServeExpertRingStableBytesGates(t *testing.T) {
	none := func(string) string { return "" }
	env := func(v string) func(string) string { return func(string) string { return v } }
	band := 183 * ringGiB
	for _, tc := range []struct {
		name string
		s    serveExpertRingSizing
		env  func(string) string
		want int64
	}{
		{"token ring stays plain", serveExpertRingSizing{Mode: serveExpertRingToken, RingBytes: 64 * ringGiB, Ceiling: band}, none, 0},
		{"small ring keeps LRU by default", serveExpertRingSizing{Mode: serveExpertRingAuto, RingBytes: 16 * ringGiB, Ceiling: band}, none, 0},
		{"small ring forced on", serveExpertRingSizing{Mode: serveExpertRingAuto, RingBytes: 16 * ringGiB, Ceiling: band}, env("on"), 8 * ringGiB},
		{"ring holds the band", serveExpertRingSizing{Mode: serveExpertRingExplicit, RingBytes: band, Ceiling: band}, none, 0},
		{"explicit 32 GiB", serveExpertRingSizing{Mode: serveExpertRingExplicit, RingBytes: 32 * ringGiB, Ceiling: band}, none, 24 * ringGiB},
		{"off", serveExpertRingSizing{Mode: serveExpertRingExplicit, RingBytes: 112 * ringGiB, Ceiling: band}, env("off"), 0},
	} {
		if got := serveExpertRingStableBytes(tc.s, tc.env); got != tc.want {
			t.Errorf("%s: stable=%d want %d", tc.name, got, tc.want)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestServeExpertRingStatsTextReportsDeviceRingHits(t *testing.T) {
	st := fakmodel.SharedExpertRingStats{Agents: 1, Ring: fakmodel.ExpertRingStats{Hits: 3, PageIns: 1, BudgetBytes: ringGiB}}
	if got := serveExpertRingStatsText(st); !strings.Contains(got, "hit_fraction=0.750") || !strings.Contains(got, "page_ins=1") {
		t.Fatalf("stats text %q", got)
	}
	if got := serveExpertRingStatsText(fakmodel.SharedExpertRingStats{}); !strings.Contains(got, "hit_fraction=n/a") {
		t.Fatalf("an untouched ring must read n/a, not 0: %q", got)
	}
}
