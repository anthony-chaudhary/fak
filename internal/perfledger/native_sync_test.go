package perfledger

import (
	"encoding/json"
	"math"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestDeviceSyncRatesRoundTrip(t *testing.T) {
	sites := []NativeSyncSite{
		{Site: "flush_status", Stage: "v41_exact_rmsnorm", SyncsPerToken: 43, WaitMSPerToken: 2.15},
		{Site: "readback", Stage: "none", SyncsPerToken: 1, WaitMSPerToken: 0.05, D2HBytesPerToken: 4096},
	}
	n := NewNativeTiming(128, 11, 10, 100).WithDeviceSync(80.5, true, 1640, true, sites)
	raw, err := json.Marshal(Record{NativeTiming: n})
	if err != nil {
		t.Fatal(err)
	}
	var back Record
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatal(err)
	}
	got := back.NativeTiming
	if got == nil || got.GPUIdleMSPerToken == nil || *got.GPUIdleMSPerToken != 80.5 ||
		got.SyncsPerToken == nil || *got.SyncsPerToken != 1640 || !reflect.DeepEqual(got.SyncSites, sites) {
		t.Fatalf("round trip = %s", raw)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestDeviceSyncRatesAbsentIsNotZero(t *testing.T) {
	base := NewNativeTiming(0, 8, 0, 2)
	want, _ := json.Marshal(base)
	off := base.WithDeviceSync(0, false, 0, false, []NativeSyncSite{{Site: "readback", Stage: "none", SyncsPerToken: 1}})
	if raw, _ := json.Marshal(off); string(raw) != string(want) {
		t.Fatalf("unmeasured sync rates changed the row: %s, want %s", raw, want)
	}
	for name, n := range map[string]*NativeTiming{
		"nan_idle":       base.WithDeviceSync(math.NaN(), true, 0, false, nil),
		"negative_syncs": base.WithDeviceSync(0, false, -1, true, nil),
		"inf_site_wait":  base.WithDeviceSync(0, false, 2, true, []NativeSyncSite{{Site: "readback", Stage: "none", SyncsPerToken: 2, WaitMSPerToken: math.Inf(1)}}),
	} {
		if n.GPUIdleMSPerToken != nil || n.SyncsPerToken != nil || n.SyncSites != nil {
			t.Fatalf("%s: %+v, want idle and syncs absent", name, n)
		}
	}
	zero := base.WithDeviceSync(0, true, 0, true, nil)
	if zero.GPUIdleMSPerToken == nil || *zero.GPUIdleMSPerToken != 0 || zero.SyncsPerToken == nil || *zero.SyncsPerToken != 0 {
		t.Fatalf("a measured zero must stay a measurement: %+v", zero)
	}
	if (*NativeTiming)(nil).WithDeviceSync(1, true, 1, true, nil) != nil {
		t.Fatal("nil timing must stay nil")
	}
}
