package model

import "testing"

func TestMeasureSustainableBandwidthSeparatesDerating(t *testing.T) {
	samples := []HardwareSample{{TimestampNS: 0, AcceptedTokens: 10, SoftwareBytes: 1000, DRAMBytes: 1200, ClockMHz: 1200, PowerWatts: 100, TemperatureC: 60, ConcurrentTenants: 1}, {TimestampNS: 1e9, AcceptedTokens: 10, SoftwareBytes: 1000, DRAMBytes: 1200, ClockMHz: 900, PowerWatts: 80, TemperatureC: 85, Throttled: true, ConcurrentTenants: 2}}
	r, err := MeasureSustainableBandwidth(samples)
	if err != nil {
		t.Fatal(err)
	}
	if r.Engine != "fak-native" || r.ByteAmplification != 1.2 || r.SustainableGBps != 0.0000024 || r.JoulesPerAccepted != 4.5 || r.MinClockMHz != 900 || r.ThrottledSamples != 1 || r.MaxConcurrentTenants != 2 {
		t.Fatalf("receipt=%+v", r)
	}
}
func TestMeasureSustainableBandwidthRejectsMisalignedTime(t *testing.T) {
	_, err := MeasureSustainableBandwidth([]HardwareSample{{TimestampNS: 1}, {TimestampNS: 1}})
	if err == nil {
		t.Fatal("non-monotonic samples accepted")
	}
}
