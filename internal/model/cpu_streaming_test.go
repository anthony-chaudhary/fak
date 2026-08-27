package model

import "testing"

func TestMeasureCPUStreamingSaturation(t *testing.T) {
	r, err := MeasureCPUStreaming([]CPUStreamingSample{{Threads: 4, LocalBytes: 900, RemoteBytes: 100, Nanoseconds: 100, Joules: 10, AcceptedTokens: 10}, {Threads: 8, LocalBytes: 1900, RemoteBytes: 100, Nanoseconds: 100, Joules: 12, AcceptedTokens: 10}, {Threads: 16, LocalBytes: 1700, RemoteBytes: 700, Nanoseconds: 130, Joules: 20, AcceptedTokens: 10}})
	if err != nil {
		t.Fatal(err)
	}
	if r.Engine != "fak-native" || r.BestThreads != 8 || r.SustainableGBps != 20 || r.RemoteTrafficRatio != .05 || !r.Saturated || !r.Regression || r.JoulesPerAccepted != 1.2 {
		t.Fatalf("receipt=%+v", r)
	}
}
func TestMeasureCPUStreamingRejectsInvalid(t *testing.T) {
	if _, err := MeasureCPUStreaming([]CPUStreamingSample{{Threads: 1}, {Threads: 2}}); err == nil {
		t.Fatal("invalid sample accepted")
	}
}
