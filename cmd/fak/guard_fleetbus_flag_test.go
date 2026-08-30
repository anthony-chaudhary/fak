package main

import (
	"flag"
	"io"
	"testing"
	"time"
)

func TestGuardFleetBusFlagComposesOperatorSettings(t *testing.T) {
	fs := flag.NewFlagSet("guard", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	got := newGuardFleetBusFlag()
	fs.Var(&got, "fleet-bus", "")
	if err := fs.Parse([]string{"--fleet-bus=on,dir=/shared/bus,id=worker-7,interval=12s"}); err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if !got.enabled || got.dir != "/shared/bus" || got.id != "worker-7" || got.interval != 12*time.Second {
		t.Fatalf("fleet-bus settings = %+v", got)
	}
}

func TestGuardFleetBusFlagPreservesBooleanSpelling(t *testing.T) {
	for _, tc := range []struct {
		args []string
		want bool
	}{
		{args: []string{"--fleet-bus"}, want: true},
		{args: []string{"--fleet-bus=false"}, want: false},
	} {
		fs := flag.NewFlagSet("guard", flag.ContinueOnError)
		fs.SetOutput(io.Discard)
		got := newGuardFleetBusFlag()
		fs.Var(&got, "fleet-bus", "")
		if err := fs.Parse(tc.args); err != nil {
			t.Fatalf("Parse(%v): %v", tc.args, err)
		}
		if got.enabled != tc.want {
			t.Fatalf("Parse(%v) enabled = %t, want %t", tc.args, got.enabled, tc.want)
		}
	}
}

func TestGuardFleetBusFlagRejectsUnknownSetting(t *testing.T) {
	got := newGuardFleetBusFlag()
	if err := got.Set("on,ttl=1m"); err == nil {
		t.Fatal("unknown fleet-bus setting accepted")
	}
}

func TestRewriteLegacyGuardFleetBusArgsPreservesSettingsAndBoundary(t *testing.T) {
	got, err := rewriteLegacyGuardFleetBusArgs([]string{
		"--fleet-bus-dir", "/shared/bus",
		"--fleet-bus-id=worker-7",
		"--fleet-bus-interval", "12s",
		"--", "agent", "--fleet-bus-dir", "child-value",
	})
	if err != nil {
		t.Fatalf("rewrite: %v", err)
	}
	want := []string{
		"--fleet-bus=dir=/shared/bus",
		"--fleet-bus=id=worker-7",
		"--fleet-bus=interval=12s",
		"--", "agent", "--fleet-bus-dir", "child-value",
	}
	if len(got) != len(want) {
		t.Fatalf("rewrite = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("rewrite[%d] = %q, want %q; all=%v", i, got[i], want[i], got)
		}
	}
}
