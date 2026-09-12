package main

import "testing"

func TestMetalGuard(t *testing.T) {
	cases := []struct {
		name      string
		compiled  bool
		available bool
		wantExit  int
		wantMsg   string
	}{
		{
			// (false,false): CGO_ENABLED=0 or non-darwin/arm64 — the stub's honest state.
			name:      "stub_no_device",
			compiled:  false,
			available: false,
			wantExit:  1,
			wantMsg:   "gpucheck: metal backend not compiled in (requires darwin/arm64 with cgo — auto-compiled, no build tag needed)",
		},
		{
			// (false,true) is unreachable in shipped builds (the stub's Available() returns
			// false), but the guard is DEFINED to name the compile state whenever
			// Compiled() says not-compiled: the stub can never be trusted to self-report
			// availability, so !compiled wins the switch. Documented, not fabricated.
			name:      "stub_self_reports_available",
			compiled:  false,
			available: true,
			wantExit:  1,
			wantMsg:   "gpucheck: metal backend not compiled in (requires darwin/arm64 with cgo — auto-compiled, no build tag needed)",
		},
		{
			// (true,false): linked into the binary but no Metal device on this host.
			name:      "compiled_no_device",
			compiled:  true,
			available: false,
			wantExit:  1,
			wantMsg:   "gpucheck: metal compiled but no usable Metal device is available on this host",
		},
		{
			// (true,true): the guard reports "proceed" with the device identity; main
			// then exits 2 because the Q4_K micro-dose lane lives in cmd/modelbench,
			// not here. The guard's contract is the decision, not main's exit code.
			name:      "compiled_and_available",
			compiled:  true,
			available: true,
			wantExit:  0,
			wantMsg:   `gpucheck: metal device "Test Metal GPU"`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			saved := metalDeviceName
			t.Cleanup(func() { metalDeviceName = saved })
			metalDeviceName = "Test Metal GPU"
			g := metalGuard(tc.compiled, tc.available)
			if g.exitCode != tc.wantExit {
				t.Errorf("exitCode = %d, want %d", g.exitCode, tc.wantExit)
			}
			if g.message != tc.wantMsg {
				t.Errorf("message = %q, want %q", g.message, tc.wantMsg)
			}
		})
	}
}
