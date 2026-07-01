package dispatchtick

import "testing"

func roomyResources() HostResources {
	return HostResources{Cores: IntPtr(64), FreeRAMMB: IntPtr(128000), TotalThreads: IntPtr(1000)}
}

func preflightInput() PreflightInput {
	return PreflightInput{
		Workspace:  "repo",
		MaxWorkers: 2,
		Host:       HostCheck{Safe: true},
		Account:    AccountCheck{Available: true, Tag: "worker-a", Tier: 1, Model: "claude"},
		Kernel:     KernelCheck{Alive: IntPtr(0), Target: IntPtr(3), Verdict: "FILLING"},
		Seat:       SeatCheck{Total: nil},
		Resources:  roomyResources(),
	}
}

func TestHostCapacityRoomyBoxBoundByCores(t *testing.T) {
	info := HostCapacity(roomyResources())
	if info.HostCap == nil || *info.HostCap != 32 {
		t.Fatalf("host cap = %v, want 32", info.HostCap)
	}
	if info.Binding != "cores" {
		t.Fatalf("binding = %q, want cores", info.Binding)
	}
}

func TestHostCapacityThreadSaturationFloors(t *testing.T) {
	info := HostCapacity(HostResources{Cores: IntPtr(8), FreeRAMMB: IntPtr(64000), TotalThreads: IntPtr(200000)})
	if info.HostCap == nil || *info.HostCap != 1 {
		t.Fatalf("host cap = %v, want floor 1", info.HostCap)
	}
	if info.Binding != "threads" || info.Components["threads"] != 0 {
		t.Fatalf("binding/components = %q/%v, want threads/0", info.Binding, info.Components)
	}
}

func TestEvaluatePreflightSpawnOK(t *testing.T) {
	got := EvaluatePreflight(preflightInput())
	if !got.OK || got.Verdict != PreflightOKVerdict {
		t.Fatalf("verdict = %s ok=%v, want SPAWN_OK", got.Verdict, got.OK)
	}
	if got.Cap != 2 || got.Live != 0 || got.Headroom != 2 {
		t.Fatalf("cap/live/headroom = %d/%d/%d, want 2/0/2", got.Cap, got.Live, got.Headroom)
	}
}

func TestEvaluatePreflightZeroTargetDoesNotCountKernelAlive(t *testing.T) {
	in := preflightInput()
	in.Kernel = KernelCheck{Alive: IntPtr(3), Target: IntPtr(0), Verdict: "OVER_TARGET"}
	got := EvaluatePreflight(in)
	if got.Cap != 2 || got.Live != 0 || got.Verdict != PreflightOKVerdict {
		t.Fatalf("cap/live/verdict = %d/%d/%s, want 2/0/SPAWN_OK", got.Cap, got.Live, got.Verdict)
	}
}

func TestEvaluatePreflightPositiveTargetCountsKernelAlive(t *testing.T) {
	in := preflightInput()
	in.Kernel = KernelCheck{Alive: IntPtr(3), Target: IntPtr(9), Verdict: "OVER_TARGET"}
	got := EvaluatePreflight(in)
	if got.Live != 3 || got.Verdict != PreflightRefuseAtCap {
		t.Fatalf("live/verdict = %d/%s, want 3/REFUSE_AT_CAP", got.Live, got.Verdict)
	}
}

func TestEvaluatePreflightHostCapFolds(t *testing.T) {
	in := preflightInput()
	in.MaxWorkers = 5
	in.Resources = HostResources{Cores: IntPtr(4), FreeRAMMB: IntPtr(3000), TotalThreads: IntPtr(1000)}
	got := EvaluatePreflight(in)
	if got.HostCap == nil || *got.HostCap != 2 || got.Cap != 2 {
		t.Fatalf("host cap/cap = %v/%d, want 2/2", got.HostCap, got.Cap)
	}
}

func TestEvaluatePreflightHostPressureTable(t *testing.T) {
	tests := []struct {
		name          string
		resources     HostResources
		osWorkerProcs int
		wantCap       int
		wantHostCap   int
		wantLive      int
		wantHeadroom  int
		wantBinding   string
		wantVerdict   string
	}{
		{
			name:         "normal resources recover to max workers",
			resources:    roomyResources(),
			wantCap:      8,
			wantHostCap:  32,
			wantHeadroom: 8,
			wantBinding:  "cores",
			wantVerdict:  PreflightOKVerdict,
		},
		{
			name:         "cpu saturation reduces effective cap",
			resources:    HostResources{Cores: IntPtr(8), FreeRAMMB: IntPtr(64000), TotalThreads: IntPtr(3000)},
			wantCap:      1,
			wantHostCap:  1,
			wantHeadroom: 1,
			wantBinding:  "threads",
			wantVerdict:  PreflightOKVerdict,
		},
		{
			name:         "memory pressure reduces effective cap",
			resources:    HostResources{Cores: IntPtr(16), FreeRAMMB: IntPtr(3000), TotalThreads: IntPtr(0)},
			wantCap:      2,
			wantHostCap:  2,
			wantHeadroom: 2,
			wantBinding:  "ram",
			wantVerdict:  PreflightOKVerdict,
		},
		{
			name:          "zero headroom refuses at cap",
			resources:     HostResources{Cores: IntPtr(8), FreeRAMMB: IntPtr(3000), TotalThreads: IntPtr(3000)},
			osWorkerProcs: 1,
			wantCap:       1,
			wantHostCap:   1,
			wantLive:      1,
			wantHeadroom:  0,
			wantBinding:   "threads",
			wantVerdict:   PreflightRefuseAtCap,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			in := preflightInput()
			in.MaxWorkers = 8
			in.Kernel.Target = IntPtr(10)
			in.Resources = tc.resources
			in.OSWorkerProcs = tc.osWorkerProcs

			got := EvaluatePreflight(in)
			if got.Cap != tc.wantCap || got.Live != tc.wantLive || got.Headroom != tc.wantHeadroom {
				t.Fatalf("cap/live/headroom = %d/%d/%d, want %d/%d/%d", got.Cap, got.Live, got.Headroom, tc.wantCap, tc.wantLive, tc.wantHeadroom)
			}
			if got.HostCap == nil || *got.HostCap != tc.wantHostCap {
				t.Fatalf("host cap = %v, want %d", got.HostCap, tc.wantHostCap)
			}
			if got.HostCapacity.Binding != tc.wantBinding {
				t.Fatalf("binding = %q, want %q; components=%v", got.HostCapacity.Binding, tc.wantBinding, got.HostCapacity.Components)
			}
			if got.Verdict != tc.wantVerdict {
				t.Fatalf("verdict = %s, want %s", got.Verdict, tc.wantVerdict)
			}
		})
	}
}

func TestEvaluatePreflightSeatPoolCapsAndDepletes(t *testing.T) {
	in := preflightInput()
	in.MaxWorkers = 100
	in.Kernel.Target = IntPtr(0)
	in.Seat = SeatCheck{Total: IntPtr(4), Free: IntPtr(0), Leased: IntPtr(4), Depleted: true}
	in.OSWorkerProcs = 4
	got := EvaluatePreflight(in)
	if got.Cap != 4 || got.Verdict != PreflightRefuseNoSeat {
		t.Fatalf("cap/verdict = %d/%s, want 4/REFUSE_NO_SEAT", got.Cap, got.Verdict)
	}
}

func TestEvaluatePreflightBlockedPoolIsNoAccount(t *testing.T) {
	in := preflightInput()
	in.MaxWorkers = 10
	in.Kernel.Target = IntPtr(0)
	in.Account = AccountCheck{Available: false, Reason: "throttled", Blocked: []string{"a", "b"}}
	in.Seat = SeatCheck{Total: IntPtr(2), Free: IntPtr(0), Leased: IntPtr(0), Depleted: true}
	got := EvaluatePreflight(in)
	if got.Verdict != PreflightRefuseNoAccount {
		t.Fatalf("verdict = %s, want REFUSE_NO_ACCOUNT", got.Verdict)
	}
}

func TestEvaluatePreflightInspectPrecedesHostFlag(t *testing.T) {
	in := preflightInput()
	in.Host = HostCheck{Safe: false, Error: "guard missing", Flagged: 1}
	got := EvaluatePreflight(in)
	if got.Verdict != PreflightRefuseInspect {
		t.Fatalf("verdict = %s, want REFUSE_INSPECT", got.Verdict)
	}
}
