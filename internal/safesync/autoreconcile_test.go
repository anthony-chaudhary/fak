package safesync

import (
	"context"
	"errors"
	"testing"
)

const autoReconcileTestHead = "1111111111111111111111111111111111111111"

func autoReconcileDisjointRefusal() PushResult {
	return PushResult{Reason: ReasonDivergedDisjoint, Divergence: string(PushDiverged)}
}

// noMergeRunner answers MERGE_HEAD probes as "absent"; any other git call fails the test.
func noMergeRunner(t *testing.T, mergeActive bool) Runner {
	return func(_ context.Context, _ string, args ...string) RunResult {
		if len(args) >= 4 && args[0] == "rev-parse" && args[3] == "MERGE_HEAD" {
			if mergeActive {
				return RunResult{Stdout: []byte(autoReconcileTestHead + "\n")}
			}
			return RunResult{Code: 1}
		}
		t.Errorf("unexpected git call %v", args)
		return RunResult{Code: 1}
	}
}

func safeDisjointBuild(head string, builds *int) func(context.Context, PacketOptions) (*ReconciliationPacket, error) {
	return func(context.Context, PacketOptions) (*ReconciliationPacket, error) {
		*builds++
		return &ReconciliationPacket{Dispatchable: true, Disposition: DispositionSafeDisjoint, LocalHead: head}, nil
	}
}

func countingExecute(executes *int, receipt *ExecutionReceipt, err error) func(context.Context, *ReconciliationPacket, ExecuteOptions) (*ExecutionReceipt, error) {
	return func(context.Context, *ReconciliationPacket, ExecuteOptions) (*ExecutionReceipt, error) {
		*executes++
		return receipt, err
	}
}

func TestAutoReconcileDisjointHealsSafeDisjoint(t *testing.T) {
	var builds, executes int
	out := AutoReconcileDisjoint(context.Background(), autoReconcileDisjointRefusal(), AutoReconcileOptions{
		Repo: t.TempDir(), Remote: "origin", Branch: "main", SourceSHA: autoReconcileTestHead,
		Runner:  noMergeRunner(t, false),
		Build:   safeDisjointBuild(autoReconcileTestHead, &builds),
		Execute: countingExecute(&executes, &ExecutionReceipt{Status: ExecuteStatusExecuted, Pushed: true}, nil),
	})
	if !out.Attempted || !out.Pushed || out.Reason != "" || builds != 1 || executes != 1 {
		t.Fatalf("outcome=%+v builds=%d executes=%d, want one attempted pushed heal", out, builds, executes)
	}
}

func TestAutoReconcileDisjointDeclines(t *testing.T) {
	cases := []struct {
		name        string
		pushed      PushResult
		mergeActive bool
		packet      *ReconciliationPacket
		receipt     *ExecutionReceipt
		execErr     error
		wantReason  string
		wantExec    int
	}{
		{name: "overlapping divergence", pushed: PushResult{Reason: ReasonDivergedOverlap}, wantReason: AutoReconcileReasonNotDisjoint},
		{name: "already pushed", pushed: PushResult{Pushed: true, Reason: ReasonDivergedDisjoint}, wantReason: AutoReconcileReasonNotDisjoint},
		{name: "active merge", pushed: autoReconcileDisjointRefusal(), mergeActive: true, wantReason: AutoReconcileReasonMergeActive},
		{name: "colliding dirty paths", pushed: autoReconcileDisjointRefusal(),
			packet:     &ReconciliationPacket{Disposition: DispositionOwnerHandoffRequired, LocalHead: autoReconcileTestHead},
			wantReason: AutoReconcileReasonNotDispatched},
		{name: "trivial superset is not this heal", pushed: autoReconcileDisjointRefusal(),
			packet:     &ReconciliationPacket{Dispatchable: true, Disposition: DispositionTrivialSuperset, LocalHead: autoReconcileTestHead},
			wantReason: AutoReconcileReasonNotDispatched},
		{name: "source moved", pushed: autoReconcileDisjointRefusal(),
			packet:     &ReconciliationPacket{Dispatchable: true, Disposition: DispositionSafeDisjoint, LocalHead: "2222222222222222222222222222222222222222"},
			wantReason: AutoReconcileReasonSourceMoved},
		{name: "executor refused", pushed: autoReconcileDisjointRefusal(),
			receipt: &ExecutionReceipt{Status: ExecuteStatusRefused, Reason: ReasonTargetMoved},
			execErr: &ExecuteError{Reason: ReasonTargetMoved}, wantReason: AutoReconcileReasonExecuteFailed, wantExec: 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			builds, executes := 0, 0
			build := safeDisjointBuild(autoReconcileTestHead, &builds)
			if tc.packet != nil {
				pkt := tc.packet
				build = func(context.Context, PacketOptions) (*ReconciliationPacket, error) { builds++; return pkt, nil }
			}
			out := AutoReconcileDisjoint(context.Background(), tc.pushed, AutoReconcileOptions{
				Repo: t.TempDir(), Remote: "origin", Branch: "main", SourceSHA: autoReconcileTestHead,
				Runner:  noMergeRunner(t, tc.mergeActive),
				Build:   build,
				Execute: countingExecute(&executes, tc.receipt, tc.execErr),
			})
			if out.Pushed || out.Reason != tc.wantReason || executes != tc.wantExec {
				t.Fatalf("outcome=%+v executes=%d, want declined %s with %d executes", out, executes, tc.wantReason, tc.wantExec)
			}
		})
	}
}

func TestAutoReconcileDisjointPacketError(t *testing.T) {
	out := AutoReconcileDisjoint(context.Background(), autoReconcileDisjointRefusal(), AutoReconcileOptions{
		Repo:   t.TempDir(),
		Runner: noMergeRunner(t, false),
		Build: func(context.Context, PacketOptions) (*ReconciliationPacket, error) {
			return nil, errors.New("fetch failed")
		},
	})
	if out.Pushed || out.Attempted || out.Reason != AutoReconcileReasonPacketError {
		t.Fatalf("outcome=%+v, want PACKET_ERROR without an attempt", out)
	}
}
