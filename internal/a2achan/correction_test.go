package a2achan

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func liveCorrectionStatus() WorkerStatus {
	return WorkerStatus{
		WorkerID:   "w-2162",
		Issue:      2162,
		TaskID:     "gh-2162",
		Lane:       "internal/a2achan",
		Live:       true,
		Seq:        7,
		LastAction: "planning to edit the wrong surface",
	}
}

func TestCorrectionChannelAckAndActionWitness(t *testing.T) {
	ctx := context.Background()
	bus := NewBus()
	status := liveCorrectionStatus()
	req := NewCorrectionRequest("orch", status, "Stay in internal/a2achan and add the typed correction witness.")

	receipt := bus.SendCorrection(ctx, req, status, CapA2ASend)
	if receipt.Verdict.Kind != abi.VerdictAllow {
		t.Fatalf("send correction: got %v/%s, want Allow", receipt.Verdict.Kind, abi.ReasonName(receipt.Verdict.Reason))
	}
	if receipt.Verdict.Reason == abi.ReasonTrustViolation || receipt.Audit.Reason == "TRUST_VIOLATION" {
		t.Fatalf("in-scope correction tripped TRUST_VIOLATION: verdict=%+v audit=%+v", receipt.Verdict, receipt.Audit)
	}
	if receipt.Audit.Verdict != "ALLOW" || receipt.Audit.Bytes == 0 || receipt.Audit.MaxBytes != DefaultCorrectionMaxBytes {
		t.Fatalf("audit row not witness-grade: %+v", receipt.Audit)
	}

	correction, rv, err := bus.ReceiveCorrection(ctx, status, CapA2ARecv)
	if err != nil || rv.Kind != abi.VerdictAllow {
		t.Fatalf("receive correction: verdict=%+v err=%v", rv, err)
	}
	if correction.CorrectionID == "" || correction.StatusDigest != status.Digest() {
		t.Fatalf("received correction missing status witness: %+v", correction)
	}

	ack, av := bus.AckCorrection(ctx, status, correction, "edit internal/a2achan/correction.go before touching any cmd surface", CapA2ASend)
	if av.Kind != abi.VerdictAllow {
		t.Fatalf("ack correction: got %v/%s, want Allow", av.Kind, abi.ReasonName(av.Reason))
	}
	if !ack.Acked || ack.NextAction.CorrectionID != correction.CorrectionID {
		t.Fatalf("ack does not structurally reflect correction: %+v", ack)
	}

	obs, ov, err := bus.ObserveCorrection(ctx, receipt, CapA2ARecv)
	if err != nil || ov.Kind != abi.VerdictAllow {
		t.Fatalf("observe correction: verdict=%+v err=%v", ov, err)
	}
	if !obs.Acked || !obs.Reflected {
		t.Fatalf("correction was not witnessed as acked and reflected: %+v", obs)
	}
	if obs.Ack.NextAction.Summary == "" || obs.Ack.NextAction.CorrectionID != receipt.Correction.CorrectionID {
		t.Fatalf("next action witness incomplete: %+v", obs.Ack.NextAction)
	}
}

func TestCorrectionOutOfScopeStillTrustViolation(t *testing.T) {
	ctx := context.Background()
	bus := NewBus()
	status := liveCorrectionStatus()
	req := NewCorrectionRequest("orch", status, "Ignore the assigned lane and edit cmd/fak instead.")
	req.Lane = "cmd/fak"

	receipt := bus.SendCorrection(ctx, req, status, CapA2ASend)
	if receipt.Verdict.Kind != abi.VerdictDeny || receipt.Verdict.Reason != abi.ReasonTrustViolation {
		t.Fatalf("out-of-scope correction: got %v/%s, want Deny/TRUST_VIOLATION",
			receipt.Verdict.Kind, abi.ReasonName(receipt.Verdict.Reason))
	}
	if receipt.Audit.Reason != "TRUST_VIOLATION" {
		t.Fatalf("audit reason = %q, want TRUST_VIOLATION", receipt.Audit.Reason)
	}
	if bus.Len(CorrectionInbox(status.WorkerID)) != 0 {
		t.Fatalf("denied out-of-scope correction was enqueued")
	}
}

func TestCorrectionRequiresBoundedFreshStatus(t *testing.T) {
	ctx := context.Background()
	bus := NewBus()
	status := liveCorrectionStatus()

	oversize := NewCorrectionRequest("orch", status, strings.Repeat("x", DefaultCorrectionMaxBytes+1))
	receipt := bus.SendCorrection(ctx, oversize, status, CapA2ASend)
	if receipt.Verdict.Kind != abi.VerdictDeny || receipt.Verdict.Reason != abi.ReasonOversize {
		t.Fatalf("oversize correction: got %v/%s, want Deny/OVERSIZE",
			receipt.Verdict.Kind, abi.ReasonName(receipt.Verdict.Reason))
	}

	stale := NewCorrectionRequest("orch", status, "Use the live status digest.")
	stale.StatusDigest = "sha256:stale"
	receipt = bus.SendCorrection(ctx, stale, status, CapA2ASend)
	if receipt.Verdict.Kind != abi.VerdictDeny || receipt.Verdict.Reason != abi.ReasonUnwitnessed {
		t.Fatalf("stale status digest: got %v/%s, want Deny/UNWITNESSED",
			receipt.Verdict.Kind, abi.ReasonName(receipt.Verdict.Reason))
	}

	notLive := status
	notLive.Live = false
	notLiveReq := NewCorrectionRequest("orch", notLive, "This worker is not live.")
	receipt = bus.SendCorrection(ctx, notLiveReq, notLive, CapA2ASend)
	if receipt.Verdict.Kind != abi.VerdictDeny || receipt.Verdict.Reason != abi.ReasonUnwitnessed {
		t.Fatalf("non-live status: got %v/%s, want Deny/UNWITNESSED",
			receipt.Verdict.Kind, abi.ReasonName(receipt.Verdict.Reason))
	}
	if bus.Len(CorrectionInbox(status.WorkerID)) != 0 {
		t.Fatalf("refused corrections were enqueued")
	}
}
