package agent

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// qwen38MTPCanaryPlanner keeps the symptom witness source-compatible with the
// pre-wiring planner. The type assertion fails at runtime on the parent commit,
// then succeeds once the production gate is connected.
type qwen38MTPCanaryPlanner interface {
	ConfigureQwen38MTPCanary(*model.Qwen38MTPCanaryManager, model.Qwen38CanaryRequest, *model.Qwen38MTPKillSwitch, ...model.MetalMTPConfig) (model.Qwen38CanaryDecision, error)
	Qwen38MTPCanaryResult() model.Qwen38CanaryDecision
	qwen38MTPCanaryAllowsExecution() bool
}

// TestInKernelPlannerMTPCanaryRequiresWitnessedEnvelope is the #12350 production-wiring
// witness: default-on MTP reaches the planner only through a fresh receipt bound to the
// exact artifact envelope. Explicit operator opt-in remains available, while the runtime
// kill switch always wins. Every refusal stays on fak-native target decode with a typed reason.
func TestInKernelPlannerMTPCanaryRequiresWitnessedEnvelope(t *testing.T) {
	now := time.Now().UTC()
	artifact := strings.Repeat("ab", 32)
	envelope := model.Qwen38CanaryEnvelope{
		ModelFamily:   "Qwen3.8",
		Format:        model.Qwen38MTPFormatQ4K,
		Backend:       model.Qwen38MTPBackendMetal,
		HeadroomBytes: 3 * 1024 * 1024 * 1024,
		ArtifactHash:  artifact,
		DraftDepth:    2,
	}
	witness := func(validUntil time.Time) model.Qwen38MTPCanaryEvidence {
		return model.Qwen38MTPCanaryEvidence{
			Receipt: model.Qwen38MTPCanaryReceipt{
				SchemaVersion:   model.Qwen38MTPCanaryReceiptSchema,
				ReceiptID:       "mtp-canary-receipt-12350",
				DefaultOn:       true,
				Engine:          model.Qwen38EngineMTP,
				Envelope:        envelope,
				Speedup:         1.1,
				TokensProduced:  2,
				TokensProposed:  1,
				TokensAccepted:  1,
				CircuitStatus:   model.CanaryCircuitClosed,
				LatencyNS:       model.Qwen38MTPLatencyNS{Setup: 1, Draft: 1, Verify: 1, Total: 3},
				MemoryBytes:     model.Qwen38MTPMemoryBytes{DraftWorkspace: 1, VerifyWorkspace: 1, Peak: 1},
				DowngradeReason: model.Qwen38MTPEligible,
			},
			ObservedAt: now.Add(-time.Hour),
			ValidUntil: validUntil,
		}
	}
	request := model.Qwen38CanaryRequest{
		Envelope:          envelope,
		EvidenceReceiptID: "mtp-canary-receipt-12350",
		ModelReady:        true,
	}
	newPlanner := func() *InKernelPlanner {
		return NewInKernelPlanner(model.NewSyntheticQwen38MTP(), nil, "qwen3.8-canary", false, nil, false)
	}
	configure := func(t *testing.T, p *InKernelPlanner, mgr *model.Qwen38MTPCanaryManager, req model.Qwen38CanaryRequest, killSwitch *model.Qwen38MTPKillSwitch) (qwen38MTPCanaryPlanner, model.Qwen38CanaryDecision) {
		t.Helper()
		gated, ok := any(p).(qwen38MTPCanaryPlanner)
		if !ok {
			t.Fatal("planner does not expose the production Qwen3.8 MTP canary gate")
		}
		decision, err := gated.ConfigureQwen38MTPCanary(mgr, req, killSwitch)
		if err != nil {
			t.Fatalf("configure: %v", err)
		}
		return gated, decision
	}

	t.Run("missing evidence", func(t *testing.T) {
		p := newPlanner()
		_, decision := configure(t, p, model.NewQwen38MTPCanaryManager(), request, nil)
		if decision.Engine != model.Qwen38EngineTargetDecode || decision.DowngradeReason != model.Qwen38MTPEvidenceMissing || p.MetalMTPCoordinator() != nil {
			t.Fatalf("missing evidence decision=%+v coordinator=%v", decision, p.MetalMTPCoordinator())
		}
	})

	t.Run("mismatched evidence", func(t *testing.T) {
		mgr := model.NewQwen38MTPCanaryManager()
		if err := mgr.RegisterCanaryEvidence(witness(now.Add(time.Hour))); err != nil {
			t.Fatalf("register evidence: %v", err)
		}
		mismatch := request
		mismatch.Envelope.ArtifactHash = strings.Repeat("cd", 32)
		p := newPlanner()
		_, decision := configure(t, p, mgr, mismatch, nil)
		if decision.Engine != model.Qwen38EngineTargetDecode || decision.DowngradeReason != model.Qwen38MTPEvidenceMismatch || p.MetalMTPCoordinator() != nil {
			t.Fatalf("mismatched evidence decision=%+v coordinator=%v", decision, p.MetalMTPCoordinator())
		}
	})

	t.Run("stale evidence", func(t *testing.T) {
		mgr := model.NewQwen38MTPCanaryManager()
		if err := mgr.RegisterCanaryEvidence(witness(now.Add(-time.Minute))); err != nil {
			t.Fatalf("register evidence: %v", err)
		}
		p := newPlanner()
		_, decision := configure(t, p, mgr, request, nil)
		if decision.Engine != model.Qwen38EngineTargetDecode || decision.DowngradeReason != model.Qwen38MTPEvidenceStale || p.MetalMTPCoordinator() != nil {
			t.Fatalf("stale evidence decision=%+v coordinator=%v", decision, p.MetalMTPCoordinator())
		}
	})

	t.Run("fresh witnessed envelope", func(t *testing.T) {
		mgr := model.NewQwen38MTPCanaryManager()
		if err := mgr.RegisterCanaryEvidence(witness(now.Add(time.Hour))); err != nil {
			t.Fatalf("register evidence: %v", err)
		}
		p := newPlanner()
		_, decision := configure(t, p, mgr, request, nil)
		if !decision.CanaryDefaultOn || decision.Engine != model.Qwen38EngineMTP || p.MetalMTPCoordinator() == nil {
			t.Fatalf("fresh evidence decision=%+v coordinator=%v", decision, p.MetalMTPCoordinator())
		}
		p.DisableMetalMTP()
	})

	t.Run("explicit operator override", func(t *testing.T) {
		p := newPlanner()
		override := request
		override.EvidenceReceiptID = ""
		override.OperatorOptIn = true
		_, decision := configure(t, p, model.NewQwen38MTPCanaryManager(), override, nil)
		if !decision.OptInActive || decision.Engine != model.Qwen38EngineMTP || p.MetalMTPCoordinator() == nil {
			t.Fatalf("override decision=%+v coordinator=%v", decision, p.MetalMTPCoordinator())
		}
		p.DisableMetalMTP()
	})

	t.Run("kill switch", func(t *testing.T) {
		mgr := model.NewQwen38MTPCanaryManager()
		if err := mgr.RegisterCanaryEvidence(witness(now.Add(time.Hour))); err != nil {
			t.Fatalf("register evidence: %v", err)
		}
		killSwitch := model.NewQwen38MTPKillSwitch()
		p := newPlanner()
		gated, decision := configure(t, p, mgr, request, killSwitch)
		if decision.Engine != model.Qwen38EngineMTP || p.MetalMTPCoordinator() == nil {
			t.Fatalf("pre-kill decision=%+v coordinator=%v", decision, p.MetalMTPCoordinator())
		}
		if err := killSwitch.Engage("operator rollback"); err != nil {
			t.Fatalf("engage kill switch: %v", err)
		}
		if gated.qwen38MTPCanaryAllowsExecution() {
			t.Fatal("engaged kill switch still admitted MTP")
		}
		decision = gated.Qwen38MTPCanaryResult()
		if decision.Engine != model.Qwen38EngineTargetDecode || decision.DowngradeReason != model.Qwen38MTPDisabledByPolicy {
			t.Fatalf("kill-switch decision=%+v", decision)
		}
		if err := killSwitch.Disengage(true); err != nil {
			t.Fatalf("disengage kill switch: %v", err)
		}
		if !gated.qwen38MTPCanaryAllowsExecution() {
			t.Fatalf("disengaged kill switch did not restore witnessed MTP: %+v", gated.Qwen38MTPCanaryResult())
		}
		p.DisableMetalMTP()
	})
}
