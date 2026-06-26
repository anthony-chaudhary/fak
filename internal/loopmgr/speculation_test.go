package loopmgr

import (
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/vdso"
)

func effectFreeCandidate() SpeculationCandidate {
	return SpeculationCandidate{
		LoopID:              "agent-loop/default",
		Tool:                "search_kb",
		ReadOnlyHint:        true,
		IdempotentHint:      true,
		CorrectProbability:  0.278,
		LatencySavedMillis:  1000,
		CostIfWrongMillis:   0,
		SlackMillis:         25,
		EstimatedWorkMillis: 5,
	}
}

func TestAdmitSpeculationAdmitsPositiveEVOnSlackEffectFree(t *testing.T) {
	d := AdmitSpeculation(effectFreeCandidate())
	if !d.Admit || d.Reason != ReasonSpecAdmitted {
		t.Fatalf("AdmitSpeculation admit=%v reason=%q summary=%q", d.Admit, d.Reason, d.Summary)
	}
	if d.EffectReason != SpeculationEffectFree {
		t.Fatalf("effect reason = %s, want %s", d.EffectReason, SpeculationEffectFree)
	}
	if math.Abs(d.ExpectedValueMillis-278) > 1e-9 {
		t.Fatalf("EV = %v, want 278", d.ExpectedValueMillis)
	}
	if d.SlackRemainingMillis != 20 {
		t.Fatalf("slack remaining = %d, want 20", d.SlackRemainingMillis)
	}
}

func TestAdmitSpeculationRefusesNoSlackBeforeEV(t *testing.T) {
	c := effectFreeCandidate()
	c.SlackMillis = 0
	d := AdmitSpeculation(c)
	if d.Admit || d.Reason != ReasonSpecNoSlack {
		t.Fatalf("no slack admit=%v reason=%q", d.Admit, d.Reason)
	}

	c = effectFreeCandidate()
	c.SlackMillis = 4
	c.EstimatedWorkMillis = 5
	d = AdmitSpeculation(c)
	if d.Admit || d.Reason != ReasonSpecNoSlack {
		t.Fatalf("over budget admit=%v reason=%q", d.Admit, d.Reason)
	}
}

func TestAdmitSpeculationDefaultWorkStillConsumesSlack(t *testing.T) {
	c := effectFreeCandidate()
	c.EstimatedWorkMillis = 0
	c.SlackMillis = 0
	d := AdmitSpeculation(c)
	if d.Admit || d.Reason != ReasonSpecNoSlack || d.EstimatedWorkMillis != 1 {
		t.Fatalf("zero estimated work decision = %+v, want no-slack with default work=1", d)
	}
}

func TestAdmitSpeculationRefusesNonPositiveEV(t *testing.T) {
	c := effectFreeCandidate()
	c.CorrectProbability = 0.2
	c.LatencySavedMillis = 100
	c.CostIfWrongMillis = 20 // equality is not enough: P*saved must be > cost.
	d := AdmitSpeculation(c)
	if d.Admit || d.Reason != ReasonSpecEVNegative {
		t.Fatalf("zero EV admit=%v reason=%q ev=%v", d.Admit, d.Reason, d.ExpectedValueMillis)
	}

	c = effectFreeCandidate()
	c.CorrectProbability = 1.1
	d = AdmitSpeculation(c)
	if d.Admit || d.Reason != ReasonSpecEVNegative || d.ExpectedValueMillis != 0 {
		t.Fatalf("invalid probability decision = %+v", d)
	}
}

func TestAdmitSpeculationRefusesEffectfulFirst(t *testing.T) {
	c := effectFreeCandidate()
	c.Tool = "delete_record"
	c.SlackMillis = 0
	d := AdmitSpeculation(c)
	if d.Admit || d.Reason != ReasonSpecEffectfulRefused {
		t.Fatalf("effectful admit=%v reason=%q", d.Admit, d.Reason)
	}
	if d.EffectReason != SpeculationEffectDestructive {
		t.Fatalf("effect reason = %s, want %s", d.EffectReason, SpeculationEffectDestructive)
	}
}

func TestClassifySpeculationEffectDefaultDeny(t *testing.T) {
	cases := []struct {
		name string
		c    SpeculationCandidate
		want SpeculationEffectReason
	}{
		{"missing read-only proof", SpeculationCandidate{Tool: "search_kb", IdempotentHint: true}, SpeculationEffectMissingReadOnly},
		{"missing idempotence proof", SpeculationCandidate{Tool: "search_kb", ReadOnlyHint: true}, SpeculationEffectMissingIdempotent},
		{"explicit destructive", SpeculationCandidate{Tool: "search_kb", ReadOnlyHint: true, IdempotentHint: true, Destructive: true}, SpeculationEffectDestructive},
		{"write-shaped name", SpeculationCandidate{Tool: "send_email", ReadOnlyHint: true, IdempotentHint: true}, SpeculationEffectDestructive},
		{"verified dry-run is effect-free", SpeculationCandidate{Tool: "payment_preview", DryRunHint: true, IdempotentHint: true}, SpeculationEffectFree},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := ClassifySpeculationEffect(tc.c)
			if got != tc.want {
				t.Fatalf("effect reason = %s, want %s", got, tc.want)
			}
			if ok == tc.want.Refused() {
				t.Fatalf("ok=%v inconsistent with reason %s", ok, tc.want)
			}
		})
	}
}

func TestSpeculationWriteShapeNeedlesMatchVDSO(t *testing.T) {
	if !reflect.DeepEqual(speculationWriteShapeNeedles, vdso.WriteShapeNeedles()) {
		t.Fatalf("loopmgr speculation write-shape needles = %v, want vDSO %v",
			speculationWriteShapeNeedles, vdso.WriteShapeNeedles())
	}
}
