package loopgate

import (
	"reflect"
	"strings"
	"testing"
)

func TestPhaseAwareDone(t *testing.T) {
	tests := []struct {
		name        string
		spec        PhaseAwareDoneSpec
		wantVerdict Verdict
		wantReason  string
		wantExplain string
	}{
		{
			name: "bug fix without repro proof nor fix proof is refused",
			spec: PhaseAwareDoneSpec{
				IsBugFix:      true,
				HasReproProof: false,
				HasFixProof:   false,
			},
			wantVerdict: VerdictRefused,
			wantReason:  ReasonSkippedReproPhase,
			wantExplain: "reproduction phase proof was skipped: bug fixes must prove the failing test before claiming done",
		},
		{
			name: "bug fix with fix proof but missing repro proof is refused",
			spec: PhaseAwareDoneSpec{
				IsBugFix:      true,
				HasReproProof: false,
				HasFixProof:   true,
				FixEvidence:   "pass",
			},
			wantVerdict: VerdictRefused,
			wantReason:  ReasonSkippedReproPhase,
			wantExplain: "reproduction phase proof was skipped: bug fixes must prove the failing test before claiming done",
		},
		{
			name: "bug fix with repro proof but missing fix proof is not yet",
			spec: PhaseAwareDoneSpec{
				IsBugFix:      true,
				HasReproProof: true,
				HasFixProof:   false,
				ReproEvidence: "fail",
			},
			wantVerdict: VerdictNotYet,
			wantReason:  ReasonFixMissing,
			wantExplain: "reproduction test witnessed, but implementation fix proof is missing or unverified",
		},
		{
			name: "bug fix with repro proof and fix proof is witnessed",
			spec: PhaseAwareDoneSpec{
				IsBugFix:      true,
				HasReproProof: true,
				HasFixProof:   true,
				ReproEvidence: "fail",
				FixEvidence:   "pass",
			},
			wantVerdict: VerdictWitnessed,
			wantReason:  "",
			wantExplain: "reproduction test failure and implementation fix both verified",
		},
		{
			name: "non-bug fix with fix proof is witnessed",
			spec: PhaseAwareDoneSpec{
				IsBugFix:    false,
				HasFixProof: true,
				FixEvidence: "pass",
			},
			wantVerdict: VerdictWitnessed,
			wantReason:  "",
			wantExplain: "implementation verified",
		},
		{
			name: "non-bug fix without fix proof is not yet",
			spec: PhaseAwareDoneSpec{
				IsBugFix:    false,
				HasFixProof: false,
			},
			wantVerdict: VerdictNotYet,
			wantReason:  ReasonImplementationMissing,
			wantExplain: "implementation proof missing",
		},
		{
			name: "non-bug fix with repro proof only without fix proof is not yet",
			spec: PhaseAwareDoneSpec{
				IsBugFix:      false,
				HasReproProof: true,
				HasFixProof:   false,
				ReproEvidence: "fail",
			},
			wantVerdict: VerdictNotYet,
			wantReason:  ReasonImplementationMissing,
			wantExplain: "implementation proof missing",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			verdict, reasonToken, explanation := EvaluatePhaseDone(tc.spec)
			if verdict != tc.wantVerdict {
				t.Fatalf("verdict = %q, want %q", verdict, tc.wantVerdict)
			}
			if reasonToken != tc.wantReason {
				t.Fatalf("reasonToken = %q, want %q", reasonToken, tc.wantReason)
			}
			if explanation != tc.wantExplain {
				t.Fatalf("explanation = %q, want %q", explanation, tc.wantExplain)
			}
		})
	}

	t.Run("clean concept tokens in exported symbols", func(t *testing.T) {
		forbidden := []string{"context", "ctx", "render", "guard", "gate"}

		check := func(name string) {
			lower := strings.ToLower(name)
			for _, tok := range forbidden {
				if strings.Contains(lower, tok) {
					t.Fatalf("symbol %q contains forbidden concept token %q", name, tok)
				}
			}
		}

		specType := reflect.TypeOf(PhaseAwareDoneSpec{})
		check(specType.Name())
		for i := 0; i < specType.NumField(); i++ {
			field := specType.Field(i)
			if field.IsExported() {
				check(field.Name)
			}
		}

		check(ReasonSkippedReproPhase)
		check(ReasonFixMissing)
		check(ReasonImplementationMissing)
		check("EvaluatePhaseDone")
	})
}
