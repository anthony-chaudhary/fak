package sessionctl

import (
	"errors"
	"fmt"
	"sync"
	"testing"
)

func TestPhaseMarker(t *testing.T) {
	t.Run("ConstantsAndMandates", func(t *testing.T) {
		cases := []struct {
			phase       WorkflowPhase
			wantStr     string
			wantMandate string
		}{
			{
				phase:       PhaseRedRepro,
				wantStr:     "RED_REPRODUCTION",
				wantMandate: "Write a test reproducing the defect. Implementation edits are locked.",
			},
			{
				phase:       PhaseGreenImpl,
				wantStr:     "GREEN_IMPLEMENTATION",
				wantMandate: "Make the minimal implementation changes to pass the reproduction test.",
			},
			{
				phase:       PhaseVerify,
				wantStr:     "VERIFY_PHASE",
				wantMandate: "Verify tests pass and capture proof artifacts.",
			},
			{
				phase:       PhaseDone,
				wantStr:     "PHASE_DONE",
				wantMandate: "Task complete. Commit and report evidence.",
			},
		}

		for _, tc := range cases {
			if string(tc.phase) != tc.wantStr {
				t.Errorf("phase %v string = %q, want %q", tc.phase, string(tc.phase), tc.wantStr)
			}
			if tc.phase.Mandate() != tc.wantMandate {
				t.Errorf("phase %v Mandate() = %q, want %q", tc.phase, tc.phase.Mandate(), tc.wantMandate)
			}
			if !tc.phase.Valid() {
				t.Errorf("phase %v Valid() = false, want true", tc.phase)
			}
		}

		invalidPhase := WorkflowPhase("UNKNOWN_PHASE")
		if invalidPhase.Valid() {
			t.Errorf("invalid phase Valid() = true, want false")
		}
		if invalidPhase.Mandate() != "" {
			t.Errorf("invalid phase Mandate() = %q, want empty", invalidPhase.Mandate())
		}
	})

	t.Run("FormatAndParseRoundtrip", func(t *testing.T) {
		phases := []WorkflowPhase{PhaseRedRepro, PhaseGreenImpl, PhaseVerify, PhaseDone}
		for _, phase := range phases {
			formatted := FormatPhaseMarker(phase)
			expected := fmt.Sprintf("[Workflow Phase: %s. Mandate: %s]", phase, phase.Mandate())
			if formatted != expected {
				t.Fatalf("FormatPhaseMarker(%v) = %q, want %q", phase, formatted, expected)
			}

			parsedPhase, mandate, ok := ParsePhaseMarker(formatted)
			if !ok {
				t.Fatalf("ParsePhaseMarker(%q) returned ok=false", formatted)
			}
			if parsedPhase != phase {
				t.Errorf("parsed phase = %v, want %v", parsedPhase, phase)
			}
			if mandate != phase.Mandate() {
				t.Errorf("parsed mandate = %q, want %q", mandate, phase.Mandate())
			}
		}
	})

	t.Run("FormatWithCustomMandate", func(t *testing.T) {
		custom := FormatPhaseMarkerWithMandate(PhaseRedRepro, "Custom mandate message")
		want := "[Workflow Phase: RED_REPRODUCTION. Mandate: Custom mandate message]"
		if custom != want {
			t.Errorf("FormatPhaseMarkerWithMandate = %q, want %q", custom, want)
		}

		phase, mandate, ok := ParsePhaseMarker(custom)
		if !ok || phase != PhaseRedRepro || mandate != "Custom mandate message" {
			t.Errorf("ParsePhaseMarker(%q) = (%v, %q, %v), want (%v, %q, true)", custom, phase, mandate, ok, PhaseRedRepro, "Custom mandate message")
		}
	})

	t.Run("ParseEdgeCases", func(t *testing.T) {
		testCases := []struct {
			name        string
			input       string
			wantPhase   WorkflowPhase
			wantMandate string
			wantOk      bool
		}{
			{
				name:        "surrounding whitespace",
				input:       "\n  [Workflow Phase: RED_REPRODUCTION. Mandate: Write a test reproducing the defect. Implementation edits are locked.]\t\n",
				wantPhase:   PhaseRedRepro,
				wantMandate: "Write a test reproducing the defect. Implementation edits are locked.",
				wantOk:      true,
			},
			{
				name:        "embedded in text block",
				input:       "Prompt preamble\n[Workflow Phase: VERIFY_PHASE. Mandate: Verify tests pass and capture proof artifacts.]\nTrailing text",
				wantPhase:   PhaseVerify,
				wantMandate: "Verify tests pass and capture proof artifacts.",
				wantOk:      true,
			},
			{
				name:        "raw prefix without brackets",
				input:       "Workflow Phase: PHASE_DONE. Mandate: Task complete. Commit and report evidence.",
				wantPhase:   PhaseDone,
				wantMandate: "Task complete. Commit and report evidence.",
				wantOk:      true,
			},
			{
				name:   "empty input",
				input:  "",
				wantOk: false,
			},
			{
				name:   "missing mandate separator",
				input:  "[Workflow Phase: RED_REPRODUCTION]",
				wantOk: false,
			},
			{
				name:   "missing phase token",
				input:  "[Workflow Phase: . Mandate: something]",
				wantOk: false,
			},
			{
				name:   "unrelated bracket text",
				input:  "[Some Other Header: text]",
				wantOk: false,
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				p, m, ok := ParsePhaseMarker(tc.input)
				if ok != tc.wantOk {
					t.Fatalf("ParsePhaseMarker(%q) ok = %v, want %v", tc.input, ok, tc.wantOk)
				}
				if tc.wantOk {
					if p != tc.wantPhase {
						t.Errorf("parsed phase = %v, want %v", p, tc.wantPhase)
					}
					if m != tc.wantMandate {
						t.Errorf("parsed mandate = %q, want %q", m, tc.wantMandate)
					}
				}
			})
		}
	})

	t.Run("Transitions", func(t *testing.T) {
		// Valid transitions
		validSteps := []struct {
			from WorkflowPhase
			to   WorkflowPhase
		}{
			{"", PhaseRedRepro},
			{PhaseRedRepro, PhaseGreenImpl},
			{PhaseGreenImpl, PhaseVerify},
			{PhaseVerify, PhaseDone},
			// Idempotent transitions
			{PhaseRedRepro, PhaseRedRepro},
			{PhaseGreenImpl, PhaseGreenImpl},
			{PhaseVerify, PhaseVerify},
			{PhaseDone, PhaseDone},
			// Valid rework transitions
			{PhaseVerify, PhaseGreenImpl},
			{PhaseGreenImpl, PhaseRedRepro},
			{PhaseVerify, PhaseRedRepro},
			{PhaseDone, PhaseRedRepro},
		}

		for _, step := range validSteps {
			res, marker, err := TransitionPhase(step.from, step.to)
			if err != nil {
				t.Errorf("TransitionPhase(%q, %q) unexpected error: %v", step.from, step.to, err)
			}
			if res != step.to {
				t.Errorf("TransitionPhase(%q, %q) phase = %q, want %q", step.from, step.to, res, step.to)
			}
			if marker != FormatPhaseMarker(step.to) {
				t.Errorf("TransitionPhase(%q, %q) marker = %q, want %q", step.from, step.to, marker, FormatPhaseMarker(step.to))
			}
		}

		// Disallowed skip transitions
		disallowedSteps := []struct {
			from WorkflowPhase
			to   WorkflowPhase
		}{
			{PhaseRedRepro, PhaseVerify},
			{PhaseRedRepro, PhaseDone},
			{PhaseGreenImpl, PhaseDone},
			{PhaseDone, PhaseGreenImpl},
			{PhaseDone, PhaseVerify},
		}

		for _, step := range disallowedSteps {
			_, _, err := TransitionPhase(step.from, step.to)
			if err == nil {
				t.Errorf("TransitionPhase(%q, %q) expected error, got nil", step.from, step.to)
			}
			if !errors.Is(err, ErrInvalidTransition) {
				t.Errorf("TransitionPhase(%q, %q) err = %v, want ErrInvalidTransition", step.from, step.to, err)
			}
		}

		// Invalid phase transitions
		_, _, err := TransitionPhase(PhaseRedRepro, WorkflowPhase("NON_EXISTENT"))
		if !errors.Is(err, ErrInvalidPhase) {
			t.Errorf("TransitionPhase with unknown next phase err = %v, want ErrInvalidPhase", err)
		}

		_, _, err = TransitionPhase(WorkflowPhase("INVALID_CURRENT"), PhaseGreenImpl)
		if !errors.Is(err, ErrInvalidPhase) {
			t.Errorf("TransitionPhase with unknown current phase err = %v, want ErrInvalidPhase", err)
		}
	})

	t.Run("TrackerLifecycle", func(t *testing.T) {
		tracker := NewPhaseMarkerTracker(PhaseRedRepro)
		if tracker.Current() != PhaseRedRepro {
			t.Fatalf("tracker.Current() = %q, want %q", tracker.Current(), PhaseRedRepro)
		}
		if tracker.Marker() != FormatPhaseMarker(PhaseRedRepro) {
			t.Fatalf("tracker.Marker() = %q, want %q", tracker.Marker(), FormatPhaseMarker(PhaseRedRepro))
		}
		if tracker.Mandate() != PhaseRedRepro.Mandate() {
			t.Fatalf("tracker.Mandate() = %q, want %q", tracker.Mandate(), PhaseRedRepro.Mandate())
		}

		// Progress through pipeline
		next, marker, err := tracker.Transition(PhaseGreenImpl)
		if err != nil || next != PhaseGreenImpl {
			t.Fatalf("transition to GreenImpl failed: %v", err)
		}
		if marker != FormatPhaseMarker(PhaseGreenImpl) {
			t.Errorf("marker = %q, want %q", marker, FormatPhaseMarker(PhaseGreenImpl))
		}

		next, marker, err = tracker.Transition(PhaseVerify)
		if err != nil || next != PhaseVerify {
			t.Fatalf("transition to Verify failed: %v", err)
		}

		next, marker, err = tracker.Transition(PhaseDone)
		if err != nil || next != PhaseDone {
			t.Fatalf("transition to Done failed: %v", err)
		}

		history := tracker.History()
		expectedHistory := []WorkflowPhase{PhaseRedRepro, PhaseGreenImpl, PhaseVerify, PhaseDone}
		if len(history) != len(expectedHistory) {
			t.Fatalf("history length = %d, want %d", len(history), len(expectedHistory))
		}
		for i, p := range history {
			if p != expectedHistory[i] {
				t.Errorf("history[%d] = %q, want %q", i, p, expectedHistory[i])
			}
		}

		// Disallowed transition does not mutate state or history
		_, _, err = tracker.Transition(PhaseVerify)
		if err == nil {
			t.Fatalf("expected error transitioning from Done to Verify")
		}
		if tracker.Current() != PhaseDone {
			t.Errorf("current phase mutated on error: got %q, want %q", tracker.Current(), PhaseDone)
		}
		if len(tracker.History()) != len(expectedHistory) {
			t.Errorf("history length mutated on error: got %d, want %d", len(tracker.History()), len(expectedHistory))
		}
	})

	t.Run("TrackerUninitializedStart", func(t *testing.T) {
		tracker := NewPhaseMarkerTracker("")
		if tracker.Current() != "" {
			t.Errorf("uninitialized tracker Current() = %q, want empty", tracker.Current())
		}
		if tracker.Marker() != "" {
			t.Errorf("uninitialized tracker Marker() = %q, want empty", tracker.Marker())
		}
		if len(tracker.History()) != 0 {
			t.Errorf("uninitialized tracker History() length = %d, want 0", len(tracker.History()))
		}

		_, marker, err := tracker.Transition(PhaseRedRepro)
		if err != nil {
			t.Fatalf("transition from uninitialized failed: %v", err)
		}
		if marker != FormatPhaseMarker(PhaseRedRepro) {
			t.Errorf("marker = %q, want %q", marker, FormatPhaseMarker(PhaseRedRepro))
		}
		if tracker.Current() != PhaseRedRepro {
			t.Errorf("tracker.Current() = %q, want %q", tracker.Current(), PhaseRedRepro)
		}
		if len(tracker.History()) != 1 || tracker.History()[0] != PhaseRedRepro {
			t.Errorf("tracker.History() = %v, want [%v]", tracker.History(), PhaseRedRepro)
		}
	})

	t.Run("TrackerConcurrency", func(t *testing.T) {
		tracker := NewPhaseMarkerTracker(PhaseRedRepro)
		var wg sync.WaitGroup
		const readers = 10
		const iterations = 50

		for i := 0; i < readers; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for j := 0; j < iterations; j++ {
					_ = tracker.Current()
					_ = tracker.Marker()
					_ = tracker.Mandate()
					_ = tracker.History()
				}
			}()
		}

		wg.Wait()
	})
}
