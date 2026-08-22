package toolcatalog

import (
	"errors"
	"testing"
)

func TestRuntimeAdmitPinsTurnsAndAutoRevertsTypedFailure(t *testing.T) {
	base, err := CompileSkill([]byte(skillFixture), "base/SKILL.md")
	if err != nil {
		t.Fatal(err)
	}
	initial, err := Expose([]Registration{base}, []string{base.Program.Name}, "openai")
	if err != nil {
		t.Fatal(err)
	}
	var swaps []RuntimeSwap
	runtime, err := NewRuntime([]Registration{base}, initial, RuntimeHooks{
		Policy:   func(Registration) error { return nil },
		Register: func(Snapshot) error { return nil },
		Swap:     func(s RuntimeSwap) { swaps = append(swaps, s) },
	})
	if err != nil {
		t.Fatal(err)
	}
	pinned := runtime.Pin()

	admitted := mustRegistration(t, "hot_tool", "hot/SKILL.md")
	next, err := runtime.Admit(admitted)
	if err != nil {
		t.Fatal(err)
	}
	if pinned.Digest != initial.Digest {
		t.Fatal("in-flight pin moved")
	}
	if runtime.Pin().Digest != next.Digest {
		t.Fatal("next turn did not see admit")
	}
	if _, err := runtime.Snapshot(initial.Digest); err != nil {
		t.Fatalf("pre-admit replay: %v", err)
	}

	reverted, err := runtime.Report(admitted.Digest, LiveFailureExecutor, errors.New("exit 1"))
	if err != nil {
		t.Fatal(err)
	}
	if !reverted || runtime.Pin().Digest != initial.Digest {
		t.Fatal("typed live failure did not revert")
	}
	replacement := mustRegistration(t, "hot_tool", "hot/v2/SKILL.md")
	if _, err := runtime.Admit(replacement); err != nil {
		t.Fatalf("re-admit after automatic revert: %v", err)
	}
	if len(swaps) != 3 || swaps[0].Kind != SwapAdmit || swaps[1].Kind != SwapAutoRevert || swaps[1].Reason != string(LiveFailureExecutor) || swaps[2].Kind != SwapAdmit {
		t.Fatalf("swaps = %#v", swaps)
	}
}

func TestRuntimeAdmitRefusesClosedClassesWithoutMovingFloor(t *testing.T) {
	base := mustRegistration(t, "base", "base/SKILL.md")
	initial, err := Expose([]Registration{base}, []string{"base"}, "openai")
	if err != nil {
		t.Fatal(err)
	}
	policyCalls := 0
	runtime, err := NewRuntime([]Registration{base}, initial, RuntimeHooks{Policy: func(r Registration) error {
		policyCalls++
		if r.Program.Name == "denied" {
			return errors.New("floor deny")
		}
		return nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	original := runtime.Pin().Digest

	cases := []struct {
		name string
		reg  Registration
		code AdmitCode
	}{
		{"digest", func() Registration { r := mustRegistration(t, "digest", "d"); r.Digest = "bad"; return r }(), AdmitDigestMismatch},
		{"version", func() Registration { r := mustRegistration(t, "version", "v"); r.Program.Version = "unknown"; return r }(), AdmitUnknownVersion},
		{"collision", mustRegistration(t, "base", "other"), AdmitNameCollision},
		{"policy", mustRegistration(t, "denied", "denied"), AdmitPolicyDeny},
		{"malformed", func() Registration { r := mustRegistration(t, "malformed", "m"); r.Program.Name = "bad name"; return r }(), AdmitMalformed},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := runtime.Admit(tc.reg); !IsAdmitCode(err, tc.code) {
				t.Fatalf("error = %v, want %s", err, tc.code)
			}
			if runtime.Pin().Digest != original {
				t.Fatal("refusal changed live snapshot")
			}
		})
	}
	if policyCalls == 0 {
		t.Fatal("policy floor was not consulted")
	}
}

func mustRegistration(t *testing.T, name, source string) Registration {
	t.Helper()
	p := SkillProgram{Version: ProgramVersion, Name: name, InputSchema: []byte(`{"type":"object"}`), Executor: Executor{Argv: []string{"fak", "version"}}}
	r, err := newRegistration(p, source)
	if err != nil {
		t.Fatal(err)
	}
	return r
}
