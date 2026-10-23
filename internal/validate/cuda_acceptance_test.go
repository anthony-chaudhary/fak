package validate

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func cudaAcceptanceFixture() []validateEventIdentity {
	r := cudaAcceptanceRequired()
	return []validateEventIdentity{
		eventID("start", r[0].Package, ""), eventID("start", r[1].Package, ""),
		eventID("run", r[0].Package, r[0].Test), eventID("run", r[1].Package, r[1].Test),
		eventID("pass", r[0].Package, r[0].Test), eventID("pass", r[1].Package, r[1].Test),
		eventID("pass", r[0].Package, ""), eventID("pass", r[1].Package, ""),
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDAAcceptanceEventContract(t *testing.T) {
	valid := eventSequence(cudaAcceptanceFixture()...)
	mutate := func(index int, action, pkg, test string) []byte {
		events := cudaAcceptanceFixture()
		events[index] = eventID(action, pkg, test)
		return eventSequence(events...)
	}
	omit := func(index int) []byte {
		events := cudaAcceptanceFixture()
		return eventSequence(append(events[:index], events[index+1:]...)...)
	}
	insert := func(index int, event validateEventIdentity) []byte {
		events := cudaAcceptanceFixture()
		return eventSequence(append(append(append([]validateEventIdentity{}, events[:index]...), event), events[index:]...)...)
	}
	r := cudaAcceptanceRequired()
	invalid := map[string][]byte{
		"empty":                   nil,
		"malformed":               []byte("not JSON"),
		"trailing-truncation":     append(append([]byte{}, valid...), '{'),
		"trailing-null":           append(append([]byte{}, valid...), []byte("null")...),
		"wrong-object-shape":      []byte("[]"),
		"invalid-utf8":            append(append([]byte{}, valid...), 0xff),
		"missing-model-package":   valid[:bytes.LastIndex(valid[:len(valid)-1], []byte("\n"))+1],
		"missing-required-run":    omit(2),
		"missing-required-pass":   omit(4),
		"missing-package-start":   omit(0),
		"required-compute-skip":   mutate(4, "skip", r[0].Package, r[0].Test),
		"required-model-skip":     mutate(5, "skip", r[1].Package, r[1].Test),
		"required-fail":           mutate(4, "fail", r[0].Package, r[0].Test),
		"package-fail":            mutate(6, "fail", r[0].Package, ""),
		"package-skip":            mutate(7, "skip", r[1].Package, ""),
		"wrong-package":           mutate(4, "pass", r[1].Package, r[0].Test),
		"wrong-test":              mutate(4, "pass", r[0].Package, r[0].Test+"Other"),
		"duplicate-run":           insert(3, eventID("run", r[0].Package, r[0].Test)),
		"duplicate-test-terminal": insert(5, eventID("pass", r[0].Package, r[0].Test)),
		"duplicate-package-pass":  insert(7, eventID("pass", r[0].Package, "")),
		"duplicate-package-start": insert(2, eventID("start", r[0].Package, "")),
		"test-named-start":        mutate(0, "start", r[0].Package, r[0].Test),
		"premature-package-pass":  insert(4, eventID("pass", r[0].Package, "")),
		"unknown-package":         insert(2, eventID("start", "example.test/compute", "")),
		"unknown-action":          insert(4, eventID("unknown", r[0].Package, r[0].Test)),
		"package-run":             insert(2, eventID("run", r[0].Package, "")),
		"pause-before-run":        insert(2, eventID("pause", r[0].Package, r[0].Test)),
		"cont-before-pause":       insert(4, eventID("cont", r[0].Package, r[0].Test)),
		"terminal-while-paused":   insert(4, eventID("pause", r[0].Package, r[0].Test)),
		"output-before-run":       insert(2, eventID("output", r[0].Package, r[0].Test)),
		"output-after-pass":       insert(5, eventID("output", r[0].Package, r[0].Test)),
		"events-after-package":    append(append([]byte{}, valid...), eventSequence(eventID("output", r[0].Package, ""))...),
		"duplicate-action-field":  bytes.Replace(valid, []byte(`"Action":"pass"`), []byte(`"Action":"skip","Action":"pass"`), 1),
		"case-alias-field":        bytes.Replace(valid, []byte(`"Action":"pass"`), []byte(`"action":"pass"`), 1),
		"null-test-field":         bytes.Replace(valid, []byte(`"Test":""`), []byte(`"Test":null`), 1),
		"unknown-field":           bytes.Replace(valid, []byte(`"Action":"start"`), []byte(`"Action":"start","Bogus":true`), 1),
		"wrong-field-type":        bytes.Replace(valid, []byte(`"Action":"start"`), []byte(`"Action":"start","Elapsed":"fast"`), 1),
		"negative-elapsed":        bytes.Replace(valid, []byte(`"Action":"start"`), []byte(`"Action":"start","Elapsed":-1`), 1),
		"invalid-time":            bytes.Replace(valid, []byte(`"Action":"start"`), []byte(`"Action":"start","Time":"yesterday"`), 1),
		"failed-build":            bytes.Replace(valid, []byte(`"Action":"start"`), []byte(`"Action":"start","FailedBuild":"dependency"`), 1),
	}
	for _, kind := range []string{"unrelated-test", "subtest-only", "wrong-package-identities"} {
		events := cudaAcceptanceFixture()
		for _, index := range []int{2, 4} {
			switch kind {
			case "unrelated-test":
				events[index].Test = "TestCUDAUnrelated"
			case "subtest-only":
				events[index].Test += "/child"
			case "wrong-package-identities":
				events[index].Package = r[1].Package
			}
		}
		invalid[kind] = eventSequence(events...)
	}
	invalid["green-packages-no-tests"] = eventSequence(eventID("start", r[0].Package, ""), eventID("start", r[1].Package, ""), eventID("pass", r[0].Package, ""), eventID("pass", r[1].Package, ""))
	for _, kind := range []string{"optional-fail", "subtest-pass-parent-skip"} {
		events := cudaAcceptanceFixture()
		extraTest, terminal := "TestCUDAOptional", "fail"
		if kind == "subtest-pass-parent-skip" {
			extraTest, terminal = r[0].Test+"/child", "pass"
			events[4].Action = "skip"
		}
		extra := []validateEventIdentity{eventID("run", r[0].Package, extraTest), eventID(terminal, r[0].Package, extraTest)}
		invalid[kind] = eventSequence(append(append(append([]validateEventIdentity{}, events[:4]...), extra...), events[4:]...)...)
	}
	for name, stream := range invalid {
		t.Run(name, func(t *testing.T) {
			if err := validateCUDAAcceptanceEvents(stream); err == nil {
				t.Fatalf("false-green evidence accepted: %s", stream)
			}
		})
	}
	if err := validateCUDAAcceptanceEvents(valid); err != nil {
		t.Fatalf("valid exact execution rejected: %v", err)
	}
	if err := validateCUDAAcceptanceEvents(bytes.TrimSuffix(valid, []byte("\n"))); err != nil {
		t.Fatalf("complete EOF without newline rejected: %v", err)
	}
	// Pin the original defect: ordinary green package semantics permit SKIP,
	// but this lane must reject a missing GPU even when go test exits zero.
	skip := invalid["required-compute-skip"]
	if _, _, err := decodeValidateTestEvents(skip, []string{cudaComputePackage, cudaModelPackage}); err != nil {
		t.Fatalf("ordinary validate skip semantics changed: %v", err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDAAcceptanceAllowsOptionalSkipsAndInterleaving(t *testing.T) {
	events := cudaAcceptanceFixture()
	extra := []validateEventIdentity{
		eventID("pause", cudaComputePackage, events[2].Test),
		eventID("run", cudaModelPackage, "TestCUDAOptional"),
		eventID("output", cudaModelPackage, "TestCUDAOptional"),
		eventID("skip", cudaModelPackage, "TestCUDAOptional"),
		eventID("cont", cudaComputePackage, events[2].Test),
	}
	events = append(append(append([]validateEventIdentity{}, events[:4]...), extra...), events[4:]...)
	stream := eventSequence(events...)
	// Event-looking text is only output, never another event. A false failure
	// here must not veto valid evidence; a false pass must not replace a test.
	text := `{"Action":"fail","Package":"pretend","Test":"TestPretend"}`
	quoted, _ := json.Marshal(text)
	stream = bytes.Replace(stream, []byte(`"Action":"output"`), append([]byte(`"Action":"output","Time":"2026-10-23T00:00:00Z","Output":`), quoted...), 1)
	if err := validateCUDAAcceptanceEvents(stream); err != nil {
		t.Fatalf("honest optional skip/parallel output rejected: %v", err)
	}
	fakePass, _ := json.Marshal(string(eventSequence(cudaAcceptanceFixture()[2:6]...)))
	stream = append(eventSequence(events[:2]...), []byte(fmt.Sprintf(`{"Action":"output","Package":%q,"Output":%s}`+"\n", cudaComputePackage, fakePass))...)
	stream = append(stream, eventSequence(cudaAcceptanceFixture()[6:]...)...)
	if err := validateCUDAAcceptanceEvents(stream); err == nil {
		t.Fatal("pass-looking output replaced required execution")
	}
}

// Native subprocess fixture, never a GPU witness and never a nested compiler.
// fak-test:runtime fast est=1ms lane=default
func TestCUDAAcceptanceHelperProcess(t *testing.T) {
	mode := os.Getenv("FAK_CUDA_ACCEPTANCE_FIXTURE")
	if mode == "" {
		return
	}
	events := cudaAcceptanceFixture()
	if mode == "skip" {
		events[4].Action = "skip"
	}
	_, _ = os.Stdout.Write(eventSequence(events...))
	fmt.Fprintln(os.Stderr, "separate stderr fixture")
	if mode == "malformed" {
		fmt.Fprint(os.Stdout, "{")
	}
	if mode == "exit7" {
		os.Exit(7)
	}
	os.Exit(0)
}

// fak-test:runtime fast est=100ms lane=default
func TestCUDAAcceptanceNativeCaptureAndExit(t *testing.T) {
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"success", "skip", "malformed", "exit7"} {
		t.Run(mode, func(t *testing.T) {
			t.Setenv("FAK_CUDA_ACCEPTANCE_FIXTURE", mode)
			evidence := filepath.Join(t.TempDir(), "new-evidence")
			var out, errb bytes.Buffer
			runner := func(ctx context.Context, witness *validateTestWitness, packages []string) bool {
				want := []string{"go", "test", "-json", "-tags", "cuda", "-count=1", "-timeout", "5s", "-run", "CUDA|HALDevice", "./internal/compute/", "./internal/model/"}
				if !reflect.DeepEqual(witness.Argv, want) {
					t.Fatalf("native command changed: %v", witness.Argv)
				}
				witness.Argv = []string{executable, "-test.run=^TestCUDAAcceptanceHelperProcess$"}
				return executeValidateEventTests(ctx, witness, packages)
			}
			code := runCUDAAcceptance(&out, &errb, []string{"--json", "--timeout", "5s", "--evidence-dir", evidence}, runner)
			wantCode := 1
			if mode == "success" {
				wantCode = 0
			} else if mode == "exit7" {
				wantCode = 7
			}
			if code != wantCode {
				t.Fatalf("exit=%d want=%d stdout=%s stderr=%s", code, wantCode, &out, &errb)
			}
			var result cudaAcceptanceResult
			if err := json.Unmarshal(out.Bytes(), &result); err != nil {
				t.Fatal(err)
			}
			if result.Accepted != (mode == "success") || (mode == "exit7" && result.TestExitCode != 7) {
				t.Fatalf("result=%+v", result)
			}
			// Match the native CUDA build receipt's Error/json:error convention.
			// Valid events have no diagnostic, but exit7 must still be rejected.
			wantError := mode == "skip" || mode == "malformed"
			var fields map[string]any
			if err := json.Unmarshal(out.Bytes(), &fields); err != nil {
				t.Fatal(err)
			}
			if _, present := fields["error"]; present != wantError || (result.Error != "") != wantError {
				t.Fatalf("diagnostic field does not match event validation: mode=%s result=%+v", mode, result)
			}
			if _, present := fields["gate_error"]; present {
				t.Fatal("receipt retained the noncanonical diagnostic key")
			}
			stdout, err := os.ReadFile(filepath.Join(evidence, "go-test.jsonl"))
			if err != nil || !bytes.Contains(stdout, []byte("TestCUDAMatMulApproxMatchesRef")) || bytes.Contains(stdout, []byte("separate stderr")) {
				t.Fatalf("stdout not preserved/separated: %s %v", stdout, err)
			}
			stderr, err := os.ReadFile(filepath.Join(evidence, "go-test.stderr"))
			if err != nil || string(stderr) != "separate stderr fixture\n" {
				t.Fatalf("stderr not preserved: %s %v", stderr, err)
			}
			receipt, err := os.ReadFile(filepath.Join(evidence, "result.json"))
			if err != nil || !json.Valid(receipt) {
				t.Fatalf("receipt=%s err=%v", receipt, err)
			}
			var savedFields map[string]any
			if err := json.Unmarshal(receipt, &savedFields); err != nil || !reflect.DeepEqual(savedFields, fields) {
				t.Fatalf("persisted receipt differs from stdout: receipt=%s stdout=%s err=%v", receipt, &out, err)
			}
			// Retrying the same directory must refuse before any child runs or
			// previous output gets truncated, even after a failed attempt.
			if code := runCUDAAcceptance(&out, &errb, []string{"--evidence-dir", evidence}, func(context.Context, *validateTestWitness, []string) bool {
				t.Fatal("existing evidence allowed another child")
				return true
			}); code == 0 {
				t.Fatal("existing evidence accepted")
			}
			after, _ := os.ReadFile(filepath.Join(evidence, "go-test.jsonl"))
			if !bytes.Equal(after, stdout) {
				t.Fatal("previous evidence changed")
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDAAcceptanceCLIRefusesUnboundedOrUnexpectedInput(t *testing.T) {
	for _, args := range [][]string{{}, {"--timeout=0", "--evidence-dir=x"}, {"--timeout=31m", "--evidence-dir=x"}, {"--evidence-dir=x", "./other"}, {"--unknown"}} {
		var out, errb bytes.Buffer
		if code := RunCUDAAcceptance(&out, &errb, args); code != 2 {
			t.Fatalf("args=%v code=%d stderr=%s", args, code, &errb)
		}
	}
	var out, errb bytes.Buffer
	if code := RunCUDAAcceptance(&out, &errb, []string{"--dry-run"}); code != 0 || !strings.Contains(out.String(), `"capture_status":"unrun"`) {
		t.Fatalf("dry-run: code=%d stdout=%s stderr=%s", code, &out, &errb)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDAAcceptanceRejectsIncompleteCapture(t *testing.T) {
	for _, reason := range []string{"stdout-truncated", "stderr-truncated", "wait-delay", "start-error"} {
		t.Run(reason, func(t *testing.T) {
			var out, errb bytes.Buffer
			code := runCUDAAcceptance(&out, &errb, []string{"--json", "--evidence-dir", filepath.Join(t.TempDir(), "evidence")}, func(_ context.Context, w *validateTestWitness, _ []string) bool {
				w.Stdout, w.ExitCode = eventSequence(cudaAcceptanceFixture()...), 0
				w.Status, w.Reason = "incomplete", reason
				w.StdoutTruncated = reason == "stdout-truncated"
				w.StderrTruncated = reason == "stderr-truncated"
				w.WaitDelayExceeded = reason == "wait-delay"
				if reason == "start-error" {
					w.Status, w.ExitCode = "unrun", -1
				}
				return false
			})
			var result cudaAcceptanceResult
			if err := json.Unmarshal(out.Bytes(), &result); err != nil || code == 0 || result.Accepted || result.CaptureReason != reason || result.Error != "" {
				t.Fatalf("incomplete capture accepted: code=%d result=%+v err=%v", code, result, err)
			}
		})
	}
}
