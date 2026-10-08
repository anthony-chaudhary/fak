package validate

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func eventFixtureEnvironment(t *testing.T) {
	t.Helper()
	t.Setenv("GOENV", "off")
	t.Setenv("GOWORK", "off")
	t.Setenv("GOTOOLCHAIN", "local")
	t.Setenv("GOFLAGS", "-p=1")
	t.Setenv("FAK_WORKSPACE_ROOT", "")
}
func eventFixtureFile(t *testing.T, root, name, body string) {
	t.Helper()
	path := filepath.Join(root, name)
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0600); err != nil {
		t.Fatal(err)
	}
}
func eventFixtureRoot(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	eventFixtureFile(t, root, "go.mod", "module fixture.test/events\n\ngo 1.26\n")
	return root
}

func TestValidateEventDecoderRetainsPassSkipAndHelperIdentities(t *testing.T) {
	raw := `{"Action":"run","Package":"p","Test":"TestPass"}
{"Action":"pass","Package":"p","Test":"TestPass","Elapsed":0.1}
{"Action":"run","Package":"p","Test":"TestSkip"}
{"Action":"output","Package":"p","Test":"TestSkip","Output":"explicit skip reason\n"}
{"Action":"skip","Package":"p","Test":"TestSkip"}
{"Action":"run","Package":"p","Test":"TestFixtureHelper"}
{"Action":"pass","Package":"p","Test":"TestFixtureHelper"}
{"Action":"pass","Package":"p"}
`
	events, tests, err := decodeValidateTestEvents([]byte(raw), []string{"p"})
	if err != nil || len(events) != strings.Count(raw, "\n") || len(tests) != 3 {
		t.Fatalf("events=%v tests=%v err=%v", events, tests, err)
	}
	if tests[1].Action != "skip" || tests[1].Test != "TestSkip" || tests[2].Test != "TestFixtureHelper" || !strings.Contains(string(events[3]), "explicit skip reason") {
		t.Fatal("lost identity/skip reason or conflated package terminal")
	}
}
func TestValidateEventDecoderRefusesIncompleteEvidence(t *testing.T) {
	for name, raw := range map[string]string{"malformed": "{broken", "missing-package": "{\"Action\":\"run\",\"Package\":\"p\",\"Test\":\"TestX\"}\n{\"Action\":\"pass\",\"Package\":\"p\",\"Test\":\"TestX\"}", "missing-test": "{\"Action\":\"run\",\"Package\":\"p\",\"Test\":\"TestX\"}\n{\"Action\":\"pass\",\"Package\":\"p\"}", "unrun": "{\"Action\":\"pass\",\"Package\":\"p\"}", "terminal-only": "{\"Action\":\"pass\",\"Package\":\"p\",\"Test\":\"TestX\"}\n{\"Action\":\"pass\",\"Package\":\"p\"}"} {
		t.Run(name, func(t *testing.T) {
			if _, _, err := decodeValidateTestEvents([]byte(raw), []string{"p"}); err == nil {
				t.Fatal("incomplete witness accepted")
			}
		})
	}
}
func TestValidateEventBufferMarksTruncationWithoutBlocking(t *testing.T) {
	b := &validateEventBuffer{limit: 4}
	n, err := b.Write([]byte("123456"))
	if n != 6 || err != nil || b.String() != "1234" || !b.truncated {
		t.Fatalf("buffer=%q n=%d err=%v", b.String(), n, err)
	}
	if n, err := b.Write([]byte("more")); n != 4 || err != nil {
		t.Fatal("writer stopped draining")
	}
}
func TestValidateEventConfigurationRefusesExternalOverrides(t *testing.T) {
	for _, tc := range []struct{ key, value string }{{"GOENV", "ambient-config"}, {"GOWORK", "auto"}, {"GOTOOLCHAIN", "auto"}, {"GOFLAGS", "-overlay=external.json"}, {"GOFLAGS", "-modfile=external.mod"}, {"FAK_WORKSPACE_ROOT", "external-root"}} {
		t.Run(tc.key+tc.value, func(t *testing.T) {
			eventFixtureEnvironment(t)
			t.Setenv(tc.key, tc.value)
			if validateEventConfiguration() == nil {
				t.Fatal("external override accepted")
			}
		})
	}
}
func TestValidateCandidateFingerprintBindsCopiedBytesBeforeTests(t *testing.T) {
	eventFixtureEnvironment(t)
	root := eventFixtureRoot(t)
	eventFixtureFile(t, root, "candidate.txt", "copied candidate")
	eventFixtureFile(t, root, ".git/owned-metadata", "generated identity")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	got, err := fingerprintValidateCandidate(ctx, root, strings.Repeat("a", 40))
	if err != nil {
		t.Fatal(err)
	}
	if len(got.SHA256) != hex.EncodedLen(sha256.Size) || len(got.GoExecutableSHA256) != hex.EncodedLen(sha256.Size) || len(got.Files) != 2 {
		t.Fatalf("fingerprint=%+v", got)
	}
	old := got.SHA256
	eventFixtureFile(t, root, "candidate.txt", "changed after snapshot")
	next, err := fingerprintValidateCandidate(ctx, root, got.Tip)
	if err != nil || next.SHA256 == old {
		t.Fatalf("snapshot did not bind copied bytes: %v", err)
	}
	if got.SHA256 != old {
		t.Fatal("pre-test identity mutated")
	}
}
func TestValidateCandidateFingerprintRefusesLocalReplacementAndSymlink(t *testing.T) {
	eventFixtureEnvironment(t)
	t.Run("local-replace", func(t *testing.T) {
		root := eventFixtureRoot(t)
		eventFixtureFile(t, root, "go.mod", "module fixture.test/events\n\ngo 1.26\n\nreplace example.test/local => ../external\n")
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := fingerprintValidateCandidate(ctx, root, strings.Repeat("a", 40)); err == nil || !strings.Contains(err.Error(), "local go.mod replacement") {
			t.Fatalf("local source accepted: %v", err)
		}
	})
	t.Run("symlink", func(t *testing.T) {
		root := eventFixtureRoot(t)
		if err := os.Symlink(filepath.Join(t.TempDir(), "absent"), filepath.Join(root, "escape")); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := fingerprintValidateCandidate(ctx, root, strings.Repeat("a", 40)); err == nil || !strings.Contains(err.Error(), "symlink") {
			t.Fatalf("symlink accepted: %v", err)
		}
	})
}

// Native test executable helper: no shell, nested compiler or fixture hooks.
// The parent helper entry is a no-op; its PASS must not be counted as assertions.
func TestValidateEventFixtureHelperProcess(t *testing.T) {
	mode := os.Getenv("FAK_VALIDATE_EVENTS_FIXTURE")
	if mode == "" {
		return
	}
	if mode == "pipe-child" {
		time.Sleep(time.Second)
		if marker := os.Getenv("FAK_VALIDATE_EVENTS_PIPE_MARKER"); marker != "" {
			_ = os.WriteFile(marker, []byte("done"), 0600)
		}
		os.Exit(0)
	}
	if mode == "binary" {
		_, _ = os.Stdout.Write([]byte{0xff, 0x00, 0xfe})
		_, _ = os.Stderr.Write([]byte{0x00, 0xff, 0x80})
		os.Exit(0)
	}
	if mode == "hold-pipes" || mode == "failed-hold-pipes" {
		executable, err := os.Executable()
		if err != nil {
			os.Exit(2)
		}
		child := exec.Command(executable, "-test.run=^TestValidateEventFixtureHelperProcess$")
		for _, entry := range os.Environ() {
			if !strings.HasPrefix(entry, "FAK_VALIDATE_EVENTS_FIXTURE=") {
				child.Env = append(child.Env, entry)
			}
		}
		child.Env = append(child.Env, "FAK_VALIDATE_EVENTS_FIXTURE=pipe-child")
		if mode != "failed-hold-pipes" {
			child.Stdout = os.Stdout
		}
		child.Stderr = os.Stderr
		if err := child.Start(); err != nil {
			os.Exit(2)
		}
	}
	fmt.Fprintln(os.Stderr, "private raw stderr sentinel")
	if mode == "timeout" {
		time.Sleep(time.Second)
		os.Exit(0)
	}
	if mode == "malformed" {
		fmt.Fprintln(os.Stdout, "{broken")
		os.Exit(0)
	}
	action := "pass"
	if mode == "failed" || mode == "failed-hold-pipes" {
		action = "fail"
	}
	fmt.Fprintf(os.Stdout, "{\"Action\":\"run\",\"Package\":\"p\",\"Test\":\"TestFixtureAssertion\"}\n{\"Action\":%q,\"Package\":\"p\",\"Test\":\"TestFixtureAssertion\"}\n{\"Action\":%q,\"Package\":\"p\"}\n", action, action)
	if mode == "failed" || mode == "failed-hold-pipes" {
		os.Exit(1)
	}
	os.Exit(0)
}
func TestValidateEventCapturePreservesRawStreamsFailureAndTimeout(t *testing.T) {
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"success", "failed", "malformed", "timeout"} {
		t.Run(mode, func(t *testing.T) {
			t.Setenv("FAK_VALIDATE_EVENTS_FIXTURE", mode)
			budget := 5 * time.Second
			if mode == "timeout" {
				budget = 200 * time.Millisecond
			}
			ctx, cancel := context.WithTimeout(context.Background(), budget)
			defer cancel()
			w := &validateTestWitness{Candidate: &validateCandidateFingerprint{Root: t.TempDir()}, Argv: []string{executable, "-test.run=^TestValidateEventFixtureHelperProcess$"}}
			ok := executeValidateEventTests(ctx, w, []string{"p"})
			if bytes.Contains(w.Stdout, []byte("private raw stderr sentinel")) {
				t.Fatal("stderr mixed into JSON stdout")
			}
			switch mode {
			case "success":
				if !ok || !w.Complete || w.Status != "complete" || w.ExitCode != 0 || !bytes.Contains(w.Stderr, []byte("private raw stderr sentinel")) {
					t.Fatalf("success=%+v", w)
				}
			case "failed":
				if ok || w.Complete || w.DrainStatus != "unknown" || w.Status != "failed" || w.ExitCode != 1 {
					t.Fatalf("failure=%+v", w)
				}
			case "malformed":
				if ok || w.Complete || w.Status != "incomplete" || w.ExitCode != 0 {
					t.Fatalf("malformed=%+v", w)
				}
			case "timeout":
				if ok || w.Complete || !w.TimedOut || w.Status != "timeout" {
					t.Fatalf("timeout=%+v", w)
				}
			}
		})
	}
}
func TestValidateEventFlagRefusesAuditOrNonJSON(t *testing.T) {
	for _, extra := range [][]string{{}, {"--json", "--audit-selection"}, {"--json", "--wsl-tests=true"}} {
		var stdout, stderr bytes.Buffer
		args := append([]string{"--mine", "source.go", "--test-events"}, extra...)
		if code := Run(&stdout, &stderr, args); code != 2 || !strings.Contains(stderr.String(), "--test-events requires") {
			t.Fatalf("code=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
		}
	}
	args := validateJSONTestArgs(validateTestArgs("^TestOwned$", time.Minute, []string{"./owned"}))
	if strings.Join(args, " ") != "test -json -count=1 -timeout 1m0s -run ^TestOwned$ ./owned" {
		t.Fatalf("unexpected selected argv %v", args)
	}
}

func eventFixtureGit(t *testing.T, root, template string, args ...string) {
	t.Helper()
	git, err := exec.LookPath("git")
	if err != nil {
		t.Fatal(err)
	}
	git, err = filepath.Abs(git)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, git, args...)
	cmd.Dir = root
	cmd.Env = []string{"PATH=" + filepath.Dir(git) + ":/usr/bin:/bin", "HOME=" + template, "XDG_CONFIG_HOME=" + template, "GIT_CONFIG_GLOBAL=" + os.DevNull, "GIT_CONFIG_SYSTEM=" + os.DevNull, "GIT_CONFIG_NOSYSTEM=1", "GIT_CONFIG_COUNT=0", "GIT_TEMPLATE_DIR=" + template, "GIT_TERMINAL_PROMPT=0", "GIT_ALLOW_PROTOCOL=file", "GIT_OPTIONAL_LOCKS=0", "TMPDIR=" + os.TempDir()}
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("fixture Git %v: %v %s", args, err, output)
	}
}
func TestValidateEventsIsolatedRunHidesPeerAndBindsRuntimeRead(t *testing.T) {
	eventFixtureEnvironment(t)
	root := eventFixtureRoot(t)
	template := t.TempDir()
	eventFixtureFile(t, root, "candidate.txt", "owned runtime bytes")
	eventFixtureFile(t, root, "app/app.go", "package app\nfunc Number() int{return 1}\n")
	eventFixtureFile(t, root, "app/app_test.go", `package app
import("os";"testing")
func TestRuntimeRead(t *testing.T){b,err:=os.ReadFile("../candidate.txt");if err!=nil||string(b)!="owned runtime bytes"{t.Fatalf("runtime read: %s %v",b,err)}}
func TestExplicitSkip(t *testing.T){t.Skip("fixture skip reason")}
`)
	eventFixtureFile(t, root, "peer/peer.go", "package peer\nfunc Clean(){}\n")
	eventFixtureGit(t, root, template, "init", "-q")
	eventFixtureGit(t, root, template, "add", ".")
	eventFixtureGit(t, root, template, "-c", "user.name=Fixture", "-c", "user.email=fixture@test", "commit", "-q", "-m", "fixture")
	eventFixtureFile(t, root, "peer/peer.go", "package peer\nfunc Broken( {\n")
	eventFixtureFile(t, root, "candidate.txt", "live root must not be read")
	var stdout, stderr bytes.Buffer
	code := Run(&stdout, &stderr, []string{"--root", root, "--mine", "app/app_test.go", "--ref", "HEAD", "--test-only", "--test-events", "--json", "--wsl-tests=false", "--test-run", "^(TestRuntimeRead|TestExplicitSkip)$", "--timeout", "1m", "--progress=false"})
	if code != 0 {
		t.Fatalf("Run=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	var got validateResult
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if !got.OK || got.TestEvents == nil || !got.TestEvents.Complete || len(got.TestEvents.Tests) != 2 || got.TestEvents.Candidate.Root == root {
		t.Fatalf("receipt=%+v", got)
	}
	if !bytes.Contains(got.TestEvents.Stdout, []byte("fixture skip reason")) || got.TestEvents.Tests[1].Action != "skip" {
		t.Fatal("explicit SKIP evidence absent")
	}
	for _, phase := range got.Phases {
		if phase.Name == "test_audit_full" || phase.Name == "build" || phase.Name == "vet" {
			t.Fatalf("duplicate/full suite phase %s", phase.Name)
		}
	}
	if _, err := os.Stat(got.TestEvents.Candidate.Root); !os.IsNotExist(err) {
		t.Fatalf("native owned extraction cleanup changed: %v", err)
	}
	b, _ := os.ReadFile(filepath.Join(root, "candidate.txt"))
	peer, _ := os.ReadFile(filepath.Join(root, "peer/peer.go"))
	if string(b) != "live root must not be read" || !strings.Contains(string(peer), "Broken(") {
		t.Fatal("live fixture root changed")
	}
}

func eventSequence(events ...validateEventIdentity) []byte {
	var stream bytes.Buffer
	encoder := json.NewEncoder(&stream)
	for _, event := range events {
		_ = encoder.Encode(event)
	}
	return stream.Bytes()
}
func eventID(action, pkg, test string) validateEventIdentity {
	return validateEventIdentity{Action: action, Package: pkg, Test: test}
}

func TestValidateEventDecoderRejectsTerminalOrdering(t *testing.T) {
	for name, events := range map[string][]validateEventIdentity{
		"premature-package":       {eventID("run", "p", "TestX"), eventID("pass", "p", ""), eventID("pass", "p", "TestX")},
		"run-after-package":       {eventID("run", "p", "TestX"), eventID("pass", "p", "TestX"), eventID("pass", "p", ""), eventID("run", "p", "TestY"), eventID("pass", "p", "TestY")},
		"output-after-test":       {eventID("run", "p", "TestX"), eventID("pass", "p", "TestX"), eventID("output", "p", "TestX"), eventID("pass", "p", "")},
		"premature-parent":        {eventID("run", "p", "TestParent"), eventID("run", "p", "TestParent/Child"), eventID("pass", "p", "TestParent"), eventID("pass", "p", "TestParent/Child"), eventID("pass", "p", "")},
		"child-after-parent":      {eventID("run", "p", "TestParent"), eventID("pass", "p", "TestParent"), eventID("run", "p", "TestParent/Child"), eventID("pass", "p", "TestParent/Child"), eventID("pass", "p", "")},
		"orphan-subtest":          {eventID("run", "p", "TestParent/Child"), eventID("pass", "p", "TestParent/Child"), eventID("pass", "p", "")},
		"cross-package-premature": {eventID("run", "p", "TestX"), eventID("run", "q", "TestY"), eventID("pass", "p", ""), eventID("pass", "q", "TestY"), eventID("pass", "p", "TestX"), eventID("pass", "q", "")},
	} {
		t.Run(name, func(t *testing.T) {
			packages := []string{"p"}
			if name == "cross-package-premature" {
				packages = append(packages, "q")
			}
			if _, _, err := decodeValidateTestEvents(eventSequence(events...), packages); err == nil {
				t.Fatal("terminal ordering accepted")
			}
		})
	}
}
func TestValidateEventDecoderAcceptsOrderedInterleavedSubtests(t *testing.T) {
	stream := eventSequence(eventID("run", "p", "TestParent"), eventID("run", "p", "TestParent/Child"), eventID("run", "q", "TestOther"), eventID("pass", "q", "TestOther"), eventID("pass", "q", ""), eventID("skip", "p", "TestParent/Child"), eventID("run", "p", "TestParent/name/with/slashes"), eventID("pass", "p", "TestParent/name/with/slashes"), eventID("pass", "p", "TestParent"), eventID("pass", "p", ""))
	_, tests, err := decodeValidateTestEvents(stream, []string{"p", "q"})
	if err != nil || len(tests) != 4 {
		t.Fatalf("ordered interleaving rejected: %v %v", tests, err)
	}
}
func TestValidateEventCaptureBinaryRoundtrip(t *testing.T) {
	t.Setenv("FAK_VALIDATE_EVENTS_FIXTURE", "binary")
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	witness := &validateTestWitness{Candidate: &validateCandidateFingerprint{Root: t.TempDir()}, Argv: []string{executable, "-test.run=^TestValidateEventFixtureHelperProcess$"}}
	if executeValidateEventTests(ctx, witness, []string{"p"}) || witness.Complete || witness.DecodeError == "" {
		t.Fatal("binary output accepted as complete JSON")
	}
	encoded, err := json.Marshal(witness)
	if err != nil {
		t.Fatal(err)
	}
	var restored validateTestWitness
	if err := json.Unmarshal(encoded, &restored); err != nil {
		t.Fatal(err)
	}
	if restored.RawEncoding != "base64" || !bytes.Equal(restored.Stdout, []byte{0xff, 0x00, 0xfe}) || !bytes.Equal(restored.Stderr, []byte{0x00, 0xff, 0x80}) {
		t.Fatalf("raw byte roundtrip lost data: stdout=%x stderr=%x", restored.Stdout, restored.Stderr)
	}
}
func TestValidateEventCaptureMissingExecutableKeepsBothDiagnostics(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	witness := &validateTestWitness{Candidate: &validateCandidateFingerprint{Root: t.TempDir()}, Argv: []string{filepath.Join(t.TempDir(), "missing-executable")}}
	if executeValidateEventTests(ctx, witness, []string{"p"}) || witness.Started || witness.Complete || witness.Status != "unrun" || witness.ExitCode != -1 {
		t.Fatalf("start failure state=%+v", witness)
	}
	if witness.ExecutionError == "" || witness.DecodeError == "" || witness.Reason != witness.ExecutionError || !strings.Contains(witness.ExecutionError, "missing-executable") {
		t.Fatalf("execution diagnostic replaced: %+v", witness)
	}
}
func TestValidateEventCaptureBoundsInheritedPipeWait(t *testing.T) {
	t.Setenv("FAK_VALIDATE_EVENTS_FIXTURE", "hold-pipes")
	marker := filepath.Join(t.TempDir(), "bounded-fixture-child-done")
	t.Setenv("FAK_VALIDATE_EVENTS_PIPE_MARKER", marker)
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	witness := &validateTestWitness{Candidate: &validateCandidateFingerprint{Root: t.TempDir()}, Argv: []string{executable, "-test.run=^TestValidateEventFixtureHelperProcess$"}}
	started := time.Now()
	ok := executeValidateEventTests(ctx, witness, []string{"p"})
	if ok || witness.Complete || !witness.WaitDelayExceeded || witness.Status != "incomplete" || witness.ExitCode != 0 || witness.ExecutionError == "" || witness.DescendantTeardown != "unproven" || witness.WaitDelayMS != 250 {
		t.Fatalf("pipe-drain witness=%+v", witness)
	}
	if time.Since(started) > 3*time.Second {
		t.Fatal("inherited pipe drain exceeded bounded fixture budget")
	}
	// The fixture child ends itself within one second; observe its owned marker
	// before t.TempDir cleanup. This does not assert generic descendant teardown.
	deadline := time.Now().Add(3 * time.Second)
	for {
		if _, err := os.Stat(marker); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("bounded native fixture child did not finish")
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func TestValidateEventCaptureNonzeroExitHeldPipeIsIncomplete(t *testing.T) {
	t.Setenv("FAK_VALIDATE_EVENTS_FIXTURE", "failed-hold-pipes")
	marker := filepath.Join(t.TempDir(), "bounded-failed-fixture-child-done")
	t.Setenv("FAK_VALIDATE_EVENTS_PIPE_MARKER", marker)
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	witness := &validateTestWitness{Candidate: &validateCandidateFingerprint{Root: t.TempDir()}, Argv: []string{executable, "-test.run=^TestValidateEventFixtureHelperProcess$"}}
	started := time.Now()
	ok := executeValidateEventTests(ctx, witness, []string{"p"})
	// Go1.26.6 may return only ExitError here, suppressing ErrWaitDelay even
	// though a descendant keeps stderr open past forced pipe closure.
	if ok || witness.Complete || witness.DrainStatus != "unknown" || witness.Status != "failed" || witness.ExitCode != 1 || witness.ExecutionError == "" || witness.DecodeError != "" || len(witness.Tests) != 1 || witness.DescendantTeardown != "unproven" {
		t.Fatalf("nonzero inherited-pipe witness=%+v", witness)
	}
	if time.Since(started) > 3*time.Second {
		t.Fatal("nonzero inherited pipe fixture exceeded its bound")
	}
	deadline := time.Now().Add(3 * time.Second)
	for {
		if _, err := os.Stat(marker); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("bounded fixture child did not finish")
		}
		time.Sleep(20 * time.Millisecond)
	}
}
