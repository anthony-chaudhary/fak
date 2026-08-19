package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
)

func sessionNewTestDeps(goos, stdin string) sessionNewDeps {
	return sessionNewDeps{
		goos:       goos,
		stdin:      strings.NewReader(stdin),
		lookPath:   func(name string) (string, error) { return "/bin/" + name, nil },
		executable: func() (string, error) { return "/opt/fak", nil },
		getwd:      func() (string, error) { return "/work/repo", nil },
		readClip:   func(string) (string, error) { return "clipboard prompt", nil },
		start:      func(string, []string, string) error { return nil },
	}
}

func TestSessionNewLaunchesGuardBoundaryWithoutShellParsing(t *testing.T) {
	prompt := "review spaces 'quotes' $HOME; echo nope\nsecond line"
	deps := sessionNewTestDeps("windows", "")
	var gotName, gotCWD string
	var gotArgs []string
	deps.start = func(name string, args []string, cwd string) error {
		gotName, gotArgs, gotCWD = name, append([]string(nil), args...), cwd
		return nil
	}
	var out, errOut bytes.Buffer
	if code := runSessionNewWith(&out, &errOut, []string{"--terminal", "windows-terminal", prompt}, deps); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if gotName != "/bin/wt.exe" || gotCWD != "/work/repo" {
		t.Fatalf("start=(%q,%q), want wt.exe in repo", gotName, gotCWD)
	}
	want := []string{"new-tab", "--startingDirectory", "/work/repo", "/opt/fak", "guard", "--", "claude", prompt}
	if !reflect.DeepEqual(gotArgs, want) {
		t.Fatalf("argv=%q\nwant=%q", gotArgs, want)
	}
	for _, arg := range gotArgs {
		if arg == "powershell" || arg == "sh" || arg == "cmd.exe" || arg == "-c" || arg == "-Command" {
			t.Fatalf("shell parser leaked into launch argv: %q", gotArgs)
		}
	}
}

func TestSessionNewStdinPreservesMultilineAndTrimsOneRecordNewline(t *testing.T) {
	deps := sessionNewTestDeps("linux", "first λ\nsecond\n")
	var got []string
	deps.start = func(_ string, args []string, _ string) error { got = append([]string(nil), args...); return nil }
	var out, errOut bytes.Buffer
	if code := runSessionNewWith(&out, &errOut, []string{"--stdin", "--terminal", "x-terminal"}, deps); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	want := []string{"-e", "/opt/fak", "guard", "--", "claude", "first λ\nsecond"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("argv=%q want=%q", got, want)
	}
}

func TestSessionNewDryRunReceiptDoesNotLeakPrompt(t *testing.T) {
	prompt := "sensitive selected text"
	deps := sessionNewTestDeps("windows", "")
	deps.start = func(string, []string, string) error { t.Fatal("dry-run started terminal"); return nil }
	var out, errOut bytes.Buffer
	if code := runSessionNewWith(&out, &errOut, []string{"--dry-run", "--json", "--terminal", "windows-terminal", prompt}, deps); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if strings.Contains(out.String(), prompt) {
		t.Fatalf("receipt leaked prompt: %s", out.String())
	}
	var got sessionNewReceipt
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != "fak-session-new/1" || got.Status != "planned" || got.Source != "argument" || got.PromptBytes != len(prompt) {
		t.Fatalf("receipt=%+v", got)
	}
	if len(got.PromptSHA256) != 64 || got.Arguments[len(got.Arguments)-1] != "<prompt:sha256:"+got.PromptSHA256[:12]+">" {
		t.Fatalf("prompt identity not stable/redacted: %+v", got)
	}
}

func TestSessionNewClipboardAndFailureContracts(t *testing.T) {
	t.Run("clipboard", func(t *testing.T) {
		deps := sessionNewTestDeps("windows", "")
		var got []string
		deps.start = func(_ string, args []string, _ string) error { got = args; return nil }
		var out, errOut bytes.Buffer
		if code := runSessionNewWith(&out, &errOut, []string{"--clipboard", "--agent", "codex", "--terminal", "windows-terminal"}, deps); code != 0 {
			t.Fatalf("code=%d stderr=%s", code, errOut.String())
		}
		if !reflect.DeepEqual(got[len(got)-3:], []string{"--", "codex", "clipboard prompt"}) {
			t.Fatalf("argv=%q", got)
		}
	})

	for _, tc := range []struct {
		name   string
		args   []string
		mutate func(*sessionNewDeps)
		want   string
	}{
		{"missing source", nil, nil, "choose exactly one"},
		{"conflicting sources", []string{"--stdin", "text"}, nil, "choose exactly one"},
		{"empty", []string{"   "}, nil, "prompt is empty"},
		{"clipboard unavailable", []string{"--clipboard"}, func(d *sessionNewDeps) {
			d.readClip = func(string) (string, error) { return "", errors.New("clipboard unavailable") }
		}, "clipboard unavailable"},
		{"terminal unavailable", []string{"text"}, func(d *sessionNewDeps) {
			d.lookPath = func(string) (string, error) { return "", errors.New("missing") }
		}, "no supported terminal found"},
		{"start failure", []string{"--terminal", "windows-terminal", "text"}, func(d *sessionNewDeps) {
			d.start = func(string, []string, string) error { return errors.New("start denied") }
		}, "start denied"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			deps := sessionNewTestDeps("windows", "")
			if tc.mutate != nil {
				tc.mutate(&deps)
			}
			var out, errOut bytes.Buffer
			if code := runSessionNewWith(&out, &errOut, tc.args, deps); code == 0 {
				t.Fatalf("code=0 out=%s", out.String())
			}
			if !strings.Contains(errOut.String(), tc.want) {
				t.Fatalf("stderr=%q want %q", errOut.String(), tc.want)
			}
			if strings.Contains(out.String(), "launched") {
				t.Fatalf("false receipt: %s", out.String())
			}
		})
	}
}

func TestBuildSessionTerminalUnixAdapters(t *testing.T) {
	look := func(name string) (string, error) { return "/usr/bin/" + name, nil }
	_, bin, args, err := buildSessionTerminal("gnome-terminal", "", "linux", look, "/work", "/opt/fak", "claude", "hi")
	if err != nil {
		t.Fatal(err)
	}
	if bin != "/usr/bin/gnome-terminal" || !reflect.DeepEqual(args, []string{"--working-directory", "/work", "--", "/opt/fak", "guard", "--", "claude", "hi"}) {
		t.Fatalf("bin=%q args=%q", bin, args)
	}
}
