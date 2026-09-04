package main

import (
	"bytes"
	"context"
	"flag"
	"os"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestManageAndGuardUsageExposeSameSurface(t *testing.T) {
	newFlags := func(name string) *flag.FlagSet {
		fs := flag.NewFlagSet(name, flag.ContinueOnError)
		fs.String("provider", "", "provider")
		return fs
	}

	var manage, guard bytes.Buffer
	printGuardUsage(&manage, newFlags("manage"), "manage", false)
	printGuardUsage(&guard, newFlags("guard"), "guard", false)

	manageText := manage.String()
	guardText := guard.String()
	for _, want := range []string{
		"usage: fak manage [flags] [--] <agent command...>",
		"e.g. fak manage claude",
		"fak manage --provider openai -- codex",
		"fak manage allow <tool> | disable [--reason TEXT] | policy explain|diff",
	} {
		if !strings.Contains(manageText, want) {
			t.Errorf("manage help missing %q\n%s", want, manageText)
		}
	}
	if strings.Contains(manageText, "usage: fak guard") {
		t.Fatalf("manage help leaked legacy command name:\n%s", manageText)
	}
	if !strings.Contains(guardText, "usage: fak guard [flags] [--] <agent command...>") || !strings.Contains(guardText, "deprecated: use fak manage (or fak m)") {
		t.Fatalf("legacy guard help no longer works:\n%s", guardText)
	}
}

func TestManageDispatchAliasesShareHandler(t *testing.T) {
	sourceBytes, err := os.ReadFile("main.go")
	if err != nil {
		t.Fatal(err)
	}
	source := string(sourceBytes)
	if !strings.Contains(source, `case "manage", "m":`) || !strings.Contains(source, "cmdManage(args)") {
		t.Fatalf("main dispatch does not bind manage and m to cmdManage")
	}
	if !strings.Contains(source, `case "guard":`) || !strings.Contains(source, "cmdGuard(args)") {
		t.Fatalf("legacy guard compatibility dispatch missing")
	}
}

func TestManageBareCodexUsesDedicatedLauncher(t *testing.T) {
	var dedicated bool
	var generic []string
	dispatchManageLaunch([]string{"codex"}, func(args []string) {
		dedicated = true
		if len(args) != 0 {
			t.Fatalf("dedicated args = %v, want none", args)
		}
	}, func(args []string) { generic = args })
	if !dedicated || generic != nil {
		t.Fatalf("dedicated = %v, generic = %v", dedicated, generic)
	}
}

func TestManageBareOpencodeUsesDedicatedLauncher(t *testing.T) {
	var dedicated bool
	var generic []string
	dispatchManageLaunchWithOpencode([]string{"opencode"}, func([]string) {}, func(args []string) {
		dedicated = true
		if len(args) != 0 {
			t.Fatalf("dedicated args = %v, want none", args)
		}
	}, func(args []string) { generic = args })
	if !dedicated || generic != nil {
		t.Fatalf("dedicated = %v, generic = %v", dedicated, generic)
	}
}

func TestManageCodexWithExplicitArgumentsStaysGeneric(t *testing.T) {
	for _, argv := range [][]string{
		{"--", "codex"},
		{"codex", "exec", "task"},
		{"--provider", "openai", "--", "codex"},
	} {
		t.Run(strings.Join(argv, "_"), func(t *testing.T) {
			var dedicated bool
			var generic []string
			dispatchManageLaunch(argv, func([]string) { dedicated = true }, func(args []string) {
				generic = append([]string(nil), args...)
			})
			if dedicated || strings.Join(generic, "\x00") != strings.Join(argv, "\x00") {
				t.Fatalf("dedicated = %v, generic = %v, want %v", dedicated, generic, argv)
			}
		})
	}
}

func TestExecOpencodeLaunchChildContextCancelsChild(t *testing.T) {
	var argv []string
	if runtime.GOOS == "windows" {
		argv = []string{"cmd", "/c", "ping -n 30 127.0.0.1 >NUL"}
	} else {
		argv = []string{"sh", "-c", "sleep 30"}
	}

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	var stdout, stderr bytes.Buffer
	start := time.Now()
	code := execOpencodeLaunchChildContext(ctx, &stdout, &stderr, argv, os.Environ())
	dur := time.Since(start)

	if code == 0 {
		t.Fatalf("expected non-zero exit code on cancelled context, got %d", code)
	}
	if dur >= 10*time.Second {
		t.Fatalf("child process did not terminate promptly on context cancel, took %v", dur)
	}
}

func TestExecCodexLaunchChildContextCancelsChild(t *testing.T) {
	var argv []string
	if runtime.GOOS == "windows" {
		argv = []string{"cmd", "/c", "ping -n 30 127.0.0.1 >NUL"}
	} else {
		argv = []string{"sh", "-c", "sleep 30"}
	}

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	var stdout, stderr bytes.Buffer
	start := time.Now()
	code := execCodexLaunchChildContext(ctx, &stdout, &stderr, argv, os.Environ())
	dur := time.Since(start)

	if code == 0 {
		t.Fatalf("expected non-zero exit code on cancelled context, got %d", code)
	}
	if dur >= 10*time.Second {
		t.Fatalf("child process did not terminate promptly on context cancel, took %v", dur)
	}
}
