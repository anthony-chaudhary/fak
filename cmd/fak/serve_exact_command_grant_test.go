package main

import (
	"context"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=1s
func TestServeNativeBashCommandGrantAndTimeoutDefaults(t *testing.T) {
	fs, sf := newServeFlagSet()
	if err := fs.Parse(nil); err != nil {
		t.Fatal(err)
	}
	if *sf.nativeBashCommandTimeout != 2*time.Minute || sf.isExplicitFlag("native-bash-command-timeout") {
		t.Fatalf("default timeout=%v explicit=%v, want 2m/false", *sf.nativeBashCommandTimeout, sf.isExplicitFlag("native-bash-command-timeout"))
	}

	fs, sf = newServeFlagSet()
	if err := fs.Parse([]string{
		"--native",
		"--native-allow-bash-command=go version",
		"--native-allow-bash-command=git status --short",
		"--native-bash-command-timeout=10m",
	}); err != nil {
		t.Fatal(err)
	}
	if got := []string(sf.nativeAllowBashCommands); len(got) != 2 || got[0] != "go version" || got[1] != "git status --short" {
		t.Fatalf("repeatable grants=%q, want both values in order", got)
	}
	if *sf.nativeBashCommandTimeout != 10*time.Minute || !sf.isExplicitFlag("native-bash-command-timeout") {
		t.Fatalf("maximum timeout=%v explicit=%v, want 10m/true", *sf.nativeBashCommandTimeout, sf.isExplicitFlag("native-bash-command-timeout"))
	}
	if err := validateServeNativeBashCommandGrants(*sf.native, *sf.nativeCodeTools, sf.nativeAllowBashCommands); err != nil {
		t.Fatalf("maximum grants validation: %v", err)
	}
	if err := validateServeNativeBashCommandTimeout(*sf.native, *sf.nativeCodeTools, *sf.nativeBashCommandTimeout, sf.isExplicitFlag("native-bash-command-timeout")); err != nil {
		t.Fatalf("maximum timeout validation: %v", err)
	}
}

// fak-test:runtime slow est=30s
func TestServeNativeBashCommandGrantAndTimeoutRejectBeforeStartup(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
		flag string
	}{
		{name: "blank grant", args: []string{"--native-allow-bash-command= "}, flag: "native-allow-bash-command"},
		{name: "grant without native", args: []string{"--native-allow-bash-command=go version"}, flag: "native-allow-bash-command"},
		{name: "grant with tools disabled", args: []string{"--native", "--native-code-tools=false", "--native-allow-bash-command=go version"}, flag: "native-allow-bash-command"},
		{name: "zero timeout", args: []string{"--native-bash-command-timeout=0s"}, flag: "native-bash-command-timeout"},
		{name: "negative timeout", args: []string{"--native-bash-command-timeout=-1s"}, flag: "native-bash-command-timeout"},
		{name: "above maximum timeout", args: []string{"--native-bash-command-timeout=10m1ms"}, flag: "native-bash-command-timeout"},
		{name: "timeout without native", args: []string{"--native-bash-command-timeout=3m"}, flag: "native-bash-command-timeout"},
		{name: "timeout with tools disabled", args: []string{"--native", "--native-code-tools=false", "--native-bash-command-timeout=3m"}, flag: "native-bash-command-timeout"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, err := runServeNativeGrantCLI(t, tc.args...)
			if err == nil {
				t.Fatalf("invalid serve argv succeeded: %v\n%s", tc.args, out)
			}
			if !strings.Contains(out, tc.flag) {
				t.Fatalf("rejection omitted actionable flag %q: %s", tc.flag, out)
			}
		})
	}
}

func runServeNativeGrantCLI(t *testing.T, args ...string) (string, error) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	childArgs := []string{"-test.run=^TestServeNativeBashCommandGrantCLIHelper$", "--"}
	args = append([]string{"--mock"}, args...)
	childArgs = append(childArgs, args...)
	cmd := exec.CommandContext(ctx, os.Args[0], childArgs...)
	cmd.Env = append(os.Environ(), "FAK_SERVE_NATIVE_GRANT_CLI_HELPER=1")
	out, err := cmd.CombinedOutput()
	if ctx.Err() != nil {
		t.Fatalf("serve validation reached startup instead of refusing: %v\n%s", ctx.Err(), out)
	}
	return string(out), err
}

// fak-test:runtime fast est=1s
func TestServeNativeBashCommandGrantCLIHelper(t *testing.T) {
	if os.Getenv("FAK_SERVE_NATIVE_GRANT_CLI_HELPER") != "1" {
		return
	}
	for i, arg := range os.Args {
		if arg == "--" {
			cmdServe(os.Args[i+1:])
			return
		}
	}
	os.Exit(97)
}
