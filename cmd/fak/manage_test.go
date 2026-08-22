package main

import (
	"bytes"
	"flag"
	"os"
	"strings"
	"testing"
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
