package main

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/devhandoff"
)

func TestMovedCommandInventoryMatchesFakDevDispatcher(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "fak-dev", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	re := regexp.MustCompile(`(?m)^\s*case ((?:"[^"]+"(?:, )?)+):`)
	quoted := regexp.MustCompile(`"([^"]+)"`)
	var got []string
	for _, match := range re.FindAllSubmatch(src, -1) {
		for _, name := range quoted.FindAllSubmatch(match[1], -1) {
			switch string(name[1]) {
			case "version", "--version", "capabilities":
				continue
			default:
				got = append(got, string(name[1]))
			}
		}
	}
	sort.Strings(got)
	want := devhandoff.Names()
	if len(got) != len(want) {
		t.Fatalf("moved inventory has %d commands, fak-dev dispatches %d\ninventory=%v\ndispatch=%v", len(want), len(got), want, got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("moved inventory and fak-dev dispatcher differ\ninventory=%v\ndispatch=%v", want, got)
		}
	}
}
