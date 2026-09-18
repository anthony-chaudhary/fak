package main

import (
	"bytes"
	"flag"
	"strings"
	"testing"
)

// TestUpHelpListsEveryRegisteredFlag guards against printUpHelp drifting from
// the actual `up` flag set. The help string is hand-maintained, so a new flag
// added to runUp without a help line becomes undiscoverable to an operator
// configuring `fak up` (e.g. from launchd).
func TestUpHelpListsEveryRegisteredFlag(t *testing.T) {
	var buf bytes.Buffer
	printUpHelp(&buf)
	help := buf.String()

	fs := flag.NewFlagSet("up", flag.ContinueOnError)
	registerUpFlags(fs)

	fs.VisitAll(func(f *flag.Flag) {
		if f.Name == "help" || f.Name == "h" {
			return
		}
		if !strings.Contains(help, "--"+f.Name) {
			t.Errorf("fak up --help omits registered flag --%s", f.Name)
		}
	})
}
