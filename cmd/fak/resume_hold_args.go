package main

import (
	"flag"
	"fmt"
	"io"
	"strings"
)

// parseInterspersedResumeHold accepts the documented "session-id --flag value"
// form while preserving the ordinary flags-first form. Exactly one session ID is
// returned; unknown flags and extra positionals remain usage errors.
func parseInterspersedResumeHold(fs *flag.FlagSet, argv []string, stderr io.Writer) (string, bool) {
	var positional []string
	var flags []string
	for i := 0; i < len(argv); i++ {
		arg := argv[i]
		if strings.HasPrefix(arg, "-") {
			flags = append(flags, arg)
			if !strings.Contains(arg, "=") {
				name := strings.TrimLeft(arg, "-")
				f := fs.Lookup(name)
				if f == nil {
					flags = append(flags, argv[i+1:]...)
					break
				}
				if _, ok := f.Value.(interface{ IsBoolFlag() bool }); !ok && i+1 < len(argv) {
					i++
					flags = append(flags, argv[i])
				}
			}
		} else {
			positional = append(positional, arg)
		}
	}
	if !parseFlags(fs, flags) {
		return "", false
	}
	positional = append(positional, fs.Args()...)
	if len(positional) != 1 {
		fmt.Fprintf(stderr, "fak resume hold: want exactly one session-id, got %d\n", len(positional))
		return "", false
	}
	return strings.TrimSpace(positional[0]), true
}

func listRequestedRaw(argv []string) bool {
	for _, arg := range argv {
		if arg == "--list" || arg == "-list" || arg == "--list=true" || arg == "-list=true" {
			return true
		}
	}
	return false
}
