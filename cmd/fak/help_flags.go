package main

import (
	"flag"
	"fmt"
	"io"
	"strings"
	"unicode"
)

// printFlagDefaults preserves authored descriptions for per-verb deep help.
// Shortening is opt-in for explicitly concise help surfaces.
func printFlagDefaults(w io.Writer, fs *flag.FlagSet, concise bool) {
	fs.VisitAll(func(f *flag.Flag) {
		arg, usage := flag.UnquoteUsage(f)
		if arg == "" && !isBoolFlag(f) {
			arg = "value"
		}
		if arg != "" {
			arg = " " + arg
		}
		if concise {
			usage = conciseFlagSummary(usage)
		}
		fmt.Fprintf(w, "  --%s%s\n      %s\n", f.Name, arg, usage)
	})
}

func conciseFlagSummary(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if i := strings.Index(s, ". "); i >= 0 {
		s = s[:i+1]
	}
	const max = 110 - 6 // Leave room for the rendered explanation's indentation.
	r := []rune(s)
	if len(r) > max {
		r = r[:max-1]
		end := len(r)
		for end > 0 && !unicode.IsSpace(r[end-1]) {
			end--
		}
		if end > 0 {
			r = r[:end]
		}
		s = strings.TrimSpace(string(r)) + "…"
	}
	if s == "" {
		return "See detailed help."
	}
	return s
}

func isBoolFlag(f *flag.Flag) bool {
	type boolFlag interface{ IsBoolFlag() bool }
	bf, ok := f.Value.(boolFlag)
	return ok && bf.IsBoolFlag()
}
