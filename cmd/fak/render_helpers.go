package main

import (
	"fmt"
	"io"
	"strings"
	"text/tabwriter"
)

// joinArgs renders CLI arguments while preserving arguments that require quoting.
func joinArgs(args []string) string {
	out := make([]string, len(args))
	for i, arg := range args {
		if strings.ContainsAny(arg, " \t\n\r\"") {
			out[i] = fmt.Sprintf("%q", arg)
		} else {
			out[i] = arg
		}
	}
	return strings.Join(out, " ")
}

func truncRunes(s string, n int) string {
	r := []rune(s)
	if n <= 0 || len(r) <= n {
		return s
	}
	if n == 1 {
		return "…"
	}
	return string(r[:n-1]) + "…"
}

func flushTab(tw *tabwriter.Writer, stderr io.Writer, label string) int {
	if err := tw.Flush(); err != nil {
		fmt.Fprintf(stderr, "%s: %v\n", label, err)
		return 1
	}
	return 0
}
