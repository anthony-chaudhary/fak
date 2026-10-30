package main

import (
	"fmt"
	"io"
	"text/tabwriter"
)

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
