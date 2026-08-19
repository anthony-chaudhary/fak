package main

import (
	"context"
	"flag"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/sessionmine"
	"io"
	"strconv"
	"strings"
)

func runSessionHistoryBenchmark(ctx context.Context, stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("session-history benchmark", flag.ContinueOnError)
	fs.SetOutput(stderr)
	sizes := fs.String("sizes", "1000,10000,100000", "comma-separated synthetic session counts (1..100000)")
	reps := fs.Int("repetitions", 3, "runs per cold/warm/change phase")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	var ns []int
	for _, raw := range strings.Split(*sizes, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(raw))
		if err != nil {
			fmt.Fprintln(stderr, "session-history benchmark: invalid sizes")
			return 2
		}
		ns = append(ns, n)
	}
	rep, err := sessionmine.BenchmarkRefresh(ctx, sessionmine.RefreshBenchmarkOptions{Sizes: ns, Repetitions: *reps})
	if err != nil {
		fmt.Fprintf(stderr, "session-history benchmark: %v\n", err)
		return 1
	}
	if err := sessionmine.WriteJSON(stdout, rep); err != nil {
		fmt.Fprintf(stderr, "session-history benchmark: %v\n", err)
		return 1
	}
	if !rep.Pass {
		return 1
	}
	return 0
}
