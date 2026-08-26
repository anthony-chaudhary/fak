package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

func runComputeTrace(stdout, stderr io.Writer, args []string) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "usage: fak compute-trace capture|summary [flags]")
		return 2
	}
	switch args[0] {
	case "capture":
		fs := flag.NewFlagSet("compute-trace capture", flag.ContinueOnError)
		fs.SetOutput(stderr)
		out := fs.String("out", "compute-trace.json", "trace artifact path")
		limit := fs.Int("limit", 128, "maximum retained events")
		run := fs.String("run", "local", "run correlation ID")
		request := fs.String("request", "capture", "request correlation ID")
		if err := fs.Parse(args[1:]); err != nil || *limit < 1 {
			return 2
		}
		r, disable := computetrace.Enable(*limit, *run, *request)
		b := compute.Default()
		w := compute.NewF32(b, []int{1, 2}, []float32{1, 2})
		x := compute.NewF32(b, []int{2}, []float32{3, 4})
		started := time.Now()
		_ = b.MatMul(w, x)
		requestNS := time.Since(started).Nanoseconds()
		disable()
		f, err := os.Create(*out)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		err = r.Write(f)
		closeErr := f.Close()
		if err == nil {
			err = closeErr
		}
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		a := r.Artifact()
		fmt.Fprintf(stdout, "wrote %s events=%d dropped=%d request_ns=%d unexplained_ns=%d observer_overhead_ns=%d\n", *out, len(a.Events), a.Dropped, requestNS, requestNS-eventDuration(a), a.ObserverOverheadNS)
		return 0
	case "summary":
		fs := flag.NewFlagSet("compute-trace summary", flag.ContinueOnError)
		fs.SetOutput(stderr)
		in := fs.String("in", "", "trace artifact path")
		if err := fs.Parse(args[1:]); err != nil || *in == "" {
			return 2
		}
		f, err := os.Open(*in)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		defer f.Close()
		a, err := computetrace.Read(f)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		fmt.Fprintf(stdout, "schema=%s events=%d dropped=%d duration_ns=%d observer_overhead_ns=%d\n", a.Schema, len(a.Events), a.Dropped, eventDuration(a), a.ObserverOverheadNS)
		return 0
	default:
		fmt.Fprintf(stderr, "unknown compute-trace action %q\n", args[0])
		return 2
	}
}
func eventDuration(a computetrace.Artifact) int64 {
	var n int64
	for _, e := range a.Events {
		n += e.DurationNS
	}
	return n
}
