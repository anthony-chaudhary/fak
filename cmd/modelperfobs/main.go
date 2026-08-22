package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "proxy":
		runProxy(os.Args[2:])
	case "report":
		runReport(os.Args[2:])
	case "cache-state-bench":
		runStateBench(os.Args[2:])
	default:
		usage()
		os.Exit(2)
	}
}

func runProxy(args []string) {
	fs := flag.NewFlagSet("proxy", flag.ExitOnError)
	listen := fs.String("listen", "127.0.0.1:8091", "proxy listen address")
	backend := fs.String("backend", "", "OpenAI-compatible backend base URL")
	ledger := fs.String("ledger", "model-perf.jsonl", "append-only observation JSONL")
	_ = fs.Parse(args)
	u, err := modelperfobs.ParseBackend(*backend)
	fatalIf(err)
	server := &http.Server{Addr: *listen, Handler: &modelperfobs.Proxy{Backend: u, Ledger: *ledger}, ReadHeaderTimeout: 10 * time.Second}
	fmt.Fprintf(os.Stderr, "modelperfobs: proxy http://%s -> %s; ledger=%s\n", *listen, u, *ledger)
	fatalIf(server.ListenAndServe())
}

func runReport(args []string) {
	fs := flag.NewFlagSet("report", flag.ExitOnError)
	input := fs.String("input", "model-perf.jsonl", "observation JSONL")
	format := fs.String("format", "md", "md or json")
	_ = fs.Parse(args)
	f, err := os.Open(*input)
	fatalIf(err)
	defer f.Close()
	rows, err := modelperfobs.ReadObservations(f)
	fatalIf(err)
	s := modelperfobs.Summarize(rows)
	if *format == "json" {
		fatalIf(json.NewEncoder(os.Stdout).Encode(s))
		return
	}
	if *format != "md" {
		fatalIf(fmt.Errorf("format must be md or json"))
	}
	fatalIf(modelperfobs.WriteMarkdown(os.Stdout, s))
}

func runStateBench(args []string) {
	fs := flag.NewFlagSet("cache-state-bench", flag.ExitOnError)
	output := fs.String("output", "", "write the observed JSON witness to this path (default: stdout)")
	verify := fs.String("verify", "", "verify a captured cache-state report instead of running")
	pretty := fs.Bool("pretty", true, "indent JSON output")
	_ = fs.Parse(args)
	if *verify != "" {
		f, err := os.Open(*verify)
		fatalIf(err)
		defer f.Close()
		report, err := modelperfobs.ReadStateReport(f)
		fatalIf(err)
		fmt.Printf("STATE_TRANSITION_WITNESS_OK arms=%d provenance=%s scope=%s\n", len(report.Arms), report.Provenance.EvidenceKind, report.Provenance.Scope)
		return
	}
	report, err := modelperfobs.RunHermeticStateBenchmark(context.Background())
	fatalIf(err)
	w := io.Writer(os.Stdout)
	var f *os.File
	if *output != "" {
		f, err = os.Create(*output)
		fatalIf(err)
		defer f.Close()
		w = f
	}
	fatalIf(modelperfobs.WriteStateReport(w, report, *pretty))
	if *output != "" {
		fmt.Fprintf(os.Stderr, "STATE_TRANSITION_WITNESS_WRITTEN path=%s arms=%d provenance=%s scope=%s\n", *output, len(report.Arms), report.Provenance.EvidenceKind, report.Provenance.Scope)
	}
}

func fatalIf(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "modelperfobs:", err)
		os.Exit(1)
	}
}
func usage() {
	fmt.Fprintln(os.Stderr, "usage: modelperfobs proxy --backend URL [--listen ADDR --ledger FILE]\n       modelperfobs report --input FILE [--format md|json]\n       modelperfobs cache-state-bench [--output FILE --pretty=true]\n       modelperfobs cache-state-bench --verify FILE")
}
