package main

import (
	"encoding/json"
	"flag"
	"fmt"
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

func fatalIf(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "modelperfobs:", err)
		os.Exit(1)
	}
}
func usage() {
	fmt.Fprintln(os.Stderr, "usage: modelperfobs proxy --backend URL [--listen ADDR --ledger FILE]\n       modelperfobs report --input FILE [--format md|json]")
}
