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

func cmdModelObserve(args []string) {
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "usage: fak model-observe proxy --backend URL [--listen ADDR --ledger FILE]\n       fak model-observe report --input FILE [--format md|json]")
		os.Exit(2)
	}
	switch args[0] {
	case "proxy":
		fs := flag.NewFlagSet("model-observe proxy", flag.ExitOnError)
		listen := fs.String("listen", "127.0.0.1:8091", "proxy listen address")
		backend := fs.String("backend", "", "OpenAI-compatible backend base URL")
		ledger := fs.String("ledger", "model-perf.jsonl", "append-only observation JSONL")
		_ = fs.Parse(args[1:])
		u, err := modelperfobs.ParseBackend(*backend)
		if err != nil {
			fmt.Fprintln(os.Stderr, "model-observe:", err)
			os.Exit(2)
		}
		server := &http.Server{Addr: *listen, Handler: &modelperfobs.Proxy{Backend: u, Ledger: *ledger}, ReadHeaderTimeout: 10 * time.Second}
		fmt.Fprintf(os.Stderr, "model-observe: proxy http://%s -> %s; ledger=%s\n", *listen, u, *ledger)
		if err := server.ListenAndServe(); err != nil {
			fmt.Fprintln(os.Stderr, "model-observe:", err)
			os.Exit(1)
		}
	case "report":
		fs := flag.NewFlagSet("model-observe report", flag.ExitOnError)
		input := fs.String("input", "model-perf.jsonl", "observation JSONL")
		format := fs.String("format", "md", "md or json")
		_ = fs.Parse(args[1:])
		f, err := os.Open(*input)
		if err != nil {
			fmt.Fprintln(os.Stderr, "model-observe:", err)
			os.Exit(1)
		}
		defer f.Close()
		rows, err := modelperfobs.ReadObservations(f)
		if err != nil {
			fmt.Fprintln(os.Stderr, "model-observe:", err)
			os.Exit(1)
		}
		s := modelperfobs.Summarize(rows)
		if *format == "json" {
			if err := json.NewEncoder(os.Stdout).Encode(s); err != nil {
				os.Exit(1)
			}
			return
		}
		if *format != "md" {
			fmt.Fprintln(os.Stderr, "model-observe: format must be md or json")
			os.Exit(2)
		}
		if err := modelperfobs.WriteMarkdown(os.Stdout, s); err != nil {
			os.Exit(1)
		}
	default:
		fmt.Fprintf(os.Stderr, "model-observe: unknown subcommand %q\n", args[0])
		os.Exit(2)
	}
}
