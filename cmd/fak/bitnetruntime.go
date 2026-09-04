package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/bitnetruntime"
)

func cmdBitnetRuntime(argv []string) {
	os.Exit(runBitnetRuntime(os.Stdout, os.Stderr, argv))
}

type bitnetRuntimeInput struct {
	Probe string              `json:"probe"`
	Host  bitnetruntime.Host  `json:"host"`
	Model bitnetruntime.Model `json:"model"`
}

func runBitnetRuntime(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("bitnetruntime", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "JSON input file ('-' for stdin)")
	contract := fs.Bool("contract", false, "print contract version and supported kernels")
	jsonOut := fs.Bool("json", false, "emit JSON output")
	_ = jsonOut

	if err := fs.Parse(argv); err != nil {
		return 2
	}

	if *contract {
		info := map[string]any{
			"contract_version":    bitnetruntime.ContractVersion,
			"runtime_name":        bitnetruntime.RuntimeName,
			"min_runtime_version": bitnetruntime.MinRuntimeVersion,
			"kernels": []string{
				string(bitnetruntime.KernelI2S),
				string(bitnetruntime.KernelTL1),
				string(bitnetruntime.KernelTL2),
			},
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(info); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		return 0
	}

	if *input == "" {
		fmt.Fprintln(stderr, "usage: fak bitnetruntime --contract | --input FILE [--json]")
		return 2
	}

	var r io.Reader
	if *input == "-" {
		r = os.Stdin
	} else {
		f, err := os.Open(*input)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		defer f.Close()
		r = f
	}

	var req bitnetRuntimeInput
	dec := json.NewDecoder(r)
	if err := dec.Decode(&req); err != nil {
		fmt.Fprintf(stderr, "bitnetruntime: decode input: %v\n", err)
		return 2
	}

	prober := func(context.Context) ([]byte, error) {
		return []byte(req.Probe), nil
	}

	res := bitnetruntime.DiscoverAndAdmit(context.Background(), prober, req.Host, req.Model)
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(res); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}

	switch res.Outcome {
	case bitnetruntime.OutcomeDelegate:
		return 0
	case bitnetruntime.OutcomeUnsupported:
		return 1
	case bitnetruntime.OutcomeAbstain:
		return 3
	case bitnetruntime.OutcomeRefuse:
		return 4
	default:
		return 2
	}
}
