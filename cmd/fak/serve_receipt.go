package main

// serve_receipt.go — the `fak serve-receipt` CLI over internal/servereceipt
// (issue #14362): plan a TP×PP serving layout against a GPU profile, validate
// a stored receipt offline, and list the built-in GPU profiles. Facts and
// arithmetic live in internal/servereceipt; this file is CLI grammar and
// exit-code plumbing only (the hwspecs.go pattern).
//
// Exit codes: plan 0 OK/TIGHT, 2 input/usage error, 3 INFEASIBLE;
// validate 0 reconciles, 1 field-by-field mismatch, 2 input error.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/servereceipt"
)

const serveReceiptUsage = `fak serve-receipt — on-demand model→GPU serving-config receipts (issue #14362)

  fak serve-receipt plan --facts <json> --gpu <profile|@file.json> [--tp N] [--pp P]
                         [--max-model-len L] [--max-num-seqs S] [--gpu-mem-util F]
                         [--activation-reserve-gb F] [--format md|json] [--out <path>]
  fak serve-receipt validate --receipt <json> [--format md|json]
  fak serve-receipt profiles [--format md|json]`

func cmdServeReceipt(args []string) {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, serveReceiptUsage)
		os.Exit(2)
	}
	switch args[0] {
	case "plan":
		os.Exit(runServeReceiptPlan(os.Stdout, os.Stderr, args[1:]))
	case "validate":
		os.Exit(runServeReceiptValidate(os.Stdout, os.Stderr, args[1:]))
	case "profiles":
		os.Exit(runServeReceiptProfiles(os.Stdout, os.Stderr, args[1:]))
	case "help", "-h", "--help":
		fmt.Println(serveReceiptUsage)
	default:
		fmt.Fprintf(os.Stderr, "fak serve-receipt: unknown subcommand %q\n\n%s\n", args[0], serveReceiptUsage)
		os.Exit(2)
	}
}

func runServeReceiptPlan(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("serve-receipt plan", flag.ExitOnError)
	factsPath := fs.String("facts", "", "path to a fak.serve-receipt.facts/1 JSON document (required)")
	gpuSpec := fs.String("gpu", "", "built-in profile key or @file.json with the same shape (required)")
	tp := fs.Int("tp", 8, "tensor parallelism")
	pp := fs.Int("pp", 1, "pipeline parallelism")
	maxModelLen := fs.Int("max-model-len", 0, "requested max model len (0 = not requested)")
	maxNumSeqs := fs.Int("max-num-seqs", 32, "max concurrent sequences")
	gpuMemUtil := fs.Float64("gpu-mem-util", 0.90, "fraction of VRAM usable for weights+state+cache")
	activationReserveGB := fs.Float64("activation-reserve-gb", 2, "activation scratch reserve, decimal GB (10^9 B)")
	format := fs.String("format", "md", "output format: md or json")
	out := fs.String("out", "", "write the receipt to this path instead of stdout")
	_ = fs.Parse(args)

	if *factsPath == "" {
		fmt.Fprintln(stderr, "fak serve-receipt plan: --facts is required")
		return 2
	}
	facts, err := servereceipt.LoadFacts(*factsPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak serve-receipt plan: %v\n", err)
		return 2
	}
	gpu, err := servereceipt.LoadGPUProfile(*gpuSpec)
	if err != nil {
		fmt.Fprintf(stderr, "fak serve-receipt plan: %v\n", err)
		return 2
	}
	cfg := servereceipt.Config{
		TP: *tp, PP: *pp, MaxModelLen: *maxModelLen, MaxNumSeqs: *maxNumSeqs,
		GPUMemUtil: *gpuMemUtil, ActivationReserveGB: *activationReserveGB,
	}
	receipt, err := servereceipt.Plan(facts, gpu, cfg)
	if err != nil {
		fmt.Fprintf(stderr, "fak serve-receipt plan: %v\n", err)
		return 2
	}
	rendered, err := renderServeReceipt(receipt, *format)
	if err != nil {
		fmt.Fprintf(stderr, "fak serve-receipt plan: %v\n", err)
		return 2
	}
	if err := emitServeReceipt(stdout, rendered, *out, fmt.Sprintf("wrote %s (verdict %s)\n", *out, receipt.Verdict)); err != nil {
		fmt.Fprintf(stderr, "fak serve-receipt plan: %v\n", err)
		return 2
	}
	if receipt.Verdict == servereceipt.VerdictInfeasible {
		return 3
	}
	return 0
}

func runServeReceiptValidate(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("serve-receipt validate", flag.ExitOnError)
	receiptPath := fs.String("receipt", "", "path to a fak.serve-receipt.receipt/1 JSON document (required)")
	format := fs.String("format", "md", "output format: md or json")
	_ = fs.Parse(args)

	if *receiptPath == "" {
		fmt.Fprintln(stderr, "fak serve-receipt validate: --receipt is required")
		return 2
	}
	receipt, err := servereceipt.LoadReceipt(*receiptPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak serve-receipt validate: %v\n", err)
		return 2
	}
	doc := servereceipt.Validate(receipt)
	var rendered []byte
	switch *format {
	case "json":
		if rendered, err = servereceipt.RenderValidateJSON(doc); err != nil {
			fmt.Fprintf(stderr, "fak serve-receipt validate: %v\n", err)
			return 2
		}
	case "md":
		rendered = []byte(servereceipt.RenderValidateMD(doc))
	default:
		fmt.Fprintf(stderr, "fak serve-receipt validate: --format %q: want md or json\n", *format)
		return 2
	}
	if _, err := stdout.Write(rendered); err != nil {
		fmt.Fprintf(stderr, "fak serve-receipt validate: %v\n", err)
		return 2
	}
	if doc.OK {
		return 0
	}
	return 1
}

func runServeReceiptProfiles(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("serve-receipt profiles", flag.ExitOnError)
	format := fs.String("format", "md", "output format: md or json")
	_ = fs.Parse(args)

	profiles := servereceipt.AllProfiles()
	if *format == "json" {
		doc := struct {
			Schema   string                    `json:"schema"`
			Units    string                    `json:"units"`
			Profiles []servereceipt.GPUProfile `json:"profiles"`
		}{Schema: servereceipt.ProfilesSchema, Units: servereceipt.UnitsNote, Profiles: profiles}
		if err := writeServeReceiptJSON(stdout, doc); err != nil {
			fmt.Fprintf(stderr, "fak serve-receipt profiles: %v\n", err)
			return 2
		}
		return 0
	}
	if *format != "md" {
		fmt.Fprintf(stderr, "fak serve-receipt profiles: --format %q: want md or json\n", *format)
		return 2
	}
	fmt.Println("fak serve-receipt built-in GPU profiles (vram_gb read as GiB = 2^30 B)")
	fmt.Printf("%-30s %8s %6s %4s %7s %s\n", "KEY", "VRAM-GB", "SM", "FP4", "NVLINK", "PCIE")
	for _, p := range profiles {
		pcie := p.PCIe
		if pcie == "" {
			pcie = "-"
		}
		fmt.Printf("%-30s %8g %6s %4t %7t %s\n", p.Key, p.VRAMGB, p.SM, p.FP4Native, p.NVLink, pcie)
	}
	fmt.Println("plan a layout: fak serve-receipt plan --facts <json> --gpu <key|@file.json>")
	return 0
}

func renderServeReceipt(receipt servereceipt.Receipt, format string) ([]byte, error) {
	switch format {
	case "json":
		return servereceipt.RenderJSON(receipt)
	case "md":
		return []byte(servereceipt.RenderMD(receipt)), nil
	default:
		return nil, fmt.Errorf("--format %q: want md or json", format)
	}
}

func writeServeReceiptJSON(w io.Writer, v any) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}

// emitServeReceipt writes the rendered receipt to stdout, or to --out plus a
// one-line confirmation on stdout.
func emitServeReceipt(stdout io.Writer, rendered []byte, outPath, confirmation string) error {
	if outPath == "" {
		_, err := stdout.Write(rendered)
		return err
	}
	if err := os.WriteFile(outPath, rendered, 0o644); err != nil {
		return fmt.Errorf("write --out: %w", err)
	}
	_, err := fmt.Fprint(stdout, confirmation)
	return err
}
