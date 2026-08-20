package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/cachevalue"
)

func runCachevalueQwen38Campaign(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("fak cachevalue qwen38-campaign", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "versioned Qwen3.8 cache campaign JSON")
	output := fs.String("output", "", "optional report output path")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *input == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak cachevalue qwen38-campaign --input CAMPAIGN.json [--output REPORT.json]")
		return 2
	}
	f, err := os.Open(*input)
	if err != nil {
		fmt.Fprintf(stderr, "fak cachevalue qwen38-campaign: %v\n", err)
		return 1
	}
	defer f.Close()
	var c cachevalue.Qwen38Campaign
	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&c); err != nil {
		fmt.Fprintf(stderr, "fak cachevalue qwen38-campaign: decode: %v\n", err)
		return 1
	}
	r, err := cachevalue.FoldQwen38Campaign(c)
	if err != nil {
		fmt.Fprintf(stderr, "fak cachevalue qwen38-campaign: %v\n", err)
		return 1
	}
	var w io.Writer = stdout
	var out *os.File
	if *output != "" {
		out, err = os.Create(*output)
		if err != nil {
			fmt.Fprintf(stderr, "fak cachevalue qwen38-campaign: %v\n", err)
			return 1
		}
		defer out.Close()
		w = io.MultiWriter(stdout, out)
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err := enc.Encode(r); err != nil {
		fmt.Fprintf(stderr, "fak cachevalue qwen38-campaign: encode: %v\n", err)
		return 1
	}
	if r.Verdict != "PASS" {
		return 3
	}
	return 0
}
