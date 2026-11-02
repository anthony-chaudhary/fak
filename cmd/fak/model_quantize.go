package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

func runModelQuantize(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("model quantize", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "local single-file GGUFv3 input; no aliases or downloads")
	output := fs.String("output", "", "new local GGUF output; must not exist")
	kind := fs.String("type", "", "reference q4_k, q5_k or q6_k; importance matrices not supported")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak model quantize --input local.gguf --output new.gguf --type q4_k|q5_k|q6_k")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	target, ok := map[string]ggufload.TensorType{"q4_k": ggufload.TensorQ4_K, "q5_k": ggufload.TensorQ5_K, "q6_k": ggufload.TensorQ6_K}[*kind]
	if !ok || *input == "" || *output == "" || fs.NArg() != 0 {
		fs.Usage()
		return 2
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()
	report, err := ggufload.ConvertGGUF(ctx, *input, *output, target)
	if err != nil {
		fmt.Fprintf(stderr, "fak model quantize: %v\n", err)
		return 1
	}
	if err := json.NewEncoder(stdout).Encode(report); err != nil {
		fmt.Fprintf(stderr, "fak model quantize: output was published; report write failed: %v\n", err)
		return 1
	}
	return 0
}
