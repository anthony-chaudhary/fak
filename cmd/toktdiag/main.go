// Command toktdiag reports whether fak can extract an embedded GGUF tokenizer.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

func main() {
	os.Exit(run(os.Stdout, os.Stderr, os.Args[1:]))
}

func run(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("toktdiag", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: toktdiag <model.gguf>")
		fmt.Fprintln(stderr, "       reports tokenizer.ggml metadata and a minimal encode smoke result")
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 1 {
		fs.Usage()
		return 2
	}
	f, err := ggufload.Open(fs.Arg(0))
	if err != nil {
		fmt.Fprintln(stderr, "open:", err)
		return 1
	}
	gt, ok := f.GGMLTokenizer()
	fmt.Fprintf(stdout, "GGMLTokenizer ok=%v\n", ok)
	if !ok {
		toks, okt := f.StringArray("tokenizer.ggml.tokens")
		mrg, okm := f.StringArray("tokenizer.ggml.merges")
		fmt.Fprintf(stdout, "  tokens present=%v n=%d ; merges present=%v n=%d\n", okt, len(toks), okm, len(mrg))
		return 0
	}
	model, _ := f.String("tokenizer.ggml.model")
	fmt.Fprintf(stdout, "  model=%q pre=%q tokens=%d merges=%d tokentypes=%d bos=%d eos=%d\n",
		model, gt.Pre, len(gt.Tokens), len(gt.Merges), len(gt.TokenTypes), gt.BOS, gt.EOS)
	tok, err := tokenizer.FromGGML(gt.Tokens, gt.Merges, gt.TokenTypes, gt.Pre)
	if err != nil {
		fmt.Fprintln(stderr, "FromGGML ERROR:", err)
		return 1
	}
	ids, err := tok.Encode("Hello, world!")
	fmt.Fprintf(stdout, "FromGGML OK; Encode('Hello, world!') -> %d ids, err=%v\n", len(ids), err)
	if err != nil {
		return 1
	}
	return 0
}
