package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"

	"github.com/anthony-chaudhary/fak/internal/hfhub"
)

// cmdModel handles `fak model <subcommand>`. Today the only subcommand is
// `load`, which resolves an hf:// URI to a locally cached file path (issue #294).
func cmdModel(args []string) {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "usage: fak model load <hf://owner/repo[@rev]/file>")
		os.Exit(2)
	}
	switch args[0] {
	case "load":
		cmdModelLoad(args[1:])
	case "-h", "--help", "help":
		fmt.Fprintln(os.Stderr, "usage: fak model load <hf://owner/repo[@rev]/file>")
	default:
		fmt.Fprintf(os.Stderr, "fak model: unknown subcommand %q\n", args[0])
		os.Exit(2)
	}
}

// cmdModelLoad downloads (or cache-hits) the file an hf:// URI names and prints
// its local path. HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) authorizes gated repos;
// FAK_MODELS_DIR overrides the cache root. Progress goes to stderr so the printed
// path on stdout stays scriptable: GGUF=$(fak model load hf://...).
func cmdModelLoad(args []string) {
	fs := flag.NewFlagSet("model load", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: fak model load <hf://owner/repo[@rev]/file>")
		fs.PrintDefaults()
	}
	_ = fs.Parse(args)
	if fs.NArg() != 1 {
		fs.Usage()
		os.Exit(2)
	}
	uri := fs.Arg(0)
	if !hfhub.IsURI(uri) {
		fmt.Fprintf(os.Stderr, "fak model load: %q is not an hf:// URI (want hf://owner/repo[@rev]/file)\n", uri)
		os.Exit(2)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	path, err := hfhub.FetchURI(ctx, uri, os.Stderr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak model load: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(path)
}
