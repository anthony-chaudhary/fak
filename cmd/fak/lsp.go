package main

// fak lsp — the in-process Language Server Protocol (LSP) server for Go and
// agent-written code, powered natively by internal/codelint.
// It speaks standard LSP framing over stdio, providing instant syntax validation,
// publishDiagnostics, and documentSymbol capabilities with zero external dependencies
// (no gopls or node installation required).

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/codelint"
)

func cmdLSP(argv []string) {
	os.Exit(runLSP(os.Stdin, os.Stdout, os.Stderr, argv))
}

func runLSP(stdin io.Reader, stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("lsp", flag.ContinueOnError)
	fs.SetOutput(stderr)
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	server := codelint.NewLSPServer(stdin, stdout, nil)
	if err := server.Run(context.Background()); err != nil && !errors.Is(err, io.EOF) {
		fmt.Fprintf(stderr, "fak lsp: %v\n", err)
		return 1
	}
	return 0
}
