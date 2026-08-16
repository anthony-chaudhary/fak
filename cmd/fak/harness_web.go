package main

import (
	"context"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/harnessweb"
)

func runHarnessWeb(stdout, stderr io.Writer, argv []string) int {
	return runHarnessWebWithCancel(context.Background(), stdout, stderr, argv)
}

func runHarnessWebWithCancel(ctx context.Context, stdout, stderr io.Writer, argv []string) int {
	return harnessweb.Run(ctx, stdout, stderr, argv)
}

func harnessWebUsage(stderr io.Writer) {
	fmt.Fprintln(stderr, "usage: fak harness web [--addr 127.0.0.1:8787] [--state FILE] [--fak-url URL] [--workspace DIR] [--selfcheck]")
}
