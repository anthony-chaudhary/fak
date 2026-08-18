package main

import (
	"context"
	"io"

	"github.com/anthony-chaudhary/fak/internal/harnessweb"
)

func runHarnessWeb(stdout, stderr io.Writer, argv []string) int {
	return runHarnessWebWithCancel(context.Background(), stdout, stderr, argv)
}

func runHarnessWebWithCancel(ctx context.Context, stdout, stderr io.Writer, argv []string) int {
	return harnessweb.Run(ctx, stdout, stderr, argv)
}
