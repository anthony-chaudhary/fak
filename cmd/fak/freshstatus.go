package main

import (
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/freshstatus"
)

func cmdFreshStatus(args []string) {
	os.Exit(runFreshStatus(os.Stdout, os.Stderr, args))
}

func runFreshStatus(stdout, stderr io.Writer, args []string) int {
	return freshstatus.Run(stdout, stderr, args)
}
