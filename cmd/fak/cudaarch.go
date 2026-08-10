package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/cudaarch"
)

func runCUDAArch(out, errOut io.Writer, args []string) int {
	fs := flag.NewFlagSet("cuda-arch-matrix", flag.ContinueOnError)
	fs.SetOutput(errOut)
	root := fs.String("root", ".", "repository root")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(errOut, "usage: fak cuda-arch-matrix [--root PATH]")
		return 2
	}
	errors, err := cudaarch.Validate(*root)
	if err != nil {
		fmt.Fprintf(errOut, "cuda-arch-matrix: %v\n", err)
		return 1
	}
	if len(errors) > 0 {
		fmt.Fprintln(out, "cuda-arch-matrix: FAIL")
		for _, validationError := range errors {
			fmt.Fprintf(out, "  - %s\n", validationError)
		}
		return 1
	}
	fmt.Fprintln(out, "cuda-arch-matrix: OK (declared SASS set + compute_120 PTX floor; Linux/Windows/Docker/docs agree)")
	return 0
}

func cmdCUDAArch(args []string) { os.Exit(runCUDAArch(os.Stdout, os.Stderr, args)) }
