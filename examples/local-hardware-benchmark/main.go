package main

import (
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/localbench"
)

func main() {
	if err := localbench.RunCLI(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
