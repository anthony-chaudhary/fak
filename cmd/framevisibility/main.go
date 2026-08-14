package main

import (
	"os"

	"github.com/anthony-chaudhary/fak/internal/framevisibility"
)

func main() {
	os.Exit(framevisibility.Run(os.Stdout, os.Stderr, os.Args[1:]))
}
