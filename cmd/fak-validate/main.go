// fak-validate exposes repository validation without linking the full fak CLI.
package main

import (
	"os"

	"github.com/anthony-chaudhary/fak/internal/validate"
)

func main() {
	os.Exit(validate.Run(os.Stdout, os.Stderr, os.Args[1:]))
}
