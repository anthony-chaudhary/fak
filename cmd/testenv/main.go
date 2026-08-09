// Command testenv runs a command after removing credential-shaped environment
// variables according to fak's repository-wide envconfiglint registry.
package main

import (
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/testenv"
)

func main() {
	args := os.Args[1:]
	if len(args) > 0 && args[0] == "--" {
		args = args[1:]
	}
	if err := testenv.Run(args); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
