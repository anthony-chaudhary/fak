// Command fak-leaseref exposes the isolated session descriptor operation without
// linking the inference runtime. Other lease operations remain in fak leaseref.
package main

import (
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/leaserefcli"
)

func main() { os.Exit(run(os.Stdout, os.Stderr, os.Args[1:])) }

const usage = "usage: fak-leaseref session-publish [flags]\nLocal session descriptor only; use fak leaseref for other lease operations."

func run(stdout, stderr io.Writer, args []string) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, usage)
		return 2
	}
	switch args[0] {
	case "session-publish":
		return leaserefcli.RunSessionPublish(stdout, stderr, args[1:])
	case "help", "-h", "--help":
		fmt.Fprintln(stdout, usage)
		return 0
	default:
		fmt.Fprintf(stderr, "fak-leaseref: unsupported command %q\n%s\n", args[0], usage)
		return 2
	}
}
