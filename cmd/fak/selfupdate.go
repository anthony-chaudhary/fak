package main

import (
	"fmt"
	"os"

	selfupdatecmd "github.com/anthony-chaudhary/fak/internal/selfupdate/cmd"
)

// cmdSelfUpdate preserves the native fak command while sharing one implementation
// with the standalone fak-selfupdate bootstrap executable.
func cmdSelfUpdate(argv []string) { selfupdatecmd.Run(argv) }

// cmdSelf handles `fak self [subcommand]`.
func cmdSelf(argv []string) {
	if len(argv) > 0 && argv[0] == "update" {
		cmdSelfUpdate(argv[1:])
		return
	}
	if len(argv) > 0 && (argv[0] == "-h" || argv[0] == "--help" || argv[0] == "help") {
		fmt.Fprintln(os.Stdout, "usage: fak self update [flags]")
		return
	}
	fmt.Fprintln(os.Stderr, "usage: fak self update [flags]")
	os.Exit(2)
}

type selfUpdateReceipt = selfupdatecmd.Receipt
type selfUpdateReceiptTarget = selfupdatecmd.ReceiptTarget

const selfUpdateReceiptSchema = selfupdatecmd.ReceiptSchema

func repoRevOf(root, ref string) string { return selfupdatecmd.RepoRevOf(root, ref) }
