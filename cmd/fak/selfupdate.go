package main

import selfupdatecmd "github.com/anthony-chaudhary/fak/internal/selfupdate/cmd"

// cmdSelfUpdate preserves the native fak command while sharing one implementation
// with the standalone fak-selfupdate bootstrap executable.
func cmdSelfUpdate(argv []string) { selfupdatecmd.Run(argv) }

type selfUpdateReceipt = selfupdatecmd.Receipt
type selfUpdateReceiptTarget = selfupdatecmd.ReceiptTarget

const selfUpdateReceiptSchema = selfupdatecmd.ReceiptSchema

func repoRevOf(root, ref string) string { return selfupdatecmd.RepoRevOf(root, ref) }
