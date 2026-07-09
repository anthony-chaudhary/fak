package main

import (
	"os"

	"github.com/anthony-chaudhary/fak/internal/refactorverify"
)

// cmdRefactorVerify is the thin shell over internal/refactorverify — the read-only
// proof that a god-split / code-motion refactor dropped no top-level declaration (Go
// port of the retired tools/refactor_verify.py). Exit 0 clean, 1 on a dropped decl
// (or a relocation under --expect-motion), 2 if --ref is not a commit.
func cmdRefactorVerify(argv []string) {
	os.Exit(refactorverify.Run(os.Stdout, os.Stderr, argv))
}
