package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
)

// parseFlagsOrHelp parses argv into fs with the standard cmd/fak exit convention:
// an explicit -h/--help is NOT a usage error (rc 0), any other parse error is (rc 2).
// On success it returns (0, true); callers do `if rc, ok := parseFlagsOrHelp(fs, argv); !ok { return rc }`.
func parseFlagsOrHelp(fs *flag.FlagSet, argv []string) (int, bool) {
	if err := fs.Parse(argv); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0, false
		}
		return 2, false
	}
	return 0, true
}

// parseFlagsRejectArgs parses argv into fs with the standard cmd/fak exit convention and
// ALSO rejects a trailing positional, returning (code, done): done=true means the caller
// should return code (a -h/--help request -> 0, a parse error or stray positional -> 2).
// Callers do `if code, done := parseFlagsRejectArgs(fs, argv, stderr); done { return code }`.
func parseFlagsRejectArgs(fs *flag.FlagSet, argv []string, stderr io.Writer) (int, bool) {
	if err := fs.Parse(argv); err != nil {
		if err == flag.ErrHelp {
			return 0, true
		}
		return 2, true
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "%s: unexpected argument %q\n", fs.Name(), fs.Arg(0))
		return 2, true
	}
	return 0, false
}

// parseFlags parses argv into fs and reports success. A parse error (already
// rendered by the FlagSet's own output) means the caller should return 2. Use for
// the flag sets that do not special-case -h/--help.
func parseFlags(fs *flag.FlagSet, argv []string) bool {
	return fs.Parse(argv) == nil
}
