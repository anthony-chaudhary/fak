package main

import (
	"errors"
	"flag"
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

// parseFlags parses argv into fs and reports success. A parse error (already
// rendered by the FlagSet's own output) means the caller should return 2. Use for
// the flag sets that do not special-case -h/--help.
func parseFlags(fs *flag.FlagSet, argv []string) bool {
	return fs.Parse(argv) == nil
}
