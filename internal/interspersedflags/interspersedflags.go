// Package interspersedflags parses Go flags around positional arguments.
package interspersedflags

import "flag"

// Parse repeatedly parses flags and collects one positional at each boundary.
func Parse(fs *flag.FlagSet, argv []string) ([]string, error) {
	var positional []string
	for rest := argv; ; {
		if err := fs.Parse(rest); err != nil {
			return nil, err
		}
		rest = fs.Args()
		if len(rest) == 0 {
			return positional, nil
		}
		positional = append(positional, rest[0])
		rest = rest[1:]
	}
}
