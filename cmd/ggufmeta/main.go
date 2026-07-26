// Command ggufmeta prints the metadata header of a GGUF checkpoint.
//
// It is the CLI surface for internal/ggufload's full-metadata export (#292),
// which until now had no command in front of it: -json emits the lossless
// snapshot (every key/value with arrays kept in full, every tensor directory
// entry, header scalars, keys sorted so two exports of one checkpoint are
// byte-identical), suitable for diffing two checkpoints or feeding a tool.
//
// The default output is the human path: sorted `key=value` lines, optionally
// narrowed with -grep to the handful of keys you actually came for
// (`ggufmeta -grep tokenizer,chat_template model.gguf`).
//
// -json and -grep are refused together on purpose. A filtered export is not a
// lossless export, and the whole value of the -json snapshot is that a reader
// can trust it is complete; emitting a partial one under the same flag would
// make that trust wrong in a way nothing downstream could detect.
//
// Usage:
//
//	ggufmeta [-json] [-grep SUBSTR[,SUBSTR...]] <model.gguf>
package main

import (
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "ggufmeta:", err)
		os.Exit(1)
	}
}

func run(argv []string, out *os.File) error {
	fs := flag.NewFlagSet("ggufmeta", flag.ContinueOnError)
	asJSON := fs.Bool("json", false, "emit the full lossless metadata export as JSON")
	grep := fs.String("grep", "", "comma-separated substrings; print only metadata keys containing one of them")
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "usage: ggufmeta [-json] [-grep SUBSTR[,SUBSTR...]] <model.gguf>")
		fs.PrintDefaults()
	}
	if err := fs.Parse(argv); err != nil {
		return err
	}

	// An arg guard, not an index panic: the previous scratch version of this
	// command died with "index out of range [1]" when run with no argument.
	if fs.NArg() != 1 {
		fs.Usage()
		os.Exit(2)
	}
	if *asJSON && *grep != "" {
		return fmt.Errorf("-json and -grep are mutually exclusive: a filtered export is not the lossless export -json promises")
	}

	f, err := ggufload.Open(fs.Arg(0))
	if err != nil {
		return err
	}

	if *asJSON {
		b, err := f.MetadataJSON()
		if err != nil {
			return err
		}
		_, err = out.Write(append(b, '\n'))
		return err
	}

	for _, k := range selectKeys(f.Metadata, splitFilters(*grep)) {
		if _, err := fmt.Fprintf(out, "%s=%v\n", k, f.Metadata[k].Value); err != nil {
			return err
		}
	}
	return nil
}

// splitFilters turns the -grep value into the substring list. An empty or
// whitespace-only flag yields no filters, which selectKeys reads as "everything"
// — distinct from a filter that happens to match nothing.
func splitFilters(s string) []string {
	out := make([]string, 0, 4)
	for _, part := range strings.Split(s, ",") {
		if p := strings.TrimSpace(part); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// selectKeys returns the metadata keys to print, sorted for a stable diff. With
// no filters every key is selected; otherwise a key is selected when it contains
// any filter substring. Split out as a pure function so the selection rule is
// testable without a GGUF fixture on disk.
func selectKeys(meta map[string]ggufload.Value, filters []string) []string {
	keys := make([]string, 0, len(meta))
	for k := range meta {
		if len(filters) == 0 || matchesAny(k, filters) {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	return keys
}

func matchesAny(k string, filters []string) bool {
	for _, f := range filters {
		if strings.Contains(k, f) {
			return true
		}
	}
	return false
}
