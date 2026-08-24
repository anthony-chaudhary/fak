package main

import (
	"bufio"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/nativefirst"
)

func cmdNativeFirstLint(args []string) int {
	if len(args) > 1 {
		fmt.Fprintln(os.Stderr, "usage: fak native-first-lint [FILE|-]")
		return 2
	}
	name := "-"
	if len(args) == 1 {
		name = args[0]
	}
	var r io.Reader = os.Stdin
	var f *os.File
	var err error
	if name != "-" {
		f, err = os.Open(name)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak native-first-lint: %v\n", err)
			return 2
		}
		defer f.Close()
		r = f
	}
	findings, err := scanNativeFirst(r)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak native-first-lint: %v\n", err)
		return 2
	}
	for _, finding := range findings {
		fmt.Printf("%s:%d: NATIVE_FIRST: %s\n  phrase: %s\n", name, finding.line, finding.reason, finding.phrase)
	}
	if len(findings) > 0 {
		return 1
	}
	fmt.Printf("native-first: OK (%s)\n", name)
	return 0
}

type nativeFirstFinding struct {
	line           int
	phrase, reason string
}

func scanNativeFirst(r io.Reader) ([]nativeFirstFinding, error) {
	var out []nativeFirstFinding
	s := bufio.NewScanner(r)
	for line := 1; s.Scan(); line++ {
		finding := nativefirst.ScanLine(s.Text())
		if finding != nil {
			out = append(out, nativeFirstFinding{line, finding.Phrase, finding.Reason})
		}
	}
	return out, s.Err()
}
