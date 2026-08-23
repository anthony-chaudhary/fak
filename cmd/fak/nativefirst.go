package main

import (
	"fmt"
	"io"
	"os"

	"bufio"
	"strings"
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
		raw := strings.TrimSpace(s.Text())
		text := strings.ToLower(raw)
		if !mentionsExternalLlama(text) || isWhitelistedReferenceUse(text) {
			continue
		}
		native := strings.Contains(text, "native") || strings.Contains(text, "performance") || strings.Contains(text, "qwen3.8") || strings.Contains(text, "qwen38")
		sub := strings.Contains(text, "default") || strings.Contains(text, "fallback") || strings.Contains(text, "fall back") || strings.Contains(text, "falls back") || strings.Contains(text, "auto") || strings.Contains(text, "delegate") || strings.Contains(text, "backend")
		if native && sub {
			out = append(out, nativeFirstFinding{line, raw, "llama.cpp may be selected only explicitly for benchmark, parity/reference diagnosis, interop/migration, or borrowing; native/performance paths must remain fak-native"})
		}
	}
	return out, s.Err()
}
func mentionsExternalLlama(s string) bool {
	return strings.Contains(s, "llama.cpp") || strings.Contains(s, "llama cpp") || strings.Contains(s, "llamacpp") || strings.Contains(s, "llama-server")
}
func isWhitelistedReferenceUse(s string) bool {
	for _, w := range []string{"benchmark", "comparison", "compare", "reference", "parity", "diagnos", "interop", "migration", "borrow", "study", "explicit"} {
		if strings.Contains(s, w) {
			return true
		}
	}
	return false
}
