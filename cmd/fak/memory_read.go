package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/memoryread"
)

func cmdMemoryRead(argv []string) { os.Exit(runMemoryRead(os.Stdout, os.Stderr, argv)) }

func runMemoryRead(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("memory-read", flag.ContinueOnError)
	fs.SetOutput(stderr)
	store := fs.String("store", "", "memory store dir (default: .claude/memory)")
	indexOnly := fs.Bool("index-only", false, "emit MEMORY.md only, skip per-fact bodies")
	maxBytes := fs.Int("max-bytes", 60000, "cap total per-fact body bytes emitted")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	dir := *store
	if dir == "" {
		root := resolveRoot("")
		if root == "" {
			root = "."
		}
		dir = memoryread.DefaultStore(root)
	} else if abs, err := filepath.Abs(dir); err == nil {
		dir = abs
	}
	_, _ = fmt.Fprint(stdout, memoryread.RenderDigest(dir, *indexOnly, *maxBytes))
	return 0
}
