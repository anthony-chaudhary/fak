package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/memoryread"
	"github.com/anthony-chaudhary/fak/internal/memq"
)

//fak:ctxplan verb=memory-read enters="the on-disk markdown memory store (MEMORY.md + per-fact files, default .claude/memory or --store DIR)" pages="the notes digest it renders to stdout — MEMORY.md plus the per-fact bodies (capped at --max-bytes, or index-only with --index-only) — the orientation block an agent pastes into its window" warms="nothing — a read-only render of the note store; it warms no prompt cache or KV"
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
	_, _ = fmt.Fprint(stdout, memq.RenderNotesDigest(dir, *indexOnly, *maxBytes))
	return 0
}
