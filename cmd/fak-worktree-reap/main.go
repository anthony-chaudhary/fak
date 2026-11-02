// Command fak-worktree-reap exposes the shared native single-worker retirement
// route without linking the inference runtime. Bulk GC remains in fak.
package main

import (
	"github.com/anthony-chaudhary/fak/internal/worktreereapcli"
	"os"
)

func main() { os.Exit(worktreereapcli.Run(os.Stdout, os.Stderr, os.Args[1:])) }
