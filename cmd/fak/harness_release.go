package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"runtime"

	"github.com/anthony-chaudhary/fak/internal/harnessrelease"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

func runHarnessRelease(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 || argv[0] != "witness" {
		fmt.Fprintln(stderr, "usage: fak harness release witness --archive A --checksum S --dir D --receipt R --rollback-command CMD")
		return 2
	}
	fs := flag.NewFlagSet("harness release witness", flag.ContinueOnError)
	fs.SetOutput(stderr)
	archive := fs.String("archive", "", "downloaded release archive")
	checksum := fs.String("checksum", "", "downloaded SHA-256 sidecar")
	target := fs.String("target", runtime.GOOS+"_"+runtime.GOARCH, "release target OS_arch")
	dir := fs.String("dir", "", "new external product directory")
	module := fs.String("module", "example.test/released-harness", "generated product module")
	receipt := fs.String("receipt", "", "machine-readable receipt output")
	rollback := fs.String("rollback-command", "", "exact operator rollback command")
	if err := fs.Parse(argv[1:]); err != nil {
		return 2
	}
	r, err := harnessrelease.Run(context.Background(), harnessrelease.Options{
		Archive: pathutil.ExpandTilde(*archive), Checksum: pathutil.ExpandTilde(*checksum), Target: *target,
		ProductDir: pathutil.ExpandTilde(*dir), Module: *module, Receipt: pathutil.ExpandTilde(*receipt), RollbackCommand: *rollback,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak harness release witness: %v\n", err)
		return 1
	}
	if err := json.NewEncoder(stdout).Encode(r); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	return 0
}
