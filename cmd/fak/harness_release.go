package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"

	"github.com/anthony-chaudhary/fak/internal/harnessartifact"
	"github.com/anthony-chaudhary/fak/internal/harnessrelease"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

func runHarnessRelease(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "model-update" {
		return runHarnessModelUpdate(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "model-cleanup" {
		return runHarnessModelCleanup(stdout, stderr, argv[1:])
	}
	if len(argv) == 0 || argv[0] != "witness" {
		fmt.Fprintln(stderr, "usage: fak harness release witness|model-update plan|model-cleanup preview|model-cleanup apply")
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

// The release CLI exposes the read-only update planner. Apply remains the
// generated harness's typed API operation because that long-lived owner holds
// the runtime mutation adapter and process handle; an ephemeral CLI must not
// invent a downloader, GPU admission, or background supervisor.
func runHarnessModelUpdate(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 || argv[0] != "plan" {
		fmt.Fprintln(stderr, "usage: fak harness release model-update plan --request FILE")
		return 2
	}
	fs := flag.NewFlagSet("harness release model-update plan", flag.ContinueOnError)
	fs.SetOutput(stderr)
	requestPath := fs.String("request", "", "JSON model update request")
	if err := fs.Parse(argv[1:]); err != nil {
		return 2
	}
	var request harnessartifact.ModelUpgradeRequest
	if err := readHarnessReleaseJSON(pathutil.ExpandTilde(*requestPath), &request); err != nil {
		fmt.Fprintf(stderr, "fak harness release model-update plan: %v\n", err)
		return 1
	}
	plan, err := harnessartifact.PrepareModelUpgrade(request)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness release model-update plan: %v\n", err)
		return 1
	}
	if err := json.NewEncoder(stdout).Encode(plan); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	return 0
}

func runHarnessModelCleanup(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 || (argv[0] != "preview" && argv[0] != "apply") {
		fmt.Fprintln(stderr, "usage: fak harness release model-cleanup preview --request FILE | apply --preview FILE")
		return 2
	}
	action := argv[0]
	fs := flag.NewFlagSet("harness release model-cleanup "+action, flag.ContinueOnError)
	fs.SetOutput(stderr)
	if action == "preview" {
		requestPath := fs.String("request", "", "JSON cleanup request")
		if err := fs.Parse(argv[1:]); err != nil {
			return 2
		}
		var request harnessartifact.ModelCleanupRequest
		if err := readHarnessReleaseJSON(pathutil.ExpandTilde(*requestPath), &request); err != nil {
			fmt.Fprintf(stderr, "fak harness release model-cleanup preview: %v\n", err)
			return 1
		}
		preview, err := harnessartifact.PreviewModelCleanup(request)
		if err != nil {
			fmt.Fprintf(stderr, "fak harness release model-cleanup preview: %v\n", err)
			return 1
		}
		if err := json.NewEncoder(stdout).Encode(preview); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		return 0
	}
	previewPath := fs.String("preview", "", "captured JSON cleanup preview")
	if err := fs.Parse(argv[1:]); err != nil {
		return 2
	}
	var preview harnessartifact.ModelCleanupPreview
	if err := readHarnessReleaseJSON(pathutil.ExpandTilde(*previewPath), &preview); err != nil {
		fmt.Fprintf(stderr, "fak harness release model-cleanup apply: %v\n", err)
		return 1
	}
	receipt, err := harnessartifact.ApplyModelCleanup(preview)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness release model-cleanup apply: %v\n", err)
		return 1
	}
	if err := json.NewEncoder(stdout).Encode(receipt); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	return 0
}

func readHarnessReleaseJSON(path string, destination any) error {
	if path == "" {
		return fmt.Errorf("JSON input path is required")
	}
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	decoder := json.NewDecoder(f)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return fmt.Errorf("trailing JSON in %s", path)
	}
	return nil
}
