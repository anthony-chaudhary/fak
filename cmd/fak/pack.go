package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/fakpack"
)

func cmdPack(argv []string) {
	os.Exit(runPack(os.Stdout, os.Stderr, argv))
}

func runPack(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		packUsage(stderr)
		return 2
	}

	switch argv[0] {
	case "create":
		return runPackCreate(stdout, stderr, argv[1:])
	case "verify":
		return runPackVerify(stdout, stderr, argv[1:])
	case "inspect":
		return runPackInspect(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		packUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak pack: unknown subcommand %q\n", argv[0])
		packUsage(stderr)
		return 2
	}
}

func packUsage(w io.Writer) {
	fmt.Fprintln(w, "Usage: fak pack <subcommand> [flags]")
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "Subcommands:")
	fmt.Fprintln(w, "  create   Create an air-gapped OCI collection bundle (.fakpack)")
	fmt.Fprintln(w, "  verify   Verify bundle digests, completeness, and air-gap safety")
	fmt.Fprintln(w, "  inspect  Inspect manifest, layer descriptors, and metadata")
}

func runPackCreate(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("pack create", flag.ContinueOnError)
	fs.SetOutput(stderr)
	lockPath := fs.String("lock", "", "path to harness.lock.json (required)")
	assetsDir := fs.String("assets", "", "optional directory containing assets")
	binDir := fs.String("bin", "", "optional directory containing binaries")
	modelPath := fs.String("model", "", "optional model weights file")
	policyPath := fs.String("policy", "", "optional policy file")
	outPath := fs.String("out", "", "output bundle path (.fakpack) (required)")

	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *lockPath == "" || *outPath == "" {
		fmt.Fprintln(stderr, "usage: fak pack create --lock <file> [--assets <dir>] [--bin <dir>] [--model <file>] [--policy <file>] --out <bundle.fakpack>")
		return 2
	}

	opts := fakpack.CreateOptions{
		LockPath:   *lockPath,
		AssetsDir:  *assetsDir,
		BinDir:     *binDir,
		ModelPath:  *modelPath,
		PolicyPath: *policyPath,
		OutPath:    *outPath,
	}

	res, err := fakpack.Create(opts)
	if err != nil {
		fmt.Fprintf(stderr, "fak pack create: %v\n", err)
		return 1
	}

	fmt.Fprintf(stdout, "Created bundle: %s (lock: %s, layers: %d, size: %d bytes)\n", res.BundlePath, res.LockID, len(res.Layers), res.TotalSize)
	return 0
}

func runPackVerify(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("pack verify", flag.ContinueOnError)
	fs.SetOutput(stderr)
	bundlePath := fs.String("bundle", "", "path to .fakpack bundle (required)")
	lockPath := fs.String("lock", "", "optional expected harness.lock.json")
	asJSON := fs.Bool("json", false, "emit verification report as JSON")

	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *bundlePath == "" {
		fmt.Fprintln(stderr, "usage: fak pack verify --bundle <bundle.fakpack> [--lock <expected.lock.json>]")
		return 2
	}

	opts := fakpack.VerifyOptions{
		BundlePath:       *bundlePath,
		ExpectedLockPath: *lockPath,
	}

	res, err := fakpack.Verify(opts)
	if err != nil {
		fmt.Fprintf(stderr, "fak pack verify: %v\n", err)
		return 1
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(res)
		return 0
	}

	fmt.Fprintf(stdout, "Bundle verified: %s (lock: %s, layers: %d, air-gap: verified)\n", res.BundlePath, res.LockID, res.LayersVerified)
	return 0
}

func runPackInspect(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("pack inspect", flag.ContinueOnError)
	fs.SetOutput(stderr)
	bundlePath := fs.String("bundle", "", "path to .fakpack bundle (required)")
	asJSON := fs.Bool("json", false, "emit inspection report as JSON")

	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *bundlePath == "" {
		fmt.Fprintln(stderr, "usage: fak pack inspect --bundle <bundle.fakpack> [--json]")
		return 2
	}

	res, err := fakpack.Inspect(*bundlePath)
	if err != nil {
		fmt.Fprintf(stderr, "fak pack inspect: %v\n", err)
		return 1
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(res)
		return 0
	}

	fmt.Fprintf(stdout, "Bundle:     %s\n", res.BundlePath)
	fmt.Fprintf(stdout, "Lock ID:    %s\n", res.LockSummary.ID)
	fmt.Fprintf(stdout, "Schema:     %s\n", res.LockSummary.Schema)
	if len(res.Platforms) > 0 {
		fmt.Fprintf(stdout, "Platforms:  %v\n", res.Platforms)
	}
	fmt.Fprintf(stdout, "Layers:     %d\n", len(res.Layers))
	for i, l := range res.Layers {
		title := l.Annotations["org.opencontainers.image.title"]
		if title != "" {
			fmt.Fprintf(stdout, "  [%d] %s (%s, %d bytes)\n", i, l.MediaType, title, l.Size)
		} else {
			fmt.Fprintf(stdout, "  [%d] %s (%d bytes)\n", i, l.MediaType, l.Size)
		}
	}
	fmt.Fprintf(stdout, "Total Size: %d bytes\n", res.TotalSize)
	if res.CreatedTime != "" {
		fmt.Fprintf(stdout, "Created:    %s\n", res.CreatedTime)
	}
	return 0
}
