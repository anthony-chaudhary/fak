package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/anthony-chaudhary/fak/internal/allinone"
	"github.com/anthony-chaudhary/fak/internal/appversion"
)

func printUpHelp(w io.Writer) {
	fmt.Fprintln(w, "Usage: fak up [flags]")
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "All-in-one one-touch bootstrap orchestrator unifying lock verification,")
	fmt.Fprintln(w, "MCP broker, memory, and gateway.")
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "Flags:")
	fmt.Fprintln(w, "  --lock <path>               path to harness product lock JSON (v2)")
	fmt.Fprintln(w, "  --bundle <path>             path to .fakpack bundle")
	fmt.Fprintln(w, "  --bundle-verify-key <key>   optional public key / signature verification key for bundle")
	fmt.Fprintln(w, "  --addr <addr>               address to bind HTTP server (default: 127.0.0.1:4000)")
	fmt.Fprintln(w, "  --policy <path>             path to security policy file")
	fmt.Fprintln(w, "  --engine <engine>           model engine ID (default: mock)")
	fmt.Fprintln(w, "  --mock                      enable mock engine and test components")
	fmt.Fprintln(w, "  --dry-run                   validate topology and print execution plan without running")
	fmt.Fprintln(w, "  --help, -h                  show this help message")
}

// cmdUp is the product entry point for the unified deployable runtime.
// When --lock or --bundle is supplied, it boots the all-in-one orchestrator.
// If neither is provided, it delegates directly to serve.
func cmdUp(argv []string) {
	for _, arg := range argv {
		if arg == "--help" || arg == "-h" || arg == "help" {
			printUpHelp(os.Stdout)
			return
		}
	}

	hasLockOrBundle := false
	for _, arg := range argv {
		if arg == "--lock" || strings.HasPrefix(arg, "--lock=") ||
			arg == "-lock" || strings.HasPrefix(arg, "-lock=") ||
			arg == "--bundle" || strings.HasPrefix(arg, "--bundle=") ||
			arg == "-bundle" || strings.HasPrefix(arg, "-bundle=") {
			hasLockOrBundle = true
			break
		}
	}

	if !hasLockOrBundle {
		cmdServe(argv)
		return
	}

	fs := flag.NewFlagSet("up", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	lockPath := fs.String("lock", "", "path to harness product lock JSON (v2)")
	bundlePath := fs.String("bundle", "", "path to .fakpack bundle")
	bundleVerifyKey := fs.String("bundle-verify-key", "", "optional verification key for bundle")
	addr := fs.String("addr", "127.0.0.1:4000", "address to bind HTTP server")
	policyPath := fs.String("policy", "", "path to security policy file")
	engineID := fs.String("engine", "mock", "model engine ID")
	dryRun := fs.Bool("dry-run", false, "validate topology and print execution plan without running")
	mock := fs.Bool("mock", false, "enable mock engine and test components")

	if err := fs.Parse(argv); err != nil {
		os.Exit(2)
	}

	cfg := allinone.Config{
		LockPath:        *lockPath,
		BundlePath:      *bundlePath,
		BundleVerifyKey: *bundleVerifyKey,
		Addr:            *addr,
		PolicyPath:      *policyPath,
		Engine:          *engineID,
		DryRun:          *dryRun,
		Mock:            *mock,
	}

	sup, err := allinone.NewSupervisor(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak up: %v\n", err)
		os.Exit(1)
	}

	if cfg.DryRun {
		plan, err := sup.DryRunTopology()
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak up dry-run failed: %v\n", err)
			os.Exit(1)
		}
		raw, err := json.MarshalIndent(plan, "", "  ")
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak up json marshal: %v\n", err)
			os.Exit(1)
		}
		fmt.Println(string(raw))
		return
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := sup.Start(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "fak up start failed: %v\n", err)
		os.Exit(1)
	}
	ver := appversion.Current()
	if id := guardShortBuildID(); id != "" {
		ver += " (" + id + ")"
	}
	fmt.Printf("fak up %s running on %s\n", ver, sup.Addr())

	<-ctx.Done()
	shutdownTimeout, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = sup.Shutdown(shutdownTimeout)
}
