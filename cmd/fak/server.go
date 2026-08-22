package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/serverlifecycle"
)

const serverUsage = `usage: fak server <init|up|status|down> [flags]

Own one loopback local-process inference server instance.

  fak server init --dir DIR --name NAME --model MODEL.gguf --sha256 HEX --executable /path/to/llama-server --json
  fak server up --dir DIR --json
  fak server status --dir DIR --json
  fak server down --dir DIR --json`

func init() {
	if len(os.Args) > 1 && os.Args[1] == "server" {
		os.Exit(runServer(os.Stdout, os.Stderr, os.Args[2:]))
	}
}

func runServer(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, serverUsage)
		return 2
	}
	var result serverlifecycle.Result
	var err error
	switch argv[0] {
	case "init":
		result, err = runServerInit(argv[1:])
	case "up":
		result, err = runServerUp(argv[1:])
	case "status":
		result, err = runServerStatus(argv[1:])
	case "down":
		result, err = runServerDown(argv[1:])
	default:
		fmt.Fprintf(stderr, "fak server: unknown command %q\n%s\n", argv[0], serverUsage)
		return 2
	}
	if encodeErr := json.NewEncoder(stdout).Encode(result); encodeErr != nil {
		fmt.Fprintf(stderr, "fak server: encode result: %v\n", encodeErr)
		return 1
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak server %s: %v\n", argv[0], err)
		return 1
	}
	return 0
}

func runServerInit(argv []string) (serverlifecycle.Result, error) {
	fs := flag.NewFlagSet("fak server init", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	dir := fs.String("dir", "", "server instance directory")
	name := fs.String("name", "", "stable server name")
	model := fs.String("model", "", "local GGUF artifact")
	sha256 := fs.String("sha256", "", "required artifact SHA-256")
	executable := fs.String("executable", "", "llama-server executable")
	alias := fs.String("model-alias", "", "served model alias (default server name)")
	port := fs.Uint("port", 0, "loopback port (0 selects a free port)")
	windowTokens := fs.Int("ctx-size", 4096, "adapter context window")
	threads := fs.Int("threads", 1, "adapter CPU threads")
	gpuLayers := fs.Int("gpu-layers", 0, "adapter GPU layers")
	versionConstraint := fs.String("version-constraint", "installed", "authored adapter version constraint")
	protocolRevision := fs.String("protocol-revision", "2026-01", "OpenAI HTTP protocol revision")
	_ = fs.Bool("json", true, "emit JSON (always enabled)")
	if err := fs.Parse(argv); err != nil {
		return serverlifecycle.Result{}, err
	}
	*dir = pathutil.ExpandTilde(*dir)
	if fs.NArg() != 0 || *port > 65535 {
		return serverlifecycle.Result{}, fmt.Errorf("unexpected arguments or invalid port")
	}
	return serverlifecycle.Init(context.Background(), serverlifecycle.InitOptions{
		InstanceDirectory: *dir,
		ServerName:        *name,
		ModelPath:         *model,
		ArtifactSHA256:    *sha256,
		AdapterExecutable: *executable,
		ModelAlias:        *alias,
		Port:              uint16(*port),
		TokenWindow:       *windowTokens,
		Threads:           *threads,
		GPULayers:         *gpuLayers,
		VersionConstraint: *versionConstraint,
		ProtocolRevision:  *protocolRevision,
	})
}

func runServerUp(argv []string) (serverlifecycle.Result, error) {
	fs := flag.NewFlagSet("fak server up", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	dir := fs.String("dir", "", "server instance directory")
	readiness := fs.Duration("readiness-timeout", 5*time.Minute, "bounded readiness deadline")
	stop := fs.Duration("stop-timeout", 10*time.Second, "failed-launch cleanup deadline")
	probe := fs.Duration("probe-interval", 100*time.Millisecond, "readiness retry interval")
	_ = fs.Bool("json", true, "emit JSON (always enabled)")
	if err := fs.Parse(argv); err != nil {
		return serverlifecycle.Result{}, err
	}
	*dir = pathutil.ExpandTilde(*dir)
	if fs.NArg() != 0 {
		return serverlifecycle.Result{}, fmt.Errorf("unexpected arguments: %v", fs.Args())
	}
	return serverlifecycle.Up(context.Background(), *dir, serverlifecycle.Options{ReadinessTimeout: *readiness, StopTimeout: *stop, ProbeInterval: *probe})
}

func runServerStatus(argv []string) (serverlifecycle.Result, error) {
	fs := flag.NewFlagSet("fak server status", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	dir := fs.String("dir", "", "server instance directory")
	probeTimeout := fs.Duration("probe-timeout", 2*time.Second, "bounded status probe timeout")
	_ = fs.Bool("json", true, "emit JSON (always enabled)")
	if err := fs.Parse(argv); err != nil {
		return serverlifecycle.Result{}, err
	}
	*dir = pathutil.ExpandTilde(*dir)
	if fs.NArg() != 0 {
		return serverlifecycle.Result{}, fmt.Errorf("unexpected arguments: %v", fs.Args())
	}
	return serverlifecycle.Status(context.Background(), *dir, serverlifecycle.Options{ReadinessTimeout: *probeTimeout})
}

func runServerDown(argv []string) (serverlifecycle.Result, error) {
	fs := flag.NewFlagSet("fak server down", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	dir := fs.String("dir", "", "server instance directory")
	stop := fs.Duration("stop-timeout", 10*time.Second, "bounded teardown deadline")
	_ = fs.Bool("json", true, "emit JSON (always enabled)")
	if err := fs.Parse(argv); err != nil {
		return serverlifecycle.Result{}, err
	}
	*dir = pathutil.ExpandTilde(*dir)
	if fs.NArg() != 0 {
		return serverlifecycle.Result{}, fmt.Errorf("unexpected arguments: %v", fs.Args())
	}
	return serverlifecycle.Down(context.Background(), *dir, serverlifecycle.Options{StopTimeout: *stop})
}
