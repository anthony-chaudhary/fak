package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/anthony-chaudhary/fak/internal/codexmcphealth"
)

func cmdCodexMCPHealth(argv []string) {
	os.Exit(runCodexMCPHealth(os.Stdout, os.Stderr, argv))
}

func runCodexMCPHealth(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("codex-mcp-health", flag.ContinueOnError)
	fs.SetOutput(stderr)
	rootFlag := fs.String("root", "", "repo root holding VERSION and examples/ (default: current repo root)")
	policy := fs.String("policy", codexmcphealth.DefaultPolicy, "policy file for the stdio smoke")
	asJSON := fs.Bool("json", false, "emit the machine diagnostic")
	transportDead := fs.Bool("transport-dead", false, "assert the in-session Codex transport is dead")
	timeout := fs.Float64("timeout", 30.0, "smoke timeout in seconds")
	reap := fs.Bool("reap", false, "explicitly reap the remaining PID arguments")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	root := *rootFlag
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}

	if *reap {
		pids, err := parsePIDs(fs.Args())
		if err != nil {
			fmt.Fprintf(stderr, "fak codex-mcp-health: %v\n", err)
			return 2
		}
		results := codexmcphealth.ReapChildren(pids)
		if *asJSON {
			return encodeJSONOrFail(stdout, stderr, map[string]any{"reaped": results}, "fak codex-mcp-health")
		}
		allOK := true
		for _, r := range results {
			if !r.Reaped {
				allOK = false
			}
			status := "OK"
			if !r.Reaped {
				status = "FAIL"
			}
			fmt.Fprintf(stdout, "reap pid=%d: %s %s\n", r.PID, status, r.Detail)
		}
		if allOK {
			return 0
		}
		return 1
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak codex-mcp-health: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	diag := codexmcphealth.Diagnose(codexmcphealth.Options{
		Root:           root,
		Policy:         *policy,
		TransportAlive: !*transportDead,
		Timeout:        time.Duration(*timeout * float64(time.Second)),
	})
	if *asJSON {
		if err := writeIndentedJSON(stdout, diag); err != nil {
			fmt.Fprintf(stderr, "fak codex-mcp-health: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintln(stdout, codexmcphealth.RenderTable(diag))
	}
	if diag.State == codexmcphealth.ReconnectOK {
		return 0
	}
	return 1
}

func parsePIDs(args []string) ([]int, error) {
	if len(args) == 0 {
		return nil, fmt.Errorf("--reap requires at least one PID")
	}
	pids := make([]int, 0, len(args))
	for _, arg := range args {
		pid, err := strconv.Atoi(arg)
		if err != nil || pid <= 0 {
			return nil, fmt.Errorf("invalid PID %q", arg)
		}
		pids = append(pids, pid)
	}
	return pids, nil
}
