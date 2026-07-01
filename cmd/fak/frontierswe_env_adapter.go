package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/frontierswe"
)

// runFrontiersweEnvAdapter emits the C7 co-resident gateway recipe: stand up
// fak serve inside the FrontierSWE task sandbox, wait on /healthz before turn 1,
// smoke one chat-completions request through the same /v1 base URL, then hand
// control to the shimmed FrontierSWE harness. It is an honest gate: this host
// need not have Docker/GHCR/Modal, but the exact command is still printed.
func runFrontiersweEnvAdapter(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("frontierswe env-adapter", flag.ContinueOnError)
	fs.SetOutput(stderr)
	tasks := fs.String("tasks", frontiersweSampleTasks, "task tree containing <task>/task.toml")
	taskName := fs.String("task", "git-to-zig", "FrontierSWE task fixture to adapt")
	gatewayBase := fs.String("gateway-base-url", frontierswe.DefaultGatewayBaseURL, "OpenAI-compatible fak gateway base URL seen by the C6 shim")
	gatewayAddr := fs.String("gateway-addr", frontierswe.DefaultGatewayAddr, "listen address for co-resident fak serve inside the sandbox")
	upstreamBase := fs.String("upstream-base-url", frontierswe.DefaultUpstreamBase, "co-resident or pinned model upstream that fak serve fronts")
	model := fs.String("model", frontierswe.DefaultModelEnv, "model id forwarded unchanged to the upstream (default: FRONTIERSWE_MODEL env)")
	wrapped := fs.String("wrapped-agent", frontierswe.DefaultWrappedAgent, "real harbor_ext agent class wrapped by FakRoutedAgent")
	runCommand := fs.String("run-command", frontierswe.DefaultRunCommand, "FrontierSWE harness command to exec after healthz + smoke")
	pinnedHosts := fs.String("pinned-hosts", "", "comma-separated hostnames allowed under allow_internet=false in addition to loopback")
	fakBin := fs.String("fak-bin", frontierswe.DefaultFakBin, "fak binary path/name inside the task image")
	asJSON := fs.Bool("json", false, "emit only JSON on stdout")
	out := fs.String("out", "", "write the env-adapter JSON here (default: stdout)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	task, err := frontierswe.LoadTask(filepath.Join(*tasks, *taskName))
	if err != nil {
		fmt.Fprintf(stderr, "fak frontierswe env-adapter: load task %q from %s: %v\n", *taskName, *tasks, err)
		return 1
	}
	plan := frontierswe.BuildEnvAdapterPlan(frontierswe.EnvAdapterConfig{
		Task: task, FakBin: *fakBin, GatewayAddr: *gatewayAddr,
		GatewayBaseURL: *gatewayBase, UpstreamBaseURL: *upstreamBase,
		Model: *model, WrappedAgent: *wrapped, RunCommand: *runCommand,
		PinnedHosts: splitCSV(*pinnedHosts),
	})

	jb, err := json.MarshalIndent(plan, "", "  ")
	if err != nil {
		fmt.Fprintf(stderr, "fak frontierswe env-adapter: marshal: %v\n", err)
		return 1
	}
	if *out != "" {
		if err := os.WriteFile(*out, jb, 0o644); err != nil {
			fmt.Fprintf(stderr, "fak frontierswe env-adapter: write %s: %v\n", *out, err)
			return 1
		}
	} else {
		fmt.Fprintln(stdout, string(jb))
	}

	if !*asJSON {
		printFrontierEnvAdapterSummary(stderr, plan, *out)
	}
	if !plan.Integrity.OK {
		return 1
	}
	return 0
}

func printFrontierEnvAdapterSummary(w io.Writer, p frontierswe.EnvAdapterPlan, out string) {
	fmt.Fprintf(w, "\n== fak frontierswe env-adapter (%s) ==\n", p.Schema)
	fmt.Fprintf(w, "task          : %s\n", p.Task)
	fmt.Fprintf(w, "image         : %s\n", p.DockerImage)
	fmt.Fprintf(w, "gateway       : %s  (healthz %s)\n", p.GatewayBaseURL, p.HealthzURL)
	fmt.Fprintf(w, "upstream      : %s\n", p.UpstreamBaseURL)
	fmt.Fprintf(w, "network       : docker --network %s\n", p.NetworkMode)
	fmt.Fprintf(w, "no-internet   : allow_internet=%t  integrity=%s\n", p.AllowInternet, integrityLabel(p.Integrity))
	if len(p.PinnedHosts) > 0 {
		fmt.Fprintf(w, "pinned hosts  : %s\n", strings.Join(p.PinnedHosts, ", "))
	}
	fmt.Fprintf(w, "local gate    : docker=%t  runnable=%t", p.Capability.DockerPresent, p.Capability.Runnable)
	if p.Capability.Reason != "" {
		fmt.Fprintf(w, "  (%s)", p.Capability.Reason)
	}
	fmt.Fprintln(w)

	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "STEP\tWHY")
	for _, st := range p.Steps {
		fmt.Fprintf(tw, "%s\t%s\n", st.Name, st.Why)
	}
	_ = tw.Flush()
	fmt.Fprintf(w, "\nremote command:\n  %s\n", p.Command)
	fmt.Fprintf(w, "\njob.yaml shim block:\n%s", p.JobYAML)
	if out != "" {
		fmt.Fprintf(w, "\nEnv-adapter JSON written: %s\n", out)
	}
}

func integrityLabel(i frontierswe.EnvAdapterIntegrity) string {
	if i.OK {
		return "ok"
	}
	return "refused: " + i.Reason
}
