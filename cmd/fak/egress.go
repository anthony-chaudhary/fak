package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/egressfloor"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

// cmdEgress — `fak egress check`: the standalone, no-GPU, no-key witness for the
// network-egress floor that makes `fak guard` useful on a random cloud VM. It asks the
// SAME kernel floor a guarded session runs whether a destination would be refused, so
// you can prove — deterministically, in milliseconds — that a (possibly prompt-injected)
// agent on an ephemeral box cannot reach the cloud-instance metadata endpoint to steal
// the VM's IAM credentials. See examples/remote-vm-guard/.
func cmdEgress(argv []string) {
	if len(argv) == 0 {
		egressUsage(os.Stderr)
		os.Exit(2)
	}
	switch argv[0] {
	case "check":
		os.Exit(runEgressCheck(argv[1:], os.Stdout, os.Stderr))
	case "-h", "--help", "help":
		egressUsage(os.Stdout)
		os.Exit(0)
	default:
		fmt.Fprintf(os.Stderr, "fak egress: unknown subcommand %q\n", argv[0])
		egressUsage(os.Stderr)
		os.Exit(2)
	}
}

func egressUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: fak egress check [--tool NAME] (--url URL | --command CMD | --host HOST | --args JSON)")
	fmt.Fprintln(w, "  prove what the kernel egress floor does to a network destination — the floor")
	fmt.Fprintln(w, "  that lets `fak guard` block cloud-metadata credential theft on a random VM.")
	fmt.Fprintln(w, "  examples:")
	fmt.Fprintln(w, "    fak egress check --url http://169.254.169.254/latest/meta-data/   # -> DENY (EGRESS_BLOCK)")
	fmt.Fprintln(w, "    fak egress check --command 'curl http://metadata.google.internal/'  # -> DENY (EGRESS_BLOCK)")
	fmt.Fprintln(w, "    fak egress check --url https://api.anthropic.com/v1/messages       # -> ALLOW")
	fmt.Fprintln(w, "    fak egress check --host 169.254.169.254                            # -> BLOCK (host classifier)")
	fmt.Fprintln(w, "  exit: 0 allowed, 1 blocked, 2 usage error.")
}

// runEgressCheck resolves the destination from the flags, runs it through the real
// adjudicator floor (or, for --host, the pure classifier), prints the verdict, and
// returns the process exit code (0 allowed, 1 blocked, 2 usage).
func runEgressCheck(argv []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("egress check", flag.ContinueOnError)
	fs.SetOutput(stderr)
	tool := fs.String("tool", "", "tool name to synthesize the call as (default: WebFetch for --url, Bash for --command)")
	url := fs.String("url", "", "a destination URL to test as a WebFetch-style arg")
	command := fs.String("command", "", "a shell command line to scan for an embedded destination (curl/wget/Invoke-WebRequest)")
	host := fs.String("host", "", "classify a bare host or IP directly via the pure egressfloor classifier (no tool call)")
	argsJSON := fs.String("args", "", "raw tool args as JSON (use with --tool)")
	asJSON := fs.Bool("json", false, "emit the result as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	// --host: the pure leaf classifier, no policy, no tool call — the deterministic
	// "is THIS address a blocked egress destination" oracle.
	if h := strings.TrimSpace(*host); h != "" {
		blocked, label := egressfloor.ClassifyHost(h)
		if *asJSON {
			emitEgressJSON(stdout, map[string]any{"host": h, "blocked": blocked, "class": label})
		} else if blocked {
			fmt.Fprintf(stdout, "BLOCK  host=%s  class=%s\n", h, label)
		} else {
			fmt.Fprintf(stdout, "ALLOW  host=%s  (not a cloud-metadata / link-local destination)\n", h)
		}
		if blocked {
			return 1
		}
		return 0
	}

	// Otherwise synthesize a tool call and run it through the SAME kernel floor `fak
	// guard` enforces, so the witness reflects the wired behavior, not just the leaf.
	toolName, args, err := egressSynthesize(*tool, *url, *command, *argsJSON)
	if err != nil {
		fmt.Fprintf(stderr, "fak egress check: %v\n", err)
		egressUsage(stderr)
		return 2
	}

	res := abi.ActiveResolver()
	ref, perr := res.Put(ctx(), []byte(args))
	if perr != nil {
		fmt.Fprintf(stderr, "fak egress check: %v\n", perr)
		return 2
	}
	tc := &abi.ToolCall{Tool: toolName, Args: ref}
	v := kernel.Fold(ctx(), abi.AdjudicatorsFor(tc), tc)

	blocked := v.Kind == abi.VerdictDeny && v.Reason == egressfloor.ReasonEgressBlock
	if *asJSON {
		emitEgressJSON(stdout, map[string]any{
			"tool":    toolName,
			"verdict": verdictName(v.Kind),
			"reason":  abi.ReasonName(v.Reason),
			"blocked": blocked,
		})
	} else {
		fmt.Fprintf(stdout, "verdict=%s reason=%s tool=%s\n", verdictName(v.Kind), abi.ReasonName(v.Reason), toolName)
		if blocked {
			fmt.Fprintln(stdout, "  -> the egress floor REFUSED this destination (cloud-metadata / link-local SSRF).")
			fmt.Fprintln(stdout, "     on a VM this is the credential-theft path a prompt-injected agent would take.")
		}
	}
	if blocked {
		return 1
	}
	return 0
}

// egressSynthesize turns the destination flags into a (tool, argsJSON) pair. --url and
// --command pick a sensible default tool (WebFetch / Bash) and wrap the value in the
// conventional arg key; --args is the escape hatch for an arbitrary tool shape (needs
// --tool). Exactly one destination source must be given.
func egressSynthesize(tool, url, command, argsJSON string) (string, string, error) {
	url, command, argsJSON = strings.TrimSpace(url), strings.TrimSpace(command), strings.TrimSpace(argsJSON)
	n := 0
	for _, s := range []string{url, command, argsJSON} {
		if s != "" {
			n++
		}
	}
	if n == 0 {
		return "", "", fmt.Errorf("give one of --url, --command, --host, or --args")
	}
	if n > 1 {
		return "", "", fmt.Errorf("--url, --command, and --args are mutually exclusive — pass one")
	}
	switch {
	case url != "":
		t := tool
		if t == "" {
			t = "WebFetch"
		}
		return t, string(mustJSON(map[string]any{"url": url})), nil
	case command != "":
		t := tool
		if t == "" {
			t = "Bash"
		}
		return t, string(mustJSON(map[string]any{"command": command})), nil
	default: // argsJSON
		if tool == "" {
			return "", "", fmt.Errorf("--args requires --tool NAME")
		}
		if !json.Valid([]byte(argsJSON)) {
			return "", "", fmt.Errorf("--args is not valid JSON")
		}
		return tool, argsJSON, nil
	}
}

func emitEgressJSON(w io.Writer, m map[string]any) {
	b, _ := json.MarshalIndent(m, "", "  ")
	fmt.Fprintln(w, string(b))
}
