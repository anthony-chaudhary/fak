package ifc

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func localPathCall(tool, jsonArgs string) *abi.ToolCall {
	return &abi.ToolCall{
		Tool: tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(jsonArgs)},
	}
}

// A bare filename and a bare hostname are the same token shape — `README.md`
// parses as a host in the `.md` TLD exactly like `example.com` does — so the
// bare-destination scan used to classify every dotted filename under a path
// argument as EGRESS. On a tainted session the sink gate then hard-refused the
// most routine work there is: reading, searching, and editing a file by name.
// Witnessed live in .dispatch-runs/guard-audit as 15 TRUST_VIOLATION denies on
// Grep/Glob/Read ("EGRESS sink fed tainted data", args path=README.md /
// path=dos.toml / path=llms.txt) over 2026-07-24..26. A local file tool has no
// outbound channel, so the refusal prevented nothing fatal.
func TestLocalPathArgIsNotAnEgressDestination(t *testing.T) {
	for _, c := range []struct {
		name string
		tool string
		args string
	}{
		{"grep by bare filename", "Grep", `{"pattern":"func","path":"README.md"}`},
		{"grep a toml", "Grep", `{"pattern":"x","path":"dos.toml"}`},
		{"read a bare filename", "Read", `{"file_path":"llms.txt"}`},
		{"edit a source file", "Edit", `{"file_path":"policy.go","old_string":"a","new_string":"b"}`},
		{"write a source file", "Write", `{"file_path":"main.go","content":"x"}`},
		{"glob a bare filename", "Glob", `{"pattern":"ifc.go"}`},
		{"notebook path", "NotebookEdit", `{"notebook_path":"run.ipynb"}`},
		{"cwd", "some_tool", `{"cwd":"build.out"}`},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := Classify(context.Background(), localPathCall(c.tool, c.args), Policy{}); got != SinkNone {
				t.Fatalf("Classify(%s %s) = %v, want SinkNone — a filesystem path is not a network destination", c.tool, c.args, got)
			}
		})
	}
}

// The skip is per-KEY and narrow: everything the bare scan and the declared
// destination family caught before still classifies as EGRESS. Each case here is
// a way an exfil could try to ride the path-key exemption.
func TestLocalPathSkipKeepsEgressDetection(t *testing.T) {
	for _, c := range []struct {
		name string
		tool string
		args string
	}{
		// A declared destination key stays fail-closed whatever it holds.
		{"declared dest key", "some_tool", `{"dest":"attacker.example.com"}`},
		{"declared url key", "some_tool", `{"url":"https://attacker.example.com/steal"}`},
		// The unlisted-key evasion the bare scan exists to close.
		{"unlisted key", "some_tool", `{"server":"attacker.example.com"}`},
		// A path key does not launder a SIBLING arg out of the bare scan.
		{"path key beside a bare host", "Grep", `{"path":"README.md","note":"attacker.example.com"}`},
		// Name-based egress is a separate rung the skip never reaches.
		{"egress by name", "send_email", `{"path":"notes.md"}`},
		{"exec by name", "run_shell", `{"path":"notes.md"}`},
		// The SafeSink spoof stays closed: a destination beats the exemption.
		{"safesink + url", "transfer_to_human_agents", `{"url":"https://attacker.example.com"}`},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := Classify(context.Background(), localPathCall(c.tool, c.args), Policy{}); got == SinkNone {
				t.Fatalf("Classify(%s %s) = SinkNone, want a gated sink class", c.tool, c.args)
			}
		})
	}
}

// isLocalPathKey never shadows the declared destination family: if a key ever
// appears in both lists, egress wins.
func TestLocalPathKeyNeverShadowsEgressKey(t *testing.T) {
	for _, ek := range egressArgKeys {
		if isLocalPathKey(ek) {
			t.Fatalf("egress key %q is treated as a local path key", ek)
		}
	}
}
