package gateway

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/egressfloor"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

func sshExecProbeWorkspace(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, ".git")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("test requires a repository workspace")
		}
		dir = parent
	}
}

func sshExecProbeArgs(command, workdir string) string {
	shell := "/bin/sh"
	if runtime.GOOS == "windows" {
		shell = "C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
	}
	args := map[string]any{
		"cmd": command, "shell": shell, "login": false,
	}
	if workdir != "" {
		args["workdir"] = workdir
	}
	raw, _ := json.Marshal(args)
	return string(raw)
}

func sshExecProbeCommand() string {
	return sshReadProbeCommand()
}

// fak-test:runtime fast est=1s
func TestSSHExecProbeCodexAliasesAdmitCanonicalEnvelope(t *testing.T) {
	root := sshExecProbeWorkspace(t)
	srv, adj := newReloadLatencyServer(t)
	installProjectedLeaseAuthority(t, root, nil)
	if srv.workspaceAdmissionPermissive {
		t.Fatal("test requires strict workspace admission")
	}
	rt, err := policy.ParseRuntime(guardTraceFloorJSON)
	if err != nil {
		t.Fatal(err)
	}
	p := rt.Adjudicator
	p.Allow = nil
	p.AllowPrefix = nil
	p.Posture = adjudicator.PostureFailClosed
	p.EgressRestrict = true
	p.EgressAllowHosts = []string{"node.example"}
	adj.SetPolicy(p)
	for _, tool := range []string{"functions.exec_command", "exec_command"} {
		t.Run(tool, func(t *testing.T) {
			verdict, repaired, err := srv.adjudicate(context.Background(), tool, sshExecProbeArgs(sshExecProbeCommand(), root), true, "", "ssh-exec-alias-probe-"+tool)
			if err != nil || verdict.Kind != "ALLOW" || verdict.By != "monitor/cli-read-only" || repaired != "" {
				t.Fatalf("canonical Codex envelope verdict=%+v repaired=%q err=%v", verdict, repaired, err)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestSSHExecProbeCodexAliasesKeepRefusalBoundaries(t *testing.T) {
	root := sshExecProbeWorkspace(t)
	srv, adj := newReloadLatencyServer(t)
	installProjectedLeaseAuthority(t, root, nil)
	rt, err := policy.ParseRuntime(guardTraceFloorJSON)
	if err != nil {
		t.Fatal(err)
	}
	base := rt.Adjudicator
	base.Allow = nil
	base.AllowPrefix = nil
	base.Posture = adjudicator.PostureFailClosed
	outside := t.TempDir()
	for _, tool := range []string{"functions.exec_command", "exec_command"} {
		for _, tc := range []struct {
			name, command, workdir string
			hint                   bool
			customPolicy           bool
			wantReason             abi.ReasonCode
		}{
			{"missing workdir", sshExecProbeCommand(), "", true, false, abi.ReasonPolicyBlock},
			{"outside workdir", sshExecProbeCommand(), outside, true, false, abi.ReasonPolicyBlock},
			{"malformed SSH", strings.Replace(sshExecProbeCommand(), "-F none ", "", 1), root, true, false, abi.ReasonPolicyBlock},
			{"opaque SSH", "ssh operator@node.example hostname", root, true, false, abi.ReasonPolicyBlock},
			{"no read hint", sshExecProbeCommand(), root, false, false, abi.ReasonDefaultDeny},
			{"metadata IPv4", strings.Replace(sshExecProbeCommand(), "node.example", "169.254.169.254", 1), root, true, false, egressfloor.ReasonEgressBlock},
			{"metadata name", strings.Replace(sshExecProbeCommand(), "node.example", "metadata", 1), root, true, false, egressfloor.ReasonEgressBlock},
			{"custom policy", sshExecProbeCommand(), root, true, true, abi.ReasonPolicyBlock},
		} {
			t.Run(tool+"/"+tc.name, func(t *testing.T) {
				p := base
				if tc.customPolicy {
					p.ArgPredicates = append(append([]adjudicator.ArgPredicate(nil), p.ArgPredicates...), adjudicator.ArgPredicate{
						Tool: "exec_command", Arg: "cmd", Kind: adjudicator.ArgDenyRegex,
						Re: regexp.MustCompile(`/bin/hostname`), Reason: abi.ReasonPolicyBlock,
					})
				}
				adj.SetPolicy(p)
				verdict, _, err := srv.adjudicate(context.Background(), tool, sshExecProbeArgs(tc.command, tc.workdir), tc.hint, "", "ssh-exec-alias-refusal-"+tool+"-"+tc.name)
				if err != nil || verdict.Kind != "DENY" || verdict.Reason != abi.ReasonName(tc.wantReason) || verdict.By == "lease-admission-permissive" {
					t.Fatalf("refusal boundary verdict=%+v err=%v", verdict, err)
				}
			})
		}
	}
}

// fak-test:runtime fast est=1s
func TestSSHExecProbeCodexAliasesRequireTrustedNativeEnvelope(t *testing.T) {
	root := sshExecProbeWorkspace(t)
	srv, adj := newReloadLatencyServer(t)
	installProjectedLeaseAuthority(t, root, nil)
	rt, err := policy.ParseRuntime(guardTraceFloorJSON)
	if err != nil {
		t.Fatal(err)
	}
	p := rt.Adjudicator
	p.Allow = nil
	p.AllowPrefix = nil
	p.Posture = adjudicator.PostureFailClosed
	adj.SetPolicy(p)
	foreignShell := "C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
	foreignExecutables := []string{"C:/Windows/System32/OpenSSH/ssh.exe"}
	if runtime.GOOS == "windows" {
		foreignShell = "/bin/sh"
		foreignExecutables = []string{"/usr/bin/ssh", "/bin/ssh"}
	}
	mutations := map[string]func(map[string]any){
		"missing shell":      func(args map[string]any) { delete(args, "shell") },
		"unknown shell":      func(args map[string]any) { args["shell"] = "/tmp/unknown-shell" },
		"opposite OS shell":  func(args map[string]any) { args["shell"] = foreignShell },
		"missing login":      func(args map[string]any) { delete(args, "login") },
		"login true":         func(args map[string]any) { args["login"] = true },
		"login string false": func(args map[string]any) { args["login"] = "false" },
	}
	for _, executable := range foreignExecutables {
		executable := executable
		mutations["opposite OS executable "+executable] = func(args map[string]any) {
			args["cmd"] = strings.Replace(sshExecProbeCommand(), sshReadProbeExecutable(), executable, 1)
		}
	}
	for _, tool := range []string{"functions.exec_command", "exec_command"} {
		for name, mutate := range mutations {
			t.Run(tool+"/"+name, func(t *testing.T) {
				var args map[string]any
				if err := json.Unmarshal([]byte(sshExecProbeArgs(sshExecProbeCommand(), root)), &args); err != nil {
					t.Fatal(err)
				}
				mutate(args)
				raw, _ := json.Marshal(args)
				verdict, _, err := srv.adjudicate(context.Background(), tool, string(raw), true, "", "ssh-native-envelope-"+tool+"-"+name)
				if err != nil || verdict.Kind != "DENY" || verdict.Reason != abi.ReasonName(abi.ReasonPolicyBlock) || verdict.By != "monitor" {
					t.Fatalf("untrusted envelope verdict=%+v err=%v", verdict, err)
				}
			})
		}
	}
}
