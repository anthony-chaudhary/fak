package benchcatalog

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"
)

// TestMain lets the test executable stand in for SSH and Tailscale without
// shell-script helpers or access to a real node.
func TestMain(m *testing.M) {
	if os.Getenv("FAK_BENCH_NODE_PROBE_HELPER") == "1" {
		switch filepath.Base(os.Args[0]) {
		case "tailscale":
			os.Exit(0)
		case "ssh":
			args := os.Args[1:]
			data, err := json.Marshal(args)
			if err != nil || os.WriteFile(os.Getenv("FAK_PROBE_ARGS"), data, 0600) != nil {
				os.Exit(99)
			}
			if len(args) == 0 || args[len(args)-1] != os.Getenv("FAK_PROBE_COMMAND") {
				fmt.Fprintln(os.Stderr, "'true' is not recognized as an internal or external command")
				os.Exit(127)
			}
			fmt.Fprintln(os.Stdout, "probe stdout must stay hidden")
			fmt.Fprint(os.Stderr, os.Getenv("FAK_PROBE_STDERR"))
			if os.Getenv("FAK_PROBE_FAIL") == "1" {
				os.Exit(255)
			}
			os.Exit(0)
		}
	}
	os.Exit(m.Run())
}

// fak-test:runtime medium est=10s lane=default
func TestBenchNodeProbeUsesRemoteShell(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("run the Bash runner witness under WSL")
	}
	root := repoRootForBenchNodeRegistryTest(t)
	bin := t.TempDir()
	// An allowlisted PATH cannot reach the real SSH or Tailscale executables.
	for _, name := range []string{"bash", "dirname", "python3", "mktemp", "rm", "head", "tr"} {
		path, err := exec.LookPath(name)
		if err != nil {
			t.Skipf("bench_node.sh witness requires %s: %v", name, err)
		}
		if err := os.Symlink(path, filepath.Join(bin, name)); err != nil {
			t.Fatal(err)
		}
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"ssh", "tailscale"} {
		if err := os.Symlink(exe, filepath.Join(bin, name)); err != nil {
			t.Fatal(err)
		}
	}
	cases := []struct {
		name, shell, command, stderr string
		state                        OnboardState
	}{
		{"bash", "bash", "true", "", OnboardAuthorized},
		{"zsh", "zsh", "true", "", OnboardAuthorized},
		{"sh", "sh", "true", "", OnboardAuthorized},
		{"windows-wsl", "windows-wsl", "cmd /c exit 0", "", OnboardAuthorized},
		{"auth-denied", "windows-wsl", "cmd /c exit 0", "Permission denied (publickey).", OnboardSSHDOnly},
		{"connection-refused", "windows-wsl", "cmd /c exit 0", "Connection refused", OnboardNoSSHD},
		{"host-key-changed", "windows-wsl", "cmd /c exit 0", "WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!", OnboardUnknown},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			registry := filepath.Join(dir, "nodes.json")
			data := fmt.Sprintf(`{"nodes":[{"name":"fixture","sanitized_name":"fixture","tailnet_ip":"192.0.2.1","ssh_user":"user","ssh_key":"unused-fixture-key","ssh_port":22,"repo_path":"/unused","toolchain_env":"GOTOOLCHAIN=local","host_key":"ssh-ed25519 fixture","remote_shell":%q}]}`, tc.shell)
			if err := os.WriteFile(registry, []byte(data), 0600); err != nil {
				t.Fatal(err)
			}
			for _, sub := range []string{"ping", "diagnose"} {
				t.Run(sub, func(t *testing.T) {
					argsPath := filepath.Join(dir, sub+"-args.json")
					ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
					defer cancel()
					cmd := exec.CommandContext(ctx, filepath.Join(bin, "bash"), filepath.Join(root, "tools", "bench_node.sh"), "fixture", sub)
					cmd.Env = []string{
						"PATH=" + bin, "HOME=" + dir, "TMPDIR=" + dir, "LC_ALL=C",
						"BENCH_NODES=" + registry, "BENCH_RUNS_DIR=" + dir,
						"FAK_BENCH_NODE_PROBE_HELPER=1", "FAK_PROBE_ARGS=" + argsPath,
						"FAK_PROBE_COMMAND=" + tc.command, "FAK_PROBE_STDERR=" + tc.stderr,
					}
					wantExit := 0
					if tc.state != OnboardAuthorized {
						cmd.Env = append(cmd.Env, "FAK_PROBE_FAIL=1")
						wantExit = 3
					}
					out, err := cmd.CombinedOutput()
					if cmd.ProcessState == nil || cmd.ProcessState.ExitCode() != wantExit {
						t.Errorf("%s: want exit %d; got %v\n%s", sub, wantExit, err, out)
					}
					wantOutput := "ONBOARD-STATE fixture " + string(tc.state) + "\nONBOARD-REMEDY " + strings.ReplaceAll(tc.state.Remedy(), " - ", " -- ") + "\n"
					if sub == "ping" {
						wantOutput = "REACHABLE fixture (as fixture)\n"
						if wantExit != 0 {
							wantOutput = "UNREACHABLE fixture\n"
						}
					}
					if string(out) != wantOutput {
						t.Errorf("output = %q, want %q", out, wantOutput)
					}
					data, err := os.ReadFile(argsPath)
					if err != nil {
						t.Fatal(err)
					}
					var args []string
					if err := json.Unmarshal(data, &args); err != nil {
						t.Fatal(err)
					}
					wantArgs := []string{"-i", "unused-fixture-key", "-p", "22", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=8", "-o", "ServerAliveCountMax=3", "-o", "", "-o", "StrictHostKeyChecking=yes", "user@192.0.2.1", tc.command}
					if len(args) != len(wantArgs) || !strings.HasPrefix(args[13], "UserKnownHostsFile=") {
						t.Fatalf("unexpected SSH arguments: %q", args)
					}
					wantArgs[13] = args[13]
					if !reflect.DeepEqual(args, wantArgs) {
						t.Errorf("SSH arguments = %q, want %q", args, wantArgs)
					}
					knownHosts := strings.TrimPrefix(args[13], "UserKnownHostsFile=")
					if _, err := os.Stat(knownHosts); !os.IsNotExist(err) {
						t.Errorf("temporary known-hosts file was not removed: %v", err)
					}
				})
			}
		})
	}
}
