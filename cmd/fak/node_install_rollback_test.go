package main

// TestNodeInstallFailedStartRollsBack is the CLI-level negative witness for the
// deploykit-service switch: `fak node install` now routes control through
// service.Update in user scope, so a start failure injected AFTER the config
// update must leave the prior registration in place AND the prior running state
// (running→running, stopped→stopped). It branches on runtime.GOOS so the native
// control plane (schtasks / systemctl --user / launchctl) is exercised on every
// host via the injected nodeServiceBoundary.

import (
	"bytes"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

// nodeRollbackFake records every command and answers the platform protocol with
// canned outputs; the START phase fails once (the apply's start) so the
// installer must roll back.
type nodeRollbackFake struct {
	calls   [][]string
	outputs map[string][][]byte
	fail    map[string]int
	creates []string // windows: task XML content captured at each /Create
}

func newNodeRollbackFake() *nodeRollbackFake {
	return &nodeRollbackFake{outputs: map[string][][]byte{}, fail: map[string]int{}}
}

func (f *nodeRollbackFake) key(args ...string) string { return strings.Join(args, "\x00") }

func (f *nodeRollbackFake) Run(args ...string) error {
	f.calls = append(f.calls, args)
	if args[0] == "/Create" {
		for i, a := range args {
			if a == "/XML" && i+1 < len(args) {
				b, err := os.ReadFile(args[i+1])
				if err != nil {
					f.creates = append(f.creates, "<unreadable: "+err.Error()+">")
				} else {
					f.creates = append(f.creates, string(b))
				}
			}
		}
	}
	key := f.key(args...)
	if f.fail[key] > 0 {
		f.fail[key]--
		return fmt.Errorf("injected %q failure", args[0])
	}
	return nil
}

func (f *nodeRollbackFake) Output(args ...string) ([]byte, error) {
	f.calls = append(f.calls, args)
	key := f.key(args...)
	if f.fail[key] > 0 {
		f.fail[key]--
		return nil, fmt.Errorf("injected output failure for %q", key)
	}
	q, ok := f.outputs[key]
	if !ok || len(q) == 0 {
		panic("nodeRollbackFake: no canned output for " + key)
	}
	if len(q) > 1 {
		f.outputs[key] = q[1:]
	}
	return q[0], nil
}

func (f *nodeRollbackFake) countCalls(verb string) int {
	n := 0
	for _, c := range f.calls {
		for _, a := range c {
			if a == verb {
				n++
				break
			}
		}
	}
	return n
}

func freePortForNodeInstallTest(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

func TestNodeInstallFailedStartRollsBack(t *testing.T) {
	for _, priorRunning := range []bool{true, false} {
		prior := "stopped"
		if priorRunning {
			prior = "running"
		}
		t.Run("prior-"+prior, func(t *testing.T) {
			tmp := t.TempDir()
			var fake *nodeRollbackFake
			switch runtime.GOOS {
			case "windows":
				t.Setenv("APPDATA", tmp)
				fake = newNodeRollbackFakeWindows(priorRunning)
			case "linux":
				t.Setenv("HOME", tmp)
				fake = newNodeRollbackFakeLinux(tmp, priorRunning)
			case "darwin":
				t.Setenv("HOME", tmp)
				fake = newNodeRollbackFakeDarwin(tmp, priorRunning)
			default:
				t.Skipf("unsupported GOOS %s", runtime.GOOS)
			}

			oldBoundary := nodeServiceBoundary
			nodeServiceBoundary = fake
			t.Cleanup(func() { nodeServiceBoundary = oldBoundary })
			oldGate := nodeHealthGate
			nodeHealthGate = func(io.Writer, string) bool { return true }
			t.Cleanup(func() { nodeHealthGate = oldGate })

			var out, errBuf bytes.Buffer
			rc := nodeInstall(&out, &errBuf, []string{"-port", strconv.Itoa(freePortForNodeInstallTest(t))})
			if rc == 0 {
				t.Fatalf("install succeeded despite the injected start failure:\nstdout:\n%s\nstderr:\n%s", out.String(), errBuf.String())
			}
			assertNodeInstallRolledBack(t, fake, tmp, priorRunning)
		})
	}
}

const nodeTestPriorTaskXML = `<Task version="1.2"><Prior>PRIOR TASK</Prior></Task>`

func newNodeRollbackFakeWindows(priorRunning bool) *nodeRollbackFake {
	fake := newNodeRollbackFake()
	status := "Ready"
	if priorRunning {
		status = "Running"
	}
	// schtasks /Query /XML carries a UTF-16 BOM on some builds; the facade must
	// strip it so the restored XML is byte-comparable.
	fake.outputs[fake.key("/Query", "/TN", nodeWindowsTaskName, "/XML")] = [][]byte{[]byte("\ufeff" + nodeTestPriorTaskXML)}
	fake.outputs[fake.key("/Query", "/TN", nodeWindowsTaskName, "/FO", "LIST")] = [][]byte{
		[]byte("Folder: \\\nTask Name:      \\" + nodeWindowsTaskName + "\nStatus:         " + status + "\n"),
	}
	// The apply's /Run fails; the restore's /Run succeeds.
	fake.fail[fake.key("/Run", "/TN", nodeWindowsTaskName)] = 1
	return fake
}

func newNodeRollbackFakeLinux(home string, priorRunning bool) *nodeRollbackFake {
	unitDir := filepath.Join(home, ".config", "systemd", "user")
	if err := os.MkdirAll(unitDir, 0o755); err != nil {
		panic(err)
	}
	if err := os.WriteFile(filepath.Join(unitDir, "fak-serve-gateway.service"), []byte("OLD UNIT"), 0o644); err != nil {
		panic(err)
	}
	active, pid := "inactive", "0"
	if priorRunning {
		active, pid = "active", "123"
	}
	showOut := "LoadState=loaded\nActiveState=" + active + "\nSubState=running\nResult=success\nMainPID=" + pid +
		"\nInvocationID=inv\nNRestarts=0\nWatchdogTimestampUSec=0\nBootID=boot\n"
	fake := newNodeRollbackFake()
	unit := "fak-serve-gateway.service"
	fake.outputs[fake.key("--user", "show", unit, "--no-pager",
		"--property=LoadState", "--property=ActiveState", "--property=SubState", "--property=Result",
		"--property=MainPID", "--property=InvocationID", "--property=NRestarts",
		"--property=WatchdogTimestampUSec", "--property=BootID")] = [][]byte{[]byte(showOut)}
	// The apply's start fails; the restore's start succeeds.
	fake.fail[fake.key("--user", "start", unit)] = 1
	return fake
}

func nodeTestLaunchdPrint(target, plistPath string, running bool) string {
	state, pid := "not running", "0"
	if running {
		state, pid = "running", "123"
	}
	return target + " = {\n" +
		"\ttype = LaunchAgent\n" +
		"\tpath = " + plistPath + "\n" +
		"\tname = " + nodeGatewayLabel + "\n" +
		"\tstate = " + state + "\n" +
		"\tpid = " + pid + "\n" +
		"\tlast exit code = 0\n" +
		"}\n"
}

func newNodeRollbackFakeDarwin(home string, priorRunning bool) *nodeRollbackFake {
	agentsDir := filepath.Join(home, "Library", "LaunchAgents")
	if err := os.MkdirAll(agentsDir, 0o755); err != nil {
		panic(err)
	}
	plistPath := filepath.Join(agentsDir, nodeGatewayLabel+".plist")
	if err := os.WriteFile(plistPath, []byte("OLD PLIST"), 0o644); err != nil {
		panic(err)
	}
	target := fmt.Sprintf("gui/%d/%s", os.Getuid(), nodeGatewayLabel)
	fake := newNodeRollbackFake()
	// Print queue: first call is the snapshot (prior state), the second is the
	// post-bootstrap probe, which reports not-running so the kickstart fires.
	fake.outputs[fake.key("print", target)] = [][]byte{
		[]byte(nodeTestLaunchdPrint(target, plistPath, priorRunning)),
		[]byte(nodeTestLaunchdPrint(target, plistPath, false)),
	}
	// The apply's kickstart fails; the restore's kickstart succeeds.
	fake.fail[fake.key("kickstart", "-k", target)] = 1
	return fake
}

func assertNodeInstallRolledBack(t *testing.T, fake *nodeRollbackFake, tmp string, priorRunning bool) {
	t.Helper()
	wantStarts := 1
	if priorRunning {
		wantStarts = 2 // the failed apply start + the restore start
	}
	switch runtime.GOOS {
	case "windows":
		xmlPath := filepath.Join(tmp, "fak", "serve-task.xml")
		got, err := os.ReadFile(xmlPath)
		if err != nil {
			t.Fatalf("task xml missing after rollback: %v", err)
		}
		if string(got) != nodeTestPriorTaskXML {
			t.Fatalf("prior task xml not restored (want BOM-free %q): %q", nodeTestPriorTaskXML, got)
		}
		if len(fake.creates) != 2 || fake.creates[0] == nodeTestPriorTaskXML || fake.creates[1] != nodeTestPriorTaskXML {
			t.Fatalf("create registrations = %v, want [new-xml prior-xml]", fake.creates)
		}
		if got := fake.countCalls("/Run"); got != wantStarts {
			t.Fatalf("/Run calls = %d, want %d (prior running=%v)", got, wantStarts, priorRunning)
		}
	case "linux":
		unitPath := filepath.Join(tmp, ".config", "systemd", "user", "fak-serve-gateway.service")
		got, err := os.ReadFile(unitPath)
		if err != nil {
			t.Fatalf("unit missing after rollback: %v", err)
		}
		if string(got) != "OLD UNIT" {
			t.Fatalf("prior unit not restored: %q", got)
		}
		if got := fake.countCalls("start"); got != wantStarts {
			t.Fatalf("start calls = %d, want %d (prior running=%v)", got, wantStarts, priorRunning)
		}
	case "darwin":
		plistPath := filepath.Join(tmp, "Library", "LaunchAgents", nodeGatewayLabel+".plist")
		got, err := os.ReadFile(plistPath)
		if err != nil {
			t.Fatalf("plist missing after rollback: %v", err)
		}
		if string(got) != "OLD PLIST" {
			t.Fatalf("prior plist not restored: %q", got)
		}
		if got := fake.countCalls("kickstart"); got != wantStarts {
			t.Fatalf("kickstart calls = %d, want %d (prior running=%v)", got, wantStarts, priorRunning)
		}
	}
}
