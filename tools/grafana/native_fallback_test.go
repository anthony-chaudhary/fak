//go:build !windows

package grafana

import (
	"bytes"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func sourceDir(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Dir(file)
}

func writeFile(t *testing.T, path, body string, mode os.FileMode) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), mode); err != nil {
		t.Fatal(err)
	}
}

func copyFile(t *testing.T, src, dst string) {
	t.Helper()
	in, err := os.Open(src)
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		t.Fatal(err)
	}
	out, err := os.Create(dst)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		t.Fatal(err)
	}
	if err := out.Close(); err != nil {
		t.Fatal(err)
	}
}

func makeFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	grafana := filepath.Join(root, "tools", "grafana")
	for _, rel := range []string{
		"up.sh", "down.sh", "prometheus.yml", "prometheus-alerts.yml",
		"provisioning/datasources/datasource.yml",
	} {
		copyFile(t, filepath.Join(sourceDir(t), filepath.FromSlash(rel)), filepath.Join(grafana, filepath.FromSlash(rel)))
	}
	writeFile(t, filepath.Join(grafana, "dashboards", "fak-native-kernel-performance.json"), "{}\n", 0o644)
	return root
}

func command(t *testing.T, env []string, name string, args ...string) (string, string) {
	t.Helper()
	cmd := exec.Command(name, args...)
	if env != nil {
		cmd.Env = env
	}
	var stdout, stderr strings.Builder
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("%s %v: %v\nstdout:\n%s\nstderr:\n%s", name, args, err, stdout.String(), stderr.String())
	}
	return stdout.String(), stderr.String()
}

func runShellFunction(t *testing.T, name, setup, result string) (string, string) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(sourceDir(t), "up.sh"))
	if err != nil {
		t.Fatal(err)
	}
	script := string(raw)
	start := strings.Index(script, name+"() {")
	if start < 0 {
		t.Fatalf("function %s not found", name)
	}
	endRel := strings.Index(script[start:], "\n}\n")
	if endRel < 0 {
		t.Fatalf("function %s end not found", name)
	}
	return command(t, nil, "bash", "-c", script[start:start+endRel+3]+"\n"+setup+"\n"+name+"\n"+result)
}

func read(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func metadata(t *testing.T, path string) map[string]string {
	t.Helper()
	out := map[string]string{}
	for _, line := range strings.Split(strings.TrimSpace(read(t, path)), "\n") {
		key, value, ok := strings.Cut(line, "=")
		if ok {
			out[key] = value
		}
	}
	return out
}

func requirePID(t *testing.T, raw string) int {
	t.Helper()
	pid, err := strconv.Atoi(raw)
	if err != nil || pid <= 0 {
		t.Fatalf("invalid positive PID %q: %v", raw, err)
	}
	return pid
}

func envWith(values map[string]string) []string {
	env := append([]string(nil), os.Environ()...)
	for key, value := range values {
		prefix := key + "="
		for i := len(env) - 1; i >= 0; i-- {
			if strings.HasPrefix(env[i], prefix) {
				env = append(env[:i], env[i+1:]...)
			}
		}
		env = append(env, prefix+value)
	}
	return env
}

func TestGeneratedNativeConfigurationIsLoopbackOnly(t *testing.T) {
	root := makeFixture(t)
	grafana := filepath.Join(root, "tools", "grafana")
	command(t, envWith(map[string]string{"FAK_GRAFANA_NATIVE_CONFIG_ONLY": "1"}), "bash", filepath.Join(grafana, "up.sh"))
	native := filepath.Join(grafana, ".run", "native")
	prometheus := read(t, filepath.Join(native, "prometheus.yml"))
	if strings.Contains(prometheus, "host.docker.internal") || strings.Contains(prometheus, "\nalerting:\n") {
		t.Fatalf("native Prometheus config retained a container-only edge:\n%s", prometheus)
	}
	for _, port := range []string{"8080", "9095", "9097", "9098"} {
		if !strings.Contains(prometheus, "127.0.0.1:"+port) {
			t.Errorf("native Prometheus config lacks loopback port %s", port)
		}
	}
	if !strings.Contains(prometheus, filepath.Join(grafana, "prometheus-alerts.yml")) {
		t.Error("native Prometheus config lacks absolute rules path")
	}
	if got := read(t, filepath.Join(native, "provisioning", "datasources", "datasource.yml")); !strings.Contains(got, "url: http://127.0.0.1:9091") {
		t.Errorf("datasource is not loopback-only:\n%s", got)
	}
	if got := read(t, filepath.Join(native, "provisioning", "dashboards", "dashboards.yml")); !strings.Contains(got, filepath.Join(grafana, "dashboards")) {
		t.Errorf("dashboard provisioning lacks fixture path:\n%s", got)
	}
	if !fileExists(filepath.Join(grafana, "dashboards", "fak-native-kernel-performance.json")) {
		t.Fatal("native configuration removed the provisioned dashboard artifact")
	}
}

func TestNativeGatewayServeIsLoopbackOnly(t *testing.T) {
	stdout, _ := runShellFunction(t, "configure_gateway_addr", "STACK_MODE=native; unset FAK_GATEWAY_ADDR", `printf "%s\n" "$GATEWAY_ADDR"`)
	if stdout != "127.0.0.1:8080\n" {
		t.Fatalf("gateway address = %q", stdout)
	}
}

func TestStackModeSelection(t *testing.T) {
	t.Run("missing-docker", func(t *testing.T) {
		stdout, stderr := runShellFunction(t, "select_stack_mode", `command() { return 1; }; uname() { printf 'Darwin\n'; }; ensure_docker_daemon() { printf 'called\n' >&2; return 0; }; die() { printf 'died\n' >&2; return 1; }; STACK_MODE=unset`, `printf "%s\n" "$STACK_MODE"`)
		if stdout != "native\n" || stderr != "" {
			t.Fatalf("stdout=%q stderr=%q", stdout, stderr)
		}
	})
	t.Run("docker-present", func(t *testing.T) {
		stdout, stderr := runShellFunction(t, "select_stack_mode", `command() { return 0; }; ensure_docker_daemon() { printf 'called\n' >&2; return 0; }; STACK_MODE=unset`, `printf "%s\n" "$STACK_MODE"`)
		if stdout != "docker\n" || stderr != "called\n" {
			t.Fatalf("stdout=%q stderr=%q", stdout, stderr)
		}
	})
}

func extractFunction(t *testing.T, name string) string {
	t.Helper()
	script := read(t, filepath.Join(sourceDir(t), "up.sh"))
	start := strings.Index(script, name+"() {")
	if start < 0 {
		t.Fatalf("cannot extract %s", name)
	}
	end := strings.Index(script[start:], "\n}\n")
	if end < 0 {
		t.Fatalf("cannot extract %s", name)
	}
	return script[start : start+end+3]
}

func stopPID(pid int) {
	if pid > 0 {
		_ = syscall.Kill(pid, syscall.SIGTERM)
	}
}

func stopAndWaitCommand(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	_ = cmd.Process.Signal(syscall.SIGTERM)
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		_ = cmd.Process.Kill()
		<-done
	}
}

func waitPIDExit(pid int, timeout time.Duration) bool {
	if pid <= 0 {
		return false
	}
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if err := syscall.Kill(pid, 0); err != nil {
			return true
		}
		time.Sleep(20 * time.Millisecond)
	}
	return false
}

func waitForOwnerMarker(owner string, attempts int, interval time.Duration, probe func() (string, error)) (string, error) {
	marker := "fak-grafana-owner=" + owner
	var last string
	for attempt := 0; attempt < attempts; attempt++ {
		command, err := probe()
		if err != nil {
			return last, fmt.Errorf("owned supervisor exited before publishing %q: %w", marker, err)
		}
		last = command
		if strings.Contains(command, marker) {
			return command, nil
		}
		if attempt+1 < attempts {
			time.Sleep(interval)
		}
	}
	return last, fmt.Errorf("owner marker %q absent after %d probe(s); last command: %s", marker, attempts, strings.TrimSpace(last))
}

func TestWaitForOwnerMarker(t *testing.T) {
	t.Run("delayed nohup exec", func(t *testing.T) {
		observations := []string{"[nohup]\n", "bash -c supervisor fak-grafana-owner=fixture\n"}
		calls := 0
		got, err := waitForOwnerMarker("fixture", len(observations), 0, func() (string, error) {
			observation := observations[calls]
			calls++
			return observation, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if calls != 2 || !strings.Contains(got, "fak-grafana-owner=fixture") {
			t.Fatalf("calls=%d command=%q", calls, got)
		}
	})

	t.Run("never ready", func(t *testing.T) {
		got, err := waitForOwnerMarker("fixture", 2, 0, func() (string, error) {
			return "[nohup]\n", nil
		})
		if err == nil || !strings.Contains(err.Error(), "absent after 2 probe(s)") || got != "[nohup]\n" {
			t.Fatalf("command=%q err=%v", got, err)
		}
	})

	t.Run("supervisor exits", func(t *testing.T) {
		probeErr := fmt.Errorf("exit status 1")
		got, err := waitForOwnerMarker("fixture", 2, 0, func() (string, error) {
			return "", probeErr
		})
		if err == nil || !strings.Contains(err.Error(), "exited before publishing") || got != "" {
			t.Fatalf("command=%q err=%v", got, err)
		}
	})
}

func waitForChild(t *testing.T, parent int, timeout time.Duration) int {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		out, err := exec.Command("pgrep", "-P", strconv.Itoa(parent)).Output()
		if err == nil {
			for _, field := range strings.Fields(string(out)) {
				if pid, err := strconv.Atoi(field); err == nil && pid > 0 {
					return pid
				}
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("no child appeared for supervisor PID %d", parent)
	return 0
}

func processCWD(t *testing.T, pid int) string {
	t.Helper()
	out, _ := command(t, nil, "lsof", "-a", "-p", strconv.Itoa(pid), "-d", "cwd", "-Fn")
	var cwd string
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(line, "n") {
			cwd = strings.TrimPrefix(line, "n")
		}
	}
	if cwd == "" {
		t.Fatalf("lsof returned no cwd for PID %d: %q", pid, out)
	}
	resolved, err := filepath.EvalSymlinks(cwd)
	if err != nil {
		t.Fatal(err)
	}
	return resolved
}

func TestNonMacOwnedProcessSurvivesLauncherExitAndDownStopsIt(t *testing.T) {
	root := makeFixture(t)
	grafana := filepath.Join(root, "tools", "grafana")
	runDir := filepath.Join(grafana, ".run")
	if err := os.Mkdir(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(runDir, "stack.mode"), "native\n", 0o644)
	owner := "native-persistence-owner"
	body := fmt.Sprintf("%s\nport_live() { return 1; }\nlog() { :; }\nRUN_DIR=%q\nRUN_ID=%q\nuname() { printf 'Linux\\n'; }\nstart_bg owned '65534 /' sleep 30\n", extractFunction(t, "start_bg"), runDir, owner)
	command(t, nil, "bash", "-c", body)
	pidFile := filepath.Join(runDir, "owned.pid")
	m := metadata(t, pidFile)
	pid := requirePID(t, m["pid"])
	t.Cleanup(func() { stopPID(pid) })
	if m["owner"] != owner {
		t.Fatalf("owner=%q", m["owner"])
	}
	stdout, err := waitForOwnerMarker(owner, 100, 20*time.Millisecond, func() (string, error) {
		out, err := exec.Command("ps", "-p", strconv.Itoa(pid), "-o", "command=").CombinedOutput()
		return string(out), err
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := syscall.Kill(pid, syscall.SIGHUP); err != nil {
		t.Fatal(err)
	}
	time.Sleep(100 * time.Millisecond)
	if err := syscall.Kill(pid, 0); err != nil {
		t.Fatalf("supervisor died after launcher exit signal: %v", err)
	}
	if got, want := read(t, pidFile), fmt.Sprintf("pid=%d\nowner=%s\n", pid, owner); got != want {
		t.Fatalf("PID metadata changed after launcher exit: got %q want %q", got, want)
	}
	stdout, _ = command(t, nil, "ps", "-p", strconv.Itoa(pid), "-o", "command=")
	if !strings.Contains(stdout, "fak-grafana-owner="+owner) {
		t.Fatalf("owner marker changed after launcher exit: %s", stdout)
	}
	command(t, nil, "bash", filepath.Join(grafana, "down.sh"))
	if !waitPIDExit(pid, time.Second) || fileExists(pidFile) {
		t.Fatal("owned supervisor was not stopped and reaped")
	}
}

func TestLaunchctlSubmitPreservesBashCArgumentPositions(t *testing.T) {
	root := makeFixture(t)
	runDir := filepath.Join(root, "tools", "grafana", ".run")
	runtimeRoot := filepath.Join(root, "tmp")
	_ = os.MkdirAll(runDir, 0o755)
	_ = os.MkdirAll(runtimeRoot, 0o755)
	owner := "launchctl-argv-owner"
	body := extractFunction(t, "start_bg") + `
port_live() { return 1; }
log() { :; }
die() { printf '%s\n' "$*" >&2; exit 1; }
uname() { printf 'Darwin\n'; }
launchctl() {
  [ "${1:-}" = submit ] || return 0
  shift
  local executable= arg0
  while [ "${1:-}" != -- ]; do [ "$1" != -p ] || executable="$2"; shift 2; done
  shift
  if [ -z "$executable" ]; then executable="$1"; shift; fi
  arg0="$1"; shift
  (exec -a "$arg0" "$executable" "$@")
}
RUN_DIR="$RUN_DIR_UNDER_TEST"
RUN_ID="$RUN_ID_UNDER_TEST"
start_bg launchctl_argv '65534 /' /usr/bin/true
`
	command(t, envWith(map[string]string{"RUN_DIR_UNDER_TEST": runDir, "RUN_ID_UNDER_TEST": owner, "TMPDIR": runtimeRoot + "/"}), "/bin/bash", "-c", body)
	m := metadata(t, filepath.Join(runDir, "launchctl_argv.pid"))
	if m["owner"] != owner || m["supervisor"] != "launchd" || m["label"] != launchdLabel("launchctl_argv", owner) {
		t.Fatalf("metadata=%v", m)
	}
	if filepath.Dir(m["runtime"]) != runtimeRoot || !strings.HasPrefix(filepath.Base(m["runtime"]), "fak-grafana-launchctl_argv.") {
		t.Fatalf("unsafe runtime=%q", m["runtime"])
	}
}

func TestDownRejectsRuntimeOutsidePrivateTmpPrefix(t *testing.T) {
	root := makeFixture(t)
	grafana := filepath.Join(root, "tools", "grafana")
	runDir, fakeBin := filepath.Join(grafana, ".run"), filepath.Join(root, "fake-bin")
	runtimeRoot, unsafe := filepath.Join(root, "tmp"), filepath.Join(root, "must-survive")
	for _, dir := range []string{runDir, fakeBin, runtimeRoot, unsafe} {
		_ = os.MkdirAll(dir, 0o755)
	}
	writeFile(t, filepath.Join(runDir, "stack.mode"), "native\n", 0o644)
	owner := "runtime-safety-owner"
	label := launchdLabel("unsafe", owner)
	pidFile := filepath.Join(runDir, "unsafe.pid")
	writeFile(t, pidFile, fmt.Sprintf("pid=4242\nowner=%s\nsupervisor=launchd\nlabel=%s\nruntime=%s\n", owner, label, unsafe), 0o644)
	calls := filepath.Join(root, "launchctl.calls")
	writeFile(t, filepath.Join(fakeBin, "uname"), "#!/usr/bin/env bash\nprintf 'Darwin\n'\n", 0o755)
	writeFile(t, filepath.Join(fakeBin, "ps"), fmt.Sprintf("#!/usr/bin/env bash\nprintf 'fak-grafana-owner=%s\n'\n", owner), 0o755)
	writeFile(t, filepath.Join(fakeBin, "launchctl"), fmt.Sprintf("#!/usr/bin/env bash\nif [ \"$1\" = print ]; then printf 'job\\n'; exit 0; fi\nprintf '%%s\\n' \"$*\" >>%q\n", calls), 0o755)
	command(t, envWith(map[string]string{"PATH": fakeBin + ":/usr/bin:/bin", "TMPDIR": runtimeRoot + "/"}), "bash", filepath.Join(grafana, "down.sh"))
	if !fileExists(unsafe) || fileExists(calls) || !fileExists(pidFile) {
		t.Fatalf("unsafe metadata was acted on: unsafe=%v calls=%v pidfile=%v", fileExists(unsafe), fileExists(calls), fileExists(pidFile))
	}
}

func TestDownStopsOnlyProcessWithMatchingOwner(t *testing.T) {
	root := makeFixture(t)
	grafana := filepath.Join(root, "tools", "grafana")
	runDir := filepath.Join(grafana, ".run")
	_ = os.MkdirAll(runDir, 0o755)
	writeFile(t, filepath.Join(runDir, "stack.mode"), "native\n", 0o644)
	owner := "native-test-owner"
	owned := exec.Command("bash", "-c", `child=""; stop_child() { [ -z "$child" ] || kill "$child" 2>/dev/null || true; [ -z "$child" ] || wait "$child" 2>/dev/null || true; exit 0; }; trap stop_child TERM INT HUP; "$@" & child=$!; wait "$child"`, "fak-grafana-owner="+owner, "sleep", "30")
	adopted := exec.Command("sleep", "30")
	if err := owned.Start(); err != nil {
		t.Fatal(err)
	}
	if err := adopted.Start(); err != nil {
		stopPID(owned.Process.Pid)
		t.Fatal(err)
	}
	ownedDone := make(chan error, 1)
	go func() { ownedDone <- owned.Wait() }()
	ownedReaped := false
	t.Cleanup(func() {
		if !ownedReaped {
			stopPID(owned.Process.Pid)
			<-ownedDone
		}
		stopAndWaitCommand(adopted)
	})
	writeFile(t, filepath.Join(runDir, "owned.pid"), fmt.Sprintf("pid=%d\nowner=%s\n", owned.Process.Pid, owner), 0o644)
	writeFile(t, filepath.Join(runDir, "adopted.pid"), fmt.Sprintf("pid=%d\nowner=not-the-owner\n", adopted.Process.Pid), 0o644)
	command(t, nil, "bash", filepath.Join(grafana, "down.sh"))
	select {
	case <-ownedDone:
		ownedReaped = true
	case <-time.After(time.Second):
		t.Fatal("owned supervisor survived down.sh")
	}
	if err := syscall.Kill(adopted.Process.Pid, 0); err != nil {
		t.Fatal("adopted process was stopped")
	}
	if fileExists(filepath.Join(runDir, "owned.pid")) || fileExists(filepath.Join(runDir, "adopted.pid")) {
		t.Fatal("down.sh retained processed pid files")
	}
}

func TestDownPreservesDockerComposePurgeBehavior(t *testing.T) {
	root := makeFixture(t)
	grafana := filepath.Join(root, "tools", "grafana")
	runDir, fakeBin := filepath.Join(grafana, ".run"), filepath.Join(root, "fake-bin")
	_ = os.MkdirAll(runDir, 0o755)
	_ = os.MkdirAll(fakeBin, 0o755)
	writeFile(t, filepath.Join(runDir, "stack.mode"), "docker\n", 0o644)
	calls := filepath.Join(root, "docker.calls")
	writeFile(t, filepath.Join(fakeBin, "docker"), fmt.Sprintf("#!/usr/bin/env bash\nif [ \"${1:-}\" = info ]; then exit 0; fi\nprintf '%%s\\n' \"$*\" >>%q\n", calls), 0o755)
	command(t, envWith(map[string]string{"PATH": fakeBin + ":" + os.Getenv("PATH")}), "bash", filepath.Join(grafana, "down.sh"), "--purge")
	if got := read(t, calls); got != "compose down -v\n" {
		t.Fatalf("docker calls=%q", got)
	}
}

func fileExists(path string) bool { _, err := os.Stat(path); return err == nil }

func launchdLabel(name, owner string) string {
	name = strings.ReplaceAll(name, "_", "-")
	return fmt.Sprintf("com.fak.grafana.%d.%s.%s", os.Getuid(), name, owner)
}

func launchdPresent(label string) bool {
	return exec.Command("launchctl", "print", fmt.Sprintf("gui/%d/%s", os.Getuid(), label)).Run() == nil
}

func removeLaunchd(label string) { _ = exec.Command("launchctl", "remove", label).Run() }

func waitHTTP(url string) bool {
	client := &http.Client{Timeout: 200 * time.Millisecond}
	for i := 0; i < 100; i++ {
		response, err := client.Get(url) // #nosec G107 -- loopback fixture URL only.
		if err == nil {
			response.Body.Close()
			if response.StatusCode == http.StatusOK {
				return true
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	return false
}

func TestRealUpDownLaunchdLifecycleAndAdoption(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("requires real launchd")
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	listener.Close()
	root := makeFixture(t)
	grafana, runDir := filepath.Join(root, "tools", "grafana"), filepath.Join(root, "tools", "grafana", ".run")
	fakeBin, modelDir, runtimeRoot := filepath.Join(root, "fake-bin"), filepath.Join(root, "model"), filepath.Join(root, "tmp")
	for _, dir := range []string{runDir, fakeBin, modelDir, runtimeRoot} {
		_ = os.MkdirAll(dir, 0o755)
	}
	writeFile(t, filepath.Join(modelDir, "weights.f32"), "fixture", 0o644)
	fakeFak := filepath.Join(runDir, "fak")
	writeFile(t, fakeFak, `#!/usr/bin/env bash
set -euo pipefail
[ "${1:-}" = serve ] || exit 64
shift
addr=127.0.0.1:8080
while [ "$#" -gt 0 ]; do if [ "$1" = --addr ]; then addr="$2"; shift 2; else shift; fi; done
host="${addr%:*}"; port="${addr##*:}"
exec /usr/bin/python3 -c 'import http.server,sys
class H(http.server.BaseHTTPRequestHandler):
 def do_GET(self): self.send_response(200); self.end_headers(); self.wfile.write(b"ok\\n")
 def log_message(self,*args): pass
http.server.ThreadingHTTPServer((sys.argv[1],int(sys.argv[2])),H).serve_forever()' "$host" "$port"
`, 0o755)
	writeFile(t, filepath.Join(fakeBin, "docker"), "#!/usr/bin/env bash\ncase \"${1:-}\" in info|compose) exit 0;; esac\nexit 64\n", 0o755)
	writeFile(t, filepath.Join(fakeBin, "curl"), fmt.Sprintf("#!/usr/bin/env bash\nfor arg in \"$@\"; do case \"$arg\" in http://127.0.0.1:%d/*|http://localhost:%d/*) exec /usr/bin/curl \"$@\";; esac; done\nexit 0\n", port, port), 0o755)
	owner := fmt.Sprintf("issue-9402-owned-%d", time.Now().UnixNano())
	adoptedOwner := owner + "-adopted"
	ownedLabel, adoptedLabel := launchdLabel("fak_gateway", owner), launchdLabel("fak_gateway", adoptedOwner)
	t.Cleanup(func() { removeLaunchd(ownedLabel); removeLaunchd(adoptedLabel) })
	baseEnv := map[string]string{
		"FAK_DOGFOOD_MODEL": "fixture", "FAK_GATEWAY_ADDR": fmt.Sprintf("127.0.0.1:%d", port),
		"FAK_GRAFANA_RUN_ID": owner, "FAK_MODEL_DIR": modelDir, "FAK_TELEMETRY_SOURCES": "fak_gateway",
		"PATH": fakeBin + ":/usr/bin:/bin:/usr/sbin:/sbin", "TMPDIR": runtimeRoot + "/",
	}
	command(t, envWith(baseEnv), "/bin/bash", filepath.Join(grafana, "up.sh"))
	pidFile := filepath.Join(runDir, "fak_gateway.pid")
	metadataText := read(t, pidFile)
	m := metadata(t, pidFile)
	pid := requirePID(t, m["pid"])
	if m["owner"] != owner || m["supervisor"] != "launchd" || m["label"] != ownedLabel || filepath.Dir(m["runtime"]) != runtimeRoot {
		t.Fatalf("launchd metadata=%v", m)
	}
	if !strings.HasPrefix(filepath.Base(m["runtime"]), "fak-grafana-fak_gateway."+owner+".") {
		t.Fatalf("runtime directory is not owner-scoped: %q", m["runtime"])
	}
	stagedFak := filepath.Join(m["runtime"], "fak")
	stagedBytes, err := os.ReadFile(stagedFak)
	if err != nil {
		t.Fatal(err)
	}
	fakeBytes, err := os.ReadFile(fakeFak)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(stagedBytes, fakeBytes) {
		t.Fatal("staged fak differs from the installed fixture")
	}
	metricsURL := fmt.Sprintf("http://127.0.0.1:%d/metrics", port)
	healthURL := fmt.Sprintf("http://127.0.0.1:%d/healthz", port)
	if !fileExists(filepath.Join(m["runtime"], "fak_gateway.log")) || !launchdPresent(ownedLabel) || !waitHTTP(metricsURL) {
		t.Fatal("owned launchd service did not become healthy")
	}
	childPID := waitForChild(t, pid, 5*time.Second)
	runtimeResolved, err := filepath.EvalSymlinks(m["runtime"])
	if err != nil {
		t.Fatal(err)
	}
	if got := processCWD(t, childPID); got != runtimeResolved {
		t.Fatalf("child cwd=%q, want runtime %q", got, runtimeResolved)
	}
	if err := syscall.Kill(pid, 0); err != nil {
		t.Fatalf("launchd wrapper is not alive: %v", err)
	}
	if err := syscall.Kill(childPID, 0); err != nil {
		t.Fatalf("launchd child is not alive: %v", err)
	}
	time.Sleep(500 * time.Millisecond)
	if got := read(t, pidFile); got != metadataText {
		t.Fatalf("launchd metadata changed after dwell: got %q want %q", got, metadataText)
	}
	if !launchdPresent(ownedLabel) {
		t.Fatal("launchd job disappeared after dwell")
	}
	if err := syscall.Kill(pid, 0); err != nil {
		t.Fatalf("launchd wrapper disappeared after dwell: %v", err)
	}
	if err := syscall.Kill(childPID, 0); err != nil {
		t.Fatalf("launchd child disappeared after dwell: %v", err)
	}
	if !waitHTTP(healthURL) {
		t.Fatal("launchd child health endpoint failed after dwell")
	}
	command(t, envWith(baseEnv), "bash", filepath.Join(grafana, "down.sh"))
	if !waitPIDExit(pid, 5*time.Second) || !waitPIDExit(childPID, 5*time.Second) || launchdPresent(ownedLabel) || fileExists(pidFile) || fileExists(m["runtime"]) {
		t.Fatal("owned launchd service was not fully reaped")
	}
	adopted := exec.Command(fakeFak, "serve", "--addr", fmt.Sprintf("127.0.0.1:%d", port), "--engine", "fixture", "--model", "fixture")
	adopted.Env = envWith(baseEnv)
	if err := adopted.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { stopAndWaitCommand(adopted) })
	if !waitHTTP(metricsURL) {
		t.Fatal("adopted listener did not become healthy")
	}
	baseEnv["FAK_GRAFANA_RUN_ID"] = adoptedOwner
	command(t, envWith(baseEnv), "bash", filepath.Join(grafana, "up.sh"))
	if err := syscall.Kill(adopted.Process.Pid, 0); err != nil || fileExists(pidFile) || launchdPresent(adoptedLabel) {
		t.Fatal("healthy listener was not adopted cleanly")
	}
	command(t, envWith(baseEnv), "bash", filepath.Join(grafana, "down.sh"))
	if err := syscall.Kill(adopted.Process.Pid, 0); err != nil || !waitHTTP(metricsURL) {
		t.Fatal("down stopped the adopted listener")
	}
}
