package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/localappcontract"
	"github.com/anthony-chaudhary/fak/internal/localapphelper"
)

type commandExecutor struct {
	command  string
	artifact string
	revision string
}

func (e commandExecutor) Execute(ctx context.Context, req localapphelper.TaskRequest) (localapphelper.TaskResult, error) {
	if strings.TrimSpace(e.command) == "" {
		return localapphelper.TaskResult{}, errors.New("localapphelper: no fak-native executor configured")
	}
	payload, _ := json.Marshal(req)
	cmd := exec.CommandContext(ctx, e.command)
	cmd.Stdin = strings.NewReader(string(payload))
	cmd.Env = append(os.Environ(), "FAK_LOCALAPP_TASK_ID="+req.TaskID, "FAK_LOCALAPP_ARTIFACT="+e.artifact)
	started := time.Now()
	out, err := cmd.Output()
	if err != nil {
		return localapphelper.TaskResult{}, fmt.Errorf("fak-native executor: %w", err)
	}
	var result localapphelper.TaskResult
	if err := json.Unmarshal(out, &result); err != nil {
		return result, fmt.Errorf("fak-native executor receipt: %w", err)
	}
	if result.Receipt.Revision != e.revision || result.Receipt.Engine != "fak-native" {
		return result, errors.New("fak-native executor identity mismatch")
	}
	if result.Receipt.ObservedEnvelope == nil {
		result.Receipt.ObservedEnvelope = map[string]int64{}
	}
	result.Receipt.ObservedEnvelope["helper_wall_ms"] = time.Since(started).Milliseconds()
	return result, nil
}

func main() { os.Exit(run()) }
func run() int {
	fs := flag.NewFlagSet("localapphelper", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	addr := fs.String("listen", "127.0.0.1:0", "loopback listen address")
	team := fs.String("team-id", "", "verified host signing team ID")
	bundle := fs.String("bundle-id", "", "verified host bundle ID")
	install := fs.String("install-id", "", "per-install identity")
	revision := fs.String("revision", "", "helper/runtime revision")
	capability := fs.String("capability-env", "FAK_APP_CAPABILITY", "environment variable containing the per-install capability")
	executor := fs.String("executor", "", "fak-native executor command (JSON request on stdin, result on stdout)")
	artifact := fs.String("artifact", "", "exact model artifact identity")
	if err := fs.Parse(os.Args[1:]); err != nil {
		return 2
	}
	secret := os.Getenv(*capability)
	if secret == "" {
		fmt.Fprintln(os.Stderr, "localapphelper: capability environment variable is empty")
		return 2
	}
	host := localapphelper.HostIdentity{TeamID: *team, BundleID: *bundle, InstallID: *install, HelperBuild: *revision}
	binding, err := localapphelper.Bind(host, []byte(secret))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	server := &localapphelper.Server{Binding: binding, Host: host, Capability: []byte(secret), Executor: commandExecutor{command: *executor, artifact: *artifact, revision: *revision}}
	handler, err := server.Handler()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	ln, err := localapphelper.ListenLocal(*addr)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	ready := map[string]any{"schema": localappcontract.Schema, "status": "ready", "address": ln.Addr().String(), "engine": "fak-native", "artifact": *artifact, "revision": *revision}
	json.NewEncoder(os.Stdout).Encode(ready)
	if err := http.Serve(ln, handler); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	return 0
}
