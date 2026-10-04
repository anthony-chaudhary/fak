package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// claude_router.go — the live-router default for `fak claude`.
//
// The router's own /v1/messages cannot serve Claude Code today: routing mode
// forwards the Anthropic body verbatim to a kind:openai backend origin
// (platform/gateway/route_wire.go), so `:18101/v1/messages` is rejected by the
// provider. Claude Code speaks the Anthropic wire only. The smallest bridge is
// therefore the same one `fak serve` already owns -- the native Anthropic
// Messages adapter -- run as a private, loopback-only front door that translates
// Anthropic Messages to the router's OpenAI /v1/chat/completions.
//
// When no explicit --gateway-url/--base-url and no FAK_MAC_GATEWAY is set, and a
// live fak router answers on FAK_ROUTER_URL (else the supervised
// http://127.0.0.1:18101), `fak claude` uses this bridge by default so the
// launcher follows the router's current model instead of the stale hardcoded
// 8080 / qwen38:27b-q4 Mac placeholder. If the router is not live the launcher
// keeps its historical direct default.

const (
	// claudeRouterDefaultOrigin is the SCM-supervised fak router. It matches the
	// private ops verbs (resolveRouterCheckEndpoint / opsQualifyDefaultRouterURL)
	// and never the cmd/fak-server bare default :8080.
	claudeRouterDefaultOrigin = "http://127.0.0.1:18101"
	// claudeRouterKeyEnv is the credential the front door presents to the router.
	// The supervised router admits keyless loopback inference, so the default is
	// a non-empty placeholder rather than a real secret.
	claudeRouterKeyEnv       = "FAK_CLAUDE_ROUTER_KEY"
	claudeRouterKeylessValue = "fak-claude-keyless"
	// claudeRouterFrontDoorKeyEnv carries the per-launch bearer the front door
	// requires of its clients (Claude Code), so no other local process can spend
	// through it.
	claudeRouterFrontDoorKeyEnv = "FAK_CLAUDE_FRONTDOOR_KEY"
)

// claudeRouterOrigin resolves the router origin: an explicit value, else
// FAK_ROUTER_URL, else the supervised 18101 listener. A trailing slash or /v1
// is stripped so /healthz and /v1/* share one origin.
func claudeRouterOrigin(explicit string) string {
	v := strings.TrimSpace(explicit)
	if v == "" {
		v = strings.TrimSpace(os.Getenv("FAK_ROUTER_URL"))
	}
	if v == "" {
		v = claudeRouterDefaultOrigin
	}
	v = strings.TrimRight(v, "/")
	return strings.TrimSuffix(v, "/v1")
}

// claudeRouterLive reports whether a router at origin serves models, returning
// the first served model id. It probes GET /v1/models, not /healthz: the router's
// /healthz runs a full census and can exceed a bounded launcher probe, while
// /v1/models is the fast, authoritative "which models can this router route"
// answer the front door needs anyway. A missing or down router means "do not use
// the router default", not a failure.
//
// It is a package var so tests override it with a stub instead of depending on a
// router that happens to be live on the test host.
var claudeRouterLive = claudeRouterLiveProbe

func claudeRouterLiveProbe(origin string, timeout time.Duration) (string, bool) {
	if strings.TrimSpace(origin) == "" {
		return "", false
	}
	client := &http.Client{Timeout: timeout}
	model := claudeRouterModelProbe(client, origin)
	if model == "" {
		return "", false
	}
	return model, true
}

// claudeRouterModel resolves the model the front door should serve. Precedence:
// FAK_ROUTER_MODEL, else the router's /v1/models (first served id). Empty means
// no router model could be discovered.
//
// It is a package var so tests override it with a stub instead of dialing.
var claudeRouterModel = claudeRouterModelProbe

func claudeRouterModelProbe(client *http.Client, origin string) string {
	if v := strings.TrimSpace(os.Getenv("FAK_ROUTER_MODEL")); v != "" {
		return v
	}
	req, err := http.NewRequest(http.MethodGet, strings.TrimRight(origin, "/")+"/v1/models", nil)
	if err != nil {
		return ""
	}
	resp, err := client.Do(req)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return ""
	}
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	var doc struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if json.Unmarshal(body, &doc) != nil || len(doc.Data) == 0 {
		return ""
	}
	return strings.TrimSpace(doc.Data[0].ID)
}

// claudeFrontDoor is a running loopback `fak serve --provider openai` bridge to
// the router: the origin Claude Code is pointed at, the per-launch bearer it
// must present, and the child process to stop when Claude Code exits.
type claudeFrontDoor struct {
	origin string
	token  string
	cmd    *exec.Cmd
	exited chan struct{}
	logf   io.Writer
}

// claudeFrontDoorArgv builds the front-door command line: Anthropic Messages in,
// the router's OpenAI wire out, loopback-bound, bearer-gated, and detached from
// shared session state.
func claudeFrontDoorArgv(addr, routerOrigin, model string) []string {
	return []string{"serve",
		"--provider", "openai",
		"--base-url", strings.TrimRight(routerOrigin, "/") + "/v1",
		"--api-key-env", claudeRouterKeyEnv,
		"--model", model,
		"--addr", addr,
		"--require-key-env", claudeRouterFrontDoorKeyEnv,
		"--session-state", "off",
		"--claude",
	}
}

// claudeFrontDoorRouterCredential is the bearer the front door presents to the
// router: an explicit value, else a non-empty placeholder that the supervised
// keyless loopback router admits.
func claudeFrontDoorRouterCredential(explicit string) string {
	if v := strings.TrimSpace(explicit); v != "" {
		return v
	}
	if v := strings.TrimSpace(os.Getenv(claudeRouterKeyEnv)); v != "" {
		return v
	}
	return claudeRouterKeylessValue
}

// startClaudeFrontDoor launches the bridge on a free loopback port, waits for it
// to answer /healthz, and returns it. The caller must stop it. A startup failure
// kills the child so no half-started listener is left behind.
func startClaudeFrontDoor(parent context.Context, fakBin, routerOrigin, model string, stderr io.Writer) (*claudeFrontDoor, error) {
	addr, err := claudeFreeLoopbackAddr()
	if err != nil {
		return nil, fmt.Errorf("pick front-door port: %w", err)
	}
	token, err := claudeFrontDoorToken()
	if err != nil {
		return nil, fmt.Errorf("mint front-door token: %w", err)
	}
	logPath := claudeFrontDoorLogPath(addr)
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return nil, fmt.Errorf("open front-door log: %w", err)
	}
	cmd := exec.Command(fakBin, claudeFrontDoorArgv(addr, routerOrigin, model)...)
	cmd.Env = append(os.Environ(),
		claudeRouterKeyEnv+"="+claudeFrontDoorRouterCredential(""),
		claudeRouterFrontDoorKeyEnv+"="+token,
	)
	cmd.Stdout, cmd.Stderr = logFile, logFile
	if err := cmd.Start(); err != nil {
		logFile.Close()
		return nil, fmt.Errorf("start front door: %w", err)
	}
	fd := &claudeFrontDoor{
		origin: "http://" + addr,
		token:  token,
		cmd:    cmd,
		exited: make(chan struct{}),
		logf:   stderr,
	}
	go func() { _ = cmd.Wait(); logFile.Close(); close(fd.exited) }()
	client := &http.Client{Timeout: 5 * time.Second}
	if err := claudeFrontDoorWaitReady(parent, client, fd.origin, token, fd.exited, 90*time.Second); err != nil {
		fd.Stop()
		return nil, fmt.Errorf("front door not ready: %w (log: %s)", err, logPath)
	}
	return fd, nil
}

// Stop terminates the front-door child if it still runs. It is idempotent.
func (f *claudeFrontDoor) Stop() {
	if f == nil || f.cmd == nil || f.cmd.Process == nil {
		return
	}
	select {
	case <-f.exited:
		return
	default:
	}
	_ = f.cmd.Process.Kill()
	select {
	case <-f.exited:
	case <-time.After(5 * time.Second):
	}
}

// claudeFrontDoorWaitReady polls the front door until it answers 200, the child
// exits, or the deadline passes.
func claudeFrontDoorWaitReady(ctx context.Context, client *http.Client, base, token string, exited <-chan struct{}, limit time.Duration) error {
	deadline := time.Now().Add(limit)
	for {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/healthz", nil)
		if err != nil {
			return err
		}
		req.Header.Set("Authorization", "Bearer "+token)
		if resp, err := client.Do(req); err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		select {
		case <-exited:
			return errors.New("fak serve exited during startup")
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(250 * time.Millisecond):
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("no 200 from %s/healthz within %s", base, limit)
		}
	}
}

func claudeFreeLoopbackAddr() (string, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return "", err
	}
	defer ln.Close()
	return ln.Addr().String(), nil
}

func claudeFrontDoorToken() (string, error) {
	b := make([]byte, 24)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return "fakc-" + hex.EncodeToString(b), nil
}

func claudeFrontDoorLogPath(addr string) string {
	return filepath.Join(os.TempDir(), "fak-claude-frontdoor-"+strings.ReplaceAll(addr, ":", "-")+".log")
}

// claudeFakBinary returns the fak executable to spawn the front door with: the
// running binary, else `fak` on PATH.
func claudeFakBinary() (string, error) {
	if exe, err := os.Executable(); err == nil {
		if info, statErr := os.Stat(exe); statErr == nil && !info.IsDir() {
			return exe, nil
		}
	}
	if path, err := exec.LookPath("fak"); err == nil {
		return path, nil
	}
	return "", errors.New("no fak binary: run the launcher from the fak executable or put fak on PATH")
}
