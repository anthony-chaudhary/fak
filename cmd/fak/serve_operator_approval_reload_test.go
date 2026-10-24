package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/journal"
)

// The initial refresh closes the floor-load to signature-snapshot race.
// fak-test:runtime fast est=2s
func TestServeAllowWatcherRefreshesApprovalWrittenDuringStartupRace(t *testing.T) {
	guardAllowWatcherFailClosed(t)
	dir := t.TempDir()
	overlayPath := filepath.Join(dir, "allow.json")
	policyPath := filepath.Join(repoRootForDoctorTest(t), "cmd", "fak", "guard-default-policy.json")
	t.Setenv(guardAllowOverlayEnv, overlayPath)
	t.Setenv(guardDenyOverlayEnv, filepath.Join(dir, "deny.json"))
	t.Setenv(policyReloadWidenConfirmEnv, "1")

	reload := policyReloader(policyPath)
	if _, err := reload(context.Background()); err != nil {
		t.Fatal(err)
	}
	const rawArgs = `{"command":"rm -rf build"}`
	call := approveTestCall("Bash", rawArgs)
	refused := adjudicator.Default.Adjudicate(context.Background(), call)
	id := refused.Meta[adjudicator.MetaOperatorApprovalID]
	if refused.Kind != abi.VerdictDeny || id == "" {
		t.Fatalf("startup floor kind=%v reason=%s by=%s approval_id=%q", refused.Kind, abi.ReasonName(refused.Reason), refused.By, id)
	}

	// Mutate after floor load and before watcher construction.
	if err := saveGuardAllowOverlay(overlayPath, guardAllowOverlay{Approve: []string{id}}); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stop := startOperatorApprovalWatcherWithInitialRefresh(ctx, serveOperatorApprovalReloader())
	defer stop()

	deadline := time.Now().Add(1500 * time.Millisecond)
	for {
		v := adjudicator.Default.Adjudicate(context.Background(), call)
		if v.Kind == abi.VerdictAllow && v.Meta["posture"] == adjudicator.PostureOperatorApproved {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("startup-race kind=%v reason=%s by=%s approval_id=%q", v.Kind, abi.ReasonName(v.Reason), v.By, v.Meta[adjudicator.MetaOperatorApprovalID])
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// Approval refresh must not restore a stale base over an explicit reload.
// fak-test:runtime fast est=2s
func TestServeOperatorApprovalReloadPreservesExplicitBaseReload(t *testing.T) {
	prior := adjudicator.Default.PolicySnapshot()
	t.Cleanup(func() { adjudicator.Default.SetPolicy(prior) })
	dir := t.TempDir()
	overlayPath := filepath.Join(dir, "allow.json")
	policyPath := filepath.Join(dir, "policy.json")
	t.Setenv(guardAllowOverlayEnv, overlayPath)
	t.Setenv(guardDenyOverlayEnv, filepath.Join(dir, "deny.json"))
	t.Setenv(policyReloadWidenConfirmEnv, "1")
	if err := saveGuardAllowOverlay(overlayPath, guardAllowOverlay{}); err != nil {
		t.Fatal(err)
	}
	base := `{"version":"fak-policy/v1","posture":"fail_closed","allow":["base_after_explicit_reload"],"deny":{"still_denied":"POLICY_BLOCK"}}`
	if err := os.WriteFile(policyPath, []byte(base), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := policyReloader(policyPath)(context.Background()); err != nil {
		t.Fatalf("explicit base reload: %v", err)
	}
	baseAfter := adjudicator.Default.PolicySnapshot()

	const id = "ap-222222222222"
	if err := saveGuardAllowOverlay(overlayPath, guardAllowOverlay{Approve: []string{id}}); err != nil {
		t.Fatal(err)
	}
	if _, err := serveOperatorApprovalReloader()(context.Background()); err != nil {
		t.Fatal(err)
	}
	got := adjudicator.Default.PolicySnapshot()
	if !got.Allow["base_after_explicit_reload"] || got.Deny["still_denied"] != abi.ReasonPolicyBlock || got.Posture != baseAfter.Posture {
		t.Fatalf("approval event changed base: allow=%v deny=%v posture=%s; base_allow=%v base_deny=%v base_posture=%s", got.Allow, got.Deny, got.Posture, baseAfter.Allow, baseAfter.Deny, baseAfter.Posture)
	}
	if len(got.OperatorApprovals) != 1 || !got.OperatorApprovals[id] {
		t.Fatalf("operator approvals = %v, want only %s", got.OperatorApprovals, id)
	}
}

// Core-lock rejects approval widening and retains the prior set.
// fak-test:runtime fast est=1s
func TestServeOperatorApprovalReloadHonorsCoreLock(t *testing.T) {
	prior := adjudicator.Default.PolicySnapshot()
	priorLock := guardCoreLockAllState.Load()
	t.Cleanup(func() {
		adjudicator.Default.SetPolicy(prior)
		guardCoreLockAllState.Store(priorLock)
	})
	dir := t.TempDir()
	overlayPath := filepath.Join(dir, "allow.json")
	t.Setenv(guardAllowOverlayEnv, overlayPath)
	t.Setenv(guardDenyOverlayEnv, filepath.Join(dir, "deny.json"))
	adjudicator.Default.SetPolicy(adjudicator.Policy{Posture: adjudicator.PostureFailClosed})
	if err := saveGuardAllowOverlay(overlayPath, guardAllowOverlay{Approve: []string{"ap-0123456789ab"}}); err != nil {
		t.Fatal(err)
	}
	guardCoreLockAllState.Store(true)
	_, err := serveOperatorApprovalReloader()(context.Background())
	if err == nil || !strings.Contains(err.Error(), "core-lock-all") {
		t.Fatalf("core-lock approval reload error = %v", err)
	}
	if got := adjudicator.Default.PolicySnapshot().OperatorApprovals; len(got) != 0 {
		t.Fatalf("core-lock refusal changed approvals: %v", got)
	}
}

// Loss of approval authority clears only approvals instead of retaining a stale grant.
// fak-test:runtime fast est=1s
func TestServeOperatorApprovalWatcherClearsApprovedCallOnAuthorityError(t *testing.T) {
	prior := adjudicator.Default.PolicySnapshot()
	t.Cleanup(func() { adjudicator.Default.SetPolicy(prior) })
	dir := t.TempDir()
	overlayPath := filepath.Join(dir, "allow.json")
	t.Setenv(guardAllowOverlayEnv, overlayPath)
	t.Setenv(guardDenyOverlayEnv, filepath.Join(dir, "deny.json"))
	adjudicator.Default.SetPolicy(adjudicator.Policy{Posture: adjudicator.PostureFailClosed})
	const id = "ap-0123456789ab"
	if err := saveGuardAllowOverlay(overlayPath, guardAllowOverlay{Approve: []string{id}}); err != nil {
		t.Fatal(err)
	}
	if _, err := serveOperatorApprovalReloader()(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(overlayPath, []byte(`{"version":`), 0o600); err != nil {
		t.Fatal(err)
	}
	w := newLoggedGuardAllowWatcher(serveOperatorApprovalReloader())
	w.signature = guardOperatorApprovalLayersSignature
	w.onSignatureError = clearServeOperatorApprovals
	event := w.Reload(context.Background())
	if !event.Rejected || len(adjudicator.Default.PolicySnapshot().OperatorApprovals) != 0 {
		t.Fatalf("corrupt authority rejected=%t approvals=%v", event.Rejected, adjudicator.Default.PolicySnapshot().OperatorApprovals)
	}
	if err := saveGuardAllowOverlay(overlayPath, guardAllowOverlay{Approve: []string{id}}); err != nil {
		t.Fatal(err)
	}
	realReload := serveOperatorApprovalReloader()
	reloadAttempts := 0
	w.reload = func(ctx context.Context) (gateway.PolicyReloadResponse, error) {
		reloadAttempts++
		if reloadAttempts == 1 {
			return gateway.PolicyReloadResponse{}, errors.New("authority changed after signature read")
		}
		return realReload(ctx)
	}
	if event = w.Reload(context.Background()); !event.Rejected {
		t.Fatalf("first authority repair reloaded=%t rejected=%t", event.Reloaded, event.Rejected)
	}
	if event = w.Reload(context.Background()); !event.Reloaded || !adjudicator.Default.PolicySnapshot().OperatorApprovals[id] {
		t.Fatalf("authority repair reloaded=%t rejected=%t approvals=%v", event.Reloaded, event.Rejected, adjudicator.Default.PolicySnapshot().OperatorApprovals)
	}
	if err := os.WriteFile(overlayPath, []byte(`{"version":`), 0o600); err != nil {
		t.Fatal(err)
	}
	w.Reload(context.Background())
	external := adjudicator.Default.PolicySnapshot()
	external.OperatorApprovals = map[string]bool{id: true}
	adjudicator.Default.SetPolicy(external)
	w.Reload(context.Background())
	if got := adjudicator.Default.PolicySnapshot().OperatorApprovals; len(got) != 0 {
		t.Fatalf("repeated authority failure retained externally restored approvals=%v", got)
	}

	if err := saveGuardAllowOverlay(overlayPath, guardAllowOverlay{Approve: []string{id}}); err != nil {
		t.Fatal(err)
	}
	if _, err := serveOperatorApprovalReloader()(context.Background()); err != nil {
		t.Fatal(err)
	}
	w.signature = func() (string, error) { return "", os.ErrPermission }
	w.signatureFailed = false
	event = w.Reload(context.Background())
	if !event.Rejected || len(adjudicator.Default.PolicySnapshot().OperatorApprovals) != 0 {
		t.Fatalf("unreadable authority rejected=%t approvals=%v", event.Rejected, adjudicator.Default.PolicySnapshot().OperatorApprovals)
	}
}

// TTL expiry changes the effective approval signature without rewriting the file.
// fak-test:runtime fast est=1s
func TestServeOperatorApprovalWatcherClearsExpiredApproval(t *testing.T) {
	prior := adjudicator.Default.PolicySnapshot()
	priorNow := guardAllowNow
	t.Cleanup(func() {
		adjudicator.Default.SetPolicy(prior)
		guardAllowNow = priorNow
	})
	dir := t.TempDir()
	overlayPath := filepath.Join(dir, "allow.json")
	t.Setenv(guardAllowOverlayEnv, overlayPath)
	t.Setenv(guardDenyOverlayEnv, filepath.Join(dir, "deny.json"))
	adjudicator.Default.SetPolicy(adjudicator.Policy{Posture: adjudicator.PostureFailClosed})
	now := time.Date(2026, 10, 24, 12, 0, 0, 0, time.UTC)
	guardAllowNow = func() time.Time { return now }
	const id = "ap-0123456789ab"
	if err := saveGuardAllowOverlay(overlayPath, guardAllowOverlay{Approve: []string{id}, Expiry: map[string]string{id: guardAllowExpiryStamp(now.Add(time.Minute))}}); err != nil {
		t.Fatal(err)
	}
	w := newGuardAllowWatcher(time.Hour, serveOperatorApprovalReloader(), nil)
	w.signature = guardOperatorApprovalLayersSignature
	if event := w.Refresh(context.Background()); !event.Reloaded {
		t.Fatalf("initial refresh reloaded=%t rejected=%t", event.Reloaded, event.Rejected)
	}
	now = now.Add(2 * time.Minute)
	if event := w.Reload(context.Background()); !event.Reloaded {
		t.Fatalf("expiry refresh reloaded=%t rejected=%t", event.Reloaded, event.Rejected)
	}
	if got := adjudicator.Default.PolicySnapshot().OperatorApprovals; len(got) != 0 {
		t.Fatalf("expired approvals=%v, want empty", got)
	}
}

// Only addition emits a grant; revocation and core-lock refusal remain audited.
// fak-test:runtime fast est=1s
func TestServeOperatorApprovalReloadAuditContract(t *testing.T) {
	prior := adjudicator.Default.PolicySnapshot()
	priorLock := guardCoreLockAllState.Load()
	journal.ResetActiveForTest()
	t.Cleanup(func() {
		adjudicator.Default.SetPolicy(prior)
		guardCoreLockAllState.Store(priorLock)
		journal.ResetActiveForTest()
	})
	dir := t.TempDir()
	overlayPath := filepath.Join(dir, "allow.json")
	t.Setenv(guardAllowOverlayEnv, overlayPath)
	t.Setenv(guardDenyOverlayEnv, filepath.Join(dir, "deny.json"))
	j, err := journal.Enable(filepath.Join(dir, "audit.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = j.Close() })
	adjudicator.Default.SetPolicy(adjudicator.Policy{Posture: adjudicator.PostureFailClosed})
	reload := serveOperatorApprovalReloader()
	const id = "ap-0123456789ab"
	if err := saveGuardAllowOverlay(overlayPath, guardAllowOverlay{Approve: []string{id}}); err != nil {
		t.Fatal(err)
	}
	if _, err := reload(context.Background()); err != nil {
		t.Fatal(err)
	}
	assertApprovalAuditRows(t, j.Recent(0), journal.ConfigSwapOK, 1)

	if err := saveGuardAllowOverlay(overlayPath, guardAllowOverlay{}); err != nil {
		t.Fatal(err)
	}
	if _, err := reload(context.Background()); err != nil {
		t.Fatal(err)
	}
	assertApprovalAuditRows(t, j.Recent(0), journal.ConfigSwapOK, 1)

	if err := saveGuardAllowOverlay(overlayPath, guardAllowOverlay{Approve: []string{id}}); err != nil {
		t.Fatal(err)
	}
	guardCoreLockAllState.Store(true)
	if _, err := reload(context.Background()); err == nil {
		t.Fatal("core-lock approval unexpectedly reloaded")
	}
	assertApprovalAuditRows(t, j.Recent(0), journal.ConfigSwapRejected, 1)
}

func assertApprovalAuditRows(t *testing.T, rows []journal.Row, wantLastSwap string, wantGrants int) {
	t.Helper()
	grants := 0
	var lastSwap *journal.ConfigSwapRow
	for i := range rows {
		if rows[i].ConfigSwap != nil {
			lastSwap = rows[i].ConfigSwap
		}
		if rows[i].Grant != nil {
			grants++
			if rows[i].Grant.Knob != "OperatorApprovals" || rows[i].Grant.Channel != journal.GrantChannelOperatorEscalation || rows[i].Grant.Actor != "operator" {
				t.Fatalf("grant knob=%q channel=%q actor=%q", rows[i].Grant.Knob, rows[i].Grant.Channel, rows[i].Grant.Actor)
			}
		}
	}
	if lastSwap == nil || lastSwap.Outcome != wantLastSwap {
		if lastSwap == nil {
			t.Fatalf("missing CONFIG_SWAP; want outcome %s", wantLastSwap)
		}
		t.Fatalf("last CONFIG_SWAP outcome=%s kind=%s source=%s, want outcome %s", lastSwap.Outcome, lastSwap.Kind, lastSwap.Source, wantLastSwap)
	}
	if grants != wantGrants {
		t.Fatalf("CAPABILITY_GRANT count=%d, want %d; row_count=%d", grants, wantGrants, len(rows))
	}
}

type approvalStdioClient struct {
	t       *testing.T
	cmd     *exec.Cmd
	in      *bufio.Writer
	out     *bufio.Reader
	nextID  int
	initIDs int
}

func (c *approvalStdioClient) call(method string, params any) json.RawMessage {
	c.t.Helper()
	c.nextID++
	id := c.nextID
	frame, err := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": id, "method": method, "params": params})
	if err != nil {
		c.t.Fatal(err)
	}
	if _, err := c.in.Write(append(frame, '\n')); err != nil {
		c.t.Fatal(err)
	}
	if err := c.in.Flush(); err != nil {
		c.t.Fatal(err)
	}
	for {
		line, err := c.out.ReadBytes('\n')
		if err != nil {
			c.t.Fatalf("read MCP response: %v", err)
		}
		var envelope struct {
			ID     int             `json:"id"`
			Result json.RawMessage `json:"result"`
			Error  json.RawMessage `json:"error"`
		}
		if err := json.Unmarshal(line, &envelope); err != nil {
			c.t.Fatalf("decode MCP frame %q: %v", line, err)
		}
		if envelope.ID == 0 { // server notification; never reconnect or reinitialize
			continue
		}
		if envelope.ID != id {
			c.t.Fatalf("MCP response id=%d, want %d: %s", envelope.ID, id, line)
		}
		if len(envelope.Error) != 0 && string(envelope.Error) != "null" {
			c.t.Fatalf("MCP %s error: %s", method, envelope.Error)
		}
		if method == "initialize" {
			c.initIDs++
		}
		return envelope.Result
	}
}

func (c *approvalStdioClient) adjudicate(command string) gateway.SyscallResponse {
	c.t.Helper()
	result := c.call("tools/call", map[string]any{
		"name": "fak_adjudicate",
		"arguments": map[string]any{
			"tool":      "Bash",
			"arguments": map[string]any{"command": command},
		},
	})
	var toolResult struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.Unmarshal(result, &toolResult); err != nil || len(toolResult.Content) != 1 {
		c.t.Fatalf("decode tool result: err=%v result=%s", err, result)
	}
	var response gateway.SyscallResponse
	if err := json.Unmarshal([]byte(toolResult.Content[0].Text), &response); err != nil {
		c.t.Fatalf("decode adjudication response: %v; text=%s", err, toolResult.Content[0].Text)
	}
	return response
}

func waitApprovalVerdict(t *testing.T, c *approvalStdioClient, command, kind, reason string, deadline time.Time) gateway.SyscallResponse {
	t.Helper()
	var got gateway.SyscallResponse
	for {
		got = c.adjudicate(command)
		if got.Verdict.Kind == kind && (reason == "" || got.Verdict.Reason == reason) {
			return got
		}
		if time.Now().After(deadline) {
			t.Fatalf("%q kind=%s reason=%s by=%s approval_id=%q, want %s/%s", command, got.Verdict.Kind, got.Verdict.Reason, got.Verdict.By, got.Verdict.Detail[adjudicator.MetaOperatorApprovalID], kind, reason)
		}
		time.Sleep(100 * time.Millisecond)
	}
}

func buildFakeDOS(t *testing.T, dir string) string {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	source := filepath.Join(dir, "dos_fixture.go")
	helper := "package main\nimport \"fmt\"\nfunc main(){fmt.Print(`[{\"holder\":\"peer\",\"lane\":\"peer\",\"tree\":[\"protected/**\"]}]`)}\n"
	if err := os.WriteFile(source, []byte(helper), 0o600); err != nil {
		t.Fatal(err)
	}
	name := "dos"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	exe := filepath.Join(dir, name)
	cmd := exec.Command("go", "build", "-o", exe, source)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build DOS fixture: %v\n%s", err, out)
	}
	return exe
}

func approvalChildEnv(overrides ...string) []string {
	replace := make(map[string]bool, len(overrides))
	for _, entry := range overrides {
		key, _, _ := strings.Cut(entry, "=")
		replace[strings.ToUpper(key)] = true
	}
	env := make([]string, 0, len(os.Environ())+len(overrides))
	for _, entry := range os.Environ() {
		key, _, _ := strings.Cut(entry, "=")
		if !replace[strings.ToUpper(key)] {
			env = append(env, entry)
		}
	}
	return append(env, overrides...)
}

// One real fak serve --stdio child, one initialize, and one continuously open
// stdin/stdout pair witness the user-visible regression. Overlay writes happen
// out of band, as an operator would make them. No MCP reconnect or process reload
// is permitted. A fake DOS read-only fixture supplies one peer-held protected/**
// lane so the same long-lived process proves approvals never lift LEASE_HELD.
// fak-test:runtime slow est=25s lane=optin
func TestServeStdioOperatorApprovalReloadWithoutReconnect(t *testing.T) {
	if testing.Short() {
		t.Skip("builds and launches the real fak binary")
	}
	root := repoRootForDoctorTest(t)
	state := t.TempDir()
	suffix := ""
	if runtime.GOOS == "windows" {
		suffix = ".exe"
	}
	fakExe := filepath.Join(state, "fak-approval-reload"+suffix)
	build := exec.Command("go", "build", "-o", fakExe, "./cmd/fak")
	build.Dir = root
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build real fak binary: %v\n%s", err, out)
	}
	fakeDOS := buildFakeDOS(t, filepath.Join(state, "dos-bin"))
	_ = fakeDOS // the executable's directory is prepended to PATH below

	workspace := filepath.Join(state, "workspace")
	if err := os.MkdirAll(filepath.Join(workspace, "protected"), 0o755); err != nil {
		t.Fatal(err)
	}
	gitInit := exec.Command("git", "init", "-q", workspace)
	if out, err := gitInit.CombinedOutput(); err != nil {
		t.Fatalf("git init fixture: %v\n%s", err, out)
	}
	overlayPath := filepath.Join(state, "allow.json")
	denyPath := filepath.Join(state, "deny.json")
	policyPath := filepath.Join(root, "cmd", "fak", "guard-default-policy.json")

	ctx, cancel := context.WithTimeout(context.Background(), 35*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, fakExe, "serve", "--stdio", "--policy", policyPath)
	cmd.Dir = workspace
	pathEnv := filepath.Dir(fakeDOS) + string(os.PathListSeparator) + os.Getenv("PATH")
	cmd.Env = approvalChildEnv(
		"PATH="+pathEnv,
		guardAllowOverlayEnv+"="+overlayPath,
		guardDenyOverlayEnv+"="+denyPath,
		guardAllowSessionIDEnv+"=",
		"FAK_LEASEPLANE_DIR="+workspace,
		"FAK_SESSION_REGISTRY="+filepath.Join(state, "sessions.json"),
		"HOME="+filepath.Join(state, "home"),
		"USERPROFILE="+filepath.Join(state, "home"),
	)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	pid := cmd.Process.Pid
	client := &approvalStdioClient{t: t, cmd: cmd, in: bufio.NewWriter(stdin), out: bufio.NewReader(stdout)}
	client.call("initialize", map[string]any{"protocolVersion": "2025-03-26", "capabilities": map[string]any{}, "clientInfo": map[string]any{"name": "approval-reload", "version": "1"}})

	const disjoint = "rm -rf build"
	const peerHeld = "rm -rf protected"
	first := client.adjudicate(disjoint)
	peer := client.adjudicate(peerHeld)
	if first.Verdict.Kind != "DENY" || first.Verdict.Reason != "POLICY_BLOCK" || peer.Verdict.Kind != "DENY" {
		t.Fatalf("initial disjoint=%s/%s/%s peer=%s/%s/%s", first.Verdict.Kind, first.Verdict.Reason, first.Verdict.By, peer.Verdict.Kind, peer.Verdict.Reason, peer.Verdict.By)
	}
	id := first.Verdict.Detail[adjudicator.MetaOperatorApprovalID]
	peerID := peer.Verdict.Detail[adjudicator.MetaOperatorApprovalID]
	if !guardOperatorApprovalIDRe.MatchString(id) || !guardOperatorApprovalIDRe.MatchString(peerID) {
		t.Fatalf("approval ids: disjoint=%q peer=%q", id, peerID)
	}

	// Unknown and malformed documents that mention an approval must not change
	// the last-good (unapproved) floor.
	if err := os.WriteFile(overlayPath, []byte(fmt.Sprintf(`{"version":%q,"approve":[%q],"unknown":true}`, guardAllowOverlayVersion, id)), 0o600); err != nil {
		t.Fatal(err)
	}
	time.Sleep(1200 * time.Millisecond)
	if got := client.adjudicate(disjoint); got.Verdict.Kind != "DENY" {
		t.Fatalf("unknown-field overlay kind=%s reason=%s by=%s", got.Verdict.Kind, got.Verdict.Reason, got.Verdict.By)
	}
	if err := os.WriteFile(overlayPath, []byte(`{"version":`), 0o600); err != nil {
		t.Fatal(err)
	}
	time.Sleep(1200 * time.Millisecond)
	if got := client.adjudicate(disjoint); got.Verdict.Kind != "DENY" {
		t.Fatalf("malformed overlay kind=%s reason=%s by=%s", got.Verdict.Kind, got.Verdict.Reason, got.Verdict.By)
	}

	// The same initialized child observes repair; peer LEASE_HELD remains final.
	if err := saveGuardAllowOverlay(overlayPath, guardAllowOverlay{Approve: []string{id, peerID}}); err != nil {
		t.Fatal(err)
	}
	waitApprovalVerdict(t, client, disjoint, "ALLOW", "", time.Now().Add(3*time.Second))
	waitApprovalVerdict(t, client, peerHeld, "DENY", "LEASE_HELD", time.Now().Add(2*time.Second))

	// Revocation is live.
	if err := saveGuardAllowOverlay(overlayPath, guardAllowOverlay{}); err != nil {
		t.Fatal(err)
	}
	waitApprovalVerdict(t, client, disjoint, "DENY", "POLICY_BLOCK", time.Now().Add(3*time.Second))

	if cmd.Process.Pid != pid || client.initIDs != 1 {
		t.Fatalf("MCP lifecycle changed: pid %d->%d initialize_count=%d", pid, cmd.Process.Pid, client.initIDs)
	}
	if err := stdin.Close(); err != nil {
		t.Fatal(err)
	}
	if err := cmd.Wait(); err != nil {
		t.Fatalf("stdio child exit: %v\nstderr:\n%s", err, stderr.String())
	}
}
