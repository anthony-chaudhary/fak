package main

import (
	"net"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/harnessres"
)

// guard_endpoints.go — builds the live accounts+nodes provider `fak guard` hands the
// gateway (SetSessionEndpointsProvider) so the status area (`fak info`) shows which
// Claude seats and which serving nodes THIS session is using. Kept out of the peer-hot
// guard.go, like guard_account_failover.go.
//
// The account roster is re-discovered on every /debug/vars pull so a mid-session
// account failover's active/walled marks stay live; the serving-node list is resolved
// once at boot from the already-decided upstream posture. Everything here is display
// metadata (seat names, an email, an endpoint host) — never a token.

// guardEndpointNodes captures the boot-resolved serving posture the node list is built
// from — the fields guard.go already has in scope where it wires the provider.
type guardEndpointNodes struct {
	provider     string // resolved upstream wire (anthropic / openai / ...)
	resolvedBase string // upstream base URL ("" for pure in-kernel)
	remoteServe  bool   // base came from --remote-serve (a lab `fak serve` box)
	localModel   bool   // --gguf: fak decodes in-kernel on this box
	localAlong   bool   // --gguf --alongside: in-kernel beside the API upstream
	localAlias   string // the --gguf alias a client addresses the local side by
}

// newGuardEndpointsProvider returns the pull provider for the /debug/vars endpoints
// block. activeDirFn returns the config dir of the seat currently serving turns (it
// follows a failover); walledFn returns the seats proven walled this session. Both are
// nil on a non-subscription session (no Claude seat is "in use"), leaving Accounts
// empty so only Nodes render. nodes is the boot-resolved serving posture.
func newGuardEndpointsProvider(activeDirFn func() string, walledFn func() map[string]bool, nodes guardEndpointNodes) func() gateway.SessionEndpoints {
	// The kernel node (this host) is stable for the session — resolve it once.
	kernelNode := gateway.SessionNode{
		Role:   "kernel",
		ID:     guardKernelNodeID(),
		Kind:   "host",
		Detail: "fak guard + agent + adjudication",
	}
	servingNodes := guardResolveServingNodes(nodes)
	allNodes := append([]gateway.SessionNode{kernelNode}, servingNodes...)
	return func() gateway.SessionEndpoints {
		return gateway.SessionEndpoints{
			Accounts: guardResolveAccounts(activeDirFn, walledFn),
			Nodes:    allNodes,
		}
	}
}

// guardResolveAccounts discovers the on-box Claude seat roster and marks each seat with
// how this session relates to it (active / walled / login readiness). Returns nil when
// no seat is in use (activeDirFn nil — a non-subscription session) or the roster cannot
// be read, so the accounts half of the block is simply absent then.
func guardResolveAccounts(activeDirFn func() string, walledFn func() map[string]bool) []gateway.SessionAccount {
	if activeDirFn == nil {
		return nil
	}
	home, err := os.UserHomeDir()
	if err != nil || strings.TrimSpace(home) == "" {
		return nil
	}
	roster, err := accounts.Discover(home)
	if err != nil || len(roster) == 0 {
		return nil
	}
	activeDir := activeDirFn()
	activeKey := accountKeyForDir(activeDir)
	activeClean := guardCleanDir(activeDir)
	var walled map[string]bool
	if walledFn != nil {
		walled = walledFn()
	}
	out := make([]gateway.SessionAccount, 0, len(roster))
	for _, h := range roster {
		key := h.Identity.AccountKey()
		// Prefer the account KEY (uuid:/tok:) to match the seat regardless of which dir
		// name it is logged in under; fall back to a cleaned-path compare when the seat
		// has no derivable identity (a half-set-up dir).
		active := (activeKey != "" && key == activeKey) ||
			(activeKey == "" && activeClean != "" && guardCleanDir(h.Dir) == activeClean)
		out = append(out, gateway.SessionAccount{
			Name:        h.Name,
			Email:       h.Identity.Email,
			Active:      active,
			Walled:      key != "" && walled[key],
			CanServe:    h.CanServe(),
			LoginStatus: string(h.LoginStatus()),
		})
	}
	return out
}

// guardResolveServingNodes turns the boot-resolved upstream posture into the serving
// node(s): the in-kernel box (--gguf), a proxied provider host, a --remote-serve lab
// box, or a detected local server. `--gguf --alongside` yields both the API host and
// the in-kernel node.
func guardResolveServingNodes(n guardEndpointNodes) []gateway.SessionNode {
	var out []gateway.SessionNode
	// Pure in-kernel (--gguf, no --alongside): the box itself is the serving node.
	if n.localModel && !n.localAlong {
		return []gateway.SessionNode{{Role: "serving", ID: "in-kernel", Kind: "in-kernel", Detail: guardNodeModelDetail(n.localAlias)}}
	}
	if host := guardHostFromBase(n.resolvedBase); host != "" {
		kind, detail := "proxy", strings.TrimSpace(n.provider)
		switch {
		case n.remoteServe:
			kind, detail = "remote-serve", "fak serve"
		case guardIsLoopbackHost(host):
			kind = "local-server"
		}
		out = append(out, gateway.SessionNode{Role: "serving", ID: host, Kind: kind, Detail: detail})
	}
	// --gguf --alongside: the in-kernel side rides beside the API upstream.
	if n.localModel && n.localAlong {
		out = append(out, gateway.SessionNode{Role: "serving", ID: "in-kernel", Kind: "in-kernel", Detail: guardNodeModelDetail(n.localAlias)})
	}
	return out
}

// guardKernelNodeID is this host's display id for the kernel node: the OS hostname, or
// a stable placeholder when it cannot be read. Kept simple (no hardware-catalog lookup)
// so the status area never depends on a repo-relative catalog file.
func guardKernelNodeID() string {
	if hostname := trimmedHostname(); hostname != "" {
		return hostname
	}
	return "this-host"
}

func trimmedHostname() string {
	hostname, err := os.Hostname()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(hostname)
}

// guardHostFromBase returns the host[:port] of an upstream base URL, or "" when the
// base is empty/unparseable. url.Parse handles the scheme; a bare host with no scheme
// falls back to the trimmed input's first path-free segment.
func guardHostFromBase(base string) string {
	base = strings.TrimSpace(base)
	if base == "" {
		return ""
	}
	if u, err := url.Parse(base); err == nil && u.Host != "" {
		return u.Host
	}
	// No scheme (e.g. "host:port/v1"): take up to the first '/'.
	if i := strings.IndexByte(base, '/'); i >= 0 {
		base = base[:i]
	}
	return base
}

// guardIsLoopbackHost reports whether host names the local machine (a detected local
// model server), so its node reads "local-server" rather than a remote "proxy".
func guardIsLoopbackHost(host string) bool {
	h := host
	// SplitHostPort strips a :port for IPv4/hostname and a bracketed [::1]:port; a bare
	// IPv6 like "::1" (many colons, no brackets) errors, so we keep it whole.
	if hh, _, err := net.SplitHostPort(host); err == nil {
		h = hh
	}
	h = strings.ToLower(strings.Trim(h, "[]"))
	return h == "127.0.0.1" || h == "localhost" || h == "::1" || strings.HasPrefix(h, "127.")
}

// guardNodeModelDetail labels an in-kernel serving node with the model alias when known.
func guardNodeModelDetail(alias string) string {
	if alias = strings.TrimSpace(alias); alias != "" {
		return "local model " + filepath.Base(alias)
	}
	return "local model"
}

// guardCleanDir normalizes a config-dir path for equality compares (Clean + drop a
// trailing separator). "" stays "".
func guardCleanDir(dir string) string {
	dir = strings.TrimSpace(dir)
	if dir == "" {
		return ""
	}
	return filepath.Clean(dir)
}

// guardHarnessToSession converts a live harnessres.Snapshot into the gateway's
// dependency-free SessionHarness shape for the /debug/vars harness block. Have* bits
// gate each axis so an unread axis stays zero/omitted rather than reading a false 0.
func guardHarnessToSession(snap harnessres.Snapshot) gateway.SessionHarness {
	out := gateway.SessionHarness{
		Samples:        snap.Samples,
		ElapsedSeconds: snap.Elapsed.Seconds(),
		GoroutinesPeak: snap.GoroutinesPeak,
		GoHeapSysBytes: snap.GoHeapSysBytes,
	}
	k := snap.Kernel
	if k.HaveCPU {
		out.KernelCPUSeconds = k.CPUSeconds()
		if pct, ok := k.CPUPercentAvg(snap.Elapsed); ok {
			out.KernelCPUPercent = pct
		}
	}
	if k.HaveRSS {
		out.KernelRSSBytes = k.RSSBytes
	}
	if k.HaveIO {
		out.KernelIOReadBytes = k.IOReadBytes
		out.KernelIOWriteBytes = k.IOWriteBytes
	}
	if k.HaveNet {
		out.NetRxBytes = k.NetRxBytes
		out.NetTxBytes = k.NetTxBytes
	}
	if snap.HaveGPU {
		out.GPUVRAMUsedBytes = snap.GPUVRAMUsedBytes
		out.GPUVRAMTotalBytes = snap.GPUVRAMTotalBytes
		out.HaveGPU = true
	}
	return out
}
