package main

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// info_endpoints.go — rendering the live accounts+nodes block (the fak guard "status
// area") in the `fak info` pane: the one-line status-line summary, and the visual panel
// showing WHICH Claude seats and WHICH serving nodes THIS session is using. All pure
// (SessionEndpoints in, strings out); the data comes from /debug/vars, so it stays
// payload-free.

// account-chip glyphs: the active seat (serving turns now), a walled seat (a 403 forced
// a failover to skip it this session), and an available seat.
const (
	guardChipActive = "●"
	guardChipWalled = "⊘"
	guardChipIdle   = "○"
)

// guardInfoEndpointsSummary is the compact status-line fragment: how many seats (with
// the active one named) and how many nodes. Empty when there is nothing to show (no
// provider, or a fak serve gateway), so the line stays clean.
func guardInfoEndpointsSummary(ep *gateway.SessionEndpoints) string {
	if ep == nil {
		return ""
	}
	var parts []string
	if n := len(ep.Accounts); n > 0 {
		s := fmt.Sprintf("accts %d", n)
		if active := guardActiveAccountName(ep.Accounts); active != "" {
			s += " (active " + active + ")"
		}
		parts = append(parts, s)
	}
	if n := len(ep.Nodes); n > 0 {
		parts = append(parts, fmt.Sprintf("nodes %d", n))
	}
	return strings.Join(parts, " · ")
}

// guardActiveAccountName returns the name of the seat currently serving turns, or "".
func guardActiveAccountName(accts []gateway.SessionAccount) string {
	for _, a := range accts {
		if a.Active {
			return a.Name
		}
	}
	return ""
}

// guardInfoEndpointsPanelRows is the accounts+nodes sub-pane. Full form is the roster +
// chips + the node list; mini is a one-row summary. Silent (nil) when the gateway
// reported no endpoints block (a fak serve gateway) — a silent panel costs zero rows.
func guardInfoEndpointsPanelRows(ctx guardInfoPanelCtx, level guardInfoPanelLevel) []string {
	ep := ctx.v.Endpoints
	if ep == nil || (len(ep.Accounts) == 0 && len(ep.Nodes) == 0) {
		return nil
	}
	if level == guardPanelMini {
		return []string{" endpoints " + guardInfoEndpointsMiniText(ep)}
	}
	var rows []string
	if len(ep.Accounts) > 0 {
		rows = append(rows, " accounts "+guardInfoAccountsHeadText(ep.Accounts))
		rows = append(rows, "          "+guardInfoAccountChipsText(ep.Accounts))
	}
	rows = append(rows, guardInfoNodeRows(ep.Nodes)...)
	return rows
}

// guardInfoEndpointsMiniText is the one-row fold: N seats (active X) · M nodes.
func guardInfoEndpointsMiniText(ep *gateway.SessionEndpoints) string {
	var parts []string
	if n := len(ep.Accounts); n > 0 {
		s := fmt.Sprintf("%d seats", n)
		if active := guardActiveAccountName(ep.Accounts); active != "" {
			s += " (active " + active + ")"
		}
		parts = append(parts, s)
	}
	if n := len(ep.Nodes); n > 0 {
		parts = append(parts, fmt.Sprintf("%d nodes", n))
	}
	return strings.Join(parts, " · ")
}

// guardInfoAccountsHeadText is the accounts header value: seat count + the active seat
// with its login readiness, plus a walled count when any seat is walled this session.
func guardInfoAccountsHeadText(accts []gateway.SessionAccount) string {
	walled := 0
	var active *gateway.SessionAccount
	for i := range accts {
		if accts[i].Active {
			active = &accts[i]
		}
		if accts[i].Walled {
			walled++
		}
	}
	head := fmt.Sprintf("%d seats", len(accts))
	if active != nil {
		head += " · active " + active.Name + " (" + guardLoginWord(active.LoginStatus) + ")"
	}
	if walled > 0 {
		head += fmt.Sprintf(" · %d walled", walled)
	}
	return head
}

// guardInfoAccountChipsText renders the seat roster as chips: ● active, ⊘ walled, ○ idle,
// with a trailing "(walled)"/login-wall note on a non-ready seat.
func guardInfoAccountChipsText(accts []gateway.SessionAccount) string {
	chips := make([]string, 0, len(accts))
	for _, a := range accts {
		glyph := guardChipIdle
		suffix := ""
		switch {
		case a.Active:
			glyph = guardChipActive
		case a.Walled:
			glyph = guardChipWalled
			suffix = " (walled)"
		}
		if suffix == "" && !a.CanServe && a.LoginStatus != "" && a.LoginStatus != string("ready") {
			suffix = " (" + guardLoginWord(a.LoginStatus) + ")"
		}
		chips = append(chips, glyph+a.Name+suffix)
	}
	return strings.Join(chips, "  ")
}

// guardInfoNodeRows renders the nodes: the kernel node + serving node(s). The first
// serving node shares the "nodes" gutter row; extra serving nodes get continuation rows.
func guardInfoNodeRows(nodes []gateway.SessionNode) []string {
	if len(nodes) == 0 {
		return nil
	}
	var kernel string
	var serving []gateway.SessionNode
	for _, n := range nodes {
		if n.Role == "kernel" {
			kernel = n.ID
			continue
		}
		serving = append(serving, n)
	}
	head := " nodes    "
	if kernel != "" {
		head += "kernel " + kernel
	}
	if len(serving) > 0 {
		if kernel != "" {
			head += " · "
		}
		head += "serving " + guardNodeText(serving[0])
	}
	rows := []string{head}
	for _, n := range serving[1:] {
		rows = append(rows, "          +"+guardNodeText(n))
	}
	return rows
}

// guardNodeText renders one serving node: its id plus a short kind/detail tag.
func guardNodeText(n gateway.SessionNode) string {
	s := n.ID
	tag := guardNodeKindWord(n.Kind)
	if d := strings.TrimSpace(n.Detail); d != "" && n.Kind == "proxy" {
		tag = d // the proxy provider name reads better than the generic "proxy"
	}
	if tag != "" {
		s += " (" + tag + ")"
	}
	return s
}

func guardNodeKindWord(kind string) string {
	switch kind {
	case "remote-serve":
		return "remote fak serve"
	case "local-server":
		return "local server"
	case "in-kernel":
		return "in-kernel"
	case "proxy":
		return "proxy"
	default:
		return ""
	}
}

// guardInfoHarnessText renders the guard harness half (kernel process CPU/RSS/IO/net,
// plus GPU VRAM when in-kernel) as one compact row — the live twin of the exit summary's
// resource line. Empty when the block is absent or has nothing worth showing.
func guardInfoHarnessText(h *gateway.SessionHarness) string {
	if h == nil {
		return ""
	}
	var parts []string
	switch {
	case h.KernelCPUPercent > 0:
		parts = append(parts, fmt.Sprintf("cpu %.0f%%", h.KernelCPUPercent))
	case h.KernelCPUSeconds > 0:
		parts = append(parts, fmt.Sprintf("cpu %.1fs", h.KernelCPUSeconds))
	}
	if h.KernelRSSBytes > 0 {
		parts = append(parts, "rss "+guardInfoBytesText(h.KernelRSSBytes))
	}
	if h.KernelIOReadBytes > 0 || h.KernelIOWriteBytes > 0 {
		parts = append(parts, "io "+guardInfoBytesText(h.KernelIOReadBytes)+"r/"+guardInfoBytesText(h.KernelIOWriteBytes)+"w")
	}
	if h.NetRxBytes > 0 || h.NetTxBytes > 0 {
		parts = append(parts, "net "+guardInfoBytesText(h.NetRxBytes)+"↓/"+guardInfoBytesText(h.NetTxBytes)+"↑")
	}
	if h.HaveGPU && h.GPUVRAMTotalBytes > 0 {
		parts = append(parts, "vram "+guardInfoBytesText(h.GPUVRAMUsedBytes)+"/"+guardInfoBytesText(h.GPUVRAMTotalBytes))
	}
	return strings.Join(parts, " · ")
}

// guardLoginWord shortens the login-status token for the pane; passes through the raw
// value for anything it does not special-case.
func guardLoginWord(status string) string {
	switch status {
	case "ready":
		return "ready"
	case "needs_login":
		return "needs login"
	case "identity_mismatch":
		return "identity mismatch"
	case "missing_dir":
		return "no dir"
	case "":
		return "unknown"
	default:
		return status
	}
}
