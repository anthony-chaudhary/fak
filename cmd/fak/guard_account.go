package main

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accountobs"
	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// guard_account.go — the ACCOUNT/NODE half of the `fak guard` exit summary. The
// harness-resources line (internal/harnessres) already answers "how loaded is the
// NODE?"; this file answers "WHICH account did this session spend, on WHICH node,
// and how loaded is that account now?" — the identity fak resolved locally
// (WITNESSED: the Claude config home, login posture, auth path) joined with the
// rate-limit / usage headers the provider relayed on every upstream response
// (OBSERVED, folded by internal/accountobs through the gateway's
// UpstreamResponseObserver seam). Split out of guard.go to keep the dispatch file
// under the steerability ceiling.

// guardAccountReport carries a guarded session's account/node identity plus the
// live upstream-response tracker. Built once in cmdGuard (both the proxy and the
// local --gguf branches) and threaded to finishGuardChildAndReport for the exit
// line. All methods are nil-safe so a path that never built one stays quiet.
type guardAccountReport struct {
	tracker *accountobs.Tracker
	// provider is the upstream wire ("anthropic", "openai", …).
	provider string
	// auth is the short human label for HOW this session authenticated upstream —
	// which account its usage accrued to (subscription OAuth vs API key vs
	// passthrough vs a local model with no account at all).
	auth string
	// home is the base name of the Claude config home (CLAUDE_CONFIG_DIR) whose
	// credential fak pinned — the fleet's per-account identity — "" when unknown.
	home string
	// login is the config home's login posture at launch ("" when not probed).
	login accounts.LoginStatus
}

// newGuardAccountReport folds the resolved upstream posture into the report. Kept
// apart from resolveGuardUpstream so the posture struct stays a pure resolver
// output and this stays a pure fold (unit-testable without touching disk).
func newGuardAccountReport(us guardUpstream) *guardAccountReport {
	auth := ""
	switch {
	case us.remoteServe:
		auth = "remote fak serve (no provider account)"
	case us.pinUpstream:
		auth = "Claude Pro/Max subscription (OAuth)"
	case us.apiKey != "":
		auth = "API key"
	case us.passthroughFallback:
		auth = "passthrough (child's own credential)"
	default:
		auth = "passthrough"
	}
	home := ""
	if d := strings.TrimSpace(us.claudeConfigDir); d != "" {
		home = filepath.Base(strings.TrimRight(d, string(os.PathSeparator)))
	}
	return &guardAccountReport{
		tracker:  accountobs.New(),
		provider: us.provider,
		auth:     auth,
		home:     home,
		login:    us.loginStatus,
	}
}

// newGuardAccountReportLocal is the --gguf branch's report: a local in-kernel model
// has NO upstream account, so the line carries only the node identity and says so
// honestly rather than rendering an empty account.
func newGuardAccountReportLocal(provider string) *guardAccountReport {
	return &guardAccountReport{provider: provider, auth: "local in-kernel model (no upstream account)"}
}

// observer is the gateway Config.UpstreamResponseObserver this report feeds from.
// nil when there is no tracker (the local branch), which the gateway treats as
// "seam off".
func (r *guardAccountReport) observer() func(status int, header http.Header) {
	if r == nil || r.tracker == nil {
		return nil
	}
	return r.tracker.Observe
}

// metricsText renders the live fak_account_* gauge block for /metrics. Empty when
// nothing was observed.
func (r *guardAccountReport) metricsText() string {
	if r == nil || r.tracker == nil {
		return ""
	}
	return r.tracker.Snapshot().PrometheusText()
}

// summaryLine renders the exit-summary account/node line, "" when there is nothing
// honest to say (no report at all). Identity fields are WITNESSED (fak resolved
// them locally); the usage half carries accountobs's own OBSERVED label. now
// anchors the reset rendering so the line is unit-testable.
func (r *guardAccountReport) summaryLine(now time.Time) string {
	if r == nil {
		return ""
	}
	var b strings.Builder
	b.WriteString("account/node — ")
	if r.home != "" {
		fmt.Fprintf(&b, "account %q", r.home)
		if r.login != "" {
			fmt.Fprintf(&b, " (login %s)", r.login)
		}
		b.WriteString(", ")
	}
	fmt.Fprintf(&b, "auth %s", r.auth)
	host, _ := os.Hostname()
	if host != "" {
		fmt.Fprintf(&b, ", node %s (%d cores)", host, runtime.NumCPU())
	}
	if r.tracker != nil {
		if usage := r.tracker.Snapshot().Report(now); usage != "" {
			b.WriteString("; ")
			b.WriteString(usage)
		} else {
			b.WriteString("; no upstream responses observed")
		}
	}
	return b.String()
}
