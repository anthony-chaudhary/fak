package benchcatalog

import (
	"fmt"
	"strings"
)

// OnboardState is the single, non-conflating answer to "where does this bench
// node ACTUALLY stand?" A binary reachable/unreachable probe is not enough: the
// fix for a box with no sshd is fundamentally different from the fix for a box
// whose sshd answers but has not authorized the driver key. Conflating the two
// sends every future agent down the wrong remediation path (issue #10: the
// premise "the desktop ships with no sshd" was falsified on the real host  -
// sshd was Running and :22 answered; the only remaining gap was the key).
//
// A node is in EXACTLY one of these states:
type OnboardState string

const (
	// OnboardNoSSHD: TCP :22 never answered, or sshd's banner never arrived -
	// connection refused / timed out / reset / no route. There is no sshd to
	// authenticate against, so the fix is "start sshd", not "add a key".
	OnboardNoSSHD OnboardState = "no-sshd"

	// OnboardSSHDOnly: sshd answered the handshake, but the driver key is not in
	// the account's authorized_keys - ssh reports "Permission denied
	// (publickey...)". The fix is "add the driver pubkey", NOT "start sshd".
	OnboardSSHDOnly OnboardState = "sshd-only"

	// OnboardAuthorized: the driver key authenticated and the remote command ran.
	OnboardAuthorized OnboardState = "authorized"

	// OnboardUnknown: evidence is ambiguous or absent (e.g. a host-key mismatch,
	// a non-auth denial, an unrecognized exit). Never collapse this into
	// sshd-only/no-sshd  -  an honest unknown is what stops a wrong fix.
	OnboardUnknown OnboardState = "unknown"
)

// ClassifyOnboardState maps an ssh probe's (exit code, stderr) to exactly one
// OnboardState. It is the authoritative, unit-tested spec of the classifier;
// tools/bench_node.sh mirrors this exact mapping in bash (its `diagnose`
// subcommand), and this function's test is the lock against silent drift.
//
// exitCode is the exit status of the probe (0 = the remote command ran).
// stderr is the probe's captured stderr (stdout is ignored; ssh reports auth
// failures on stderr). The signature is forward-compatible: callers may pass the
// combined output in stderr and rely only on the substrings below.
//
// Rules, in order:
//  1. exit 0                     -> authorized
//  2. stderr contains sshd's publickey-denial marker -> sshd-only
//  3. stderr contains a connect/banner failure marker -> no-sshd
//  4. otherwise                  -> unknown
//
// The order matters: an auth denial proves sshd answered, so a publickey denial
// must be tested BEFORE the connect markers (a wrapped multiplexer or proxy can
// emit both a connect notice and the denial).
func ClassifyOnboardState(exitCode int, stderr string) OnboardState {
	if exitCode == 0 {
		return OnboardAuthorized
	}
	low := strings.ToLower(stderr)
	if isPublickeyDenial(low) {
		return OnboardSSHDOnly
	}
	if isConnectFailure(low) {
		return OnboardNoSSHD
	}
	return OnboardUnknown
}

// publickeyDenialMarkers are the strings OpenSSH emits when sshd answered and
// rejected the key. "permission denied (publickey" is the load-bearing prefix;
// the bare "permission denied" is deliberately NOT matched (a remote command
// exiting 1 can print that without an auth failure ever occurring), so only the
// publickey form - which can only come from sshd's auth exchange - qualifies.
var publickeyDenialMarkers = []string{
	"permission denied (publickey",
	"too many authentication failures",
}

// connectFailureMarkers are the strings ssh emits when it never reached a working
// sshd: TCP refused, timed out, the banner never arrived, or the host could not
// be reached at all. A node answering ICMP but with no sshd lands here (issue
// #10's original - now falsified - premise), and so does a tailnet-offline peer.
var connectFailureMarkers = []string{
	"connection refused",
	"connection timed out",
	"connection closed",
	"operation timed out",
	"no route to host",
	"network is unreachable",
	"host is down",
	"connection reset",
	"broken pipe",
	"connection to", // e.g. "ssh: connect to host x port 22: ..."
	"kex_exchange_identification",
	"banner exchange",
	"port 22:",
}

func isPublickeyDenial(low string) bool {
	for _, m := range publickeyDenialMarkers {
		if strings.Contains(low, m) {
			return true
		}
	}
	return false
}

func isConnectFailure(low string) bool {
	for _, m := range connectFailureMarkers {
		if strings.Contains(low, m) {
			return true
		}
	}
	return false
}

// OnboardRemedy is the one-line, state-specific fix a human (or agent) should
// apply next. It is what makes the classification actionable rather than a
// label: the whole point of separating sshd-only from no-sshd is that they get
// DIFFERENT remedies.
func (s OnboardState) Remedy() string {
	switch s {
	case OnboardNoSSHD:
		return "start sshd on the node and confirm :22 answers (a TCP probe, not ICMP) - the driver key is not the problem yet"
	case OnboardSSHDOnly:
		return "sshd is up; append the driver pubkey to the node account's authorized_keys, then re-run diagnose"
	case OnboardAuthorized:
		return "no action - the driver key authenticates and the remote command runs"
	default:
		return "inspect the raw ssh stderr; the classifier could not tell a connect failure from an auth denial"
	}
}

// OnboardStateLine renders the single machine-greppable line the runner prints.
// Format is stable so callers can `grep '^ONBOARD-STATE'`.
func OnboardStateLine(node string, s OnboardState) string {
	return fmt.Sprintf("ONBOARD-STATE %s %s", node, s)
}
