package accounts

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// keychain_state.go — the typed outcome of one Keychain probe (fak-private#3063). The
// historical reader answered a bare ok=false for three very different situations, and
// every one of them surfaced as the same "needs_login":
//
//   - the item does not exist (`security` exits 44) — a real "log in" answer;
//   - the read timed out — on an unattended box that is almost always a securityd ACL
//     consent prompt nobody is there to click, and `/login` does not fix it;
//   - any other failure (locked keychain, bad argv, exec failure).
//
// The `security` exec itself lives behind keychainExecRunner so the classification is
// platform-neutral and unit-tested everywhere with a fake runner; only
// keychain_darwin.go wires the real exec into claudeKeychainReadPassword.

// KeychainState is the closed vocabulary for one keychain probe's outcome.
type KeychainState string

const (
	// KeychainFound: the item exists and carried a usable value.
	KeychainFound KeychainState = "found"
	// KeychainUnsupported: this build has no keychain seam (every non-darwin GOOS).
	KeychainUnsupported KeychainState = "unsupported"
	// KeychainMissing: no such item (`security` exit 44).
	KeychainMissing KeychainState = "missing"
	// KeychainTimeout: the read exceeded its deadline — typically a securityd ACL prompt
	// on an unattended box. The item very likely exists; fak is just not trusted yet.
	KeychainTimeout KeychainState = "timeout"
	// KeychainError: any other read failure.
	KeychainError KeychainState = "error"
	// KeychainNoToken: the item exists but its value is a placeholder / mis-shaped.
	KeychainNoToken KeychainState = "no_token"
	// KeychainExpired: the item's OAuth credential carries a recorded expiry in the past.
	KeychainExpired KeychainState = "expired"
)

// keychainStateRank orders miss states by how much they tell an operator, so a probe
// across several candidate services reports the most informative miss: a timeout on one
// service (the item is probably there behind an ACL) outranks a plain missing on another.
var keychainStateRank = map[KeychainState]int{
	KeychainUnsupported: 0,
	KeychainMissing:     1,
	KeychainNoToken:     2,
	KeychainExpired:     3,
	KeychainError:       4,
	KeychainTimeout:     5,
	KeychainFound:       6,
}

var (
	// ErrKeychainItemNotFound is what the exec seam returns for `security` exit 44.
	ErrKeychainItemNotFound = errors.New("keychain item not found")
	// ErrKeychainTimeout is what the exec seam returns when the read hit its deadline.
	ErrKeychainTimeout = errors.New("keychain read timed out (likely a securityd ACL prompt with nobody to answer it)")
)

// KeychainProbe is the credential-safe result of one keychain lookup: the typed state,
// the service it concerns, and a non-secret detail (exit code, macOS version). It never
// carries the item's value.
type KeychainProbe struct {
	State   KeychainState `json:"state"`
	Service string        `json:"service,omitempty"`
	Detail  string        `json:"detail,omitempty"`
}

// OK reports whether the probe found a usable value.
func (p KeychainProbe) OK() bool { return p.State == KeychainFound }

func (p KeychainProbe) worse(q KeychainProbe) KeychainProbe {
	if keychainStateRank[q.State] > keychainStateRank[p.State] {
		return q
	}
	return p
}

// withOSVersion stamps the macOS version onto a miss so a keychain diagnostic names the
// OS it was observed on (Keychain/securityd behavior differs across releases). A found
// or unsupported probe is returned unchanged.
func (p KeychainProbe) withOSVersion() KeychainProbe {
	if p.State == KeychainFound || p.State == KeychainUnsupported {
		return p
	}
	v := keychainOSVersion()
	if v == "" {
		return p
	}
	if p.Detail == "" {
		p.Detail = "macOS " + v
	} else {
		p.Detail += "; macOS " + v
	}
	return p
}

// Reason renders the operator-facing sentence for a non-found probe, naming the repair
// that actually fits the failure instead of a blanket "log in".
func (p KeychainProbe) Reason() string {
	var s string
	switch p.State {
	case KeychainFound:
		return "keychain item found"
	case KeychainUnsupported:
		return "no keychain on this platform"
	case KeychainMissing:
		s = "keychain item is missing"
	case KeychainTimeout:
		s = "keychain read timed out — most likely a securityd access prompt nobody answered"
	case KeychainError:
		s = "keychain read failed"
	case KeychainNoToken:
		s = "keychain item holds no usable credential"
	case KeychainExpired:
		s = "keychain credential has expired"
	default:
		s = "keychain state " + string(p.State)
	}
	if p.Service != "" {
		s += " (service " + p.Service + ")"
	}
	if p.Detail != "" {
		s += ": " + p.Detail
	}
	return s
}

// classifyKeychainErr maps an exec-seam error onto the typed state plus a non-secret
// detail. A seam that returns a plain error (not one of the sentinels) reads as
// KeychainError, never silently as missing.
func classifyKeychainErr(err error) (KeychainState, string) {
	switch {
	case err == nil:
		return KeychainFound, ""
	case errors.Is(err, ErrKeychainItemNotFound):
		return KeychainMissing, ""
	case errors.Is(err, ErrKeychainTimeout):
		return KeychainTimeout, ""
	default:
		return KeychainError, err.Error()
	}
}

// securityExitItemNotFound is `security find-generic-password`'s exit status for "The
// specified item could not be found in the keychain." (errSecItemNotFound).
const securityExitItemNotFound = 44

// claudeKeychainReadTimeout bounds one `security` exec. An item that does not exist
// returns in milliseconds with no prompt; the slow case is an item whose ACL makes
// securityd raise the GUI consent dialog (fak is not in the ACL Claude Code created, so
// the FIRST read on a box prompts until the operator clicks "Always Allow"). 15s gives
// an attended operator time to click; an unattended box times out and the probe reports
// KeychainTimeout — never a wedge.
const claudeKeychainReadTimeout = 15 * time.Second

// keychainExecResult is what one exec of an external tool produced, reduced to the facts
// the classifier needs. A fake runner builds it directly, so tests never exec `security`.
type keychainExecResult struct {
	Out      []byte
	ExitCode int
	TimedOut bool
	Err      error
}

// keychainExecRunner execs name+args under ctx.
type keychainExecRunner func(ctx context.Context, name string, args ...string) keychainExecResult

// execKeychainRunner is the real runner (wired only on darwin).
func execKeychainRunner(ctx context.Context, name string, args ...string) keychainExecResult {
	out, err := exec.CommandContext(ctx, name, args...).Output()
	res := keychainExecResult{Out: out, Err: err}
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		res.TimedOut = true
	}
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		res.ExitCode = ee.ExitCode()
	}
	return res
}

// securityReadPassword builds the claudeKeychainReadPassword seam over a runner: exec
// `/usr/bin/security find-generic-password [-a account] -w -s service` with a deadline
// and map the outcome onto the sentinel errors classifyKeychainErr understands. An empty
// account matches the service under any account (the fak-owned durable items).
func securityReadPassword(run keychainExecRunner, timeout time.Duration) func(service, account string) ([]byte, error) {
	return func(service, account string) ([]byte, error) {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		args := []string{"find-generic-password"}
		if account != "" {
			args = append(args, "-a", account)
		}
		// -w prints the password only (never item metadata), so the value can flow
		// straight into the parser; the absolute path pins Apple's binary.
		args = append(args, "-w", "-s", service)
		res := run(ctx, "/usr/bin/security", args...)
		switch {
		case res.TimedOut:
			return nil, fmt.Errorf("%w after %s", ErrKeychainTimeout, timeout)
		case res.Err == nil:
			return res.Out, nil
		case res.ExitCode == securityExitItemNotFound:
			return nil, ErrKeychainItemNotFound
		case res.ExitCode != 0:
			return nil, fmt.Errorf("security find-generic-password exited %d", res.ExitCode)
		default:
			return nil, fmt.Errorf("security find-generic-password: %w", res.Err)
		}
	}
}

// keychainOSVersionFunc is the macOS-version seam for keychain diagnostics. nil (every
// non-darwin build) means no version is recorded; keychain_darwin.go wires sw_vers.
var keychainOSVersionFunc func() string

var (
	keychainOSVersionOnce sync.Once
	keychainOSVersionVal  string
)

// keychainOSVersion returns the cached macOS product version, or "" when unknown.
func keychainOSVersion() string {
	f := keychainOSVersionFunc
	if f == nil {
		return ""
	}
	keychainOSVersionOnce.Do(func() { keychainOSVersionVal = f() })
	return keychainOSVersionVal
}

// swVersProductVersion runs `sw_vers -productVersion` through a runner, returning the
// trimmed version or "" on any failure (a diagnostic nicety, never load-bearing).
func swVersProductVersion(run keychainExecRunner) string {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	res := run(ctx, "/usr/bin/sw_vers", "-productVersion")
	if res.Err != nil || res.TimedOut {
		return ""
	}
	v := strings.TrimSpace(string(res.Out))
	if v == "" || strings.ContainsAny(v, " \t\n\r") {
		return ""
	}
	return v
}
