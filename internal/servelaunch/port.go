package servelaunch

import (
	"errors"
	"fmt"
	"net"
)

// Port claiming is fail-closed: a port is accepted only on a positively
// confirmed free result. Anything the prober cannot confirm — an error, a
// timeout, a non-positive answer — is treated as busy, because a harness
// that launches onto a held port corrupts the measurement it exists to take.

var (
	// ErrPortProbeInconclusive reports that a probe could not positively
	// confirm the port free; the port is treated as busy, never accepted.
	ErrPortProbeInconclusive = errors.New("servelaunch: port probe inconclusive; treating port as busy")
	// ErrNoFreePort reports that the bounded scan window drained without a
	// single positively-free port.
	ErrNoFreePort = errors.New("servelaunch: no positively-free port in scan window")
)

// DefaultWindow is the scan width when ClaimOptions.Window <= 0: ten ports,
// the pinned study's port..port+9 auto-selection span.
const DefaultWindow = 10

// Prober reports one port's state. Only (true, nil) is an actionable answer:
// a positive confirmation that the port is free. (false, nil) is a
// conclusive "held"; any error is inconclusive.
type Prober func(port int) (free bool, err error)

// ProbeFree is the fail-closed decision over one probe result: the port is
// accepted only on positive confirmation. A prober error surfaces as
// ErrPortProbeInconclusive (treated as busy); a conclusive "held" is plain
// not-free with no error.
func ProbeFree(p Prober, port int) (bool, error) {
	if p == nil {
		return false, fmt.Errorf("%w: port %d: no prober wired", ErrPortProbeInconclusive, port)
	}
	free, err := p(port)
	if err != nil {
		return false, fmt.Errorf("%w: port %d: %w", ErrPortProbeInconclusive, port, err)
	}
	return free, nil
}

// ClaimOptions bounds one claim scan.
type ClaimOptions struct {
	// Start is the first port to try, inclusive.
	Start int
	// Window is how many ports to try. <= 0 takes DefaultWindow.
	Window int
	// Prober confirms freeness. Nil uses ListenProber.
	Prober Prober
}

// ClaimOutcome records what the scan saw.
type ClaimOutcome struct {
	// Port is the claimed port; 0 when none was claimed.
	Port int
	// Busy lists the ports conclusively reported held.
	Busy []int
	// Inconclusive lists the ports whose probe errored; they were treated as
	// busy, never accepted.
	Inconclusive []int
}

// PortClaim scans [Start, Start+Window) and claims the first port the prober
// positively confirms free. Inconclusive probes are treated as busy and
// recorded. When the window drains without a positive confirmation the error
// always wraps ErrNoFreePort and additionally wraps
// ErrPortProbeInconclusive when any probe was inconclusive, so callers can
// tell "all genuinely held" from "the prober could not confirm anything" via
// errors.Is. Invalid windows are refused up front with a config error that
// matches neither sentinel.
func PortClaim(opts ClaimOptions) (*ClaimOutcome, error) {
	if opts.Window <= 0 {
		opts.Window = DefaultWindow
	}
	if opts.Start < 1 || opts.Start+opts.Window-1 > 65535 {
		return nil, fmt.Errorf("servelaunch: invalid scan window [%d, %d): ports must stay within 1..65535",
			opts.Start, opts.Start+opts.Window)
	}
	prober := opts.Prober
	if prober == nil {
		prober = ListenProber
	}

	out := &ClaimOutcome{}
	for i := 0; i < opts.Window; i++ {
		port := opts.Start + i
		free, err := ProbeFree(prober, port)
		switch {
		case err != nil:
			out.Inconclusive = append(out.Inconclusive, port)
		case free:
			out.Port = port
			return out, nil
		default:
			out.Busy = append(out.Busy, port)
		}
	}

	errs := []error{ErrNoFreePort}
	if len(out.Inconclusive) > 0 {
		errs = append(errs, ErrPortProbeInconclusive)
	}
	return out, fmt.Errorf("%w: scanned %d port(s) starting at %d (%d busy, %d inconclusive)",
		errors.Join(errs...), opts.Window, opts.Start, len(out.Busy), len(out.Inconclusive))
}

// ListenProber is the production prober: it tries to bind the port on the
// loopback interface. A successful bind positively confirms the port was
// free at bind time and is released immediately; the claimer is expected to
// launch the server that takes the port right away (the bind-to-launch gap
// is the intrinsic TOCTOU of any listen-based claim). Any bind failure is
// returned as an error, which PortClaim treats as inconclusive → busy.
func ListenProber(port int) (bool, error) {
	l, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		return false, err
	}
	_ = l.Close()
	return true, nil
}
