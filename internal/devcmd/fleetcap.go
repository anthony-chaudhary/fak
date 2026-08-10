package devcmd

import (
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/fleetcap"
)

// RunFleetcap is the dry-run fleet-capacity lens: it turns a target issue-
// resolution rate and a median session duration into the concurrent workers
// Little's law requires (L = lambda*W), and — when the operator supplies the
// available concurrency (a host cap, a seat count, or an explicit ceiling) — judges
// whether the fleet can sustain that rate (SUFFICIENT vs UNDER_CAPACITY + the
// shortfall to close). It is one leaf of the safe 400 GitHub issues/hour throughput
// program (fleet-400iph). It launches, counts, and observes NO real worker: pure
// planning arithmetic, safe to run at any time — which is exactly why it belongs to
// the development artifact rather than the served runtime (#6022).
func RunFleetcap(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fleetcap", flag.ContinueOnError)
	fs.SetOutput(stderr)
	rate := fs.Float64("rate", 400, "target issue-resolution rate in issues/hour")
	session := fs.Float64("session", 10, "median agent-session duration in minutes (used for the assessment line)")
	hostCap := fs.Int("cap", 0, "host concurrency cap — an available-worker ceiling (0 = unset)")
	seats := fs.Int("seats", 0, "account seat inventory — an available-worker ceiling (0 = unset)")
	available := fs.Int("available", 0, "explicit available concurrent workers; overrides --cap/--seats when > 0")
	asJSON := fs.Bool("json", false, "emit the capacity table (and assessment, if any) as JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}

	table := fleetcap.Table(*rate)

	// Resolve the available concurrency: an explicit --available wins, else fold the
	// supplied ceilings (cap/seats) to their tightest. Only assess when we have one.
	avail := *available
	if avail <= 0 {
		avail = fleetcap.AvailableFrom(*hostCap, *seats)
	}
	var est *fleetcap.Estimate
	if avail > 0 {
		e := fleetcap.Assess(*rate, *session, avail)
		est = &e
	}

	if *asJSON {
		out := map[string]any{
			"target_rate_per_hour": *rate,
			"table":                table,
		}
		if est != nil {
			out["assessment"] = est
		}
		if err := writeIndentedJSON(stdout, out); err != nil {
			fmt.Fprintf(stderr, "fleetcap: encode json: %v\n", err)
			return 1
		}
		return 0
	}

	fmt.Fprint(stdout, fleetcap.Render(*rate))
	if est != nil {
		fmt.Fprintf(stdout, "\nassessment  %s\n", est.Line())
	}
	return 0
}
