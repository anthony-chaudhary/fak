package main

// doctor_trust.go — `fak doctor trust`, the read-only answer to "why does this work
// in my shell and not inside fak" on a managed host (#8172).
//
// It is the check the two enterprise config bails point at: both
// UPSTREAM_TRUST_UNVERIFIED and UPSTREAM_UNSUPPORTED name `fak doctor trust` as
// their `check:` line, because a refusal is only as good as the read-only command
// that shows the operator what fak saw. What it reports:
//
//   - the trust source fak is validating with, and the CA subjects it carries;
//   - per upstream endpoint, whether the chain validates against the platform trust
//     store, whether it validates only once the declared bundle is appended, and the
//     CA the chain terminates at — the interceptor's own name;
//   - which sibling runtimes on this host already point at a corporate bundle while
//     fak does not (the no-network signal, and the one that catches the common case);
//   - which child runtimes would NOT inherit fak's trust source, computed from the
//     environment a wrapped child would ACTUALLY receive;
//   - a request-signed cloud model route, which no certificate can fix.
//
// Probing is on by default and bounded. An endpoint that cannot be reached at all is
// reported as "no verdict" rather than as a trust failure, so an offline, air-gapped,
// or firewalled host produces zero findings from this check instead of a false alarm.
// `--probe=false` makes the run strictly local.

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cloudroute"
	"github.com/anthony-chaudhary/fak/internal/httptrust"
	"github.com/anthony-chaudhary/fak/internal/secretload"
)

const doctorTrustSchema = "fak.doctor-trust.v1"

// doctorTrustProbeHost is the endpoint every run probes: the model upstream fak
// itself talks to, which is the connection whose failure the operator is actually
// debugging.
const doctorTrustProbeHost = "api.anthropic.com:443"

// doctorTrustRouteProbe adds the CONTROL-PLANE endpoint for a detected cloud route.
// It is deliberately a second host rather than a nicety: a site routinely runs more
// than one interceptor, and the proxy in front of the model endpoint is frequently
// not the one in front of the cloud control plane. A bundle that covers only the
// first is the failure mode that reads as "fak broke my network", so the check has
// to look at both.
var doctorTrustRouteProbe = map[cloudroute.Kind]string{
	cloudroute.KindBedrock: "sts.amazonaws.com:443",
	cloudroute.KindVertex:  "oauth2.googleapis.com:443",
}

type doctorTrustReport struct {
	Schema          string                  `json:"schema"`
	OK              bool                    `json:"ok"`
	Declared        bool                    `json:"declared"`
	Path            string                  `json:"path,omitempty"`
	Subjects        []string                `json:"subjects,omitempty"`
	Intercepted     bool                    `json:"intercepted"`
	Probes          []httptrust.ProbeResult `json:"probes,omitempty"`
	Recommendations []Recommendation        `json:"recommendations"`
	Findings        int                     `json:"findings"`
}

func runDoctorTrust(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak doctor trust", flag.ContinueOnError)
	fs.SetOutput(stderr)
	probe := fs.Bool("probe", true, "handshake with the upstream endpoints to witness TLS interception; --probe=false stays strictly local")
	var hosts stringListFlag
	fs.Var(&hosts, "host", "additional host:port to probe (repeatable)")
	timeout := fs.Duration("timeout", httptrust.DefaultProbeTimeout, "per-endpoint handshake timeout")
	asJSON := fs.Bool("json", false, "emit stable JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak doctor trust: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	facts := doctorTrustFactsFn(*probe, hosts, *timeout)
	rep := httptrust.Assess(facts)
	out := doctorTrustReport{
		Schema:      doctorTrustSchema,
		OK:          rep.Warnings == 0,
		Declared:    rep.Declared,
		Path:        rep.Path,
		Subjects:    rep.Subjects,
		Intercepted: rep.Intercepted,
		Probes:      facts.Probes,
		Findings:    rep.Warnings,
	}
	for _, f := range rep.Findings {
		out.Recommendations = append(out.Recommendations, Recommendation{
			Check: f.Check, Severity: f.Severity, Finding: f.Finding, Recommend: f.Recommend,
		})
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, out, "fak doctor trust")
	}
	writeDoctorTrustHuman(stdout, out)
	if out.Findings > 0 {
		return 1
	}
	return 0
}

// doctorTrustFactsFn is the fact-collection seam. The verdict is pure and tested in
// internal/httptrust; this indirection lets the CLI's own tests exercise the render
// and exit-code contract against a synthetic intercepted host, which is the one thing
// that cannot be observed from whatever machine the test happens to run on.
var doctorTrustFactsFn = gatherDoctorTrustFacts

// gatherDoctorTrustFacts is the impure half: resolve the declared source, model the
// child environment through the REAL guard spawn path, read the host's sibling trust
// variables, detect a cloud route, and (unless --probe=false) handshake.
//
// The child environment is taken from guardChildCommandEnv rather than re-derived
// here on purpose. A diagnostic that models the spawn path approximately is worse
// than none: it can only ever tell the operator about a child fak does not actually
// launch. This calls the same function `fak guard` calls.
func gatherDoctorTrustFacts(probe bool, extraHosts []string, timeout time.Duration) httptrust.Facts {
	environ := os.Environ()
	_, childEnv := guardChildCommandEnv([]string{"claude"}, nil, false)
	facts := httptrust.Facts{
		Source:   httptrust.Resolve(),
		Siblings: httptrust.HostSiblingTrustVars(environ),
		ChildEnv: childEnv,
		Now:      time.Now(),
	}
	route, routed := cloudroute.Detect(environ)
	if routed {
		facts.CloudRoute = route.Selector
		facts.CloudRouteWaived = route.Waived
		if !facts.CloudRouteWaived {
			if v, found := secretload.Default().Lookup(cloudroute.WaiverKey); found {
				facts.CloudRouteWaived = guardTruthyKnob(v)
			}
		}
	}
	if !probe {
		return facts
	}
	hosts := []string{doctorTrustProbeHost}
	if routed {
		if h, ok := doctorTrustRouteProbe[route.Kind]; ok {
			hosts = append(hosts, h)
		}
	}
	hosts = append(hosts, extraHosts...)
	facts.Probes = probeDoctorTrustHosts(hosts, facts.Source, timeout)
	return facts
}

// probeDoctorTrustHosts handshakes every endpoint concurrently and returns the
// results in the order the hosts were given, so the report is byte-stable across
// runs regardless of which proxy answers first.
func probeDoctorTrustHosts(hosts []string, src httptrust.Source, timeout time.Duration) []httptrust.ProbeResult {
	pool := src.RootCAs()
	results := make([]httptrust.ProbeResult, len(hosts))
	var wg sync.WaitGroup
	for i, host := range hosts {
		wg.Add(1)
		go func(i int, host string) {
			defer wg.Done()
			results[i] = httptrust.ProbeHost(context.Background(), host, pool, timeout)
		}(i, host)
	}
	wg.Wait()
	return results
}

func writeDoctorTrustHuman(w io.Writer, rep doctorTrustReport) {
	fmt.Fprintln(w, "== fak doctor: upstream trust ==")
	if rep.Declared {
		fmt.Fprintf(w, "trust source: %s=%s\n", httptrust.BundleKey, rep.Path)
	} else {
		fmt.Fprintf(w, "trust source: none declared (%s unset) — platform trust store only\n", httptrust.BundleKey)
	}
	if rep.Intercepted {
		fmt.Fprintln(w, "interception: WITNESSED on this host (see the upstream-tls rows)")
	}
	for _, r := range rep.Recommendations {
		tag := "OK  "
		if r.Severity == sevWarn {
			tag = "WARN"
		}
		fmt.Fprintf(w, "[%s] %-19s %s\n", tag, r.Check, r.Finding)
		if r.Recommend != "" {
			fmt.Fprintf(w, "       recommend: %s\n", r.Recommend)
		}
	}
	if rep.Findings == 0 {
		fmt.Fprintln(w, "doctor: healthy (0 findings)")
	} else {
		fmt.Fprintf(w, "doctor: %d finding(s)\n", rep.Findings)
	}
}
