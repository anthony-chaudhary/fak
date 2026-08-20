package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/cloudroute"
	"github.com/anthony-chaudhary/fak/internal/httptrust"
	"github.com/anthony-chaudhary/fak/internal/secretload"
)

// guard_upstream_trust.go — the two enterprise-posture gates that run BEFORE the
// credential gates in resolveGuardUpstreamPosture (#8172).
//
// Order is the whole point of this file. The credential gates below it are
// correct for the posture they were written for and wrong for this one:
//
//   - A Bedrock/Vertex-routed box has no .credentials.json and no
//     ANTHROPIC_API_KEY, so the noTokenAnywhere gate refuses with "no Claude
//     subscription token found" and tells the operator to run `claude
//     setup-token` — advice that cannot apply, for a session holding a perfectly
//     good working credential.
//   - Worse, the #2260 STALE_CRED park then waits up to 24h, polling every 2m,
//     for a re-login to land in a file that will never exist. Witnessed on a
//     corporate Windows workstation: `fak guard --quiet --probe claude` on a
//     Bedrock-routed environment printed "parked — STALE_CRED: waiting for a
//     re-login to land at …\.credentials.json (poll every 2m0s, ceiling 24h0m0s)"
//     and hung. The park is right about human-paced recovery and simply has the
//     wrong precondition: the same reasoning guard_child.go's #3267 early return
//     already applies when an env token outranks the credential file — checking a
//     file that cannot change which credential is sent can only produce a false
//     STALE_CRED.
//
// So both gates run first, name the real posture, and exit before any of that.
// Neither fires on a box that has no bundle declared and no cloud selector set,
// which is every non-enterprise host: the historical path is byte-for-byte intact.

// guardUpstreamTrustGate refuses the launch when a corporate trust source was
// DECLARED and fak is not validating with it, and otherwise returns the startup-report
// line naming the trust store this session validates with ("" when nothing is
// declared, which is every non-enterprise host).
//
// It is a refusal rather than a warning because the alternative is worse in a
// specific way: fak would fall back to the platform trust store, every outbound
// call would fail on a chain it cannot verify, and the resulting error names
// neither the bundle nor fak. The operator configured trust and would be debugging
// the network. Nothing is refused when no bundle is declared.
//
// The informational line is RETURNED rather than printed so it lands in the durable
// startup report (`fak info --startup`, the Slack control replies) and obeys
// --banner, instead of being a stderr line that a compact launch loses and a full
// launch prints twice. The expiry WARNING is printed here regardless of banner mode,
// like the other launch-time credential warnings: it is news, not posture.
func guardUpstreamTrustGate(quiet bool) string {
	src := httptrust.Resolve()
	if !src.Declared() {
		return "" // no trust source configured: the clean box, unchanged
	}
	if src.Usable() {
		if src.Bundle.Expired > 0 && !quiet {
			fmt.Fprintf(os.Stderr, "fak guard: %d certificate(s) in %s are past their expiry — the chain they signed will stop validating.\n", src.Bundle.Expired, src.Path)
		}
		detail := strings.Join(src.Bundle.Subjects, ", ")
		if detail == "" {
			detail = fmt.Sprintf("%d certificate(s)", len(src.Bundle.Subjects))
		}
		return fmt.Sprintf("fak guard: upstream trust — validating against the platform store PLUS %s (%s); children inherit it via %s\n",
			src.Path, detail, strings.Join(httptrust.ChildTrustVars, ", "))
	}
	writeConfigBail(os.Stderr, configBail{
		Verb:    "fak guard",
		Reason:  bailUpstreamTrustUnverified,
		Summary: "a corporate CA bundle was declared and fak cannot validate with it — refusing to launch a session whose every upstream call would fail on a chain fak cannot verify",
		Knobs: []bailKnob{
			bailEnv(httptrust.BundleKey, src.Path).want("a readable PEM file containing at least one CERTIFICATE block"),
			bailFile(src.Path, bailLoadDetail(src.Err)),
		},
		Check: "fak doctor trust",
		Bind:  []string{"path=" + src.Path},
	})
	os.Exit(2)
	return ""
}

// bailLoadDetail renders a load failure as the one-line observation a bail's
// `file` knob carries. The error already names the path, so strip the package
// prefix rather than repeating it on the same line.
func bailLoadDetail(err error) string {
	if err == nil {
		return ""
	}
	return strings.TrimPrefix(err.Error(), "httptrust: ")
}

// guardCloudRouteGate refuses (or, when waived, warns about) a launch whose
// wrapped agent is routed to a request-signed cloud gateway.
//
// The environment inspected is the PARENT's, deliberately: this runs during
// posture resolution, before the child environment is assembled, and the selectors
// it looks for are inherited rather than injected — so the parent's environ is
// exactly what the child will be selected by. It returns true when the launch was
// waived, so the caller can record that the session is unadjudicated.
func guardCloudRouteGate(quiet bool) bool {
	r, ok := cloudroute.Detect(os.Environ())
	if !ok {
		return false
	}
	// The waiver is resolved through the config surface rather than a bare
	// os.Getenv so it can also come from an encrypted store or .env, the same way
	// every other non-credential knob in this binary is meant to resolve.
	if !r.Waived {
		if v, found := secretload.Default().Lookup(cloudroute.WaiverKey); found {
			r.Waived = guardTruthyKnob(v)
		}
	}
	if r.Waived {
		fmt.Fprintf(os.Stderr, "fak guard: %s — proceeding under %s=1: the hook floor, tool brokering, transcript, and sandbox still apply, but fak will see NONE of this session's model traffic.\n",
			bailUpstreamUnsupported, cloudroute.WaiverKey)
		return true
	}
	writeConfigBail(os.Stderr, configBail{
		Verb:    "fak guard",
		Reason:  bailUpstreamUnsupported,
		Summary: r.Explain(),
		Knobs: []bailKnob{
			bailEnv(r.Selector, "on").want("unset, to route this session through fak's gateway with a bearer credential"),
			bailEnv(cloudroute.InertRepointVars[0], "(would be injected, and ignored by the child)"),
			bailEnv(cloudroute.WaiverKey, os.Getenv(cloudroute.WaiverKey)).want("1 to launch anyway with the model traffic unadjudicated"),
		},
		Check: "fak doctor trust",
	})
	fmt.Fprintln(os.Stderr, "  works:  fak serve --stdio --policy FILE   (fak as an MCP server the agent calls — provider-agnostic, so the capability floor is enforced whatever the model wire is)")
	os.Exit(2)
	return false
}

// guardTruthyKnob is the shared on/off reading for a guard knob, matching the
// vocabulary internal/secretload uses for FAK_SANDBOX_INHERIT_ENV so an operator
// meets one convention across the tool.
func guardTruthyKnob(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "1", "true", "yes", "y", "on":
		return true
	default:
		return false
	}
}

// guardChildTrustEnv returns the per-runtime trust variables to add to a child
// environment, derived from the declared bundle and skipping every name the parent
// already set. Empty when no bundle is declared or the parent already covers every
// runtime — so a call site can append it unconditionally.
func guardChildTrustEnv(parent []string) []string {
	src := httptrust.Resolve()
	if !src.Usable() {
		return nil
	}
	return httptrust.ChildEnv(src.Path, parent)
}
