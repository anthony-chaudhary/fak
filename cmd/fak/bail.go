package main

// bail.go — the standard shape of a fak CONFIG BAIL: an early, non-negotiable
// exit whose cause is the operator's CONFIGURATION rather than the run.
//
// The complaint this answers is not that fak refuses. A malformed capability
// floor or a half-resolved tenant keyset SHOULD refuse, loudly and before the
// listener binds. The complaint is that the refusal used to be where the
// operator's information ENDED. `fak: parse policy.json: unexpected end of
// JSON input` names the failure and nothing else: not which knob carried it,
// not what to look at, not what clears it. The operator's next move is a guess,
// and an agent's next move is a retry of the same command.
//
// So every config bail renders the same block:
//
//	fak serve: --reset-on-budget has no budget to reset against
//	  reason: BUDGET_FLAG_INCOHERENT
//	  config: flag --reset-on-budget = true
//	          flag --context-budget-tokens = 0  (want: a positive token budget)
//	  check:  fak serve --help
//	  next:   fak recover BUDGET_FLAG_INCOHERENT
//
// `reason` is a stable, greppable token from the same closed vocabulary
// `fak recover` already resolves for the guard/DOS refusals (OFF_TRUNK,
// COLLISION_RISK, …). That makes the last line not advice but a COMMAND: it
// prints the concrete checks, and where a mechanical repair is safe it runs
// them under --execute. recover_config.go carries those plans.
//
// The `config:` block is the part a bare error cannot have: every knob that
// participated in the decision, its ORIGIN (flag / env / file / cwd), the value
// fak actually observed, and — when the knob is the one to change — what it
// wants instead. An operator who mis-set a flag two shells ago sees the
// observed value, not their intent.
//
// bailReasons is the vocabulary and TestEveryConfigBailReasonHasARecoveryPlan is
// the ratchet: the tokens here and the plans in configRecoveryPlans must be the
// same set, in both directions. A new config bail therefore cannot ship without
// a recovery behind its token, and a plan cannot rot after its bail is deleted.

import (
	"fmt"
	"io"
	"sort"
	"strings"
)

// Config-bail reason tokens. Closed vocabulary, additive only: each one resolves
// in configRecoveryPlans, and each is stable enough to grep a fleet log for.
const (
	bailNotAWorkspace          = "NOT_A_WORKSPACE"
	bailPolicyLoadFailed       = "POLICY_LOAD_FAILED"
	bailPolicyCheckNoFile      = "POLICY_CHECK_NO_FILE"
	bailKeyEnvUnset            = "KEY_ENV_UNSET"
	bailKeyPrincipalUnresolved = "KEY_PRINCIPAL_UNRESOLVED"
	bailBudgetFlagIncoherent   = "BUDGET_FLAG_INCOHERENT"
	bailAddrRequired           = "ADDR_REQUIRED"
	bailBackendUnavailable     = "BACKEND_UNAVAILABLE"
	bailRouteManifestInvalid   = "ROUTE_MANIFEST_INVALID"
	bailWeightsRequired        = "WEIGHTS_REQUIRED"

	// bailUpstreamTrustUnverified (#8172) is the intercepted-chain case: a corporate
	// trust source was DECLARED and fak is not using it — the bundle did not read,
	// parsed to no certificate, or the platform trust store it must widen was
	// unavailable. It is its own token because the operator's next step is unlike
	// every other config bail here: nothing about fak is misconfigured, a file on
	// the host is, and the fix is a certificate rather than a flag.
	bailUpstreamTrustUnverified = "UPSTREAM_TRUST_UNVERIFIED"

	// bailUpstreamUnsupported (#8172) is the request-signed-upstream case: the
	// wrapped agent is routed to a cloud gateway (Bedrock SigV4, Vertex ADC) that
	// authenticates each request itself, so fak's base-URL repoint is inert and the
	// gateway would adjudicate nothing. Naming it is the whole point — the launch
	// used to refuse for a credential the operator did not lack, or park for a day
	// waiting for a re-login that could not change which token was sent.
	bailUpstreamUnsupported = "UPSTREAM_UNSUPPORTED"

	// bailUnauthenticatedBind is NOT a new spelling: serve_bind_safety.go has
	// named this exact token in its refusal text since the bind guard landed. It
	// was a token with nothing behind it — `fak recover UNAUTHENTICATED_OFF_HOST_BIND`
	// answered "unknown recovery reason". Adopting the existing constant here is
	// the point: the vocabulary was already right, only the lookup was missing.
	bailUnauthenticatedBind = serveBindRefusalToken
)

// bailReasons is every token a config bail may cite. The ratchet walks this list
// against configRecoveryPlans in BOTH directions, so a token added here without a
// plan fails, and a plan left behind after its bail is deleted fails too.
var bailReasons = []string{
	bailNotAWorkspace,
	bailPolicyLoadFailed,
	bailPolicyCheckNoFile,
	bailKeyEnvUnset,
	bailKeyPrincipalUnresolved,
	bailBudgetFlagIncoherent,
	bailAddrRequired,
	bailBackendUnavailable,
	bailRouteManifestInvalid,
	bailWeightsRequired,
	bailUpstreamTrustUnverified,
	bailUpstreamUnsupported,
	bailUnauthenticatedBind,
}

// Knob origins — where a configured value came from. The origin matters as much
// as the value: "--policy" and "FAK_POLICY" are fixed in different places, and an
// operator who set the env var will not find it by re-reading their command line.
const (
	bailOriginFlag = "flag"
	bailOriginEnv  = "env"
	bailOriginFile = "file"
	bailOriginCWD  = "cwd"
)

// bailKnob is one configuration input that participated in the refusal: its
// origin, its name, the value fak OBSERVED (not the value the operator meant),
// and — only on the knob that should change — what an admissible value looks like.
type bailKnob struct {
	Origin   string
	Name     string
	Observed string
	Want     string
}

// configBail is a refusal whose cause is configuration. Verb and Summary are the
// headline; Reason is the recovery token; Knobs is the evidence; Check is an
// optional read-only command that shows the operator the same thing fak saw.
//
// Bind carries NAME=VALUE bindings for the recovery plan's placeholders, so the
// `next:` line is a command about THIS failure rather than a template. It is the
// difference between telling an operator to run `fak policy --check <path>` and
// handing them the path they actually passed — the bail site is the only place
// that knows it, so it is the only place that can close that gap.
type configBail struct {
	Verb    string
	Reason  string
	Summary string
	Knobs   []bailKnob
	Check   string
	Bind    []string
}

// bailFlag/bailEnv/bailFile/bailCWD are the knob constructors. Want is optional
// and set with want() so the common "this knob is just context" case stays terse.
func bailFlag(name, observed string) bailKnob {
	return bailKnob{Origin: bailOriginFlag, Name: "--" + strings.TrimPrefix(name, "--"), Observed: observed}
}

func bailEnv(name, observed string) bailKnob {
	return bailKnob{Origin: bailOriginEnv, Name: name, Observed: observed}
}

func bailFile(path, observed string) bailKnob {
	return bailKnob{Origin: bailOriginFile, Name: path, Observed: observed}
}

func bailCWD(dir, observed string) bailKnob {
	return bailKnob{Origin: bailOriginCWD, Name: dir, Observed: observed}
}

// want marks this knob as the one to change and states an admissible value.
func (k bailKnob) want(w string) bailKnob {
	k.Want = w
	return k
}

// bailUnset is the rendered stand-in for a knob the operator never set. It is
// deliberately not the empty string: "--addr = " reads as a rendering bug, while
// "--addr = (unset)" reads as the finding it is.
const bailUnset = "(unset)"

// bailValue renders an observed value, mapping empty to the explicit (unset)
// marker so a blank never looks like a formatting slip.
func bailValue(s string) string {
	if strings.TrimSpace(s) == "" {
		return bailUnset
	}
	return s
}

// writeConfigBail renders the bail block to w. It never exits — the caller owns
// the exit code, because the same block precedes a usage exit (2) at flag-parse
// time and a load failure (1) once a file has been read.
func writeConfigBail(w io.Writer, b configBail) {
	verb := strings.TrimSpace(b.Verb)
	if verb == "" {
		verb = "fak"
	}
	fmt.Fprintf(w, "%s: %s\n", verb, b.Summary)
	if r := strings.TrimSpace(b.Reason); r != "" {
		fmt.Fprintf(w, "  reason: %s\n", r)
	}
	for i, k := range b.Knobs {
		label := "  config:"
		if i > 0 {
			label = "         " // continuation lines align under the first knob
		}
		line := fmt.Sprintf("%s %s %s = %s", label, k.Origin, k.Name, bailValue(k.Observed))
		if strings.TrimSpace(k.Want) != "" {
			line += fmt.Sprintf("  (want: %s)", k.Want)
		}
		fmt.Fprintln(w, line)
	}
	if c := strings.TrimSpace(b.Check); c != "" {
		fmt.Fprintf(w, "  check:  %s\n", c)
	}
	if r := strings.TrimSpace(b.Reason); r != "" {
		next := "fak recover " + r
		for _, bind := range b.Bind {
			if strings.TrimSpace(bind) == "" {
				continue
			}
			next += " --set " + shellQuote(bind)
		}
		fmt.Fprintf(w, "  next:   %s\n", next)
	}
}

// bailReasonSet is the emitted vocabulary as a set, for the recovery-plan ratchet
// and for any caller validating a token it read back out of a log.
func bailReasonSet() map[string]bool {
	set := make(map[string]bool, len(bailReasons))
	for _, r := range bailReasons {
		set[r] = true
	}
	return set
}

// sortedBailReasons returns the emitted vocabulary in a stable order for help
// text and test failure messages.
func sortedBailReasons() []string {
	out := append([]string(nil), bailReasons...)
	sort.Strings(out)
	return out
}
