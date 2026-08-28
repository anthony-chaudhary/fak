package agentreadinessscore

import (
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/mathx"
)

func kpiAgentsEntrypoint(agentsText string, present bool) KPI {
	defects := []string{}
	if !present || strings.TrimSpace(agentsText) == "" {
		return KPI{"agents_entrypoint", "discover", 0, "no " + agentsFile + " at the repo root",
			[]string{"missing " + agentsFile + " — the agents.md machine-read entry point (agents.md)"}, []string{}}
	}
	if !has(agentsText, "what this project is", "what this is", "is an agent", "**fak**") {
		defects = append(defects, agentsFile+" does not state what the project is (a 'what this is' line)")
	}
	if !has(agentsText, "go build", "make build", "make test-fast") {
		defects = append(defects, agentsFile+" has no build command (go build / make …)")
	}
	if !has(agentsText, "make test", "go test", "test.ps1", "make ci") {
		defects = append(defects, agentsFile+" has no test command (make test / go test / ./test.ps1)")
	}
	if !has(agentsText, "go run ./cmd/fak", "fak preflight", "fak serve", "fak agent") {
		defects = append(defects, agentsFile+" has no runnable first verb (go run ./cmd/fak … / fak …)")
	}
	detail := "AGENTS.md states identity + build/test/run"
	if len(defects) > 0 {
		detail = strconv.Itoa(len(defects)) + " missing entry-point element(s)"
	}
	return KPI{"agents_entrypoint", "discover", mathx.ClampScore(100 - 22*float64(len(defects))), detail, defects, []string{}}
}

// kpiCoverage renders the "N of M families are covered" KPI shape: one defect per
// missing label, score = covered/total. `defect` and `noun` carry each caller's own
// wording so the rendered strings are unchanged.
func kpiCoverage(kpi, group string, total int, missing []string, defect func(label string) string, noun string) KPI {
	defects := []string{}
	for _, label := range missing {
		defects = append(defects, defect(label))
	}
	covered := total - len(missing)
	return KPI{kpi, group,
		mathx.ClampScore(100 * float64(covered) / float64(max1(total))),
		strconv.Itoa(covered) + "/" + strconv.Itoa(total) + " " + noun,
		defects, []string{}}
}

func kpiAgentConfig(missing []string) KPI {
	return kpiCoverage("agent_config", "discover", len(agentConfigs), missing,
		func(label string) string {
			return "no auto-discovered config for " + label + " — add it so that harness drops in with no setup"
		},
		"agent harnesses have a zero-setup config")
}

func kpiLLMSMap(llmsPresent, llmsFullPresent bool) KPI {
	defects := []string{}
	soft := []string{}
	if !llmsPresent {
		defects = append(defects, "missing "+llmsFile+" — the agent / answer-engine doc-map")
	}
	if !llmsFullPresent {
		soft = append(soft, "no "+llmsFullFile+" (the inlined full doc-map an answer engine ingests)")
	}
	detail := llmsFile + " present"
	if len(defects) > 0 {
		detail = "no " + llmsFile
	}
	return KPI{"llms_map", "discover", mathx.ClampScore(100 - 60*float64(len(defects)) - 8*float64(len(soft))), detail, defects, soft}
}

func kpiIdentityStatement(presentIn, missingFrom []string) KPI {
	evidence := make([]identityEvidence, 0, len(presentIn))
	for _, doc := range presentIn {
		evidence = append(evidence, identityEvidence{Doc: doc, Statement: "recognized identity statement"})
	}
	return kpiIdentityStatementWithEvidence(evidence, missingFrom)
}

func kpiIdentityStatementWithEvidence(evidence []identityEvidence, missingFrom []string) KPI {
	defects := []string{}
	for _, doc := range missingFrom {
		defects = append(defects, "no quotable fak identity in the first "+strconv.Itoa(identityHeadLines)+" lines or under a named identity heading in "+doc+" — state what fak is or does")
	}
	presentIn := make([]string, 0, len(evidence))
	matched := make([]string, 0, len(evidence))
	for _, match := range evidence {
		presentIn = append(presentIn, match.Doc)
		location := match.Doc
		if match.Line > 0 {
			location += ":" + strconv.Itoa(match.Line)
		}
		matched = append(matched, location+" "+strconv.Quote(match.Statement))
	}
	score := 100
	var detail string
	if len(missingFrom) == 0 {
		detail = "identity resolves in all " + strconv.Itoa(len(identityDocs)) + " orientation docs: " + strings.Join(matched, "; ")
	} else {
		score = int(math.RoundToEven(100 * float64(len(presentIn)) / float64(len(identityDocs))))
		pres := strings.Join(presentIn, ", ")
		if pres == "" {
			pres = "none"
		}
		detail = "identity missing from " + strings.Join(missingFrom, ", ") + " (matched: " + pres + ")"
		if len(matched) > 0 {
			detail += ": " + strings.Join(matched, "; ")
		}
	}
	return KPI{"identity_statement", "discover", score, detail, defects, []string{}}
}

func kpiEntryLinksResolve(dead []string) KPI {
	sorted := append([]string{}, dead...)
	sort.Strings(sorted)
	defects := []string{}
	for _, d := range sorted {
		defects = append(defects, "dead orientation link: "+d)
	}
	detail := "every orientation link resolves"
	if len(defects) > 0 {
		detail = strconv.Itoa(len(defects)) + " dead orientation link(s)"
	}
	return KPI{"entry_links_resolve", "discover", mathx.ClampScore(100 - 14*float64(len(defects))), detail, defects, []string{}}
}

// kpiPresence renders the "the one runnable thing is either there or it is not" KPI
// shape: 100 and a locating detail when found, otherwise a fixed floor plus one defect
// naming what to add. `absentScore` keeps each caller's own floor — 20 for a missing
// first command, 40 for a missing install one-liner — which is the only number the two
// KPIs disagree on.
func kpiPresence(kpi string, found bool, absentScore int, absentDefect, absentDetail, presentDetail string) KPI {
	defects := []string{}
	if !found {
		defects = append(defects, absentDefect)
	}
	score := absentScore
	detail := absentDetail
	if found {
		score = 100
		detail = presentDetail
	}
	return KPI{kpi, "adopt", score, detail, defects, []string{}}
}

func kpiFirstCommand(found bool, where string) KPI {
	return kpiPresence("first_command", found, 20,
		"no copy-pasteable no-key/no-model/no-GPU first command in a fenced block of "+strings.Join(firstCommandDocs, ", ")+" (e.g. `fak preflight …`)",
		"no runnable no-setup first command",
		"first command present in "+where)
}

func kpiInstallOneliner(found bool, where string) KPI {
	return kpiPresence("install_oneliner", found, 40,
		"no one-line install (`go install …@latest`) in "+strings.Join(installDocs, ", ")+" — give an agent the one-command install",
		"no one-line install",
		"install one-liner present in "+where)
}

func kpiHonestyLedger(present bool, claimRecords int, violations []string) KPI {
	defects := []string{}
	if !present {
		defects = append(defects, "missing "+claimsFile+" — the honesty ledger an agent trusts (every claim tagged shipped/simulated/stub)")
	} else {
		limit := violations
		if len(limit) > 8 {
			limit = limit[:8]
		}
		defects = append(defects, limit...)
	}
	soft := []string{}
	if present && len(violations) > 8 {
		soft = append(soft, "... and "+strconv.Itoa(len(violations)-8)+" more honesty-ledger violation(s)")
	}
	base := 0.0
	nPresentDefects := 0
	if present {
		base = 100
		nPresentDefects = len(defects)
	}
	detail := "no " + claimsFile
	if present {
		detail = claimsFile + " document set present, " + strconv.Itoa(claimRecords) + " claim record(s), " + strconv.Itoa(len(violations)) + " violation(s)"
	}
	return KPI{"honesty_ledger", "adopt", mathx.ClampScore(base - 12*float64(nPresentDefects)), detail, defects, soft}
}

func kpiIntegrationRecipes(missing []string) KPI {
	return kpiCoverage("integration_recipes", "adopt", len(requiredRecipes), missing,
		func(label string) string {
			return "no integration recipe for " + label + " — add one under docs/integrations/"
		},
		"agent families have an integration recipe")
}

func kpiCodexRecipeCurrent(gaps []string) KPI {
	n := len(gaps)
	detail := "Codex recipe covers MCP, AGENTS.md, exec JSON, proxy URL, and Responses fence"
	if n > 0 {
		detail = strconv.Itoa(n) + " Codex currentness gap(s)"
	}
	return KPI{"codex_recipe_current", "adopt", mathx.ClampScore(100 - 15*float64(n)), detail, gaps, []string{}}
}

func kpiExtensionScaffold(scaffold, extending bool) KPI {
	defects := []string{}
	if !scaffold {
		defects = append(defects, "no "+leafScaffold+" — the leaf scaffolder an agent runs to add a feature additively")
	}
	if !extending {
		defects = append(defects, "no "+extendingFile+" — the doc that teaches the plug-in/prove-it path")
	}
	detail := "leaf scaffolder + EXTENDING.md present"
	if len(defects) > 0 {
		detail = strconv.Itoa(len(defects)) + " missing extension affordance(s)"
	}
	return KPI{"extension_scaffold", "build", mathx.ClampScore(100 - 50*float64(len(defects))), detail, defects, []string{}}
}

func kpiGuardrailsSurfaced(missing []string) KPI {
	defects := []string{}
	for _, label := range missing {
		defects = append(defects, "enforced rule not surfaced in "+agentsFile+": "+label)
	}
	covered := len(guardrailClusters) - len(missing)
	return KPI{"guardrails_surfaced", "build",
		mathx.ClampScore(100 * float64(covered) / float64(max1(len(guardrailClusters)))),
		strconv.Itoa(covered) + "/" + strconv.Itoa(len(guardrailClusters)) + " enforced rules surfaced up front",
		defects, []string{}}
}

func kpiContributorContract(contributing, linked, greenGate bool) KPI {
	defects := []string{}
	if !contributing {
		defects = append(defects, "no "+contribFile+" — the contributor contract")
	} else if !linked {
		defects = append(defects, contribFile+" exists but is not linked from "+agentsFile+"/README — an agent can't find it")
	}
	if !greenGate {
		defects = append(defects, "no one-command green gate documented (make ci / make test / ./test.ps1) — the feedback loop an agent runs before shipping")
	}
	detail := "CONTRIBUTING linked + green gate documented"
	if len(defects) > 0 {
		detail = strconv.Itoa(len(defects)) + " missing contract/feedback affordance(s)"
	}
	return KPI{"contributor_contract", "build", mathx.ClampScore(100 - 30*float64(len(defects))), detail, defects, []string{}}
}

func kpiMachineConsumable(jsonTools, totalTools int, missing []string) KPI {
	soft := []string{}
	lim := missing
	if len(lim) > 8 {
		lim = lim[:8]
	}
	for _, m := range lim {
		soft = append(soft, "tool without a --json surface: "+m)
	}
	rate := 1.0
	if totalTools != 0 {
		rate = float64(jsonTools) / float64(totalTools)
	}
	detail := "no measurement tools found"
	if totalTools != 0 {
		detail = strconv.Itoa(jsonTools) + "/" + strconv.Itoa(totalTools) + " measurement tools expose --json (" + pctString(rate) + ")"
	}
	return KPI{"machine_consumable", "build", mathx.ClampScore(math.RoundToEven(100 * rate)), detail, []string{}, soft}
}

func kpiCommandVerbsResolve(unknown []string) KPI {
	defects := []string{}
	for _, u := range unknown {
		defects = append(defects, "unknown CLI verb an agent would paste: "+u)
	}
	n := len(defects)
	detail := "every pasted `fak <verb>` resolves to a real dispatched verb"
	if n > 0 {
		detail = strconv.Itoa(n) + " pasted `fak <verb>` command(s) don't resolve to a dispatched verb"
	}
	return KPI{"command_verbs_resolve", "adopt", mathx.ClampScore(100 - 20*float64(n)), detail, defects, []string{}}
}

func kpiRecipeLinksResolve(dead []string) KPI {
	sorted := append([]string{}, dead...)
	sort.Strings(sorted)
	defects := []string{}
	for _, d := range sorted {
		defects = append(defects, "dead recipe link: "+d)
	}
	detail := "every link inside every integration recipe resolves"
	if len(defects) > 0 {
		detail = strconv.Itoa(len(defects)) + " dead link(s) inside the integration recipes"
	}
	return KPI{"recipe_links_resolve", "discover", mathx.ClampScore(100 - 12*float64(len(defects))), detail, defects, []string{}}
}

func kpiAgentConfigValid(bad []string) KPI {
	defects := append([]string{}, bad...)
	detail := mcpConfigFile + " parses and every server names a launch command"
	if len(defects) > 0 {
		detail = strconv.Itoa(len(defects)) + " agent-config integrity defect(s)"
	}
	return KPI{"agent_config_valid", "discover", mathx.ClampScore(100 - 34*float64(len(defects))), detail, defects, []string{}}
}

func kpiFencedPathsResolve(badPaths []string) KPI {
	defects := []string{}
	for _, b := range badPaths {
		defects = append(defects, "unresolvable fenced path: "+b)
	}
	n := len(defects)
	detail := "every fenced command path resolves from a clean clone"
	if n > 0 {
		detail = strconv.Itoa(n) + " fenced command path(s) don't resolve from a clean clone"
	}
	return KPI{"fenced_paths_resolve", "adopt", mathx.ClampScore(100 - 10*float64(n)), detail, defects, []string{}}
}

func kpiFirstCommandRuns(found, policyOK bool, policyRef string, needsKey bool) KPI {
	if !found {
		return KPI{"first_command_runs", "adopt", 100, "no first command to check (see first_command)", []string{}, []string{}}
	}
	defects := []string{}
	if !policyOK {
		ref := policyRef
		if ref == "" {
			ref = "<none parsed>"
		}
		defects = append(defects, "the first command names a policy that doesn't exist on disk: "+ref+" — an agent that pastes it hits a missing-file error")
	}
	if needsKey {
		defects = append(defects, "the first command sold as the no-setup proof secretly needs a key (--api-key-env / --provider) — it is not the no-key/no-model/no-GPU form an agent can run cold")
	}
	detail := "the first command runs cold (policy " + policyRef + " resolves, no key)"
	if len(defects) > 0 {
		detail = strconv.Itoa(len(defects)) + " runnability gap(s) in the first command"
	}
	return KPI{"first_command_runs", "adopt", mathx.ClampScore(100 - 50*float64(len(defects))), detail, defects, []string{}}
}

func kpiPlatformGuidanceConsistent(sellsMake, hasBridge bool) KPI {
	defects := []string{}
	if sellsMake && !hasBridge {
		defects = append(defects, "AGENTS.md sells `make ci` as the green gate but names no native-Windows bridge (scripts/ci.ps1 / ./test.ps1 under WSL) — a Windows agent can't run the gate it's told to run")
	}
	score := 100
	detail := "the green gate names its native-Windows bridge"
	if len(defects) > 0 {
		score = 40
		detail = "make ci sold without a native-Windows bridge"
	}
	return KPI{"platform_guidance_consistent", "build", score, detail, defects, []string{}}
}

func kpiRefusalRecoveryMapped(unmapped []string, total int) KPI {
	defects := []string{}
	for _, t := range unmapped {
		defects = append(defects, "refusal token has no query-backed recovery: "+t+" — add summary/fix metadata to its dos.toml [reasons] block and keep the AGENTS.md query commands usable")
	}
	mapped := total - len(unmapped)
	score := 100
	detail := "no dos.toml [reasons.*] parsed — abstain"
	if total != 0 {
		score = mathx.ClampScore(100 * float64(mapped) / float64(total))
		detail = strconv.Itoa(mapped) + "/" + strconv.Itoa(total) + " kernel refusal tokens have an agent-facing recovery"
	}
	return KPI{"refusal_recovery_mapped", "build", score, detail, defects, []string{}}
}

func kpiQuickstartSuccessSignal(found, hasSignal bool) KPI {
	if !found {
		return KPI{"quickstart_success_signal", "adopt", 100, "no first command to check (see first_command)", []string{}, []string{}}
	}
	defects := []string{}
	if !hasSignal {
		defects = append(defects, "the first-command proof block shows no expected-result marker (`-> DENY/ALLOW`, an exit code, a result line) — an agent can run it but can't tell whether it succeeded; annotate each command with its expected outcome")
	}
	detail := "the proof block shows an observable success signal"
	if len(defects) > 0 {
		detail = "the proof block names no expected outcome"
	}
	return KPI{"quickstart_success_signal", "adopt", mathx.ClampScore(100 - 60*float64(len(defects))), detail, defects, []string{}}
}

func kpiToolchainPinned(hasDirective, docNamed bool) KPI {
	defects := []string{}
	if !hasDirective {
		defects = append(defects, goModFile+" has no `go <version>` directive — an agent can't know which Go toolchain to provision")
	}
	if !docNamed {
		defects = append(defects, "no entry doc (AGENTS.md / README / GETTING-STARTED) names the required Go version — pin it so an agent provisions the right toolchain")
	}
	detail := "go.mod pins the Go version and an entry doc names it"
	if len(defects) > 0 {
		detail = strconv.Itoa(len(defects)) + " toolchain-pin gap(s)"
	}
	return KPI{"toolchain_pinned", "adopt", mathx.ClampScore(100 - 50*float64(len(defects))), detail, defects, []string{}}
}

// ---------------------------------------------------------------------------
// Fold: KPIs -> composite score, grade, friction-debt, control-pane payload.
// ---------------------------------------------------------------------------
