package main

// bail_test.go — the witnesses for the config-bail path.
//
// The load-bearing one is TestEveryConfigBailReasonHasARecoveryPlan. Everything
// else here checks that a block renders the way it should; that one checks the
// property the whole change exists to create — that a config refusal always has
// somewhere to send you. It is a BIJECTION, not a one-way check, because both
// directions are real failures: a token with no plan is a bail that dead-ends
// again, and a plan with no token is a recovery for a refusal that no longer
// exists, which is worse than nothing (it is confidently wrong advice).

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

func TestEveryConfigBailReasonHasARecoveryPlan(t *testing.T) {
	plans := configRecoveryPlans()
	for _, reason := range sortedBailReasons() {
		plan, ok := plans[reason]
		if !ok {
			t.Errorf("config bail reason %s has no recovery plan — a bail citing it dead-ends at `fak recover: unknown recovery reason`", reason)
			continue
		}
		if plan.Reason != reason {
			t.Errorf("plan keyed %s carries Reason %q — the key and the rendered reason must match or the operator is told to run a token that does not resolve", reason, plan.Reason)
		}
		if strings.TrimSpace(plan.Summary) == "" {
			t.Errorf("plan %s has no summary — `fak recover --list` would show a blank row", reason)
		}
		if len(plan.Steps) == 0 && len(plan.Notes) == 0 {
			t.Errorf("plan %s offers neither a command nor a note, so it is a lookup that answers nothing", reason)
		}
		if plan.Executable {
			safe := false
			for _, step := range plan.Steps {
				if step.Safe {
					safe = true
					break
				}
			}
			if !safe {
				t.Errorf("plan %s claims Executable but has no Safe step, so --execute would run nothing and report success", reason)
			}
		}
	}

	known := bailReasonSet()
	for reason := range plans {
		if !known[reason] {
			t.Errorf("recovery plan %s is not in the bail vocabulary — either wire a bail that cites it or delete the plan; a recovery for a refusal fak no longer makes is advice that cannot be followed", reason)
		}
	}
}

// bailConstNames maps each reason to the identifier a bail site writes to cite it.
// Scanning for the CONSTANT rather than the string is deliberate: a site that
// hard-codes "POLICY_LOAD_FAILED" as a literal has opted out of the vocabulary and
// should not count as wired.
var bailConstNames = map[string]string{
	bailNotAWorkspace:          "bailNotAWorkspace",
	bailPolicyLoadFailed:       "bailPolicyLoadFailed",
	bailPolicyCheckNoFile:      "bailPolicyCheckNoFile",
	bailKeyEnvUnset:            "bailKeyEnvUnset",
	bailKeyPrincipalUnresolved: "bailKeyPrincipalUnresolved",
	bailBudgetFlagIncoherent:   "bailBudgetFlagIncoherent",
	bailAddrRequired:           "bailAddrRequired",
	bailBackendUnavailable:     "bailBackendUnavailable",
	bailRouteManifestInvalid:   "bailRouteManifestInvalid",
	bailWeightsRequired:        "bailWeightsRequired",
	bailUnauthenticatedBind:    "bailUnauthenticatedBind",

	bailUpstreamTrustUnverified: "bailUpstreamTrustUnverified",
	bailUpstreamUnsupported:     "bailUpstreamUnsupported",
}

// bailEmitterAliases lets a reason be cited through an equivalent constant. The
// bind guard named its token before this vocabulary existed and still spells it
// serveBindRefusalToken, which bail.go adopts verbatim — the same constant under
// two names is wired, not missing.
var bailEmitterAliases = map[string][]string{
	bailUnauthenticatedBind: {"serveBindRefusalToken"},
}

// pendingBailWiring holds reasons whose recovery plan is written and resolvable
// but whose bail SITE still prints a bare sentence. Each entry names the site, so
// the remaining work is a lookup rather than a re-hunt.
//
// It is EMPTY, and that is the point: every reason in the vocabulary is emitted
// by a real site. The map stays because the ratchet below needs somewhere to
// record a deliberate gap — a reason whose site is blocked (in this shared
// checkout, usually a file another session is mid-edit) can be declared here
// instead of silently shipping as a recovery for a refusal fak never makes.
//
// TestConfigBailReasonsAreEmitted ratchets in BOTH directions: wiring one of
// these fails until it is removed from this map, and adding a reason without
// either a site or an entry here fails too. Growing the list is still possible —
// what it cannot be is silent, since the entry has to be written and reviewed.
var pendingBailWiring = map[string]string{}

// TestConfigBailReasonsAreEmitted keeps the vocabulary honest about itself. A
// reason with a plan but no site is a recovery for a refusal fak never makes:
// harmless in the catalog, actively misleading in `fak recover --list`, where it
// reads as shipped. Rather than let that rot silently, every reason must either
// be cited by a real bail site or be declared pending with its site named.
func TestConfigBailReasonsAreEmitted(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read package dir: %v", err)
	}
	type source struct{ name, body string }
	var sources []source
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		// The files that DEFINE the vocabulary and the catalog mention every
		// token by construction; counting them would make the check vacuous.
		if name == "bail.go" || name == "recover_config.go" {
			continue
		}
		body, err := os.ReadFile(filepath.Join(".", name))
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		sources = append(sources, source{name: name, body: string(body)})
	}
	if len(sources) == 0 {
		t.Fatal("scanned no package sources; the check would pass vacuously")
	}

	emittedBy := func(reason string) string {
		idents := append([]string{bailConstNames[reason]}, bailEmitterAliases[reason]...)
		for _, src := range sources {
			for _, ident := range idents {
				if ident != "" && strings.Contains(src.body, ident) {
					return src.name
				}
			}
		}
		return ""
	}

	for _, reason := range sortedBailReasons() {
		if bailConstNames[reason] == "" {
			t.Errorf("reason %s has no entry in bailConstNames, so this check cannot see whether it is wired", reason)
			continue
		}
		site, pending := pendingBailWiring[reason]
		switch {
		case emittedBy(reason) != "" && pending:
			t.Errorf("%s is now emitted by a real bail site — delete its pendingBailWiring entry (%s); leaving it there hides the next gap behind a stale one", reason, site)
		case emittedBy(reason) == "" && !pending:
			t.Errorf("%s has a recovery plan but no bail site cites %s — wire a site, or record the intended site in pendingBailWiring so `fak recover --list` does not advertise a refusal fak never makes", reason, bailConstNames[reason])
		}
	}

	for reason := range pendingBailWiring {
		if !bailReasonSet()[reason] {
			t.Errorf("pendingBailWiring names %s, which is not in the bail vocabulary", reason)
		}
	}
}

// The pending list is a promise about coverage, so it must be readable as one:
// every entry names a concrete site rather than a shrug.
func TestPendingBailWiringNamesItsSites(t *testing.T) {
	names := make([]string, 0, len(pendingBailWiring))
	for reason, site := range pendingBailWiring {
		if !strings.Contains(site, "cmd/fak/") {
			t.Errorf("pendingBailWiring[%s] = %q does not name a file; the entry has to be actionable without re-hunting the site", reason, site)
		}
		names = append(names, reason)
	}
	sort.Strings(names)
	t.Logf("config bails still to wire (%d of %d): %s", len(names), len(bailReasons), strings.Join(names, ", "))
}

// The two catalog halves share one namespace, so a token defined in both would
// have the config plan silently overwrite the tree plan at merge time — and the
// operator hitting OFF_TRUNK would be handed a config recovery.
func TestTreeAndConfigRecoveryVocabulariesAreDisjoint(t *testing.T) {
	tree := treeRecoveryPlans("main")
	config := configRecoveryPlans()
	for reason := range config {
		if _, clash := tree[reason]; clash {
			t.Errorf("%s is defined in BOTH the tree and config catalogs; the merge in recoveryPlans would drop the tree plan", reason)
		}
	}
	merged := recoveryPlans("main")
	for reason := range tree {
		if _, ok := merged[reason]; !ok {
			t.Errorf("tree recovery plan %s missing from merged catalog", reason)
		}
	}
	for reason := range config {
		if _, ok := merged[reason]; !ok {
			t.Errorf("config recovery plan %s missing from merged catalog", reason)
		}
	}

	dosCount := 0
	if root := guardFindReasonRoot(); root != "" {
		for token := range guardReadReasonDocs(root) {
			if _, inTree := tree[token]; !inTree {
				if _, inConfig := config[token]; !inConfig {
					dosCount++
				}
			}
		}
	}
	if want := len(tree) + len(config) + dosCount; len(merged) != want {
		t.Fatalf("merged catalog has %d plans, want %d — a token collided", len(merged), want)
	}
}

// Every config-bail token must resolve end to end through the real `fak recover`
// entry point, not merely exist in the map: the bail prints that exact command,
// so an operator pasting it has to reach a plan.
func TestConfigBailReasonsResolveThroughRecover(t *testing.T) {
	for _, reason := range sortedBailReasons() {
		var out, errb bytes.Buffer
		if rc := runRecover(&out, &errb, []string{reason, "--dry-run", "--trunk", "main"}); rc != 0 {
			t.Errorf("fak recover %s: rc = %d, stderr = %s", reason, rc, errb.String())
			continue
		}
		if got := out.String(); !strings.Contains(got, "recover "+reason) {
			t.Errorf("fak recover %s printed no plan header:\n%s", reason, got)
		}
	}
}

func TestWriteConfigBailRendersEveryPart(t *testing.T) {
	var buf bytes.Buffer
	writeConfigBail(&buf, configBail{
		Verb:    "fak serve",
		Reason:  bailBudgetFlagIncoherent,
		Summary: "--reset-on-budget has no budget to reset against",
		Knobs: []bailKnob{
			bailFlag("reset-on-budget", "true"),
			bailFlag("context-budget-tokens", "0").want("a positive token budget"),
			bailEnv("FAK_CONTEXT_BUDGET", ""),
		},
		Check: "fak serve --help",
	})
	got := buf.String()
	for _, want := range []string{
		"fak serve: --reset-on-budget has no budget to reset against",
		"reason: " + bailBudgetFlagIncoherent,
		"config: flag --reset-on-budget = true",
		"flag --context-budget-tokens = 0  (want: a positive token budget)",
		"env FAK_CONTEXT_BUDGET = " + bailUnset,
		"check:  fak serve --help",
		"next:   fak recover " + bailBudgetFlagIncoherent,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("rendered bail missing %q:\n%s", want, got)
		}
	}
	// The knob block is one labelled line plus aligned continuations; a second
	// "config:" label would read as a second finding.
	if n := strings.Count(got, "config:"); n != 1 {
		t.Errorf("expected exactly one config: label, got %d:\n%s", n, got)
	}
}

// A flag knob renders with the leading dashes whether or not the caller passed
// them, so a bail never shows a knob spelled differently from the flag itself.
func TestBailFlagNormalizesDashes(t *testing.T) {
	if got := bailFlag("addr", "").Name; got != "--addr" {
		t.Errorf("bailFlag(\"addr\").Name = %q, want --addr", got)
	}
	if got := bailFlag("--addr", "").Name; got != "--addr" {
		t.Errorf("bailFlag(\"--addr\").Name = %q, want --addr", got)
	}
}

// An empty observed value is the common case (the knob the operator forgot), so
// it must read as a finding rather than as a truncated line.
func TestBailValueMarksUnset(t *testing.T) {
	for _, blank := range []string{"", " ", "\t"} {
		if got := bailValue(blank); got != bailUnset {
			t.Errorf("bailValue(%q) = %q, want %q", blank, got, bailUnset)
		}
	}
	if got := bailValue("127.0.0.1:8080"); got != "127.0.0.1:8080" {
		t.Errorf("bailValue passed through wrong: %q", got)
	}
}

// A bail with no reason still renders its headline, but must not print a `next:`
// line pointing at an empty recovery token.
func TestWriteConfigBailWithoutReasonOmitsNext(t *testing.T) {
	var buf bytes.Buffer
	writeConfigBail(&buf, configBail{Verb: "fak serve", Summary: "something went wrong"})
	got := buf.String()
	if !strings.Contains(got, "fak serve: something went wrong") {
		t.Fatalf("headline missing:\n%s", got)
	}
	if strings.Contains(got, "next:") {
		t.Errorf("a reasonless bail must not advertise a recovery:\n%s", got)
	}
}

func TestRecoverBindsPlaceholders(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runRecover(&out, &errb, []string{bailPolicyLoadFailed, "--dry-run", "--trunk", "main", "--set", "path=guard-policy.json"})
	if rc != 0 {
		t.Fatalf("rc = %d, stderr = %s", rc, errb.String())
	}
	got := out.String()
	if !strings.Contains(got, "fak policy --check guard-policy.json") {
		t.Errorf("--set path did not bind into the command:\n%s", got)
	}
	if strings.Contains(got, "<path>") {
		t.Errorf("a bound plan still shows the placeholder:\n%s", got)
	}
}

// Printing a template is fine; SHELLING OUT a literal <path> is not. The refusal
// is itself a next step, so it must name the placeholder to bind.
func TestRecoverRefusesExecuteWithUnboundPlaceholder(t *testing.T) {
	old := recoverRunStep
	t.Cleanup(func() { recoverRunStep = old })
	ran := 0
	recoverRunStep = func(dir string, argv []string, stdout, stderr io.Writer) int {
		ran++
		return 0
	}

	var out, errb bytes.Buffer
	rc := runRecover(&out, &errb, []string{bailPolicyLoadFailed, "--execute", "--trunk", "main"})
	if rc != 2 {
		t.Fatalf("rc = %d, want 2; stderr = %s", rc, errb.String())
	}
	if ran != 0 {
		t.Fatalf("ran %d step(s) with an unbound placeholder", ran)
	}
	if msg := errb.String(); !strings.Contains(msg, "path") || !strings.Contains(msg, "--set") {
		t.Errorf("refusal must name the placeholder and how to bind it, got: %s", msg)
	}
}

func TestRecoverExecutesOnceBound(t *testing.T) {
	old := recoverRunStep
	t.Cleanup(func() { recoverRunStep = old })
	var ran [][]string
	recoverRunStep = func(dir string, argv []string, stdout, stderr io.Writer) int {
		ran = append(ran, append([]string(nil), argv...))
		return 0
	}

	var out, errb bytes.Buffer
	rc := runRecover(&out, &errb, []string{bailPolicyLoadFailed, "--execute", "--trunk", "main", "--set", "path=floor.json"})
	if rc != 0 {
		t.Fatalf("rc = %d, stderr = %s", rc, errb.String())
	}
	if len(ran) == 0 {
		t.Fatal("no safe step ran")
	}
	if got := strings.Join(ran[0], " "); got != "fak policy --check floor.json" {
		t.Errorf("first safe step = %q, want the bound policy check", got)
	}
}

func TestRecoverRejectsMalformedSet(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runRecover(&out, &errb, []string{bailPolicyLoadFailed, "--dry-run", "--trunk", "main", "--set", "nonsense"}); rc != 2 {
		t.Fatalf("rc = %d, want 2 for a --set with no '='; stderr = %s", rc, errb.String())
	}
}

// The bind guard has named UNAUTHENTICATED_OFF_HOST_BIND since it shipped; what
// it lacked was a lookup behind the token. This is the regression guard for the
// join.
func TestBindRefusalPointsAtItsRecovery(t *testing.T) {
	refusal := serveBindRefusal("0.0.0.0:8080", false, false)
	if refusal == "" {
		t.Fatal("an unauthenticated off-host bind must be refused")
	}
	if !strings.Contains(refusal, "fak recover "+serveBindRefusalToken) {
		t.Errorf("bind refusal names its token but not its recovery:\n%s", refusal)
	}
	var out, errb bytes.Buffer
	if rc := runRecover(&out, &errb, []string{serveBindRefusalToken, "--dry-run", "--trunk", "main"}); rc != 0 {
		t.Fatalf("the token the refusal prints does not resolve: rc = %d, stderr = %s", rc, errb.String())
	}
}
