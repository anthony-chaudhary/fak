package adjudicator

import (
	"encoding/json"
	"os"
	"regexp"
	"strings"
	"testing"
)

// TestEveryShippedStructuralRuleIsRecognised makes the surface-parity defect class
// impossible to reintroduce silently.
//
// A structural decider only runs when its RECOGNISER accepts the shipped rule; a
// rule it rejects falls back to the raw regex and loses every carve-out the decider
// grants. Two rules have already shipped on more surfaces than their recogniser
// accepted — the delete rule and the download-pipe rule — and in both cases the
// mirror surfaces quietly kept the raw-regex path. Nothing failed loudly: the same
// command simply got a different verdict depending on which tool NAME the harness
// used, and under `fak guard -- claude` the strict side reads as an agent-chosen
// end_turn rather than a refusal.
//
// The same silence hides a second drift: editing a deny_regex in the shipped policy
// by one character disables its decider outright, because the recogniser matches the
// spelling EXACTLY. Both directions are asserted here.
func TestEveryShippedStructuralRuleIsRecognised(t *testing.T) {
	b, err := os.ReadFile("../../cmd/fak/guard-default-policy.json")
	if err != nil {
		t.Fatalf("read shipped policy: %v", err)
	}
	var manifest struct {
		ArgRules []struct {
			Tool      string `json:"tool"`
			Arg       string `json:"arg"`
			DenyRegex string `json:"deny_regex"`
		} `json:"arg_rules"`
	}
	if err := json.Unmarshal(b, &manifest); err != nil {
		t.Fatalf("parse shipped policy: %v", err)
	}

	families := []struct {
		name      string
		recognise func(*ArgPredicate) bool
		spellings []string
	}{
		{"rm_rf", isRmRfArgRule, []string{defaultRmRfDenyRegex, defaultPSDeleteDenyRegex}},
		{"rce_pipe", isRCEPipeArgRule, []string{legacyRCEPipeDenyRegex, defaultRCEPipeDenyRegex, defaultPSRCEPipeDenyRegex}},
		{"sudo", isSudoArgRule, []string{defaultSudoDenyRegex}},
		{"runas", isRunAsArgRule, []string{defaultRunAsDenyRegex}},
		{"terraform_destroy", isTerraformDestroyArgRule, []string{terraformDestroyDenyRegex, terraformDestroyDenyRegexCI}},
		{"shell_dialect", isShellDialectArgRule, []string{defaultShellDialectDenyRegex}},
		{"out_of_tree_write", isOutOfTreeWriteArgRule, []string{ootDashORegex, ootOutputRegex, ootRedirectRegex, ootCopyVerbRegex}},
	}

	// undecided inventories the shipped deny_regex rules that have NO structural
	// decider today, keyed by a distinguishing substring. They are listed rather
	// than merely skipped so that adding a rule to the policy forces a conscious
	// choice — write a decider, or record here why the raw regex is the whole
	// truth for it. Each entry is a standing candidate for the same treatment the
	// families above already got.
	undecided := []struct{ marker, why string }{
		{"mkfs", "disk-formatting verbs; a quoted mention is still refused"},
		{"Format-Volume", "PowerShell disk verbs: they destroy a filesystem outright, so the raw regex IS the whole truth and operator-only is the right posture"},
		{"Clear-Content", "split out of the disk rule, which claimed a file truncation was a 'disk/volume operation' and sent the caller to an operator with nothing to approve. Still denied — an in-place truncation has no preview and no undo — but its remedy now names the in-tree routes that were always admitted. A containment decider (empty a file under the working tree or a scratchpad root, deny outside) is the obvious next step and needs a new dispatch call site"},
		{`os\.system`, "the execute_code surface, which is not a shell command line"},
		{`:\(\)`, "fork bomb; prose describes the shape well enough that the remedy needs no literal, which is why its fix text now omits one"},
	}

	shipped := map[string]map[string]bool{}
	for _, fam := range families {
		shipped[fam.name] = map[string]bool{}
	}

	for _, r := range manifest.ArgRules {
		if r.DenyRegex == "" {
			continue
		}
		matched := false
		for _, fam := range families {
			for _, spelling := range fam.spellings {
				if r.DenyRegex != spelling {
					continue
				}
				matched = true
				pr := &ArgPredicate{
					Tool: r.Tool, Arg: r.Arg, Kind: ArgDenyRegex,
					Re: regexp.MustCompile(r.DenyRegex),
				}
				if !fam.recognise(pr) {
					t.Errorf("%s: tool %q arg %q ships this deny_regex but %s rejects it — the rule silently falls back to the raw regex on that surface and loses every carve-out the decider grants",
						fam.name, r.Tool, r.Arg, fam.name+"'s recogniser")
				}
				shipped[fam.name][strings.ToLower(r.Tool)] = true
			}
		}
		if matched {
			continue
		}
		for _, u := range undecided {
			if strings.Contains(r.DenyRegex, u.marker) {
				matched = true
				break
			}
		}
		if !matched {
			t.Errorf("tool %q arg %q ships a deny_regex with no structural decider and no inventory entry: %q\n"+
				"Decide it: either add a recogniser+decider family above, or add a marker to the `undecided` inventory saying why the raw regex is the whole truth for it. A raw regex refuses every quoted MENTION of the pattern too.",
				r.Tool, r.Arg, r.DenyRegex)
		}
	}

	for _, fam := range families {
		if len(shipped[fam.name]) == 0 {
			t.Errorf("%s: no shipped rule matches any spelling this package decides — a policy edit has disabled the decider outright, since the recogniser matches the spelling EXACTLY", fam.name)
		}
	}
}

// TestEveryShippedDenyRuleNamesARemedy makes the positive-funnel defect class
// impossible to reintroduce silently.
//
// A refusal that names no route is, from the agent's side, indistinguishable from
// one that HAS no route. Under `fak guard -- claude` a POLICY_BLOCK reads as an
// agent-chosen end_turn, so the agent does not get a second look at the rule: the
// fix text is the entire interface between the floor and the caller. The fork-bomb
// rule shipped on three surfaces with no fix field at all, which meant the only
// thing a blocked agent learned was that something it could not name was refused.
//
// This does not judge the QUALITY of the remedy, only that one was written. Two
// weaker properties are worth stating because they are what the reviewer should
// check by hand: a remedy is only useful if it is TRUE of the shipped rule (the
// deny-all breaker shipped a delete remedy the adjudicator's own carve-out
// contradicted), and it must not embed the literal that trips its own rule, or
// quoting the refusal re-trips it.
func TestEveryShippedDenyRuleNamesARemedy(t *testing.T) {
	b, err := os.ReadFile("../../cmd/fak/guard-default-policy.json")
	if err != nil {
		t.Fatalf("read shipped policy: %v", err)
	}
	var manifest struct {
		ArgRules []struct {
			Tool      string `json:"tool"`
			Arg       string `json:"arg"`
			DenyRegex string `json:"deny_regex"`
			Reason    string `json:"reason"`
			Fix       string `json:"fix"`
		} `json:"arg_rules"`
	}
	if err := json.Unmarshal(b, &manifest); err != nil {
		t.Fatalf("parse shipped policy: %v", err)
	}
	seen := 0
	for _, r := range manifest.ArgRules {
		if r.DenyRegex == "" {
			continue
		}
		seen++
		if strings.TrimSpace(r.Fix) == "" {
			t.Errorf("tool %q arg %q reason %q ships deny_regex %q with NO fix text — a refused agent is told only that it was refused, and a route nobody names is a route that does not exist",
				r.Tool, r.Arg, r.Reason, r.DenyRegex)
			continue
		}
		// A remedy that trips its own rule sends the caller straight back into the
		// refusal the moment they quote it into a commit message, a note or a grep.
		//
		// This is only a DEFECT where the rule cannot tell a mention from a use. A
		// rule with a structural decider already admits the quoted mention — the
		// terraform remedy names `terraform plan -destroy` and the sudo remedy names
		// `sudo`, and both are admitted precisely because their deciders resolve real
		// command words. Demanding those rules avoid their own subject would make the
		// refusal less intelligible to buy nothing. So the hard assertion is scoped to
		// the raw-regex rules, where the match IS the verdict.
		re, err := regexp.Compile(r.DenyRegex)
		if err != nil || !re.MatchString(r.Fix) {
			continue
		}
		pr := &ArgPredicate{Tool: r.Tool, Arg: r.Arg, Re: re}
		decided := isRmRfArgRule(pr) || isRCEPipeArgRule(pr) || isSudoArgRule(pr) ||
			isRunAsArgRule(pr) || isTerraformDestroyArgRule(pr) ||
			isShellDialectArgRule(pr) || isOutOfTreeWriteArgRule(pr)
		if !decided {
			t.Errorf("tool %q arg %q: the fix text for %q MATCHES its own deny_regex, and this rule has NO structural decider — the match IS the verdict, so quoting this refusal re-trips the rule that produced it:\n  fix: %s",
				r.Tool, r.Arg, r.DenyRegex, r.Fix)
			continue
		}
		// Tolerated, but recorded: the decider is what makes it safe, so if one is
		// ever narrowed the note below is where to look first.
		//
		// One of these is NOT actually safe today and is left open deliberately
		// rather than papered over: the shell-dialect rule's decider does not grant
		// mention-immunity, so a cmdlet name quoted inside an argument (a regex
		// literal, a grep pattern) still trips it — observed while writing this test.
		// Fixing that means teaching that decider to skip quoted words, which is a
		// separate change from this one.
		t.Logf("tolerated: tool %q fix text matches its own deny_regex, but the rule is decided structurally so a quoted mention is admitted (%q)", r.Tool, r.DenyRegex)
	}
	if seen == 0 {
		t.Fatal("no deny_regex arg rules found in the shipped policy — the manifest shape changed and this test is now vacuous")
	}
}
