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
		{"rce_pipe", isRCEPipeArgRule, []string{legacyRCEPipeDenyRegex, defaultRCEPipeDenyRegex}},
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
		{"Format-Volume", "PowerShell disk verbs; Clear-Content is a FILE verb mis-filed here"},
		{"Invoke-Expression", "the PowerShell download-pipe mirror; psSegments would serve it"},
		{`os\.system`, "the execute_code surface, which is not a shell command line"},
		{`:\(\)`, "fork bomb; the spelling is narrow enough that a mention is unlikely"},
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
