package adjudicator

import (
	"encoding/json"
	"os"
	"regexp"
	"strings"
	"testing"
)

// shippedUndecidedRules inventories the shipped deny_regex rules that have NO
// structural decider today, keyed by a distinguishing substring. They are listed
// rather than merely skipped so that adding a rule to the policy forces a conscious
// choice — write a decider, or record here why the raw regex is the whole truth for
// it. Each entry is a standing candidate for the same treatment the decided
// families already got.
//
// The list is package-level because two tests need the same answer to "does this
// rule tell a use from a MENTION": the parity test above, and the remedy test that
// holds these rules to a stricter standard for exactly that reason.
//
// The disk/device verbs are no longer here. They were the two longest-standing
// entries — the POSIX raw-device rule and the PowerShell volume rule — and both are
// now decided structurally (device_op.go, #5429), so they have moved into the
// families list below. What they DENY is unchanged; only the mention stopped being
// reported as a use.
var shippedUndecidedRules = []struct{ marker, why string }{
	{"Clear-Content", "split out of the disk rule, which claimed a file truncation was a 'disk/volume operation' and sent the caller to an operator with nothing to approve. Still denied — an in-place truncation has no preview and no undo — but its remedy now names the in-tree routes that were always admitted. A containment decider (empty a file under the working tree or a scratchpad root, deny outside) is the obvious next step and needs a new dispatch call site"},
	{`os\.system`, "the execute_code surface, which is not a shell command line"},
	{`:\(\)`, "fork bomb; prose describes the shape well enough that the remedy needs no literal, which is why its fix text now omits one"},
	{`(?i:true)`, "the effectful-apply flag on fak_memory_run. Unlike every other entry here this one is not a decider waiting to be written: the arg is a BOOLEAN, so the anchored regex has exactly two possible inputs and no room to quote anything. The raw regex is the whole truth by construction, and the mention hazard the other entries carry does not exist for it (see freeTextArg)"},
	{`\bfind\s+`, "fence against broad recursive searches on user root"},
	{`USERPROFILE`, "PowerShell fence against broad recursive searches on user root"},
}

// freeTextArg reports whether an arg's VALUE can carry a quoted MENTION of the
// pattern its rule matches — a shell command line or a code body, where `grep
// 'mkfs'` and a real mkfs are the same bytes to a raw regex. The hazard is a
// property of the VALUE's shape, not of the rule: a scalar flag (apply=true) has
// exactly two possible values and nowhere to quote a mention, so a rule over one
// cannot confuse a use with a mention — and its remedy must not claim it can.
func freeTextArg(arg string) bool {
	switch strings.ToLower(arg) {
	case "command", "cmd", "code":
		return true
	}
	return false
}

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
		{"device_op", isDeviceOpArgRule, []string{defaultDeviceOpDenyRegex, defaultPSDiskOpDenyRegex}},
		{"build_cache_clean", isBuildCacheCleanArgRule, []string{defaultBuildCacheCleanDenyRegex}},
		{"git_push", isGitPushArgRule, []string{defaultGitPushDenyRegex, defaultPSGitPushDenyRegex}},
	}

	undecided := shippedUndecidedRules

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
			isShellDialectArgRule(pr) || isOutOfTreeWriteArgRule(pr) ||
			isDeviceOpArgRule(pr) || isBuildCacheCleanArgRule(pr) ||
			isGitPushArgRule(pr)
		if !decided {
			t.Errorf("tool %q arg %q: the fix text for %q MATCHES its own deny_regex, and this rule has NO structural decider — the match IS the verdict, so quoting this refusal re-trips the rule that produced it:\n  fix: %s",
				r.Tool, r.Arg, r.DenyRegex, r.Fix)
			continue
		}
		// Tolerated, but recorded: the decider is what makes it safe, so if one is
		// ever narrowed the note below is where to look first.
		//
		// An earlier revision of this note recorded the shell-dialect rule as the one
		// entry here that was NOT safe — the claim being that its decider refused a
		// cmdlet name quoted inside an argument (a regex literal, a grep pattern).
		// That claim was wrong, and it was already contradicted by a test one file
		// over: commandLeadsWithPowerShellCmdlet resolves the command WORD of each
		// segment and matches a cmdlet only in that position, so a quoted mention
		// never reaches it. TestShellDialectStructuralNoFalsePositive has pinned that
		// since the decider shipped; its corpus now also carries the commit-message
		// and git-grep shapes, which are how a mention usually arrives in practice.
		t.Logf("tolerated: tool %q fix text matches its own deny_regex, but the rule is decided structurally so a quoted mention is admitted (%q)", r.Tool, r.DenyRegex)
	}
	if seen == 0 {
		t.Fatal("no deny_regex arg rules found in the shipped policy — the manifest shape changed and this test is now vacuous")
	}
}

// TestUndecidedRuleRemediesAreNotSelfRefuting holds the raw-regex rules to the one
// standard their decided siblings get for free.
//
// A decided rule can afford a remedy that names its own subject: the decider
// resolves real command words, so `grep -rn "sudo" docs/` is admitted and quoting
// the refusal costs nothing. A rule with NO decider has no such slack — the match
// IS the verdict, so every route its remedy names must survive the rule itself.
//
// Two shipped remedies did not. The disk/device and disk/volume rules both ended
// "print the exact command and what it is for, and ask the operator" — and printing
// it is a shell command line containing the pattern, so the rule refuses that too.
// The caller is sent in a circle, and because a POLICY_BLOCK under `fak guard --
// claude` reads upstream as an agent-chosen end_turn, they do not get a second look
// to notice. Worse, the refusal misdescribes what happened: a `grep` for the string
// is reported as an attempted "disk/device operation", so an agent that never went
// near a device is told it tried to destroy one.
//
// Those two rules have since been DECIDED (device_op.go, #5429), which is the real
// cure rather than the disclosure, so they no longer fall under this test — the
// assertions below now cover the remaining raw-regex rules. Their shipped fix text
// still carries the "no structural decider ... CANNOT tell a use from a MENTION"
// disclosure, which is now stale prose in cmd/fak/guard-default-policy.json; it
// under-promises rather than over-promises, so it is safe but wants a follow-up edit
// landed in the same change as the decider.
//
// So this asserts both halves of an honest raw-regex remedy:
//
//	NEGATIVE — it must not route the caller through printing or echoing the
//	command, because the same rule refuses that.
//	POSITIVE — it must SAY that it cannot tell a use from a mention, because that
//	is the fact the caller needs in order to pick a route that works.
//
// Neither half asks the rule to be more permissive. Every verb these rules name —
// the in-place truncation cmdlet included — stays denied on every surface it ships
// on; only the remedy stops lying about the way out.
func TestUndecidedRuleRemediesAreNotSelfRefuting(t *testing.T) {
	b, err := os.ReadFile("../../cmd/fak/guard-default-policy.json")
	if err != nil {
		t.Fatalf("read shipped policy: %v", err)
	}
	var manifest struct {
		ArgRules []struct {
			Tool      string `json:"tool"`
			Arg       string `json:"arg"`
			DenyRegex string `json:"deny_regex"`
			Fix       string `json:"fix"`
		} `json:"arg_rules"`
	}
	if err := json.Unmarshal(b, &manifest); err != nil {
		t.Fatalf("parse shipped policy: %v", err)
	}

	// Imperative forms only: these are remedies that TELL the caller to print the
	// command. A remedy that instead DISCLOSES the limitation ("echoing the command
	// is refused for that same reason") is the cure, not the defect, and reads as
	// "echoing the", which none of these match.
	selfRefuting := []string{
		"print the exact command",
		"print the command",
		"echo the command",
	}

	hits := map[string]int{}
	for _, r := range manifest.ArgRules {
		if r.DenyRegex == "" {
			continue
		}
		var why string
		for _, u := range shippedUndecidedRules {
			if strings.Contains(r.DenyRegex, u.marker) {
				why, hits[u.marker] = u.why, hits[u.marker]+1
				break
			}
		}
		if why == "" {
			continue // decided elsewhere, or unknown — the parity test owns that case
		}
		fix := strings.ToLower(r.Fix)
		for _, bad := range selfRefuting {
			if strings.Contains(fix, bad) {
				t.Errorf("tool %q arg %q: deny_regex %q has NO structural decider, yet its remedy routes the caller through %q — a shell command line that prints this pattern trips this very rule, so the only route it names is one it refuses:\n  fix: %s",
					r.Tool, r.Arg, r.DenyRegex, bad, r.Fix)
			}
		}
		if freeTextArg(r.Arg) && !strings.Contains(fix, "mention") {
			t.Errorf("tool %q arg %q: deny_regex %q has NO structural decider, so it refuses every quoted MENTION of the pattern — a grep, a commit message, a here-doc — but its remedy never says so. The caller is left to infer it from a refusal that claims they performed the operation:\n  fix: %s",
				r.Tool, r.Arg, r.DenyRegex, r.Fix)
		}
	}

	for _, u := range shippedUndecidedRules {
		if hits[u.marker] == 0 {
			t.Errorf("undecided marker %q matches no shipped rule — either it gained a decider (move it to the families list in TestEveryShippedStructuralRuleIsRecognised) or it was dropped from the policy; either way this test is silently covering nothing.\n  recorded rationale: %s", u.marker, u.why)
		}
	}
}
