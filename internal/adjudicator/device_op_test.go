package adjudicator

import (
	"context"
	"encoding/json"
	"os"
	"regexp"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// device_op_test.go — the two faces of the disk/device verb decider (#5429), plus the
// fail-closed face that the sibling families do not have.
//
// Face (1) MENTION: naming one of these verbs in a quoted search pattern, a commit
// message or a here-doc body is not a device operation and must be ADMITTED.
// Face (2) USE: the verb at a resolved command word, or a redirect into a raw block
// device, is the operation itself and must stay REFUSED on every surface.
// Face (3) FAIL-CLOSED: a mention this walk cannot vouch for — an evaluator, a
// launcher, an unresolvable command word, an unparseable quote — must be REFUSED,
// because admitting is the only outcome that could be wrong on a refusal path.

// deviceOpAdj builds an adjudicator carrying ONLY the named shipped spelling, so
// isDeviceOpArgRule recognises it and the structural path decides.
func deviceOpAdj(tool, spelling string) *Adjudicator {
	return New(Policy{
		Allow: map[string]bool{tool: true},
		ArgPredicates: []ArgPredicate{{
			Tool: tool, Arg: "command", Kind: ArgDenyRegex,
			Re: regexp.MustCompile(spelling), Reason: abi.ReasonPolicyBlock,
		}},
	})
}

// TestDiskDeviceVerbMentionIsAdmitted is face (1).
//
// Every command here MATCHES one of the two shipped raw regexes — that is the point,
// it is what made each of them a POLICY_BLOCK before the decider — and every one of
// them performs no device operation whatsoever. The three shapes that bite in
// practice are all represented: a quoted search pattern over the policy file that
// ships the rule, a commit message explaining why the rule exists, and a here-doc
// body writing that explanation into a document.
//
// MUTANT THAT REDS THIS: make commandPerformsDeviceOperation return true
// unconditionally (or delete the deviceOpInertHeads allow-list consultation so every
// mention-carrying segment fails closed). Both restore the raw-regex verdict and every
// case below flips to a refusal.
func TestDiskDeviceVerbMentionIsAdmitted(t *testing.T) {
	mentions := []struct {
		name string
		cmd  string
	}{
		// A quoted search pattern over the checked-in policy file that ships the rule.
		{"posix verb in a grep pattern", `grep -rn 'mkfs' cmd/fak/guard-default-policy.json`},
		{"byte-copy verb in a search pattern", `rg -n "dd if=" docs/`},
		{"device redirect in a search pattern", `grep -rn '> /dev/sd' cmd/fak/guard-default-policy.json`},
		{"volume cmdlet in a search pattern", `rg -n 'Clear-Disk' docs/`},
		{"volume cmdlet in a windows-path search", `Select-String -Pattern 'Format-Volume' -Path cmd\fak\guard-default-policy.json`},

		// A commit message. Documenting the guard is routine work on this repo, and it
		// is the shape most likely to name a guarded pattern.
		{"commit message naming the posix verb", `git commit -m "docs(guard): explain why mkfs is refused (fak adjudicator)"`},
		{"commit message naming the byte-copy verb", `git commit -m "docs(guard): note that dd if=/dev/zero of=/dev/sda is operator-only"`},
		{"commit message naming a volume cmdlet", `git commit -m "docs(guard): Initialize-Disk stays operator-only (fak adjudicator)"`},
		{"log search naming the verb", `git log --grep 'mkfs'`},

		// Printing the refusal for a human to read — the route the rule's own remedy
		// used to forbid because printing it re-tripped the rule.
		{"echoing the refusal", `echo 'dd if=/dev/zero of=/dev/sda is operator-only'`},
		{"echoing a device path", `echo "never redirect into /dev/sda"`},
		{"printing a volume cmdlet", `Write-Output "Initialize-Disk is refused"`},

		// A here-doc body is file CONTENT, not a command line.
		{"here-doc body naming the verbs", "cat > docs/guard.md <<'EOF'\nThe floor refuses mkfs and dd if=/dev/zero of=/dev/sda outright.\nEOF"},
		{"here-doc body naming a volume cmdlet", "cat > docs/guard.md <<'EOF'\nFormat-Volume and Clear-Disk are operator-only.\nEOF"},

		// A read-only pipeline threaded through the text utilities.
		{"search piped through a pager stage", `grep -rn 'mkfs' cmd/fak | head -5`},
		{"search counted", `grep -rn 'Clear-Disk' docs/ | wc -l`},
	}
	for _, tc := range mentions {
		t.Run(tc.name, func(t *testing.T) {
			if commandPerformsDeviceOperation(tc.cmd) {
				t.Errorf("a MENTION was decided as a device operation: %q", tc.cmd)
			}
		})
	}
}

// TestDiskDeviceVerbUseStaysRefused is face (2): nothing this family denies today
// becomes admitted. Each case is a real operation, on every launder the walk unwraps.
//
// MUTANT THAT REDS THIS: make commandPerformsDeviceOperation return false
// unconditionally, or drop isDeviceOpCommandWord's check from posixDeviceOperation /
// psDeviceOperation, or delete the redirectWriteTargets scan (which reds only the
// redirect cases — they are the ones no command-word check can catch).
func TestDiskDeviceVerbUseStaysRefused(t *testing.T) {
	uses := []struct {
		name string
		cmd  string
	}{
		{"filesystem creation at the command word", `mkfs /dev/sda1`},
		{"per-filesystem variant", `mkfs.ext4 /dev/sdb1`},
		{"absolute path to the verb", `/sbin/mkfs.xfs /dev/sdc`},
		{"escalated", `sudo mkfs.ext4 /dev/sdb1`},
		{"env-prefixed", `env FORCE=1 mkfs /dev/sda`},
		{"byte copy onto a device", `dd if=/dev/zero of=/dev/sda`},
		{"byte copy with block size", `dd if=/dev/zero of=/dev/sda bs=1M`},
		{"laundered one level through a shell", `sh -c 'mkfs /dev/sda'`},
		{"reached through a transparent wrapper", `find . -name '*.img' | xargs mkfs`},
		{"a later pipeline stage", `cat image.bin | dd if=/dev/stdin of=/dev/sda`},

		// The redirect alternative: no command word is the verb, the TARGET is.
		{"redirect into a raw device", `echo x > /dev/sda`},
		{"append into a raw device", `cat payload.img >> /dev/sdb`},

		// The PowerShell volume rule's own surface.
		{"volume format cmdlet", `Format-Volume -DriveLetter D`},
		{"disk wipe cmdlet", `Clear-Disk -Number 1 -RemoveData`},
		{"disk initialise cmdlet", `Initialize-Disk -Number 2`},
		{"call operator does not hide the cmdlet", `& Format-Volume -DriveLetter D`},
		{"nested in a host payload", `powershell -Command "Format-Volume -DriveLetter D"`},
	}
	for _, tc := range uses {
		t.Run(tc.name, func(t *testing.T) {
			if !commandPerformsDeviceOperation(tc.cmd) {
				t.Errorf("a real device operation was admitted as a mention: %q", tc.cmd)
			}
		})
	}
}

// TestDiskDeviceDeciderFailsClosed is face (3), and it is the face the sibling
// families do NOT assert: every case here is a command whose verb sits inside an
// operand rather than at a command word, so a decider that only asked "is the command
// word one of these verbs" would ADMIT all of them. Each is an indirection this walk
// does not follow, or a line it cannot parse, and a refusal path must resolve
// uncertainty toward refusing.
//
// MUTANT THAT REDS THIS: replace the fail-closed branch in posixDeviceOperation
// (`if i < 0 || !deviceOpInertHeads[…] { return true }`) with a `continue`, or make
// psDeviceOperation return false when psSegments reports an unterminated quote. Note
// that the mutants which red TestDiskDeviceVerbUseStaysRefused do NOT red these cases
// and vice versa — the two faces are pinned by different code.
func TestDiskDeviceDeciderFailsClosed(t *testing.T) {
	unproven := []struct {
		name string
		cmd  string
	}{
		{"evaluator re-executes its operand", `eval "mkfs /dev/sda"`},
		{"awk can shell out", `awk 'BEGIN{system("mkfs /dev/sda")}'`},
		{"an interpreter program is not data", `python3 -c "import os; os.system('mkfs /dev/sda')"`},
		{"a stream editor that can execute", `sed -e 's/x/mkfs \/dev\/sda/e' file`},
		{"a remote payload this local walk does not open", `ssh gpu-box 'mkfs /dev/sda'`},
		{"variable indirection", `M=mkfs; $M /dev/sda`},
		{"an unterminated quote hides the rest of the line", `echo 'mkfs /dev/sda`},
		{"a host launcher payload", `cmd /c "Format-Volume -DriveLetter D"`},
		{"an unreadable encoded payload", `powershell -EncodedCommand RgBvAHIAbQBhAHQALQBWAG8AbAB1AG0AZQA=  # Format-Volume`},
	}
	for _, tc := range unproven {
		t.Run(tc.name, func(t *testing.T) {
			if !commandPerformsDeviceOperation(tc.cmd) {
				t.Errorf("an unprovable mention was ADMITTED; uncertainty on a refusal path must keep the deny: %q", tc.cmd)
			}
		})
	}
}

// TestDiskDeviceDeciderIsSubtractive pins the one-directional contract: the decider is
// consulted only after the raw regex has matched, and it must never turn a command the
// regex would NOT have flagged into a refusal. These commands name none of the verbs,
// so no rule fires and the structural walk must not invent one.
//
// MUTANT THAT REDS THIS: drop the `pr.Re.MatchString(canon)` gate from the dispatch
// branch in evalArgPredicates and widen deviceOpCommandWords to a shape match (e.g.
// any `mkfs`-prefixed or any `*-Disk` head).
func TestDiskDeviceDeciderIsSubtractive(t *testing.T) {
	a := deviceOpAdj("Bash", defaultDeviceOpDenyRegex)
	for _, cmd := range []string{
		`ls -la`,
		`go build ./...`,
		`git status --porcelain`,
		`mkswap /dev/sdz`,     // a neighbouring verb the shipped policy does NOT name
		`ddrescue a b`,        // no word boundary after the byte-copy verb
		`echo "add a widget"`, // "dd" only as a substring
	} {
		v := a.Adjudicate(context.Background(), inlineCall("Bash", jsonCmd(cmd)))
		if v.Kind != abi.VerdictAllow {
			t.Errorf("the decider added a deny the raw regex never made for %q: got %v/%s",
				cmd, v.Kind, abi.ReasonName(v.Reason))
		}
	}
}

// TestDiskDeviceDispatchIsWired proves the branch is reached through the real
// adjudicate path on both surfaces, not merely that the pure predicate is correct — a
// decider nothing calls is a decider that does nothing.
//
// MUTANT THAT REDS THIS: delete the `if isDeviceOpArgRule(pr)` branch from
// evalArgPredicates, which drops both rules back to the raw-regex fall-through.
func TestDiskDeviceDispatchIsWired(t *testing.T) {
	cases := []struct {
		name     string
		tool     string
		spelling string
		cmd      string
		want     abi.VerdictKind
	}{
		{"posix mention admitted", "Bash", defaultDeviceOpDenyRegex,
			`grep -rn 'mkfs' cmd/fak/guard-default-policy.json`, abi.VerdictAllow},
		{"posix use refused", "Bash", defaultDeviceOpDenyRegex,
			`mkfs /dev/sda1`, abi.VerdictDeny},
		{"posix device redirect refused", "Bash", defaultDeviceOpDenyRegex,
			`echo x > /dev/sda`, abi.VerdictDeny},
		{"powershell mention admitted", "PowerShell", defaultPSDiskOpDenyRegex,
			`Select-String -Pattern 'Format-Volume' -Path cmd\fak\guard-default-policy.json`, abi.VerdictAllow},
		{"powershell use refused", "PowerShell", defaultPSDiskOpDenyRegex,
			`Format-Volume -DriveLetter D`, abi.VerdictDeny},
		// The mirror surface ships BOTH spellings; a rule recognised on Bash but not
		// here is the surface-parity defect the package already closed twice.
		{"mirror surface mention admitted", "shell_command", defaultDeviceOpDenyRegex,
			`git commit -m "docs(guard): explain why mkfs is refused (fak adjudicator)"`, abi.VerdictAllow},
		{"mirror surface use refused", "shell_command", defaultDeviceOpDenyRegex,
			`dd if=/dev/zero of=/dev/sda`, abi.VerdictDeny},
		{"mirror surface volume mention admitted", "functions.shell_command", defaultPSDiskOpDenyRegex,
			`rg -n 'Clear-Disk' docs/`, abi.VerdictAllow},
		{"mirror surface volume use refused", "functions.shell_command", defaultPSDiskOpDenyRegex,
			`Clear-Disk -Number 1 -RemoveData`, abi.VerdictDeny},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := deviceOpAdj(tc.tool, tc.spelling)
			v := a.Adjudicate(context.Background(), inlineCall(tc.tool, jsonCmd(tc.cmd)))
			if v.Kind != tc.want {
				t.Errorf("%s: %q got %v/%s, want %v", tc.tool, tc.cmd, v.Kind, abi.ReasonName(v.Reason), tc.want)
			}
		})
	}
}

// TestDiskDeviceRuleRecognisedByExactSpelling guards against sentinel drift in BOTH
// directions: the shipped spellings must be recognised on every surface that ships
// them, and a differently-spelled or differently-scoped rule must keep the raw-regex
// path. The recogniser compares Re.String() EXACTLY, so a one-character policy edit
// disables the decider outright and nothing else would say so.
//
// MUTANT THAT REDS THIS: change one byte of defaultDeviceOpDenyRegex or
// defaultPSDiskOpDenyRegex, or widen isDeviceOpArgRule to accept any tool.
func TestDiskDeviceRuleRecognisedByExactSpelling(t *testing.T) {
	recognised := []struct{ tool, spelling string }{
		{"Bash", defaultDeviceOpDenyRegex},
		{"shell_command", defaultDeviceOpDenyRegex},
		{"functions.shell_command", defaultDeviceOpDenyRegex},
		{"PowerShell", defaultPSDiskOpDenyRegex},
		{"shell_command", defaultPSDiskOpDenyRegex},
		{"functions.shell_command", defaultPSDiskOpDenyRegex},
	}
	for _, tc := range recognised {
		pr := &ArgPredicate{Tool: tc.tool, Arg: "command", Kind: ArgDenyRegex,
			Re: regexp.MustCompile(tc.spelling)}
		if !isDeviceOpArgRule(pr) {
			t.Errorf("tool %q: the shipped spelling is not recognised — the rule silently falls back to the raw regex on that surface", tc.tool)
		}
	}
	rejected := []struct {
		name     string
		tool     string
		arg      string
		spelling string
	}{
		{"posix spelling on the PowerShell surface", "PowerShell", "command", defaultDeviceOpDenyRegex},
		{"volume spelling on the Bash surface", "Bash", "command", defaultPSDiskOpDenyRegex},
		{"a non-shell tool", "Read", "command", defaultDeviceOpDenyRegex},
		{"a non-command arg", "Bash", "file_path", defaultDeviceOpDenyRegex},
		{"a differently-spelled rule", "Bash", "command", `\bmkfs\b`},
	}
	for _, tc := range rejected {
		pr := &ArgPredicate{Tool: tc.tool, Arg: tc.arg, Kind: ArgDenyRegex,
			Re: regexp.MustCompile(tc.spelling)}
		if isDeviceOpArgRule(pr) {
			t.Errorf("%s: must keep the raw-regex path, not the structural one", tc.name)
		}
	}
}

// TestDiskDeviceSpellingsAreByteIdenticalToTheShippedPolicy is the other half of the
// drift guard, read from the shipped file rather than restated: each constant must
// appear VERBATIM as a deny_regex in cmd/fak/guard-default-policy.json, on the exact
// set of surfaces this decider claims. The parity test asserts that no shipped rule
// goes undecided; this asserts that no CONSTANT goes unshipped, which is the failure
// mode where the decider compiles, passes its own unit tests, and never fires in
// production.
func TestDiskDeviceSpellingsAreByteIdenticalToTheShippedPolicy(t *testing.T) {
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
	want := map[string]map[string]bool{
		defaultDeviceOpDenyRegex: {"bash": false, "shell_command": false, "functions.shell_command": false, "exec_command": false},
		defaultPSDiskOpDenyRegex: {"powershell": false, "shell_command": false, "functions.shell_command": false, "exec_command": false},
	}
	for _, r := range manifest.ArgRules {
		surfaces, ok := want[r.DenyRegex]
		if !ok {
			continue
		}
		tool := strings.ToLower(r.Tool)
		if _, expected := surfaces[tool]; !expected {
			t.Errorf("deny_regex %q ships on tool %q, which this decider does not claim — extend isDeviceOpArgRule or that surface keeps the raw-regex path", r.DenyRegex, r.Tool)
			continue
		}
		surfaces[tool] = true
	}
	for spelling, surfaces := range want {
		for tool, seen := range surfaces {
			if !seen {
				t.Errorf("no shipped rule pairs tool %q with %q — either the policy spelling drifted (the recogniser matches EXACTLY, so the decider is now dead on that surface) or the rule was dropped", tool, spelling)
			}
		}
	}
}
