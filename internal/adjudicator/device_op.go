package adjudicator

import (
	"regexp"
	"strings"
)

// device_op.go — structural decision for the DISK/DEVICE VERB arg-rule family: the
// POSIX raw-device rule (`\bmkfs\b|\bdd\s+if=|>\s*/dev/sd`, shipped on Bash,
// shell_command, functions.shell_command, and exec_command) and its PowerShell volume sibling
// (`(?i)\b(Format-Volume|Clear-Disk|Initialize-Disk)\b`, shipped on PowerShell and
// the same three mirrors) in cmd/fak/guard-default-policy.json.
//
// WHY THIS EXISTS (#5429). Both rules were RAW pattern matches with no structural
// decider, so the match WAS the verdict and a MENTION could not be told from a USE.
// Naming one of these verbs anywhere on a shell command line was refused — inside a
// quoted grep pattern, inside a commit message, inside a here-doc body — and the
// refusal then reported a documentation edit as an attempted device operation, which
// is a claim about the caller that is simply false. Under `fak guard -- claude` a
// POLICY_BLOCK reads upstream as an agent-chosen end_turn rather than a refusal, so
// the caller never got a second look to notice. The rules' own remedies had already
// been amended to DISCLOSE the limitation; this closes it instead, the same way the
// decided families (delete, download-pipe, out-of-tree-write, elevation, terraform,
// shell-dialect) closed theirs — by resolving the COMMAND WORD.
//
// WHAT CHANGES, AND WHAT DOES NOT. What CHANGES is the verdict on a MENTION: a
// command line that only NAMES one of these verbs is denied today and is admitted
// after this, which is the whole point of the change. What does NOT change is the
// verdict on an OPERATION — no real device operation that is denied today becomes
// admitted. The decider is consulted only AFTER the rule's own raw regex has already
// matched (decide_argpredicates.go gates on it), it can only ever DOWNGRADE that
// match, and it downgrades only a mention it can affirmatively vouch for: an
// unresolvable command word, an evaluator or launcher head, and an unterminated
// quote each keep the deny. It never introduces a new deny, and it never widens what
// a real device operation may do: laying a filesystem, the raw byte-copy verb, a
// redirect into a raw block device and the PowerShell volume cmdlets all stay
// operator-only on every surface they ship on.
//
// THE POSTURE IS FAIL-CLOSED, and deliberately stricter than the deny-list shape the
// sibling deciders use. This is a refusal path: uncertainty must resolve toward
// REFUSING, so a mention is admitted only against an ALLOW-LIST of command words that
// provably cannot turn an operand back into a command. A deny-list of evaluators
// would be fail-OPEN on every construct nobody thought of.

const (
	// defaultDeviceOpDenyRegex is the one canonical spelling of the shipped POSIX
	// raw-device / filesystem-creation deny_regex. cmd/fak/guard-default-policy.json
	// ships it four times — Bash, shell_command, functions.shell_command, exec_command —
	// byte-identical. The rule is RECOGNISED by this exact string and then decided
	// structurally; a policy that ships a different spelling is unaffected and keeps
	// the raw-regex path.
	defaultDeviceOpDenyRegex = `\bmkfs\b|\bdd\s+if=|>\s*/dev/sd`

	// defaultPSDiskOpDenyRegex is the PowerShell volume/disk sibling, shipped four
	// times in the same file — PowerShell, shell_command, functions.shell_command,
	// exec_command.
	// Recognising it here is what stops the same command getting two different
	// verdicts decided by nothing but which tool NAME the harness happens to use,
	// the surface-parity defect pinned by TestEveryShippedStructuralRuleIsRecognised.
	defaultPSDiskOpDenyRegex = `(?i)\b(Format-Volume|Clear-Disk|Initialize-Disk)\b`

	// rawBlockDevicePrefix is the block-device path family the third alternative of
	// the POSIX regex names (`>\s*/dev/sd`). That alternative is about a REDIRECT
	// TARGET rather than a command word, so it is decided by the quote-aware
	// redirect extractor rather than by command-word resolution.
	rawBlockDevicePrefix = "/dev/sd"
)

// isDeviceOpArgRule reports whether pr is one of the two shipped disk/device
// deny_regexes on a shell command arg, on a surface that ships it. The recogniser
// matches the spelling EXACTLY: editing the shipped policy by one character disables
// this decider outright and silently drops the rule back to the raw-regex path, which
// is why the parity test asserts both directions.
func isDeviceOpArgRule(pr *ArgPredicate) bool {
	if pr == nil || pr.Re == nil || (pr.Arg != "command" && pr.Arg != "cmd") {
		return false
	}
	re := pr.Re.String()
	switch strings.ToLower(pr.Tool) {
	case "bash":
		return re == defaultDeviceOpDenyRegex
	case "powershell":
		return re == defaultPSDiskOpDenyRegex
	case "shell_command", "functions.shell_command", "exec_command":
		// This surface is POSIX on one host and PowerShell on another, so the
		// shipped policy gives it BOTH rules; recognise both.
		return re == defaultDeviceOpDenyRegex || re == defaultPSDiskOpDenyRegex
	default:
		return false
	}
}

// deviceOpCommandWords are the verbs that, standing at a RESOLVED command-word
// position, ARE the operation these rules refuse. Keyed by the lowered basename both
// rceProgramBasename and psProgramBasename produce, so `/sbin/mkfs`, `dd.exe` and
// `Format-Volume` all land here.
//
// The set is exactly what the two shipped regexes name — no more. Adding a verb the
// policy does not ship (mkswap, newfs, diskpart, wipefs) would be a WIDENING of the
// floor smuggled in under a false-positive fix, which is not what this change is.
var deviceOpCommandWords = map[string]bool{
	"dd":              true,
	"format-volume":   true,
	"clear-disk":      true,
	"initialize-disk": true,
}

// isDeviceOpCommandWord reports whether an already-lowered program basename is one of
// the verbs above, including the filesystem-creation verb's per-filesystem variants
// (`mkfs.ext4`, `mkfs.xfs`, `mkfs-foo`). Those variants are in scope because `.` and
// `-` are word boundaries, so the shipped `\bmkfs\b` alternative matches them too;
// `mkfs2` is NOT, for the same reason (no boundary), and is left exactly as the raw
// regex left it.
func isDeviceOpCommandWord(base string) bool {
	if deviceOpCommandWords[base] {
		return true
	}
	if base == "mkfs" {
		return true
	}
	return strings.HasPrefix(base, "mkfs.") || strings.HasPrefix(base, "mkfs-")
}

// deviceOpMentionRe reports whether a single TOKEN names any verb this family
// governs. It is deliberately BROADER than the two shipped regexes — it needs no
// `if=` after the byte-copy verb and no `>` before the device path, and it is
// case-insensitive on both dialects — because it decides only whether a segment must
// be VOUCHED FOR before the raw match may be downgraded. Over-reporting there keeps a
// deny the raw match had already made; under-reporting would admit one.
var deviceOpMentionRe = regexp.MustCompile(`(?i)\bmkfs|\bdd\b|/dev/sd|\bformat-volume\b|\bclear-disk\b|\binitialize-disk\b`)

// deviceOpInertHeads is the ALLOW-LIST of command words for which an operand naming
// one of these verbs is provably DATA rather than a command.
//
// It is an allow-list, not a deny-list of evaluators, on purpose: this is a refusal
// path, and a deny-list is fail-OPEN on every construct nobody thought of. The base
// set is ootInertMentionVerbs verbatim (oot_mention.go) — echo, printf, the greps, rg
// and git, none of which executes an operand — so the two families cannot drift on
// what "inert" means. Layered on top are the read-only text utilities that carry the
// shapes THIS rule actually bites on and that likewise never execute an operand:
// `cat`, whose redirected here-doc is how a document naming these verbs gets written;
// the pipeline stages a search is normally threaded through; and PowerShell's own
// read-only search/print cmdlets, because the volume rule ships on the PowerShell
// surface where `Select-String -Pattern '<cmdlet>'` is the shape of a mention.
//
// Deliberately ABSENT, and therefore REFUSED: eval, the shells, xargs with a
// replacement string, find (-exec), awk (system()), sed (GNU's `e`), the language
// interpreters, ssh, every launcher, and the PowerShell cmdlets that WRITE
// (Out-File, Set-Content, Add-Content). Each can turn a quoted operand back into a
// command word this walk never sees, or can write to a device path.
var deviceOpInertHeads = func() map[string]bool {
	heads := make(map[string]bool, len(ootInertMentionVerbs)+20)
	for verb := range ootInertMentionVerbs {
		heads[verb] = true
	}
	for _, verb := range []string{
		// POSIX read-only text utilities.
		"cat", "head", "tail", "wc", "sort", "uniq", "cut", "nl", "diff", "comm", "tr",
		// PowerShell's read-only search/print cmdlets, for the volume rule's surface.
		"select-string", "get-content", "write-output", "write-host",
		"select-object", "sort-object", "measure-object",
	} {
		heads[verb] = true
	}
	return heads
}()

// commandPerformsDeviceOperation reports whether cmd actually PERFORMS one of the
// disk/device operations these rules refuse, as opposed to merely NAMING one.
//
// It is used SUBTRACTIVELY: decide_argpredicates.go consults it only once the rule's
// own raw regex has already matched, and a false result downgrades that match to an
// admit. So it can never introduce a NEW deny for a command the regex would not have
// flagged, and every uncertainty resolves toward keeping the deny.
//
// The command is decided under BOTH dialects and the deny is kept if EITHER reads a
// real operation, exactly as commandAppliesTerraformDestroy does: these rules ship on
// a POSIX surface, a PowerShell surface, and two mirrors that are either depending on
// the host, so deciding under only one lexer would let the other dialect's quoting
// hide an invocation.
func commandPerformsDeviceOperation(cmd string) bool {
	return posixDeviceOperation(cmd) || psDeviceOperation(cmd)
}

// posixDeviceOperation decides cmd under POSIX lexing, reusing the shared rce walker
// so `sh -c '…'`, `$(…)` and backtick payloads are unwrapped and decided as
// statements in their own right, and a provably inert here-doc body is stripped
// before anything is read as a command line.
func posixDeviceOperation(cmd string) bool {
	sources := rceShellSources(cmd)
	if len(sources) == 0 {
		// FAIL CLOSED. The raw regex matched something, and the tokenizer read no
		// source at all out of it — there is nothing to base an admit on, and an
		// admit is the only outcome that could be wrong here. Refuse.
		return true
	}
	for _, src := range sources {
		// The third alternative of the POSIX regex is a REDIRECT into a raw block
		// device, not a command word, so it is decided by the quote-aware redirect
		// extractor. A `>` inside a quoted word is not a redirection to it, which is
		// what keeps `echo '> /dev/sda'` and a grep pattern out of this branch.
		for _, target := range redirectWriteTargets(src) {
			if isRawBlockDevicePath(target) {
				return true
			}
		}
		for _, seg := range rceShellSegments(src) {
			// Resolve past env-assignments, env, sudo, `command`, and the transparent
			// wrappers (xargs/nohup/time/…) — the same resolver the delete family
			// uses, so `sudo mkfs …` and `find … | xargs mkfs …` stay denied.
			i := rmDeleteCommandWord(seg.argv)
			if i >= 0 && isDeviceOpCommandWord(rceProgramBasename(seg.argv[i])) {
				return true // a real device verb at the resolved command word
			}
			if !argvNamesDeviceVerb(seg.argv) {
				continue // this segment names nothing this family governs
			}
			// This segment CARRIES the mention the raw regex fired on. FAIL CLOSED:
			// downgrade the deny only against the inert allow-list. An unresolvable
			// command word, or any head this package cannot vouch for — an
			// evaluator, a launcher, an indirection this walk does not unwrap —
			// keeps the deny. Uncertainty in a refusal path must never resolve
			// toward admitting.
			if i < 0 || !deviceOpInertHeads[rceProgramBasename(seg.argv[i])] {
				return true
			}
		}
	}
	return false
}

// psDeviceOperation decides cmd under PowerShell lexing (backtick escape, backslash
// as a path byte, doubled quotes), which is the dialect the volume rule's own surface
// speaks. The inert here-doc strip runs first for the same reason it does on the
// POSIX side: a `cat > file <<'EOF'` body is file CONTENT, and PowerShell's lexer
// would otherwise read each of its lines as a statement.
func psDeviceOperation(cmd string) bool {
	segs, ok := psSegments(stripInertHeredocBodies(cmd))
	if !ok {
		// FAIL CLOSED. An unterminated quoted span means the rest of the line folds
		// into one inert-looking token, which is precisely the shape that would let a
		// real device operation read as a quoted mention. Refuse.
		return true
	}
	for _, seg := range segs {
		head, _, resolved := psCommandWord(seg)
		if resolved && isDeviceOpCommandWord(head) {
			return true
		}
		if !psTokensNameDeviceVerb(seg) {
			continue
		}
		// FAIL CLOSED, for the same reason as the POSIX walk above.
		if !resolved || !deviceOpInertHeads[head] {
			return true
		}
	}
	return false
}

// argvNamesDeviceVerb reports whether any token of a POSIX-lexed segment names a verb
// this family governs.
func argvNamesDeviceVerb(argv []string) bool {
	for _, tok := range argv {
		if deviceOpMentionRe.MatchString(tok) {
			return true
		}
	}
	return false
}

// psTokensNameDeviceVerb is the PowerShell-lexed twin of argvNamesDeviceVerb.
func psTokensNameDeviceVerb(seg []psToken) bool {
	for _, tok := range seg {
		if deviceOpMentionRe.MatchString(tok.text) {
			return true
		}
	}
	return false
}

// isRawBlockDevicePath reports whether a redirect target is a raw block device of the
// family the shipped regex names. Matched by PREFIX so it tracks the regex's own
// anchoring (`>\s*/dev/sd` requires the path to START there): a target that merely
// contains the string further along, such as a file called `notes/dev/sda.md`, is not
// what the raw regex flagged and must not be treated as one.
func isRawBlockDevicePath(target string) bool {
	t := strings.ToLower(strings.TrimSpace(strings.ReplaceAll(target, `\`, "/")))
	return strings.HasPrefix(t, rawBlockDevicePrefix)
}
