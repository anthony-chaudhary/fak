package adjudicator

import (
	"regexp"
	"strings"
)

// familyMatchInput carries the precomputed views of one pending call that
// family matchers key on: the payload-scan view of the command (quoted
// mentions blanked, live client payloads kept — #2752) raw and lowered, the
// quote-aware operator-split segments, the view's word stream, the original
// command for operand-aware matchers, and the lowered tool name.
type familyMatchInput struct {
	rawCmd    string
	cmd       string
	lowerCmd  string
	segs      [][]string
	words     []string
	lowerTool string
}

// reversibilityFamily declares one escalated family in a single place: every
// surface that blocks it AND the redirect surfaced with the escalation. The
// classifier and the dry-run hint both read the SAME entry, so a family
// cannot be added to the block without deciding its redirect, and a redirect
// cannot name a surface the block never fires on (#2748, #2746).
type reversibilityFamily struct {
	name  string
	class ReversibilityClass
	// Block surfaces — any non-zero surface participates in the match.
	heads        []string                       // command segment head verbs
	prefixes     [][]string                     // command segment word prefixes
	cmdContains  []string                       // lowered whole-command substrings
	toolContains []string                       // lowered tool-name substrings
	matchCmd     func(in familyMatchInput) bool // payload-shaped matchers (curl flags, SQL words)
	// Redirect surfaced when this family blocks ("" = no sanctioned sidestep).
	hint string
}

func (f *reversibilityFamily) matches(in familyMatchInput) bool {
	if segmentHeadIs(in.segs, f.heads...) {
		return true
	}
	for _, p := range f.prefixes {
		if segmentHasPrefix(in.segs, p...) {
			return true
		}
	}
	for _, sub := range f.cmdContains {
		if strings.Contains(in.lowerCmd, sub) {
			return true
		}
	}
	for _, sub := range f.toolContains {
		if strings.Contains(in.lowerTool, sub) {
			return true
		}
	}
	return f.matchCmd != nil && f.matchCmd(in)
}

// reversibilityClassOrder fixes the escalation precedence: a call matching
// both an outward-facing family and an irreversible family previews as
// outward-facing, whatever the table order.
var reversibilityClassOrder = []ReversibilityClass{ReversibilityOutwardFacing, ReversibilityIrreversible}

func classifyAgainstFamilies(families []reversibilityFamily, tool, cmd string) (ReversibilityClass, string) {
	// Payload-shaped scans (curl flags, SQL words, cmdContains substrings) read
	// the quote-stripped view so a trigger mentioned inside a quoted argument —
	// a commit message, a grep pattern, an echoed string — is not a live token,
	// while a quoted payload handed to a DB/shell client stays live (#2752).
	view := payloadScanView(cmd)
	in := familyMatchInput{
		rawCmd:    cmd,
		cmd:       view,
		lowerCmd:  strings.ToLower(view),
		segs:      commandSegments(cmd),
		words:     commandWords(view),
		lowerTool: strings.ToLower(tool),
	}
	for _, class := range reversibilityClassOrder {
		for i := range families {
			if families[i].class != class {
				continue
			}
			if families[i].matches(in) {
				return class, families[i].hint
			}
		}
	}
	return ReversibilityReversible, ""
}

// reversibilityFamilies is the single source of truth for the escalated
// outward/destructive families (#2748). Within a class, earlier entries win
// the redirect when a call matches several families, so the hinted families
// stay first, in redirect-priority order.
//
// Most of the `gh` surface is deliberately NOT a family (operator decision,
// 2026-07-05): issue/release create·comment·edit·close·reopen·merge·upload
// targets the operator's OWN authenticated GitHub and is reversible in practice
// (issues edit·close·reopen; a release can be deleted), so the preview-confirm
// pause was pure friction on routine fleet work — the #2650/#2651 confirm-loop
// lesson — while the Claude Code allow-list already admits `Bash(gh …)`.
//
// Pull-request creation is the ONE withdrawn relaxation (operator decision,
// 2026-09-11): the pr-create family below re-escalates `gh pr create` to
// outward-facing because an agent-opened PR is not operator-requested landing
// work — it sits unattended, its checks fail, and it rebase-decays beside a
// trunk that lands direct through the fak-flow lease/land protocol (the
// audited stuck-PR classes on both repos: private fak/ticket-866 · 814 · 812
// · 729, public fak/ticket-12733 and codex/issue-12550). `gh pr` READS
// (list/view/status/checks/diff) stay reversible.
//
// The gh-write carve-out remains (#3560): `gh api` with a write
// method (--method/-X POST|PUT|PATCH|DELETE) and `gh repo fork|rename|delete` are
// re-escalated to the SAME outward-facing class as `git push`, because those
// escape the relaxation's premise. `gh api` mutations reach ARBITRARY third-party
// repos and the Git Data API (author commits/branches/trees, open cross-repo PRs,
// PATCH-rename or delete a repo) and `gh repo fork|rename|delete` always mutate —
// a strictly LARGER blast radius than the git-CLI writes the relaxation was
// matching, so leaving them allowed reroutes capability DOWNWARD past the git-push
// gate. A `gh api` READ (GET / no --method) stays reversible. curl/httpie POSTs
// likewise stay gated (http-write below): those reach ARBITRARY hosts, not the
// authenticated gh surface, so the outbound floor still holds for the general case.
var reversibilityFamilies = []reversibilityFamily{
	// An MCP tool literally named create_issue / issue_create is still
	// escalated; the redirect points at the compiled `fak issue create` verb
	// (cmd/fak/issue_create.go), a trusted-binary sidestep the kernel admits.
	// Raw `gh issue create` is no longer escalated (see the gh note above), so
	// a Bash gh call never reaches this family.
	{
		name:         "issue-create-tool",
		class:        ReversibilityOutwardFacing,
		toolContains: []string{"create_issue", "issue_create"},
		hint:         "file it with the sanctioned compiled verb: fak issue create --title … --body-file … (a trusted-binary path the kernel admits)",
	},
	// A Bash `slack …` HEAD (or an MCP tool whose name contains "slack") is
	// escalated; the redirect points at the compiled `fak slack send` verb
	// (cmd/fak/slack.go). Matched on the segment HEAD — not a bare substring —
	// so `git push origin slack-feature` keeps the git-push redirect below,
	// and sendmail/mail/mutt (no fak equivalent) stay a separate family with
	// no redirect.
	{
		name:         "slack",
		class:        ReversibilityOutwardFacing,
		heads:        []string{"slack"},
		toolContains: []string{"slack"},
		hint:         "send it with the sanctioned compiled verb: fak slack send (a trusted-binary path the kernel admits)",
	},
	// git push is escalated; the redirect names the compiled sidestep FIRST —
	// a safe, non-force push the kernel admits because its command head is
	// `fak`, not `git push` — with the --dry-run preview kept as the secondary
	// option. Before #2651's pattern was generalized here, this hint named
	// only `git push --dry-run`, which previews nothing and funnels the agent
	// straight back to the gated real push (the confirm loop in
	// docs/notes/CONFIRM-GATE-DEADLOCK-2026-07-04.md).
	{
		name:         "git-push",
		class:        ReversibilityOutwardFacing,
		prefixes:     [][]string{{"git", "push"}},
		toolContains: []string{"git_push"},
		hint:         "push with the safe compiled verb: fak sync push (a trusted-binary non-force push the kernel admits), or preview first with git push --dry-run",
	},
	{
		name:         "git-pull",
		class:        ReversibilityOutwardFacing,
		prefixes:     [][]string{{"git", "pull"}},
		toolContains: []string{"git_pull"},
		hint:         "integrate with the safe compiled verb: fak sync apply --fetch (a trusted-binary fast-forward that preserves peer WIP), or preview first with fak sync check",
	},
	// The gh-write carve-out (#3560, see the gh note above): a `gh api` mutation
	// (--method/-X POST|PUT|PATCH|DELETE) or `gh repo fork|rename|delete` is the
	// SAME outward-facing class as git push — a larger-blast-radius write than the
	// git-CLI path, so it may not slip under the gh relaxation. A `gh api` read
	// (GET / no --method) is untouched and stays reversible.
	{
		name:     "gh-write",
		class:    ReversibilityOutwardFacing,
		matchCmd: func(in familyMatchInput) bool { return ghWriteMutation(in.cmd) },
		hint:     "gh api writes and gh repo fork/rename/delete reach arbitrary GitHub repos with a larger blast radius than the git-CLI path; read instead (gh api with GET / no --method), or review the exact repo, endpoint, and method before confirming the live mutation",
	},
	{
		name:     "npm-publish",
		class:    ReversibilityOutwardFacing,
		prefixes: [][]string{{"npm", "publish"}},
		hint:     "preview the package publish first: npm publish --dry-run",
	},
	{
		name:  "mail",
		class: ReversibilityOutwardFacing,
		heads: []string{"sendmail", "mail", "mutt"},
		hint:  "review the exact recipient and body in a draft before confirming the live mail send",
	},
	{
		name:         "webhook",
		class:        ReversibilityOutwardFacing,
		cmdContains:  []string{"webhook"},
		toolContains: []string{"webhook"},
		hint:         "send to a test endpoint or use the endpoint's dry-run mode before confirming the live webhook",
	},
	{
		name:         "registry-publish",
		class:        ReversibilityOutwardFacing,
		prefixes:     [][]string{{"docker", "push"}, {"cargo", "publish"}, {"gem", "push"}, {"twine", "upload"}},
		toolContains: []string{"publish", "upload"},
		hint:         "use the registry's check or dry-run path before publishing, such as cargo publish --dry-run or twine check",
	},
	{
		name:  "http-write",
		class: ReversibilityOutwardFacing,
		matchCmd: func(in familyMatchInput) bool {
			return curlWrites(in.cmd) || httpieWrites(in.segs)
		},
		hint: "use a read-only request, a test endpoint, or the service's dry-run mode before confirming the live HTTP write",
	},
	{
		name:         "messaging-tool",
		class:        ReversibilityOutwardFacing,
		toolContains: []string{"send_email", "sendemail", "email", "send_mail", "post_message"},
		hint:         "use the sanctioned compiled verb when one exists, such as fak slack send for Slack, or review recipient/body before confirming",
	},
	{
		name:         "pr-create-tool",
		class:        ReversibilityOutwardFacing,
		toolContains: []string{"create_pr", "pr_create"},
		hint:         "preview the pull request title, body, and target branch with the host's draft or dry-run path before confirming creation",
	},
	// pr-create-bash re-escalates the Bash `gh pr create` form (operator
	// decision, 2026-09-11): an agent-opened PR is not operator-requested
	// landing work — it sits unattended while the trunk lands direct through
	// the lease/land protocol, so the fleet no longer carries this capability.
	// Matched on the segment HEAD pair so `git push origin pr-branch` and a
	// quoted mention inside a commit message stay clear.
	{
		name:     "pr-create-bash",
		class:    ReversibilityOutwardFacing,
		matchCmd: func(in familyMatchInput) bool { return ghPRCreate(in.segs) },
		hint:     "landing is operator-authorized direct to trunk via the lease/land protocol (fak-flow land); do not open a pull request unless the operator explicitly asked for one `",
	},
	{
		name:  "fs-destroy",
		class: ReversibilityIrreversible,
		heads: []string{"rm", "rmdir", "del", "erase", "shred", "truncate", "mkfs", "remove-item"},
		hint:  "use a preview or quarantine path first, such as Remove-Item -WhatIf or git clean -nd for git cleanup",
	},
	{
		name:     "git-destroy",
		class:    ReversibilityIrreversible,
		prefixes: [][]string{{"git", "clean"}, {"git", "reset", "hard"}},
		hint:     "inspect first with git status and a non-mutating preview such as git clean -nd; preserve shared work instead of resetting it",
	},
	{
		name:     "infra-destroy",
		class:    ReversibilityIrreversible,
		prefixes: [][]string{{"terraform", "destroy"}, {"kubectl", "delete"}},
		hint:     "preview infrastructure deletion first with terraform plan -destroy or kubectl diff/delete --dry-run=server",
	},
	// Payload scans stay whole-command on purpose: SQL statements and dd
	// targets arrive as arguments to a client binary, never as the head.
	{
		name:  "sql-drop",
		class: ReversibilityIrreversible,
		matchCmd: func(in familyMatchInput) bool {
			return orderedWords(in.words, "drop", "database") || orderedWords(in.words, "drop", "table")
		},
		hint: "inspect or back up the schema first, then use a transaction that rolls back before confirming the DROP",
	},
	{
		name:  "sql-truncate",
		class: ReversibilityIrreversible,
		matchCmd: func(in familyMatchInput) bool {
			return orderedWords(in.words, "truncate", "table") || (containsWord(in.words, "truncate") && (containsWord(in.words, "table") || containsWord(in.words, "database")))
		},
		hint: "inspect or back up data first, then use a transaction with rollback before confirming the TRUNCATE",
	},
	{
		name:  "sql-alter",
		class: ReversibilityIrreversible,
		matchCmd: func(in familyMatchInput) bool {
			return orderedWords(in.words, "alter", "table") || orderedWords(in.words, "alter", "database")
		},
		hint: "verify schema compatibility or capture a snapshot first, then use a transaction with rollback before confirming the ALTER",
	},
	{
		name:  "db-migration-down",
		class: ReversibilityIrreversible,
		matchCmd: func(in familyMatchInput) bool {
			return (containsWord(in.words, "prisma") && containsWord(in.words, "reset")) ||
				(containsWord(in.words, "alembic") && containsWord(in.words, "downgrade")) ||
				(containsWord(in.words, "goose") && (containsWord(in.words, "down") || containsWord(in.words, "reset"))) ||
				(containsWord(in.words, "flyway") && containsWord(in.words, "clean"))
		},
		hint: "downward database migration or database reset is irreversible; capture a database snapshot before confirming",
	},
	{
		name:  "dd-device-write",
		class: ReversibilityIrreversible,
		matchCmd: func(in familyMatchInput) bool {
			return (containsWord(in.words, "dd") && strings.Contains(in.lowerCmd, "of=/dev/")) ||
				ddQuotedDeviceWrite(in.rawCmd)
		},
		hint: "write to an image file first and inspect the target device before confirming a raw device write",
	},
	{
		name:         "destructive-tool",
		class:        ReversibilityIrreversible,
		toolContains: []string{"delete", "remove", "destroy", "truncate", "unlink", "rmdir"},
		hint:         "use the tool's preview or dry-run mode, or move the target to quarantine first, before confirming destruction",
	},
}

// ddQuotedDeviceWrite reads dd's actual operands without exposing all its quoted
// data to the payload scans. The older whole-command matcher remains conservative
// for launchers; this additional path closes the quoted-target hole (#3138) without
// treating a quoted filename such as "image dd of=/dev/sda" as another operation.
func ddQuotedDeviceWrite(cmd string) bool {
	// A here-document is input, not an argv list; leave its execution semantics
	// to the existing payload matcher rather than inventing command heads in it.
	cmd = stripShellComments(stripHeredocBodies(cmd))
	for _, seg := range quoteAwareSegmentTexts(cmd) {
		fields := reversibilityCommandFields(seg)
		for i, raw := range fields {
			head, ok := reversibilityShellWord(raw)
			if !ok {
				break
			}
			if envAssignmentRE.MatchString(raw) || commandWrapperHeads[strings.ToLower(head)] {
				continue
			}
			if ghHead(head) == "dd" {
				for _, arg := range fields[i+1:] {
					if word, ok := reversibilityShellWord(arg); ok && strings.HasPrefix(word, "of=/dev/") {
						return true
					}
				}
			}
			break // An argument naming dd is not an executable command head.
		}
	}
	return false
}

// reversibilityCommandFields retains raw quoting while separating command words
// from redirection operators and their operands. A quoted redirection filename or
// here-document delimiter is shell syntax, never an argument supplied to dd.
func reversibilityCommandFields(seg string) []string {
	var fields []string
	var b strings.Builder
	redirect := false
	flush := func() {
		if b.Len() == 0 {
			return
		}
		if !redirect {
			fields = append(fields, b.String())
		}
		b.Reset()
		redirect = false
	}
	for i := 0; i < len(seg); {
		c := seg[i]
		switch {
		case c == '\\' && i+1 < len(seg):
			b.WriteString(seg[i : i+2])
			i += 2
		case c == '\'' || c == '"':
			end := quotedSpanEnd(seg, i, c == '\'' && i > 0 && seg[i-1] == '$')
			b.WriteString(seg[i:end])
			i = end
		case c == '<' || c == '>':
			// Unquoted digits adjoining a redirect are its file descriptor.
			if !redirect && b.Len() > 0 && strings.Trim(b.String(), "0123456789") == "" {
				b.Reset()
			} else {
				flush()
			}
			redirect = true
			start := i
			for i < len(seg) && strings.ContainsRune("<>&|", rune(seg[i])) {
				i++
			}
			if seg[start:i] == "<<" && i < len(seg) && seg[i] == '-' {
				i++
			}
		case c == ' ' || c == '\t':
			flush()
			i++
		case c == '(' || c == ')' || c == '`':
			// Substitutions require another grammar, not ordinary argv splitting.
			return nil
		default:
			b.WriteByte(c)
			i++
		}
	}
	flush()
	if redirect { // A missing redirection operand cannot execute.
		return nil
	}
	return fields
}

// reversibilityShellWord removes POSIX quote/escape syntax from one already-split
// word, without evaluating expansions or executing anything. Unlike unquoteSpans,
// it removes escapes only where the shell does:
// a backslash before / inside double quotes is literal, not a /dev path prefix.
func reversibilityShellWord(raw string) (string, bool) {
	var b strings.Builder
	var quote byte
	for i := 0; i < len(raw); i++ {
		c := raw[i]
		if c == '\\' && quote != '\'' {
			if i+1 == len(raw) {
				return "", false
			}
			next := raw[i+1]
			if quote == 0 || strings.ContainsRune("$`\"\\\n", rune(next)) {
				i++
				if next != '\n' {
					b.WriteByte(next)
				}
				continue
			}
		}
		if quote != 0 {
			if c == quote {
				quote = 0
			} else {
				b.WriteByte(c)
			}
			continue
		}
		if c == '\'' || c == '"' {
			quote = c
		} else {
			b.WriteByte(c)
		}
	}
	return b.String(), quote == 0
}

var curlWriteFlagRE = regexp.MustCompile(`(?i)(^|[;&|()]|[[:space:]])curl(\.exe)?\b.*([[:space:]]-[Xx][[:space:]]*(POST|PUT|PATCH|DELETE)\b|[[:space:]]--request(=|[[:space:]]+)(POST|PUT|PATCH|DELETE)\b|[[:space:]](-d|--data|--data-raw|--data-binary|--form)(=|[[:space:]]|$))`)

func curlWrites(cmd string) bool {
	return curlWriteFlagRE.MatchString(cmd)
}

func httpieWrites(segs [][]string) bool {
	for _, seg := range segs {
		if seg[0] != "http" && seg[0] != "https" {
			continue
		}
		for _, w := range seg[1:] {
			if httpWriteVerb(w) {
				return true
			}
		}
	}
	return false
}

func httpWriteVerb(w string) bool {
	switch strings.ToLower(w) {
	case "post", "put", "patch", "delete":
		return true
	default:
		return false
	}
}

// ghWriteMutation reports whether the command invokes a MUTATING gh surface that
// the operator relaxation (see the gh note on reversibilityFamilies) must NOT
// cover (#3560): `gh api` with a write method (--method/-X POST|PUT|PATCH|DELETE,
// the flag anywhere in argv), or `gh repo fork|rename|delete` (always mutating).
// A `gh api` READ (GET / no --method) is a read and returns false. It reads the
// payload-scan view — quoted MENTIONS are already blanked (so a commit message or
// grep pattern naming `gh api -X POST` does not classify), while a live-payload
// statement stays visible — and parses raw segments so flag dashes survive
// (commandWords strips them). It splits on the same sequencing operators and
// strips the same leading env/wrapper heads as commandSegments/gitDryRunPreview,
// so a `sudo`/`FOO=1` prefix does not hide the gh call.
func ghWriteMutation(cmd string) bool {
	for _, raw := range commandSegmentRE.Split(cmd, -1) {
		fields := strings.Fields(raw)
		for len(fields) > 0 &&
			(envAssignmentRE.MatchString(fields[0]) || commandWrapperHeads[strings.ToLower(fields[0])]) {
			fields = fields[1:]
		}
		if len(fields) < 2 || ghHead(fields[0]) != "gh" {
			continue
		}
		switch strings.ToLower(fields[1]) {
		case "api":
			if ghAPIArgsWrite(fields[2:]) {
				return true
			}
		case "repo":
			if len(fields) >= 3 && ghRepoSubMutates(fields[2]) {
				return true
			}
		}
	}
	return false
}

// ghPRCreate reports whether any single command segment is a `gh pr create`
// invocation. It reads the quote-aware segment view (not the whole-command
// scan) so a quoted MENTION of the form — in a commit message, a grep pattern —
// stays clear, while env-assignment and wrapper heads are stripped the same way
// ghWriteMutation strips them.
func ghPRCreate(segs [][]string) bool {
	for _, seg := range segs {
		if len(seg) < 3 {
			continue
		}
		i := 0
		for i < len(seg) &&
			(envAssignmentRE.MatchString(seg[i]) || commandWrapperHeads[strings.ToLower(seg[i])]) {
			i++
		}
		if rest := seg[i:]; len(rest) >= 3 &&
			ghHead(rest[0]) == "gh" &&
			strings.EqualFold(rest[1], "pr") &&
			strings.EqualFold(rest[2], "create") {
			return true
		}
	}
	return false
}

// ghHead normalizes a command head to its bare program name so `gh`, `gh.exe`,
// and a path like /usr/bin/gh all read as "gh".
func ghHead(tok string) string {
	h := strings.ToLower(tok)
	h = strings.TrimSuffix(h, ".exe")
	if k := strings.LastIndexAny(h, `/\`); k >= 0 {
		h = h[k+1:]
	}
	return h
}

// ghAPIArgsWrite reports whether a `gh api` argument list carries a write method
// flag — `--method POST`, `--method=POST`, `-X POST`, or the glued `-XPOST` — for
// any of POST/PUT/PATCH/DELETE, robust to the flag appearing ANYWHERE in argv. The
// default gh api method is GET (a read), so no method flag means not a write.
func ghAPIArgsWrite(args []string) bool {
	const eq = "--method="
	for i, a := range args {
		switch {
		case strings.EqualFold(a, "--method"), strings.EqualFold(a, "-X"):
			if i+1 < len(args) && ghMethodWrites(args[i+1]) {
				return true
			}
		case len(a) > len(eq) && strings.EqualFold(a[:len(eq)], eq):
			if ghMethodWrites(a[len(eq):]) {
				return true
			}
		case len(a) > 2 && (strings.HasPrefix(a, "-X") || strings.HasPrefix(a, "-x")):
			if ghMethodWrites(a[2:]) {
				return true
			}
		}
	}
	return false
}

func ghMethodWrites(m string) bool {
	switch strings.ToUpper(m) {
	case "POST", "PUT", "PATCH", "DELETE":
		return true
	default:
		return false
	}
}

func ghRepoSubMutates(sub string) bool {
	switch strings.ToLower(sub) {
	case "fork", "rename", "delete":
		return true
	default:
		return false
	}
}
