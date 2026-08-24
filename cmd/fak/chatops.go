package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/chatops"
	"github.com/anthony-chaudhary/fak/internal/chatrelay"
	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/slackenv"
)

// `fak chatops` — the inbound READ-ONLY control door: drive fak from a Slack channel with a
// closed command grammar and a fail-closed admin allowlist. It is the impure shell around the
// pure parse+authorize kernel (internal/chatops, epic #2259 C4): the kernel decides, this
// command does the Slack I/O, the audit journal, and the idempotent poll.
//
// It answers the READ verbs inline — `help`, `ping`, `status`, `fleet` — and DECLINES the
// act/control verbs (`dispatch`, `resume`, `bench`, `halt`): the mutating path detaches
// through guarded dispatch (internal/chatopsdetach, #2264) and lands separately, so this
// spine can never start or stop work. Every inbound message is folded through the kernel's
// ordered fence (bot-loop → wrong-channel → not-addressed → empty → not-admin → grammar) and
// journaled; channel text is only ever parsed as a closed verb, never executed as free-form.
//
// Nothing is baked into source: the bot token, channel id, bot-user id, and admin allowlist
// all resolve at runtime from flags → env → .env.slack.local (the internal/slackenv idiom the
// scoreboard/blockerpost/chatrelay surfaces share), so the public tree stays clear of secrets
// and operator ids.
//
//	# see what would connect, and the exact grammar, without touching Slack:
//	fak chatops --dry-run
//
//	# run the door against the control channel (token/channel/admins from .env.slack.local):
//	fak chatops --channel C0XXXX --admins U0AAA,U0BBB --audit .chatops-audit.jsonl
//
//	# or under the guard's capability floor:
//	fak guard -- fak chatops --channel C0XXXX
func cmdChatOps(argv []string) {
	fs := flag.NewFlagSet("chatops", flag.ExitOnError)
	channel := fs.String("channel", "", "Slack control channel id to listen in (default: $FAK_CHATOPS_CHANNEL / .env.slack.local). REQUIRED — no silent fallback.")
	token := fs.String("token", "", "Slack bot token (default: $FAK_CHATOPS_TOKEN, then .env.slack.local, then the shared scoreboard bot token). Needs conversations.history + chat:write.")
	botUser := fs.String("bot-user", "", "the door's own Slack user id, for the @mention gate + self-loop fence (default: $FAK_CHATOPS_BOT_USER / .env.slack.local)")
	admins := fs.String("admins", "", "comma-separated allowlist of Slack user ids permitted to command the door (default: $FAK_CHATOPS_ADMINS / .env.slack.local). EMPTY ⇒ the door refuses everyone (fail-closed).")
	audit := fs.String("audit", "", "append one JSONL row per handled message to this file (empty ⇒ no journal)")
	interval, prime, once, dryRun := parseChatPollingFlags(fs, argv,
		"on start, skip the existing channel backlog and only answer messages posted after launch (pass --prime=false to answer the visible history too)",
		"print the resolved config + the verb registry and exit without connecting")

	tok := *token
	if tok == "" {
		tok = resolveChatopsToken()
	}
	ch := *channel
	if ch == "" {
		ch = resolveChatopsChannel()
	}
	bu := *botUser
	if bu == "" {
		bu = resolveChatopsBotUser()
	}
	adminList := resolveChatopsAdmins(*admins)

	if *dryRun {
		printChatopsDryRun(os.Stdout, ch, tok, bu, adminList, *audit)
		return
	}

	requireSlackTokenAndChannel("fak chatops", "FAK_CHATOPS", tok, ch)
	if len(adminList) == 0 {
		// Fail-closed is the kernel's contract, but a door with no admins can never answer a
		// single command — that is an operator mistake, not a running mode, so we refuse to start.
		fmt.Fprintln(os.Stderr, "fak chatops: admin allowlist is empty — set --admins or FAK_CHATOPS_ADMINS (the door would refuse every command)")
		os.Exit(2)
	}

	var journal io.Writer
	if *audit != "" {
		f, err := os.OpenFile(*audit, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak chatops: cannot open audit journal %s: %v\n", *audit, err)
			os.Exit(1)
		}
		defer f.Close()
		journal = f
	}

	door := &chatopsDoor{
		Slack: &chatrelay.HTTPSlack{Token: tok},
		Cfg: chatops.Config{
			BotUserID:      bu,
			ControlChannel: ch,
			Admins:         adminList,
		},
		Channel: ch,
		Audit:   journal,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	fmt.Printf("fak chatops: read-only door listening in %s (%d verbs, %d admin(s)); read verbs answer, act verbs are declined (#2264)\n",
		ch, len(chatops.Grammar()), len(adminList))

	runSlackPollLifecycle(ctx, door, "fak chatops", "handled", *prime, *once, *interval)
}

func parseChatPollingFlags(fs *flag.FlagSet, argv []string, primeHelp, dryRunHelp string) (*time.Duration, *bool, *bool, *bool) {
	interval := fs.Duration("interval", 3*time.Second, "poll interval between conversations.history fetches")
	prime := fs.Bool("prime", true, primeHelp)
	once := fs.Bool("once", false, "run a single poll and exit (smoke test) instead of looping")
	dryRun := fs.Bool("dry-run", false, dryRunHelp)
	_ = fs.Parse(argv)
	return interval, prime, once, dryRun
}

// --- config resolution (flags → env → .env.slack.local) ------------------------------
//
// Mirrors internal/chatrelay's resolver idiom, but with FAK_CHATOPS_* keys and NO channel
// fallback: a control door must target a DELIBERATE channel and must never inherit another
// surface's default. The token falls back to the shared scoreboard bot so one workspace bot
// can serve the door and the status feeders alike.

// resolveChatopsEnv is the env->file half every FAK_CHATOPS_* setting shares: the process
// environment wins, otherwise a matching line in .env.slack.local. Empty means "not
// configured" — the settings that have a further fallback say so themselves, below.
func resolveChatopsEnv(name string) string {
	if v := strings.TrimSpace(os.Getenv(name)); v != "" {
		return v
	}
	return slackenv.FileValue(name)
}

// resolveChatopsToken is the one setting with a third source: an unset door token falls back
// to the shared scoreboard bot.
func resolveChatopsToken() string {
	if v := resolveChatopsEnv("FAK_CHATOPS_TOKEN"); v != "" {
		return v
	}
	return scoreboard.ResolveToken()
}

func resolveChatopsChannel() string { return resolveChatopsEnv("FAK_CHATOPS_CHANNEL") }

func resolveChatopsBotUser() string { return resolveChatopsEnv("FAK_CHATOPS_BOT_USER") }

// resolveChatopsAdmins returns the admin allowlist: the explicit --admins flag when non-empty,
// else $FAK_CHATOPS_ADMINS, else a FAK_CHATOPS_ADMINS= line in .env.slack.local. The value is a
// comma-separated list of Slack user ids; whitespace and empty entries are dropped so a stray
// trailing comma or space can never seed a blank id (which the kernel already refuses, but the
// door should not carry junk into its config either).
func resolveChatopsAdmins(flagVal string) []string {
	raw := strings.TrimSpace(flagVal)
	if raw == "" {
		raw = strings.TrimSpace(os.Getenv("FAK_CHATOPS_ADMINS"))
	}
	if raw == "" {
		raw = slackenv.FileValue("FAK_CHATOPS_ADMINS")
	}
	return parseAdminList(raw)
}

// parseAdminList splits a comma-separated allowlist into trimmed, non-empty ids, preserving
// order and dropping duplicates. Pure: string in, slice out.
func parseAdminList(raw string) []string {
	var out []string
	seen := map[string]bool{}
	for _, part := range strings.Split(raw, ",") {
		id := strings.TrimSpace(part)
		if id == "" || seen[id] {
			continue
		}
		seen[id] = true
		out = append(out, id)
	}
	return out
}

// printChatopsDryRun renders the resolved connection facts (secret redacted) and the full verb
// registry, so an operator can confirm what a live run WOULD do without connecting to Slack.
func printChatopsDryRun(w io.Writer, channel, token, botUser string, admins []string, auditPath string) {
	fmt.Fprintf(w, "fak chatops (dry-run) — the inbound read-only control door:\n")
	fmt.Fprintf(w, "  channel  : %s\n", orUnset(channel))
	fmt.Fprintf(w, "  token    : %s\n", redactSecret(token))
	fmt.Fprintf(w, "  bot-user : %s\n", orUnset(botUser))
	fmt.Fprintf(w, "  admins   : %s\n", orUnset(strings.Join(admins, ", ")))
	fmt.Fprintf(w, "  audit    : %s\n", orUnset(auditPath))
	fmt.Fprintf(w, "  verbs    : (closed grammar — channel text only ever parses as one of these)\n")
	for _, s := range chatops.Grammar() {
		lane := chatopsClassLane(s.Class)
		operand := ""
		if s.NeedsOperand {
			operand = " <arg>"
		}
		fmt.Fprintf(w, "    %-9s %-6s %s%s — %s\n", s.Verb, "["+lane+"]", "", operand, s.Help)
	}
	if channel == "" {
		fmt.Fprintln(w, "  (channel is UNSET — set --channel or FAK_CHATOPS_CHANNEL before a live run)")
	}
	if token == "" {
		fmt.Fprintln(w, "  (token is UNSET — set --token or FAK_CHATOPS_TOKEN before a live run)")
	}
	if len(admins) == 0 {
		fmt.Fprintln(w, "  (admins are UNSET — the door is fail-closed and would refuse EVERY command)")
	}
}

// chatopsClassLane names the answer lane a verb class rides, for the dry-run registry.
func chatopsClassLane(c chatops.Class) string {
	switch c {
	case chatops.ClassRead:
		return "read"
	case chatops.ClassControl:
		return "control"
	case chatops.ClassAct:
		return "act"
	default:
		return "refuse"
	}
}

// --- the door: the impure poll loop around the pure kernel -----------------------------

// chatopsDoor polls ONE control channel, folds each new message through chatops.Parse, replies
// to the read verbs, declines the act/control verbs, journals every decision, and advances a
// high-water mark so each message is handled exactly once across polls. It is not safe for
// concurrent Tick calls (single-goroutine poll loop by design), matching chatrelay.Relay.
type chatopsDoor struct {
	Slack   chatrelay.SlackClient
	Cfg     chatops.Config
	Channel string
	Audit   io.Writer // nil ⇒ no journal

	// HistoryLimit bounds one conversations.history fetch (default chatopsHistoryLimit).
	HistoryLimit int

	// lastTS is the high-water mark: only messages with ts > lastTS are considered.
	lastTS string
}

const (
	chatopsHistoryLimit = 50
	chatopsPollDefault  = 3 * time.Second
)

// env captures the door's static facts the pure reply builder needs (channel, grammar size,
// admin count) without handing it the whole door.
func (d *chatopsDoor) env() chatopsEnv {
	return chatopsEnv{Channel: d.Channel, Admins: len(d.Cfg.Admins), Grammar: len(chatops.Grammar())}
}

func (d *chatopsDoor) limit() int {
	if d.HistoryLimit > 0 {
		return d.HistoryLimit
	}
	return chatopsHistoryLimit
}

// Prime advances the high-water mark to the newest message currently in the channel WITHOUT
// answering any of it, so a freshly started door does not reply to the whole backlog.
// Best-effort: a fetch error leaves the mark unset (the first Tick then answers the visible
// history) and is returned for the caller to log.
func (d *chatopsDoor) Prime(ctx context.Context) error {
	msgs, err := d.Slack.History(ctx, d.Channel, "", d.limit())
	if err != nil {
		return err
	}
	for _, m := range msgs {
		if chatrelay.TSAfter(m.TS, d.lastTS) {
			d.lastTS = m.TS
		}
	}
	return nil
}

// Tick performs ONE poll: fetch new messages, fold each through the kernel, reply/journal, and
// return how many were replied to. A Post failure is returned WITHOUT advancing the mark past
// that message, so a transient failure is retried on the next Tick rather than dropping a turn.
func (d *chatopsDoor) Tick(ctx context.Context) (replied int, err error) {
	// Oldest-first so we answer in conversational order and advance the mark monotonically.
	msgs, err := chatrelay.PollHistory(ctx, d.Slack, d.Channel, d.lastTS, d.limit())
	if err != nil {
		return 0, err
	}

	for _, m := range msgs {
		if !chatrelay.TSAfter(m.TS, d.lastTS) {
			continue // already seen (or the inclusive oldest echo) — idempotent re-poll
		}
		// Message edits/joins/system posts carry a subtype; they are not fresh human turns.
		// Mark them seen and move on (the kernel's loop fence also covers bot posts).
		if (m.Type != "" && m.Type != "message") || m.Subtype != "" {
			d.lastTS = m.TS
			continue
		}
		res := chatops.Parse(chatops.Message{
			User: m.User, BotID: m.BotID, Channel: d.Channel, TS: m.TS, Text: m.Text,
		}, d.Cfg)
		text, post := chatopsReply(res, d.env())
		didReply := false
		if post {
			if _, perr := d.Slack.Post(ctx, d.Channel, m.TS, text); perr != nil {
				return replied, fmt.Errorf("post ts=%s: %w", m.TS, perr)
			}
			didReply = true
		}
		d.writeAudit(res, didReply)
		d.lastTS = m.TS
		if didReply {
			replied++
		}
	}
	return replied, nil
}

// Run polls every interval until ctx is cancelled. A Tick error is delivered to onErr (when
// non-nil) and the loop continues — a single bad poll must not tear down a long-lived door.
func (d *chatopsDoor) Run(ctx context.Context, interval time.Duration, onErr func(error)) error {
	// chatopsPollDefault (not the relay's) stays this door's cadence — a control door polls
	// faster than a chat bridge; only the ticker/keep-going lifecycle is shared.
	return chatrelay.PollLoop(ctx, interval, chatopsPollDefault, func(ctx context.Context) error {
		_, err := d.Tick(ctx)
		return err
	}, onErr)
}

// chatopsAuditRow is one journaled decision: what the kernel decided and whether the door
// replied. It is the audit trail the read-only spine leaves — every inbound message, accepted
// or refused, produces exactly one row.
type chatopsAuditRow struct {
	TS      string `json:"ts"`
	User    string `json:"user"`
	Channel string `json:"channel"`
	Class   string `json:"class"`
	Verb    string `json:"verb,omitempty"`
	Operand string `json:"operand,omitempty"`
	Refused bool   `json:"refused"`
	Reason  string `json:"reason,omitempty"`
	Replied bool   `json:"replied"`
}

func (d *chatopsDoor) writeAudit(res chatops.Result, replied bool) {
	if d.Audit == nil {
		return
	}
	row := chatopsAuditRow{
		TS:      res.Nonce,
		User:    res.User,
		Channel: res.Channel,
		Class:   res.Class.String(),
		Verb:    string(res.Verb),
		Operand: res.Operand,
		Refused: res.Refused,
		Reason:  res.Reason,
		Replied: replied,
	}
	if b, err := json.Marshal(row); err == nil {
		_, _ = d.Audit.Write(append(b, '\n'))
	}
}

// chatopsEnv is the door's static context the pure reply builder reads.
type chatopsEnv struct {
	Channel string
	Admins  int
	Grammar int
}

// chatopsReply is the door's PURE reply decision: given the kernel's Result, it returns the
// text to post and whether to post at all. Read verbs answer inline; act/control verbs are
// DECLINED (this is the read-only spine — the mutating path lands with #2264); refusals reply
// only when the sender got far enough to be addressing the door in its channel (a helpful
// nudge), and stay SILENT for the pre-addressing fence (bot-loop / wrong-channel / not-
// addressed) and for the fail-closed authz gate (not-admin / empty), so the door never leaks
// its grammar to an unauthorized poke and never chatters at rooms it was merely invited to.
func chatopsReply(res chatops.Result, env chatopsEnv) (text string, post bool) {
	if res.Refused {
		switch res.Reason {
		case chatops.ReasonUnknownVerb:
			return "unknown command — reply `help` for the list of verbs I answer.", true
		case chatops.ReasonMissingOperand:
			return "that command needs an argument (e.g. `dispatch #2265`) — reply `help` for the grammar.", true
		default:
			// BOT_LOOP, WRONG_CHANNEL, NOT_ADDRESSED, EMPTY, NOT_ADMIN — silent.
			return "", false
		}
	}
	switch res.Class {
	case chatops.ClassRead:
		switch res.Verb {
		case chatops.VerbHelp:
			return renderChatopsHelp(), true
		case chatops.VerbPing:
			return "pong", true
		case chatops.VerbStatus:
			return fmt.Sprintf("chatops door online — listening in %s, %d verbs in the grammar, %d admin(s) seeded. Read verbs answer here; act verbs route through guarded dispatch once #2264 lands.",
				env.Channel, env.Grammar, env.Admins), true
		case chatops.VerbFleet:
			return "the cross-machine fleet aggregate is not surfaced through the read-only door yet (follow-on to #2650) — run `fak fleet` / `fak ps` locally; `status` shows this door's own liveness.", true
		default:
			return "", false
		}
	case chatops.ClassControl:
		return fmt.Sprintf("the read-only door does not execute `%s` yet — the kill-switch lands with the guarded act path (#2264).", res.Verb), true
	case chatops.ClassAct:
		return fmt.Sprintf("the read-only door does not execute act verbs yet — `%s` will route through guarded dispatch once the act path (#2264) lands.", res.Verb), true
	default:
		return "", false
	}
}

// renderChatopsHelp lists the closed grammar, tagged by answer lane, so an admin can see
// exactly what the door accepts. Pure: reads only the compiled-in grammar.
func renderChatopsHelp() string {
	var b strings.Builder
	b.WriteString("chatops — the read-only control door. Address me with a mention, then one verb:\n")
	for _, s := range chatops.Grammar() {
		operand := ""
		if s.NeedsOperand {
			operand = " <arg>"
		}
		fmt.Fprintf(&b, "• `%s%s` [%s] — %s\n", s.Verb, operand, chatopsClassLane(s.Class), s.Help)
	}
	b.WriteString("read verbs answer here; act/control verbs are declined until the guarded act path (#2264) lands.")
	return b.String()
}

// The door's Slack high-water comparison is chatrelay.TSAfter — the ONE ordering rule, shared
// with `fak chatrelay`'s bridge instead of re-derived here. It used to be a byte-identical
// private copy (chatopsTSAfter); the rule (numeric compare across micros widths, "" is the
// zero mark, an unparseable ts falls back to a string compare) lives with its table test in
// internal/chatrelay.
