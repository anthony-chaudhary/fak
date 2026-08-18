package main

// `fak slack shot` — LAUNCH a Slack channel and SCREENSHOT what fak actually posted.
//
// Every other `fak slack` verb WRITES (send, trajectory, refresh) or checks CONFIG
// (check, health). None of them let you SEE what landed. So when a feed job posts a
// scoreboard card or a dispatch result, the only way to know it rendered — and reads
// sensibly to a human — was to alt-tab into Slack and scroll. `fak slack shot` closes
// that dogfooding loop from the terminal:
//
//   - it READS the channel back (conversations.history + conversations.replies via
//     internal/slackwire) and renders a faithful transcript to stdout — a text
//     "screenshot" an agent or human can eyeball;
//   - with --out FILE.html it writes a self-contained HTML capture that reproduces the
//     channel visually (author, timestamp, message bubbles) and opens in any browser
//     WITHOUT Slack — the artifact you attach to a dogfood note;
//   - with --launch it prints (and, on a desktop, opens) the Slack deep link and web
//     URL for the channel so you can pop the REAL client open next to the capture and
//     compare.
//
//	fak slack shot dispatch                 # transcript of the #dispatch surface to stdout
//	fak slack shot dispatch --out shot.html # + a visual HTML screenshot you can open
//	fak slack shot --channel C0ABC123 -n 40 # any channel id, last 40 messages
//	fak slack shot scoreboard --launch      # also open the real Slack client to compare
//	fak slack shot dispatch --dry-run       # resolve channel + URLs + out path, no network
//
// It resolves token/channel through the SAME registry `fak slack check` walks
// (slackSurfaces), so "which channel does the dispatch surface post to?" has one answer.
// Read-only against Slack (history + auth.test); it never posts.

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/interspersedflags"
	"html"
	"io"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/slackenv"
	"github.com/anthony-chaudhary/fak/internal/slackwire"
)

// slackShotDefaultLimit is how many recent messages a shot captures when -n is unset.
// Twenty covers "did the last few posts render right?" without a wall of backlog.
const slackShotDefaultLimit = 20

// slackShotReplyLimit bounds one conversations.replies call. A capture is an operator
// sample, not an archive export; 100 rows show a substantial thread without letting one
// runaway conversation dominate memory or API time.
const slackShotReplyLimit = 100

// teamEnv names the workspace team id used to build the Slack deep link / web URL. It is
// public (T…), not a secret; when unset the shot resolves it from auth.test.
const teamEnv = "FAK_SLACK_TEAM"

// slackShotHistory reads a channel's recent messages. It is a package var so tests inject
// a fixed transcript without a network or a token (mirrors trajectoryPost's seam).
var slackShotHistory = func(token, apiBase, channel string, limit int) ([]slackwire.Message, error) {
	var opts []slackwire.Option
	if apiBase != "" {
		opts = append(opts, slackwire.WithAPIBase(apiBase))
	}
	c := slackwire.New(token, opts...)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	return c.History(ctx, channel, "", limit)
}

// slackShotReplies reads one thread. Kept as a separate seam so tests can prove the
// operator capture nests real replies without touching Slack.
var slackShotReplies = func(token, apiBase, channel, threadTS string, limit int) ([]slackwire.Message, error) {
	var opts []slackwire.Option
	if apiBase != "" {
		opts = append(opts, slackwire.WithAPIBase(apiBase))
	}
	c := slackwire.New(token, opts...)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	return c.Replies(ctx, channel, threadTS, limit)
}

// slackShotTeam resolves the workspace team id (for the launch URLs) from auth.test. It is
// a package var so tests skip the network; a failure yields "" and the URLs fall back to a
// <team> placeholder rather than erroring — the transcript is the point, the URLs a bonus.
var slackShotTeam = func(token, apiBase string) string {
	var opts []slackwire.Option
	if apiBase != "" {
		opts = append(opts, slackwire.WithAPIBase(apiBase))
	}
	c := slackwire.New(token, opts...)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	info, err := c.AuthTest(ctx)
	if err != nil || info == nil {
		return ""
	}
	return info.TeamID
}

// shotResult is the machine-readable outcome (--json): what was captured and where.
type shotResult struct {
	Surface  string `json:"surface,omitempty"`
	Channel  string `json:"channel"`
	Team     string `json:"team,omitempty"`
	Count    int    `json:"count"`
	Replies  int    `json:"replies,omitempty"`
	DeepLink string `json:"deep_link"`
	WebURL   string `json:"web_url"`
	Out      string `json:"out,omitempty"`
	DryRun   bool   `json:"dry_run,omitempty"`
}

// runSlackShot resolves a channel, reads its recent messages, and renders the transcript
// (stdout) plus an optional HTML capture, and optionally launches the real client.
func runSlackShot(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak slack shot", flag.ContinueOnError)
	fs.SetOutput(stderr)
	channel := fs.String("channel", "", "channel id to capture (e.g. C0ABC123); overrides a surface positional arg")
	token := fs.String("token", "", "bot token (default: the surface's token, then $FAK_SCOREBOARD_TOKEN / "+slackenv.EnvFileName+")")
	apiBase := fs.String("api-base", "", "override the Slack API base URL (for testing/proxying)")
	team := fs.String("team", "", "workspace team id T… for the launch URLs (default $"+teamEnv+", then auth.test)")
	limit := fs.Int("n", slackShotDefaultLimit, "how many recent messages to capture")
	out := fs.String("out", "", "write a self-contained HTML screenshot to this path")
	launch := fs.Bool("launch", false, "also print the Slack deep link + web URL, and open the client on a desktop")
	asJSON := fs.Bool("json", false, "emit the capture summary as JSON")
	dryRun := fs.Bool("dry-run", false, "resolve channel + URLs + out path and exit; no Slack calls, no launch")
	// Parse flags allowing them on EITHER side of the surface positional — Go's flag
	// package stops at the first non-flag, so a plain fs.Parse would silently drop the
	// flags in `fak slack shot dispatch --out x.html`. Collect positionals across the gaps.
	positional, err := interspersedflags.Parse(fs, argv)
	if err != nil {
		return 2
	}

	// Resolve the target: an explicit --channel wins; otherwise the first positional is a
	// surface name (dispatch, scoreboard, …) or a bare channel id.
	surfaceName, chID, tok := "", strings.TrimSpace(*channel), strings.TrimSpace(*token)
	if chID == "" {
		arg := ""
		if len(positional) > 0 {
			arg = strings.TrimSpace(positional[0])
		}
		switch {
		case arg == "":
			fmt.Fprintln(stderr, "fak slack shot: name a surface (e.g. `fak slack shot dispatch`) or pass --channel C0ABC123")
			fmt.Fprintln(stderr, "  surfaces: "+strings.Join(shotSurfaceNames(), ", "))
			return 2
		case looksLikeChannelID(arg):
			chID = arg
		default:
			s, ok := findSurface(arg)
			if !ok {
				fmt.Fprintf(stderr, "fak slack shot: unknown surface %q; one of: %s\n", arg, strings.Join(shotSurfaceNames(), ", "))
				return 2
			}
			surfaceName = s.Name
			chID = s.channel().Value
			if tok == "" {
				tok = s.token().Value
			}
			if chID == "" {
				fmt.Fprintf(stderr, "fak slack shot: surface %q has no channel configured; set %s or pass --channel\n", s.Name, s.ChannelEnv)
				return 2
			}
		}
	}
	if tok == "" {
		tok = scoreboard.ResolveToken()
	}

	// Team id for the launch URLs: flag, then env, then (unless dry-run) auth.test.
	teamID := strings.TrimSpace(*team)
	if teamID == "" {
		teamID = strings.TrimSpace(os.Getenv(teamEnv))
	}
	if teamID == "" && !*dryRun && tok != "" {
		teamID = slackShotTeam(tok, *apiBase)
	}

	res := shotResult{
		Surface:  surfaceName,
		Channel:  chID,
		Team:     teamID,
		DeepLink: slackDeepLink(teamID, chID),
		WebURL:   slackWebURL(teamID, chID),
		Out:      strings.TrimSpace(*out),
		DryRun:   *dryRun,
	}

	if *dryRun {
		if *asJSON {
			return emitShotJSON(stdout, stderr, res)
		}
		fmt.Fprintf(stdout, "fak slack shot (dry-run):\n")
		fmt.Fprintf(stdout, "  surface : %s\n", orDash(surfaceName))
		fmt.Fprintf(stdout, "  channel : %s\n", chID)
		fmt.Fprintf(stdout, "  token   : %s\n", tokenState(tok))
		fmt.Fprintf(stdout, "  capture : last %d messages\n", *limit)
		if res.Out != "" {
			fmt.Fprintf(stdout, "  out     : %s (HTML)\n", res.Out)
		}
		fmt.Fprintf(stdout, "  launch  : %s\n", res.DeepLink)
		fmt.Fprintf(stdout, "            %s\n", res.WebURL)
		return 0
	}

	if tok == "" {
		fmt.Fprintln(stderr, "fak slack shot: no bot token — set --token, $FAK_SCOREBOARD_TOKEN, or add it to "+slackenv.EnvFileName)
		return 2
	}

	msgs, err := slackShotHistory(tok, *apiBase, chID, *limit)
	if err != nil {
		fmt.Fprintf(stderr, "fak slack shot: history: %v\n", err)
		return 1
	}
	msgs, err = expandSlackShotThreads(tok, *apiBase, chID, msgs)
	if err != nil {
		fmt.Fprintf(stderr, "fak slack shot: replies: %v\n", err)
		return 1
	}
	msgs = sortMessagesChrono(msgs)
	res.Count, res.Replies = slackShotCounts(msgs)

	title := chID
	if surfaceName != "" {
		title = "#" + surfaceName + " (" + chID + ")"
	}

	if res.Out != "" {
		if err := os.WriteFile(res.Out, []byte(renderShotHTML(title, res.WebURL, msgs)), 0o644); err != nil {
			fmt.Fprintf(stderr, "fak slack shot: write %s: %v\n", res.Out, err)
			return 1
		}
	}

	if *launch {
		if err := openURL(res.DeepLink); err != nil {
			// A missing desktop / handler is not a failure of the capture; note it and move on.
			fmt.Fprintf(stderr, "fak slack shot: could not open %s: %v (open it manually)\n", res.DeepLink, err)
		}
	}

	if *asJSON {
		return emitShotJSON(stdout, stderr, res)
	}

	fmt.Fprint(stdout, renderShotText(title, msgs))
	if res.Out != "" {
		fmt.Fprintf(stdout, "\nwrote HTML screenshot: %s\n", res.Out)
	}
	if *launch {
		fmt.Fprintf(stdout, "launch: %s\n       %s\n", res.DeepLink, res.WebURL)
	}
	return 0
}

// emitShotJSON writes r as indented JSON; a small shared helper for the two JSON exits.
func emitShotJSON(stdout, stderr io.Writer, r shotResult) int {
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(r); err != nil {
		fmt.Fprintf(stderr, "fak slack shot: encode json: %v\n", err)
		return 1
	}
	return 0
}

// findSurface returns the registered surface with the given name.
func findSurface(name string) (slackSurface, bool) {
	for _, s := range slackSurfaces {
		if s.Name == name {
			return s, true
		}
	}
	return slackSurface{}, false
}

// shotSurfaceNames lists the registered surface names for help/error text.
func shotSurfaceNames() []string {
	out := make([]string, 0, len(slackSurfaces))
	for _, s := range slackSurfaces {
		out = append(out, s.Name)
	}
	return out
}

// looksLikeChannelID reports whether s is a bare Slack channel/group id (C…/G…/D…, all
// upper-case alnum) — so `fak slack shot C0ABC123` is read as a channel, not a surface.
func looksLikeChannelID(s string) bool {
	if len(s) < 8 {
		return false
	}
	if s[0] != 'C' && s[0] != 'G' && s[0] != 'D' {
		return false
	}
	for _, r := range s {
		if !(r >= 'A' && r <= 'Z') && !(r >= '0' && r <= '9') {
			return false
		}
	}
	return true
}

// sortMessagesChrono returns msgs oldest-first (conversations.history returns newest-first)
// so a capture reads top-to-bottom like the channel does.
func sortMessagesChrono(msgs []slackwire.Message) []slackwire.Message {
	out := make([]slackwire.Message, len(msgs))
	copy(out, msgs)
	sort.SliceStable(out, func(i, j int) bool { return tsSeconds(out[i].TS) < tsSeconds(out[j].TS) })
	return out
}

// expandSlackShotThreads joins history roots with their replies. History exposes a
// reply_count but not the reply bodies; conversations.replies returns the parent as its
// first row, so that duplicate is dropped. A row missing thread_ts is defensively bound
// to the root being read — Slack normally sets it on replies, but the capture should keep
// the hierarchy even when a test/proxy omits the redundant field.
func expandSlackShotThreads(token, apiBase, channel string, roots []slackwire.Message) ([]slackwire.Message, error) {
	out := append([]slackwire.Message(nil), roots...)
	for _, root := range roots {
		if root.ThreadTS != "" || root.ReplyCount <= 0 || root.TS == "" {
			continue
		}
		rows, err := slackShotReplies(token, apiBase, channel, root.TS, slackShotReplyLimit)
		if err != nil {
			return nil, fmt.Errorf("thread %s: %w", root.TS, err)
		}
		for _, reply := range rows {
			if reply.TS == root.TS {
				continue // conversations.replies repeats the parent first
			}
			if reply.ThreadTS == "" {
				reply.ThreadTS = root.TS
			}
			out = append(out, reply)
		}
	}
	return out, nil
}

func slackShotCounts(msgs []slackwire.Message) (roots, replies int) {
	for _, m := range msgs {
		if m.ThreadTS == "" {
			roots++
		} else {
			replies++
		}
	}
	return roots, replies
}

// slackShotThreaded returns top-level roots plus replies keyed by parent, all ordered
// oldest-first. Orphan replies are promoted into the root list rather than hidden.
func slackShotThreaded(msgs []slackwire.Message) ([]slackwire.Message, map[string][]slackwire.Message) {
	ordered := sortMessagesChrono(msgs)
	knownRoots := make(map[string]bool)
	for _, m := range ordered {
		if m.ThreadTS == "" {
			knownRoots[m.TS] = true
		}
	}
	var roots []slackwire.Message
	replies := make(map[string][]slackwire.Message)
	for _, m := range ordered {
		if m.ThreadTS != "" && knownRoots[m.ThreadTS] {
			replies[m.ThreadTS] = append(replies[m.ThreadTS], m)
			continue
		}
		roots = append(roots, m)
	}
	return roots, replies
}

// tsSeconds parses a Slack ts ("1719600000.000100") to a float for ordering; unparseable
// ts sort first (0) so a malformed row never hides a real one at the bottom.
func tsSeconds(ts string) float64 {
	f, err := strconv.ParseFloat(ts, 64)
	if err != nil {
		return 0
	}
	return f
}

// formatSlackTS renders a Slack ts as a human UTC stamp; a bad ts passes through verbatim.
func formatSlackTS(ts string) string {
	sec := int64(tsSeconds(ts))
	if sec == 0 {
		return ts
	}
	return time.Unix(sec, 0).UTC().Format("2006-01-02 15:04:05 UTC")
}

// messageAuthor labels who posted a message: the bot id for a bot post, else the user id,
// else "(system)" for a subtype-only event (joins, etc.).
func messageAuthor(m slackwire.Message) string {
	switch {
	case m.BotID != "":
		return "bot:" + m.BotID
	case m.User != "":
		return m.User
	default:
		return "(system)"
	}
}

// renderShotText is the stdout transcript "screenshot": a header plus one block per
// message (stamp, author, text). Pure — the unit tests pin its exact shape.
func renderShotText(title string, msgs []slackwire.Message) string {
	var b strings.Builder
	roots, replies := slackShotThreaded(msgs)
	_, replyCount := slackShotCounts(msgs)
	fmt.Fprintf(&b, "%s — %d message(s), oldest first · %d threaded %s\n", title, len(roots), replyCount, pluralWord(replyCount, "reply", "replies"))
	b.WriteString(strings.Repeat("─", 60) + "\n")
	if len(msgs) == 0 {
		b.WriteString("(channel is empty, or the bot cannot read its history)\n")
		return b.String()
	}
	for _, m := range roots {
		fmt.Fprintf(&b, "[%s] %s\n", formatSlackTS(m.TS), messageAuthor(m))
		text := strings.TrimRight(m.Text, "\n")
		if text == "" {
			text = "(no text — blocks/attachment only)"
		}
		for _, line := range strings.Split(text, "\n") {
			b.WriteString("    " + line + "\n")
		}
		for _, reply := range replies[m.TS] {
			fmt.Fprintf(&b, "    ↳ [%s] %s\n", formatSlackTS(reply.TS), messageAuthor(reply))
			replyText := strings.TrimRight(reply.Text, "\n")
			if replyText == "" {
				replyText = "(no text — blocks/attachment only)"
			}
			for _, line := range strings.Split(replyText, "\n") {
				b.WriteString("        " + line + "\n")
			}
		}
	}
	return b.String()
}

// renderShotHTML is the self-contained visual capture: one HTML document with inline CSS,
// no external assets, that reproduces the channel as author/timestamp/message rows and
// links back to the live channel. Pure and fully escaped.
func renderShotHTML(title, webURL string, msgs []slackwire.Message) string {
	var b strings.Builder
	roots, replies := slackShotThreaded(msgs)
	_, replyCount := slackShotCounts(msgs)
	b.WriteString("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\n")
	b.WriteString("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
	fmt.Fprintf(&b, "<title>%s — fak slack shot</title>\n", html.EscapeString(title))
	b.WriteString("<style>\n")
	b.WriteString("body{font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#1a1d21;color:#d1d2d3}\n")
	b.WriteString("header{padding:16px 20px;border-bottom:1px solid #35373b;position:sticky;top:0;background:#1a1d21}\n")
	b.WriteString("header h1{font-size:18px;margin:0 0 4px}header a{color:#1d9bd1;text-decoration:none;font-size:13px}\n")
	b.WriteString(".feed{padding:12px 20px;max-width:900px}\n")
	b.WriteString(".msg{display:flex;gap:10px;padding:8px 0;border-bottom:1px solid #26282c}\n")
	b.WriteString(".reply{margin-left:34px;padding-left:12px;border-left:2px solid #35373b}\n")
	b.WriteString(".who{color:#e8e8e8;font-weight:700}.ts{color:#9a9c9f;font-size:12px;margin-left:8px}\n")
	b.WriteString(".body{white-space:pre-wrap;word-break:break-word;margin-top:2px}\n")
	b.WriteString(".empty{color:#9a9c9f;padding:20px}\n")
	b.WriteString("</style></head><body>\n")
	fmt.Fprintf(&b, "<header><h1>%s</h1>", html.EscapeString(title))
	if webURL != "" {
		fmt.Fprintf(&b, "<a href=\"%s\">open in Slack ↗</a>", html.EscapeString(webURL))
	}
	fmt.Fprintf(&b, "<div class=\"ts\">%d message(s), %d threaded %s, captured by fak slack shot</div></header>\n", len(roots), replyCount, pluralWord(replyCount, "reply", "replies"))
	b.WriteString("<div class=\"feed\">\n")
	if len(msgs) == 0 {
		b.WriteString("<div class=\"empty\">Channel is empty, or the bot cannot read its history.</div>\n")
	}
	for _, m := range roots {
		renderShotHTMLMessage(&b, m, false)
		for _, reply := range replies[m.TS] {
			renderShotHTMLMessage(&b, reply, true)
		}
	}
	b.WriteString("</div></body></html>\n")
	return b.String()
}

func renderShotHTMLMessage(b *strings.Builder, m slackwire.Message, reply bool) {
	text := m.Text
	if strings.TrimSpace(text) == "" {
		text = "(no text — blocks/attachment only)"
	}
	class := "msg"
	if reply {
		class += " reply"
	}
	fmt.Fprintf(b, "<div class=\"%s\"><div><div><span class=\"who\">", class)
	b.WriteString(html.EscapeString(messageAuthor(m)))
	b.WriteString("</span><span class=\"ts\">")
	b.WriteString(html.EscapeString(formatSlackTS(m.TS)))
	b.WriteString("</span></div><div class=\"body\">")
	b.WriteString(html.EscapeString(text))
	b.WriteString("</div></div></div>\n")
}

// slackDeepLink builds the desktop-client deep link for a channel (opens the installed
// Slack app). With no team id it uses a <team> placeholder so the shape is still legible.
func slackDeepLink(team, channel string) string {
	return fmt.Sprintf("slack://channel?team=%s&id=%s", orPlaceholder(team, "team"), channel)
}

// slackWebURL builds the browser client URL for a channel.
func slackWebURL(team, channel string) string {
	return fmt.Sprintf("https://app.slack.com/client/%s/%s", orPlaceholder(team, "team"), channel)
}

func orPlaceholder(v, name string) string {
	if strings.TrimSpace(v) == "" {
		return "<" + name + ">"
	}
	return v
}

// tokenState describes token presence for the dry-run report without ever printing it.
func tokenState(tok string) string {
	if tok == "" {
		return "(unset — set --token or $FAK_SCOREBOARD_TOKEN before a live shot)"
	}
	return "resolved (" + redactToken(tok) + ")"
}

// openURL opens url in the platform's default handler — the desktop half of --launch. It
// is best-effort: a headless host has no handler and the caller degrades gracefully.
func openURL(url string) error {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		cmd = exec.Command("cmd", "/c", "start", "", url)
	case "darwin":
		cmd = exec.Command("open", url)
	default:
		cmd = exec.Command("xdg-open", url)
	}
	return cmd.Start()
}
