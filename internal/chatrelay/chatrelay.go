// Package chatrelay bridges ONE Slack channel to an OpenAI-compatible chat endpoint:
// it reads new human messages from the channel (conversations.history), forwards each to
// a served /v1/chat/completions model, and posts the reply back in-thread
// (chat.postMessage). It is what makes a `fak serve`-hosted model — e.g. GLM-5.2 on the
// pure in-kernel forward — USABLE from a Slack channel: a person types, the model answers.
//
// It is deliberately a GENERIC chatbot front-end, the Slack analogue of `fak c` / the
// console chat client, NOT the lab control bridge:
//
//   - endpoint-agnostic: the model endpoint is a plain --endpoint URL (default the local
//     `fak serve` at http://127.0.0.1:8080); it carries no lab host/path/protocol.
//   - identifier-free: the bot token and channel id resolve at runtime from env /
//     .env.slack.local (the same idiom internal/scoreboard + internal/blockerpost use), so
//     no token, channel id, or hostname ever sits in source.
//   - it runs NO shell and executes NO commands — it only sends a PROMPT to an inference
//     endpoint and posts back TEXT.
//
// Those three properties keep it on the PUBLIC side of the GPU-server/Slack boundary
// (docs/gpu-server-private-boundary.md): the private piece is the lab *control* bridge (it runs
// commands on lab boxes and speaks a private protocol); this is *chat*. The path-based
// commit gate (tools/check_committed_files.py) refuses a package whose name carries the
// private GPU-server token or slack+{bridge,control,gc}; "chatrelay" is neither, the same way
// internal/blockerpost holds a chat.postMessage client in the public tree.
package chatrelay

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/slackwire"
)

// Message is one Slack channel message, the subset of conversations.history this relay
// needs to decide whether to answer it and how to thread the reply.
type Message struct {
	Type     string // "message" for a real post
	Subtype  string // non-empty for edits/joins/bot posts — those are skipped
	TS       string // Slack timestamp "1719600000.000100"; the in-channel message id + thread anchor
	ThreadTS string // parent thread ts when the message is itself a threaded reply
	User     string // posting user id ("" for a bot post)
	BotID    string // non-empty when posted by a bot (including THIS relay's own replies)
	Text     string // message body
}

// SlackClient is the inbound+outbound Slack surface the relay drives. It is an interface
// so a test can drive the relay against an in-memory channel with no network. HTTPSlack is
// the live implementation.
type SlackClient interface {
	// History returns the channel's messages with ts strictly after oldestTS (oldestTS=""
	// means "from the beginning"), oldest-first, capped at limit. The implementation may
	// over-return; the relay re-filters by ts itself.
	History(ctx context.Context, channel, oldestTS string, limit int) ([]Message, error)
	// Post sends text to the channel as a reply in threadTS's thread (threadTS="" posts at
	// top level). It returns the posted message ts.
	Post(ctx context.Context, channel, threadTS, text string) (string, error)
}

// ModelClient turns a user prompt into a model completion. HTTPModel calls a served
// OpenAI-compatible /v1/chat/completions (the `fak serve` endpoint).
type ModelClient interface {
	Complete(ctx context.Context, prompt string) (string, error)
}

// Relay forwards new human messages from one Slack channel to a model and posts the
// replies back. It tracks the last-seen ts so each message is answered exactly once across
// polls. It is not safe for concurrent Tick calls (single-goroutine poll loop by design).
type Relay struct {
	Slack   SlackClient
	Model   ModelClient
	Channel string

	// BotUserID, when set, is this relay's own bot user id; a message from it is skipped as
	// a belt-and-suspenders guard in addition to the BotID!="" skip (covers a deployment
	// where replies are posted as a user token rather than a bot token).
	BotUserID string

	// Mention, when non-empty, gates responses: only a message whose text contains this
	// token (e.g. "<@U07BOT>") is answered, and the token is stripped before the prompt is
	// built. Empty means answer every human message in the channel (a dedicated chat room).
	Mention string

	// HistoryLimit bounds a single conversations.history fetch (default historyLimitDefault).
	HistoryLimit int

	// lastTS is the high-water mark: only messages with ts > lastTS are considered. It is
	// seeded by Prime (skip the backlog) or left "" to answer the whole visible history.
	lastTS string
}

const (
	historyLimitDefault = 50
	// pollIntervalDefault is the default gap between conversations.history polls in Run.
	pollIntervalDefault = 3 * time.Second
)

// Prime sets the high-water mark to the newest message currently in the channel WITHOUT
// answering any of it, so a freshly started relay does not reply to the whole backlog. It
// is best-effort: a fetch error leaves the mark unset (the first Tick then answers the
// visible history) and is returned for the caller to log.
func (r *Relay) Prime(ctx context.Context) error {
	limit := r.HistoryLimit
	if limit <= 0 {
		limit = historyLimitDefault
	}
	msgs, err := r.Slack.History(ctx, r.Channel, "", limit)
	if err != nil {
		return err
	}
	for _, m := range msgs {
		if TSAfter(m.TS, r.lastTS) {
			r.lastTS = m.TS
		}
	}
	return nil
}

// Tick performs ONE poll: fetch new messages, answer each genuine human message exactly
// once (advancing the high-water mark), and return how many were answered. A model or post
// failure for one message is returned but does NOT advance the mark past that message, so a
// transient failure is retried on the next Tick rather than silently dropping the turn.
func (r *Relay) Tick(ctx context.Context) (handled int, err error) {
	limit := r.HistoryLimit
	if limit <= 0 {
		limit = historyLimitDefault
	}
	// Oldest-first so we answer in conversational order and advance the mark monotonically.
	msgs, err := PollHistory(ctx, r.Slack, r.Channel, r.lastTS, limit)
	if err != nil {
		return 0, err
	}

	for _, m := range msgs {
		if !TSAfter(m.TS, r.lastTS) {
			continue // already seen (or the inclusive oldest echo)
		}
		prompt, ok := r.promptFor(m)
		if !ok {
			r.lastTS = m.TS // not for us (bot post, subtype, unaddressed) — mark seen, never answer
			continue
		}
		reply, cerr := r.Model.Complete(ctx, prompt)
		if cerr != nil {
			return handled, fmt.Errorf("complete ts=%s: %w", m.TS, cerr)
		}
		reply = strings.TrimSpace(reply)
		if reply == "" {
			reply = "(the model returned an empty completion)"
		}
		// Defang any Slack mention/broadcast control tokens the model echoed back before
		// posting, so an answer can never re-ping the addressed app ("@app") or the whole
		// channel — the bridge posts chat, not notifications (see defangMentions).
		reply = defangMentions(reply)
		// Reply in the message's own thread: a top-level message anchors a new thread on its
		// ts; a threaded message keeps its parent thread.
		threadTS := m.ThreadTS
		if threadTS == "" {
			threadTS = m.TS
		}
		if _, perr := r.Slack.Post(ctx, r.Channel, threadTS, reply); perr != nil {
			return handled, fmt.Errorf("post ts=%s: %w", m.TS, perr)
		}
		r.lastTS = m.TS
		handled++
	}
	return handled, nil
}

// Run polls every interval until ctx is cancelled. A Tick error is delivered to onErr (when
// non-nil) and the loop continues — a single bad poll must not tear down a long-lived relay.
// interval<=0 uses pollIntervalDefault. It returns ctx.Err() when cancelled.
func (r *Relay) Run(ctx context.Context, interval time.Duration, onErr func(error)) error {
	return PollLoop(ctx, interval, pollIntervalDefault, func(ctx context.Context) error {
		_, err := r.Tick(ctx)
		return err
	}, onErr)
}

// promptFor decides whether a message should be answered and, if so, returns the prompt to
// send the model. It returns ok=false for anything that is not a fresh human turn for this
// relay: a non-"message" type, any subtype (edits/joins/bot posts), a bot post (BotID set),
// this relay's own user id, an empty body, or — when Mention gating is on — a message that
// does not address the bot. When addressed, the mention token is stripped from the prompt.
func (r *Relay) promptFor(m Message) (string, bool) {
	if m.Type != "" && m.Type != "message" {
		return "", false
	}
	if m.Subtype != "" || m.BotID != "" {
		return "", false
	}
	if r.BotUserID != "" && m.User == r.BotUserID {
		return "", false
	}
	text := strings.TrimSpace(m.Text)
	if text == "" {
		return "", false
	}
	if r.Mention != "" {
		if !strings.Contains(text, r.Mention) {
			return "", false
		}
		text = strings.TrimSpace(strings.ReplaceAll(text, r.Mention, ""))
		if text == "" {
			return "", false
		}
	}
	return text, true
}

// The Slack control tokens that turn POSTED text into a notification. chat.postMessage
// is sent WITHOUT link_names (see slackwire), so a bare "@app"/"@channel" is inert
// literal text — only these angle-bracket forms are re-linked by Slack and actually
// ping someone. A served model, addressed as "<@U07BOT>", routinely echoes that token
// back (or emits a "<!here>" it saw in the room), which is exactly the "the bridge keeps
// tagging @app" notification noise this defangs.
var (
	// reUserMention: <@U07BOT> or <@U07BOT|label> — a user/bot mention (the "@app" ping).
	reUserMention = regexp.MustCompile(`<@([^>|]*)(?:\|([^>]*))?>`)
	// reSubteamMention: <!subteam^S123> or <!subteam^S123|handle> — pings a whole group.
	reSubteamMention = regexp.MustCompile(`<!subteam\^([^>|]*)(?:\|([^>]*))?>`)
	// reBroadcast: <!here>, <!channel>, <!everyone> (optionally <!here|here>) — the
	// channel/workspace-wide broadcast pings. Note this deliberately does NOT match other
	// <!…> tokens (e.g. <!date^…>), which do not notify and must render unchanged.
	reBroadcast = regexp.MustCompile(`<!(here|channel|everyone)(?:\|[^>]*)?>`)
)

// defangMentions rewrites the notification-triggering Slack control tokens in text into
// their plain-text "@label" form. Because the bridge posts with link_names off, the
// result is inert — it reads like a normal "@app" / "@here" but Slack no longer re-links
// it, so a reply cannot ping the addressed app or the whole channel. A token carrying a
// display label (`<@U07BOT|app>`, `<!subteam^S1|team>`) keeps the friendly label; a bare
// id falls back to the id itself. Pure: text in, text out, no I/O.
func defangMentions(text string) string {
	text = reUserMention.ReplaceAllStringFunc(text, func(m string) string {
		return atLabel(reUserMention.FindStringSubmatch(m))
	})
	text = reSubteamMention.ReplaceAllStringFunc(text, func(m string) string {
		return atLabel(reSubteamMention.FindStringSubmatch(m))
	})
	return reBroadcast.ReplaceAllString(text, "@$1")
}

// atLabel renders a defanged mention from a [full, id, label] submatch: the label when
// present (with any redundant leading "@" collapsed so it never doubles), else the id.
func atLabel(sub []string) string {
	if len(sub) >= 3 {
		if label := strings.TrimSpace(sub[2]); label != "" {
			return "@" + strings.TrimPrefix(label, "@")
		}
	}
	return "@" + sub[1]
}

// TSAfter reports whether Slack ts a is strictly after b. Slack timestamps are
// "<seconds>.<micros>" decimals; a numeric compare is correct across widths (string compare
// is not, e.g. "1000000000.0" vs "999999999.9"). A "" b is the zero mark (everything is
// after it); an unparseable ts falls back to a string compare so a malformed value still
// orders deterministically rather than panicking.
//
// Exported because every Slack poll door needs the SAME high-water comparison: this relay,
// and `fak chatops`'s read-only door, which previously carried a byte-identical private copy
// (chatopsTSAfter). One definition, one test, one ordering rule.
func TSAfter(a, b string) bool {
	if b == "" {
		return a != ""
	}
	if a == "" {
		return false
	}
	af, aerr := strconv.ParseFloat(a, 64)
	bf, berr := strconv.ParseFloat(b, 64)
	if aerr != nil || berr != nil {
		return a > b
	}
	return af > bf
}

// PollHistory fetches one page of channel history strictly after sinceTS and returns it
// OLDEST-FIRST. Both halves matter and neither is optional for a high-water-mark poller:
// the fetch error is wrapped "history: …" so a caller's error names the failing call, and
// the sort is by TSAfter (numeric ts order, not lexical) so a door answers in conversational
// order and advances its mark monotonically. Slack's API may over-return; re-filtering by
// the caller's own mark is still the caller's job.
//
// Exported because every high-water Slack door needs the SAME fetch-and-order preamble:
// this relay and `fak chatops`'s read-only door carried byte-identical private copies, and
// a door that sorted differently would have advanced its mark past unanswered messages.
func PollHistory(ctx context.Context, slack SlackClient, channel, sinceTS string, limit int) ([]Message, error) {
	msgs, err := slack.History(ctx, channel, sinceTS, limit)
	if err != nil {
		return nil, fmt.Errorf("history: %w", err)
	}
	sort.SliceStable(msgs, func(i, j int) bool { return TSAfter(msgs[j].TS, msgs[i].TS) })
	return msgs, nil
}

// PollLoop runs tick every interval until ctx is cancelled, returning ctx.Err(). A tick
// error is delivered to onErr (when non-nil) and the loop CONTINUES — a single bad poll
// must not tear down a long-lived door. The first tick fires immediately, before the first
// interval elapses, so a door starts working the moment it is run.
//
// fallback is the interval used when the caller passes interval<=0. It is a parameter, not
// a package constant, because the two doors deliberately poll at different cadences (a chat
// relay answers a human; the chatops control door polls faster) — that divergence is real
// and is preserved by making the caller name its own default.
//
// Exported because the LIFECYCLE is the shared part: this relay and `fak chatops`'s door
// carried byte-identical private copies of the ticker/select/keep-going rule.
func PollLoop(ctx context.Context, interval, fallback time.Duration, tick func(context.Context) error, onErr func(error)) error {
	if interval <= 0 {
		interval = fallback
	}
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		if err := tick(ctx); err != nil && onErr != nil {
			onErr(err)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-t.C:
		}
	}
}

// --- live HTTP implementations -------------------------------------------------------

// HTTPSlack is the live SlackClient over the Slack Web API. Token is a bot token with the
// conversations.history (channels:history / groups:history) and chat:write scopes. APIBase
// defaults to https://slack.com/api/ and is overridable for tests.
//
// It is a thin adapter over internal/slackwire — the ONE Slack transport (#2261): the
// HTTP exchange, ok:false envelope decode, and 429/Retry-After handling live there; this
// type keeps only the relay-shaped surface (its Message mapping and threadTS-in-Post).
// The hand-rolled conversations.history/chat.postMessage HTTP that used to live here was
// the second parallel Slack client the epic retired.
type HTTPSlack struct {
	Token   string
	APIBase string
	HTTP    *http.Client
}

// wire builds the slackwire client for one call. Construction is cheap (the injected —
// or default — http.Client carries the connection pool, not this struct), and building
// per call keeps HTTPSlack a plain value type with no lazy-init state to guard.
func (s *HTTPSlack) wire() *slackwire.Client {
	opts := make([]slackwire.Option, 0, 2)
	if s.APIBase != "" {
		opts = append(opts, slackwire.WithAPIBase(s.APIBase))
	}
	if s.HTTP != nil {
		opts = append(opts, slackwire.WithHTTPClient(s.HTTP))
	}
	return slackwire.New(s.Token, opts...)
}

// UnmarshalJSON maps Slack's snake_case message fields onto Message.
func (m *Message) UnmarshalJSON(b []byte) error {
	var raw struct {
		Type     string `json:"type"`
		Subtype  string `json:"subtype"`
		TS       string `json:"ts"`
		ThreadTS string `json:"thread_ts"`
		User     string `json:"user"`
		BotID    string `json:"bot_id"`
		Text     string `json:"text"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	m.Type, m.Subtype, m.TS = raw.Type, raw.Subtype, raw.TS
	m.ThreadTS, m.User, m.BotID, m.Text = raw.ThreadTS, raw.User, raw.BotID, raw.Text
	return nil
}

// History calls conversations.history. oldestTS is passed as the `oldest` bound; the relay
// still re-filters by ts so the inclusive/exclusive nuance never double-answers a message.
func (s *HTTPSlack) History(ctx context.Context, channel, oldestTS string, limit int) ([]Message, error) {
	wire, err := s.wire().History(ctx, channel, oldestTS, limit)
	if err != nil {
		return nil, err
	}
	msgs := make([]Message, len(wire))
	for i, m := range wire {
		msgs[i] = Message{
			Type: m.Type, Subtype: m.Subtype, TS: m.TS,
			ThreadTS: m.ThreadTS, User: m.User, BotID: m.BotID, Text: m.Text,
		}
	}
	return msgs, nil
}

// Post calls chat.postMessage, threading the reply under threadTS when set.
func (s *HTTPSlack) Post(ctx context.Context, channel, threadTS, text string) (string, error) {
	return s.wire().PostMessage(ctx, channel, text, nil, threadTS)
}

// HTTPModel calls a served OpenAI-compatible /v1/chat/completions. Endpoint is the serve
// base (e.g. http://127.0.0.1:8080); Model is the advertised id; System, when set, is sent
// as a leading system turn; MaxTokens/Temperature bound the completion. APIKey, when set,
// is sent as a Bearer token (for a --require-key-env serve).
type HTTPModel struct {
	Endpoint    string
	Model       string
	System      string
	MaxTokens   int
	Temperature float64
	APIKey      string
	HTTP        *http.Client
}

func (m *HTTPModel) httpClient() *http.Client {
	if m.HTTP != nil {
		return m.HTTP
	}
	return &http.Client{Timeout: 5 * time.Minute} // a CPU GLM-5.2 turn is slow; do not clip it
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature float64       `json:"temperature,omitempty"`
	Stream      bool          `json:"stream"`
}

type chatResponse struct {
	Choices []struct {
		Message chatMessage `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// Complete sends a single-user-turn chat completion and returns the assistant text.
func (m *HTTPModel) Complete(ctx context.Context, prompt string) (string, error) {
	msgs := make([]chatMessage, 0, 2)
	if strings.TrimSpace(m.System) != "" {
		msgs = append(msgs, chatMessage{Role: "system", Content: m.System})
	}
	msgs = append(msgs, chatMessage{Role: "user", Content: prompt})
	reqBody := chatRequest{
		Model:       m.Model,
		Messages:    msgs,
		MaxTokens:   m.MaxTokens,
		Temperature: m.Temperature,
		Stream:      false,
	}
	raw, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}
	endpoint := strings.TrimRight(m.Endpoint, "/") + "/v1/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(raw))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	if m.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+m.APIKey)
	}
	resp, err := m.httpClient().Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	var cr chatResponse
	if err := json.Unmarshal(data, &cr); err != nil {
		return "", fmt.Errorf("chat/completions: decode: %w (status=%d body=%.200s)", err, resp.StatusCode, string(data))
	}
	if cr.Error != nil && cr.Error.Message != "" {
		return "", fmt.Errorf("chat/completions: %s", cr.Error.Message)
	}
	if resp.StatusCode/100 != 2 {
		return "", fmt.Errorf("chat/completions: status %d (body=%.200s)", resp.StatusCode, string(data))
	}
	if len(cr.Choices) == 0 {
		return "", fmt.Errorf("chat/completions: no choices (body=%.200s)", string(data))
	}
	return cr.Choices[0].Message.Content, nil
}
