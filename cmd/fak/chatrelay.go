package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"time"

	"github.com/anthony-chaudhary/fak/internal/chatrelay"
)

// `fak chatrelay` — make a `fak serve`-hosted model usable from a Slack channel.
//
// It polls ONE Slack channel, forwards each new human message to a served
// OpenAI-compatible /v1/chat/completions endpoint (the in-kernel `fak serve`, e.g. GLM-5.2
// on the pure CPU forward), and posts the model's reply back in-thread. It is a generic
// chatbot front-end — no shell, no lab identifiers in source, endpoint/token/channel all
// resolved at runtime — so it stays on the PUBLIC side of the GPU-server/Slack boundary
// (the lab *control* bridge is the private piece; this is *chat*). See internal/chatrelay.
//
// End to end (the "GLM-5.2 usable via Slack" path):
//
//	# 1. serve GLM-5.2 on the pure in-kernel forward (CPU; the ~30x-faster path vs device offload)
//	FAK_Q4K=1 fak serve --gguf <glm-5.2-shard1.gguf> --addr 127.0.0.1:8080
//
//	# 2. bridge a Slack channel to it (token+channel from .env.slack.local or flags)
//	fak chatrelay --endpoint http://127.0.0.1:8080 --channel C0XXXX --model glm-5.2
//
//	# or, run the bridge under the guard defaults:
//	fak guard -- fak chatrelay --endpoint http://127.0.0.1:8080 --channel C0XXXX
func cmdChatRelay(argv []string) {
	fs := flag.NewFlagSet("chatrelay", flag.ExitOnError)
	endpoint := fs.String("endpoint", "http://127.0.0.1:8080", "served OpenAI-compatible base URL (the `fak serve` --addr); /v1/chat/completions is appended")
	model := fs.String("model", "glm-5.2", "model id sent in the chat request (a label; the in-kernel serve uses its loaded model regardless)")
	channel := fs.String("channel", "", "Slack channel id to bridge (default: $FAK_CHATRELAY_CHANNEL / .env.slack.local). REQUIRED — no silent fallback.")
	token := fs.String("token", "", "Slack bot token (default: $FAK_CHATRELAY_TOKEN, then .env.slack.local, then the scoreboard bot token). Needs conversations.history + chat:write scopes.")
	botUser := fs.String("bot-user", "", "this bot's own Slack user id, to skip its own posts (belt-and-suspenders; bot_id posts are skipped regardless)")
	mention := fs.String("mention", "", "only answer messages containing this mention token (e.g. <@U07BOT>), and strip it from the prompt; empty answers every human message")
	system := fs.String("system", "", "optional system prompt prepended to every turn")
	maxTokens := fs.Int("max-tokens", 512, "max_tokens for each completion")
	temperature := fs.Float64("temperature", 0, "sampling temperature (0 = greedy)")
	apiKeyEnv := fs.String("api-key-env", "", "env var holding a bearer token to send to the serve (for a --require-key-env serve)")
	interval, prime, once, dryRun := parseChatPollingFlags(fs, argv,
		"on start, skip the existing channel backlog and only answer messages posted after launch (pass --prime=false to answer the latest backlog too — handy with --once)",
		"print the resolved config and exit without connecting")

	tok := *token
	if tok == "" {
		tok = chatrelay.ResolveToken()
	}
	ch := *channel
	if ch == "" {
		ch = chatrelay.ResolveChannel()
	}

	apiKey := ""
	if *apiKeyEnv != "" {
		apiKey = os.Getenv(*apiKeyEnv)
		if apiKey == "" {
			fmt.Fprintf(os.Stderr, "fak chatrelay: --api-key-env %s is set but the env var is empty\n", *apiKeyEnv)
			os.Exit(2)
		}
	}

	mentionMode := "every human message"
	if *mention != "" {
		mentionMode = "messages addressed to " + *mention
	}
	if *dryRun {
		fmt.Printf("fak chatrelay (dry-run):\n")
		fmt.Printf("  endpoint : %s/v1/chat/completions\n", *endpoint)
		fmt.Printf("  model    : %s\n", *model)
		fmt.Printf("  channel  : %s\n", orUnset(ch))
		fmt.Printf("  token    : %s\n", redact(tok))
		fmt.Printf("  answers  : %s\n", mentionMode)
		fmt.Printf("  interval : %s   prime=%v once=%v\n", *interval, *prime, *once)
		if ch == "" {
			fmt.Fprintln(os.Stderr, "  (channel is UNSET — set --channel or FAK_CHATRELAY_CHANNEL before a live run)")
		}
		if tok == "" {
			fmt.Fprintln(os.Stderr, "  (token is UNSET — set --token or FAK_CHATRELAY_TOKEN before a live run)")
		}
		return
	}

	requireSlackTokenAndChannel("fak chatrelay", "FAK_CHATRELAY", tok, ch)

	relay := &chatrelay.Relay{
		Slack: &chatrelay.HTTPSlack{Token: tok},
		Model: &chatrelay.HTTPModel{
			Endpoint:    *endpoint,
			Model:       *model,
			System:      *system,
			MaxTokens:   *maxTokens,
			Temperature: *temperature,
			APIKey:      apiKey,
		},
		Channel:   ch,
		BotUserID: *botUser,
		Mention:   *mention,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	fmt.Printf("fak chatrelay: bridging channel %s <-> %s/v1/chat/completions (model=%s), answering %s\n",
		ch, *endpoint, *model, mentionMode)

	runSlackPollLifecycle(ctx, relay, "fak chatrelay", "answered", *prime, *once, *interval)
}

// slackPoller is the poll lifecycle the Slack-polling fak commands share: skip the
// backlog once, then either take a single poll or loop on a ticker. `fak chatops`'s
// read-only door (chatopsDoor) and `fak chatrelay`'s bridge (chatrelay.Relay) implement
// it independently — the interface exists so the SHELL around them is written once.
type slackPoller interface {
	Prime(ctx context.Context) error
	Tick(ctx context.Context) (int, error)
	Run(ctx context.Context, interval time.Duration, onErr func(error)) error
}

// requireSlackTokenAndChannel refuses to start a Slack-polling command that resolved no
// bot token or no channel, printing the exact refusal each command used to print itself:
// `label` is the command name ("fak chatops") and `envPrefix` its env namespace
// ("FAK_CHATOPS"). Both refusals exit 2 — an unconfigured door/bridge is an operator
// mistake, not a running mode. NOT folded in: `fak chatops`' third refusal (an empty
// admin allowlist), which only the fail-closed door has.
func requireSlackTokenAndChannel(label, envPrefix, token, channel string) {
	if token == "" {
		fmt.Fprintf(os.Stderr, "%s: no Slack token — set --token, %s_TOKEN, or add it to .env.slack.local\n",
			label, envPrefix)
		os.Exit(2)
	}
	if channel == "" {
		fmt.Fprintf(os.Stderr, "%s: no channel — set --channel or %s_CHANNEL (no silent fallback to another channel)\n",
			label, envPrefix)
		os.Exit(2)
	}
}

// runSlackPollLifecycle drives a slackPoller through the prime -> (once | loop) shell both
// commands had copied: a failed prime is a warning and the run continues, `--once` takes a
// single poll and reports the count, and the long-lived loop exits 1 on a real error but
// stays silent when the operator interrupted it (ctx cancelled).
//
// `tickedVerb` is the past-tense verb the --once summary uses: the door says "handled" and
// the bridge says "answered" — a real wording divergence, kept as a parameter so each
// command prints exactly what it printed before rather than being quietly unified.
func runSlackPollLifecycle(ctx context.Context, p slackPoller, label, tickedVerb string,
	prime, once bool, interval time.Duration) {
	if prime {
		if err := p.Prime(ctx); err != nil {
			fmt.Fprintf(os.Stderr, "%s: prime (skip backlog) failed: %v — will answer the visible history on the first poll\n",
				label, err)
		}
	}

	if once {
		n, err := p.Tick(ctx)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", label, err)
			os.Exit(1)
		}
		fmt.Printf("%s: %s %d message(s)\n", label, tickedVerb, n)
		return
	}

	err := p.Run(ctx, interval, func(e error) {
		fmt.Fprintf(os.Stderr, "%s: poll error (continuing): %v\n", label, e)
	})
	if err != nil && ctx.Err() == nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", label, err)
		os.Exit(1)
	}
}

// orUnset renders an empty value as "(unset)" for the dry-run summary.
func orUnset(s string) string {
	if s == "" {
		return "(unset)"
	}
	return s
}

// redact shows only that a token is present and its last 4 chars, never the secret.
func redact(s string) string {
	return redactSecret(s)
}
