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
	interval := fs.Duration("interval", 3*time.Second, "poll interval between conversations.history fetches")
	prime := fs.Bool("prime", true, "on start, skip the existing channel backlog and only answer messages posted after launch (pass --prime=false to answer the latest backlog too — handy with --once)")
	once := fs.Bool("once", false, "run a single poll and exit (smoke test) instead of looping")
	dryRun := fs.Bool("dry-run", false, "print the resolved config and exit without connecting")
	_ = fs.Parse(argv)

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

	if tok == "" {
		fmt.Fprintln(os.Stderr, "fak chatrelay: no Slack token — set --token, FAK_CHATRELAY_TOKEN, or add it to .env.slack.local")
		os.Exit(2)
	}
	if ch == "" {
		fmt.Fprintln(os.Stderr, "fak chatrelay: no channel — set --channel or FAK_CHATRELAY_CHANNEL (no silent fallback to another channel)")
		os.Exit(2)
	}

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

	if *prime {
		if err := relay.Prime(ctx); err != nil {
			fmt.Fprintf(os.Stderr, "fak chatrelay: prime (skip backlog) failed: %v — will answer the visible history on the first poll\n", err)
		}
	}

	if *once {
		n, err := relay.Tick(ctx)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak chatrelay: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("fak chatrelay: answered %d message(s)\n", n)
		return
	}

	err := relay.Run(ctx, *interval, func(e error) {
		fmt.Fprintf(os.Stderr, "fak chatrelay: poll error (continuing): %v\n", e)
	})
	if err != nil && ctx.Err() == nil {
		fmt.Fprintf(os.Stderr, "fak chatrelay: %v\n", err)
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
	if s == "" {
		return "(unset)"
	}
	if len(s) <= 4 {
		return "****"
	}
	return "****" + s[len(s)-4:]
}
