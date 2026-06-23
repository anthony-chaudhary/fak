// Command dgxbridge is the pure-Go driver for the Slack control-bridge that fronts
// the lab DGX (8x A100). The DGX is reachable only through a slack-helpers "control
// session" thread in #dgx-control: a message posted to the thread is typed as stdin
// into a persistent remote shell. Because the bridge runs in PTY mode its live
// stdout mirror wedges on Slack's msg_too_long limit, so this tool reads results back
// via the bridge's !dump transcript file (limit-free) — see internal/dgxbridge.
//
// A control banner can outlive its shell, so discovery prefers a LIVE session:
// status -probe (and exec/readfile/pull auto-pick) probe each candidate with !status
// and use the newest one that actually answers.
//
// Token resolution: -token, then FAK_SLACK_BOT_TOKEN, then SLACK_BOT_TOKEN, then a
// gitignored .env.slack.local walked up from the cwd.
//
// Examples:
//
//	go run ./cmd/dgxbridge -dgx-host dgx-a100.example.lab status -probe
//	go run ./cmd/dgxbridge -thread-ts 1781964542.658809 exec 'nvidia-smi -L'
//	go run ./cmd/dgxbridge -thread-ts <ts> readfile /tmp/MATRIX.json
//	go run ./cmd/dgxbridge -thread-ts <ts> pull /srv/run/GATE.json ./GATE.json
//	go run ./cmd/dgxbridge -thread-ts <ts> ship ./run.sh /tmp/run.sh
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dgxbridge"
)

func main() {
	var (
		token     = flag.String("token", "", "Slack bot token (default: env/.env.slack.local)")
		channel   = flag.String("channel", "", "Slack control channel id (default: FAK_SLACK_CHANNEL/SLACK_CHANNEL env or .env.slack.local)")
		threadTS  = flag.String("thread-ts", "", "control-session thread ts (default: auto-discover via -dgx-host)")
		dgxHost   = flag.String("dgx-host", "", "DGX host to discover the live control thread for")
		timeout   = flag.Duration("timeout", 4*time.Minute, "overall per-command timeout")
		settle    = flag.Duration("settle", 6*time.Second, "wait after posting before first !dump")
		probe     = flag.Bool("probe", false, "probe each candidate session for liveness (status), and auto-pick a live one")
		probeWait = flag.Duration("probe-wait", 15*time.Second, "how long to wait for a session's !status reply")
		scratch   = flag.String("scratch", "", "dir to cache downloaded transcripts (debug)")
		sessionID = flag.String("session-id", "", "hub session id (e.g. default-1); enables multi-session `!dump <id>`. Auto-discovered (newest running) when empty")
	)
	flag.Parse()
	args := flag.Args()
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "usage: dgxbridge [flags] <status|sessions|selftest|exec|readfile|pull|ship> [args]")
		os.Exit(2)
	}

	// Resolve the channel the dead-simple way: explicit -channel wins, else the
	// FAK_SLACK_CHANNEL/SLACK_CHANNEL env or a CHANNEL line in .env.slack.local, and only
	// then the placeholder. The placeholder is never reachable, so fail fast with a clear
	// hint instead of letting every Slack call return channel_not_found.
	if *channel == "" {
		*channel = dgxbridge.ResolveChannel()
	}
	if *channel == "" || *channel == dgxbridge.DefaultChannel {
		fatal(fmt.Errorf("no Slack channel set: pass -channel <id>, or add SLACK_CHANNEL=<id> to .env.slack.local (or set FAK_SLACK_CHANNEL)"))
	}

	client, err := dgxbridge.NewClient(*token)
	if err != nil {
		fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), *timeout+2*time.Minute)
	defer cancel()

	newBridge := func(ts string) *dgxbridge.Bridge {
		return &dgxbridge.Bridge{
			Client: client, Channel: *channel, ThreadTS: ts, SessionID: *sessionID,
			Timeout: *timeout, Settle: *settle, ScratchDir: *scratch,
		}
	}

	// resolveThread returns an explicit -thread-ts, else discovers. It first tries the
	// multi-session hub (!sessions): with -session-id it resolves that id's thread, else
	// it picks the newest running session and sets *sessionID so !dump <id> is used. It
	// falls back to legacy banner discovery for a single-session bridge.
	resolveThread := func() string {
		if *threadTS != "" {
			return *threadTS
		}
		hub := &dgxbridge.Bridge{Client: client, Channel: *channel, Settle: *settle, Timeout: *timeout}
		if sessions, err := hub.ListSessions(ctx); err == nil && len(sessions) > 0 {
			if *sessionID != "" {
				for _, s := range sessions {
					if s.ID == *sessionID {
						return s.ThreadTS
					}
				}
				fatal(fmt.Errorf("session %q not in hub listing", *sessionID))
			}
			if s, ok := dgxbridge.PickRunning(sessions, ""); ok {
				*sessionID = s.ID
				fmt.Fprintf(os.Stderr, "dgxbridge: picked running session %s (thread %s)\n", s.ID, s.ThreadTS)
				return s.ThreadTS
			}
			fatal(fmt.Errorf("hub has %d session(s) but none running — start one or pass -session-id/-thread-ts", len(sessions)))
		}
		sessions, err := client.FindControlSessions(ctx, *channel, *dgxHost)
		if err != nil {
			fatal(fmt.Errorf("discover control session: %w", err))
		}
		if len(sessions) == 0 {
			fatal(fmt.Errorf("no control session banner for host %q in channel %s", *dgxHost, *channel))
		}
		if !*probe {
			return sessions[0].ThreadTS
		}
		for _, s := range sessions {
			alive, _, err := newBridge(s.ThreadTS).Alive(ctx, *probeWait)
			if err == nil && alive {
				return s.ThreadTS
			}
		}
		fatal(fmt.Errorf("no LIVE control session for host %q (%d stale banner(s) found) — the DGX shell/bridge needs (re)starting by an operator", *dgxHost, len(sessions)))
		return ""
	}

	switch verb := args[0]; verb {
	case "status":
		sessions, err := client.FindControlSessions(ctx, *channel, *dgxHost)
		if err != nil {
			fatal(err)
		}
		if len(sessions) == 0 {
			fmt.Printf("no control session banner for host %q in channel %s\n", *dgxHost, *channel)
			os.Exit(1)
		}
		anyLive := false
		for i, s := range sessions {
			line := fmt.Sprintf("[%d] thread_ts=%s host=%s", i, s.ThreadTS, s.Host)
			if *probe {
				alive, detail, _ := newBridge(s.ThreadTS).Alive(ctx, *probeWait)
				if alive {
					anyLive = true
					line += "  LIVE"
				} else {
					line += "  STALE (" + firstLine(detail) + ")"
				}
			}
			fmt.Println(line)
		}
		if *probe && !anyLive {
			fmt.Println("\nNo live session: a banner exists but no shell answers. An operator must (re)start the DGX control shell/bridge.")
			os.Exit(1)
		}

	case "sessions":
		// Enumerate the hub's sessions (id, status, thread) — the id is what !dump needs.
		hub := &dgxbridge.Bridge{Client: client, Channel: *channel, Settle: *settle, Timeout: *timeout}
		sessions, err := hub.ListSessions(ctx)
		if err != nil {
			fatal(err)
		}
		for _, s := range sessions {
			marker := ""
			if s.Running() {
				marker = "  RUNNING"
			}
			fmt.Printf("%-14s %-8s profile=%s mode=%s thread=%s%s\n", s.ID, s.Status, s.Profile, s.Mode, s.ThreadTS, marker)
		}

	case "selftest":
		// Verify the *readback* path (not just liveness): round-trip an echo and report
		// a typed reason if results can't be read back from this session.
		res, err := newBridge(resolveThread()).SelfTest(ctx)
		if err != nil {
			fatal(err)
		}
		if res.OK {
			fmt.Println("readback OK:", res.Detail)
		} else {
			fmt.Printf("readback BROKEN: %s — %s\n  %s\n", res.Reason, readbackHint(res.Reason), firstLine(res.Detail))
			os.Exit(1)
		}

	case "exec":
		if len(args) < 2 {
			fatal(fmt.Errorf("exec needs a command"))
		}
		out, err := newBridge(resolveThread()).Exec(ctx, args[1])
		if err != nil {
			fatal(err)
		}
		fmt.Println(out)

	case "readfile":
		if len(args) < 2 {
			fatal(fmt.Errorf("readfile needs a remote path"))
		}
		data, err := newBridge(resolveThread()).ReadFile(ctx, args[1])
		if err != nil {
			fatal(err)
		}
		os.Stdout.Write(data)

	case "pull":
		if len(args) < 3 {
			fatal(fmt.Errorf("pull needs <remote> <local>"))
		}
		if err := newBridge(resolveThread()).PullArtifact(ctx, args[1], args[2]); err != nil {
			fatal(err)
		}
		fmt.Fprintf(os.Stderr, "pulled %s -> %s\n", args[1], args[2])

	case "ship":
		if len(args) < 3 {
			fatal(fmt.Errorf("ship needs <local> <remote>"))
		}
		if err := newBridge(resolveThread()).Ship(ctx, args[1], args[2]); err != nil {
			fatal(err)
		}
		fmt.Fprintf(os.Stderr, "shipped %s -> %s\n", args[1], args[2])

	default:
		fatal(fmt.Errorf("unknown verb %q", verb))
	}
}

// readbackHint maps a SelfTest reason to a one-line operator action.
func readbackHint(reason string) string {
	switch reason {
	case dgxbridge.ReasonNoSessionTranscript:
		return "this session's !dump isn't uploading a transcript bound to its thread — restart the DGX control bridge, ideally in pipe (--no-pty) mode"
	case dgxbridge.ReasonSentinelMissing:
		return "the shell isn't executing/echoing commands — the PTY may be wedged; restart the control bridge"
	case dgxbridge.ReasonEchoMismatch:
		return "the read path returned unexpected content — check for two bridges sharing this thread"
	default:
		return "control transport error — check the token, channel, and bridge process"
	}
}

func firstLine(s string) string {
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			return s[:i]
		}
	}
	if len(s) > 80 {
		return s[:80]
	}
	return s
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "dgxbridge:", err)
	os.Exit(1)
}
