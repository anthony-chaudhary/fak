package dgxbridge

import (
	"context"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// Session is one hub-managed control session, parsed from a "!sessions" listing.
// The multi-session control_hub identifies sessions by a profile-scoped id like
// "default-1" (NOT the thread ts), which is the argument the hub's !dump/!clear verbs
// require.
type Session struct {
	ID       string // e.g. "default-1"
	Status   string // running | exited | stopped | killed
	Profile  string // e.g. "default"
	Mode     string // pipe | tmux
	ThreadTS string // the session's own Slack thread
}

// Running reports whether the session is currently driveable.
func (s Session) Running() bool { return s.Status == "running" }

// sessionLineRe matches one hub listing entry, e.g.
//
//	`default-1` running profile=`default` mode=`pipe` thread=`1781964298.319749`
var sessionLineRe = regexp.MustCompile(
	"`([^`]+)`\\s+(\\w+)\\s+profile=`([^`]*)`\\s+mode=`([^`]*)`\\s+thread=`([^`]*)`")

// parseSessions extracts sessions from a hub "!sessions" reply. It is tolerant of the
// listing being collapsed onto one line (Slack joins the reply's newlines) or spread
// across lines, since it matches per-entry rather than per-line.
func parseSessions(text string) []Session {
	var out []Session
	for _, m := range sessionLineRe.FindAllStringSubmatch(text, -1) {
		out = append(out, Session{
			ID:       m[1],
			Status:   m[2],
			Profile:  m[3],
			Mode:     m[4],
			ThreadTS: m[5],
		})
	}
	return out
}

// ListSessions asks the hub to enumerate its sessions (!sessions) and parses the reply.
// It posts the command at channel top level (where the hub processes commands) and polls
// channel history for the "Known control sessions" listing.
func (b *Bridge) ListSessions(ctx context.Context) ([]Session, error) {
	b.normalize()
	postTS, err := b.Client.Post(ctx, b.Channel, "", "!sessions")
	if err != nil {
		return nil, fmt.Errorf("post !sessions: %w", err)
	}
	after, _ := strconv.ParseFloat(postTS, 64)
	deadline := b.now().Add(30 * time.Second)
	for b.now().Before(deadline) {
		time.Sleep(3 * time.Second)
		msgs, err := b.Client.History(ctx, b.Channel, 40)
		if err != nil {
			continue
		}
		for _, m := range msgs {
			if mts, _ := strconv.ParseFloat(m.TS, 64); mts <= after {
				continue
			}
			if strings.Contains(m.Text, "Known control sessions") {
				if s := parseSessions(m.Text); len(s) > 0 {
					return s, nil
				}
			}
		}
	}
	return nil, fmt.Errorf("no !sessions reply within timeout")
}

// PickRunning returns the newest running session (optionally restricted to a profile),
// or ok=false if none qualifies. "Newest" is the largest thread ts.
func PickRunning(sessions []Session, profile string) (Session, bool) {
	var best Session
	var bestTS float64
	found := false
	for _, s := range sessions {
		if !s.Running() {
			continue
		}
		if profile != "" && s.Profile != profile {
			continue
		}
		ts, _ := strconv.ParseFloat(s.ThreadTS, 64)
		if !found || ts > bestTS {
			best, bestTS, found = s, ts, true
		}
	}
	return best, found
}
