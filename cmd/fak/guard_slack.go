package main

import (
	"context"
	"fmt"
	"os"
	"path"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/slackenv"
	"github.com/anthony-chaudhary/fak/internal/slackoutbox"
)

const (
	guardSessionsChannelEnv     = "FAK_GUARD_SESSIONS_CHANNEL"
	guardSessionsTokenEnv       = "FAK_GUARD_SESSIONS_TOKEN"
	guardSessionsChannelDefault = "C0BEZ7513UM"
	guardSessionThreadSource    = "guard-session"
)

type guardSessionThreadMeta struct {
	Channel   string
	Nonce     string
	TraceID   string
	Agent     string
	Provider  string
	AuditPath string
	StartedAt time.Time
	PID       int
}

func resolveGuardSessionsChannel() string {
	if r := slackenv.Lookup(guardSessionsChannelEnv); r.Set() {
		return strings.TrimSpace(r.Value)
	}
	return guardSessionsChannelDefault
}

func resolveGuardSessionsToken() string {
	if r := slackenv.Lookup(guardSessionsTokenEnv); r.Set() {
		return strings.TrimSpace(r.Value)
	}
	if r := slackenv.Lookup(scoreboardTokenKey); r.Set() {
		return strings.TrimSpace(r.Value)
	}
	return ""
}

func enqueueGuardSessionThread(traceID, provider string, command []string, auditPath string, startedAt time.Time) (slackoutbox.Row, error) {
	row := guardSessionThreadRow(guardSessionThreadMeta{
		Channel:   resolveGuardSessionsChannel(),
		Nonce:     slackoutbox.NewNonce(),
		TraceID:   traceID,
		Agent:     guardSessionAgentName(command),
		Provider:  provider,
		AuditPath: auditPath,
		StartedAt: startedAt,
		PID:       os.Getpid(),
	})
	ob, err := openOutbox()
	if err != nil {
		return row, err
	}
	if _, err := ob.Enqueue(row); err != nil {
		return row, err
	}
	return row, nil
}

func guardSessionThreadRow(m guardSessionThreadMeta) slackoutbox.Row {
	channel := strings.TrimSpace(m.Channel)
	if channel == "" {
		channel = guardSessionsChannelDefault
	}
	nonce := strings.TrimSpace(m.Nonce)
	if nonce == "" {
		nonce = slackoutbox.NewNonce()
	}
	started := m.StartedAt.UTC()
	if started.IsZero() {
		started = time.Now().UTC()
	}
	text := fmt.Sprintf(
		"fak guard session started\nsession_thread_id: %s\ntrace_id: %s\nagent: %s\nprovider: %s\npid: %d\nstarted_utc: %s\naudit: %s",
		guardSlackField(nonce),
		guardSlackField(m.TraceID),
		guardSlackField(m.Agent),
		guardSlackField(m.Provider),
		m.PID,
		started.Format(time.RFC3339),
		guardSlackAuditRef(m.AuditPath),
	)
	return slackoutbox.Row{
		Nonce:   nonce,
		Channel: channel,
		Text:    text,
		Source:  guardSessionThreadSource,
	}
}

func guardSessionAgentName(command []string) string {
	if len(command) == 0 {
		return "unknown"
	}
	name := strings.TrimSpace(command[0])
	if name == "" {
		return "unknown"
	}
	name = strings.ReplaceAll(name, "\\", "/")
	return path.Base(name)
}

func guardSlackField(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "-"
	}
	return strings.Join(strings.Fields(s), " ")
}

func guardSlackAuditRef(s string) string {
	s = guardSlackField(s)
	if s == "-" {
		return s
	}
	if strings.Contains(strings.ToLower(s), "off") {
		return "off"
	}
	return path.Base(strings.ReplaceAll(s, "\\", "/"))
}

func startGuardSessionThreadDrain() {
	token := resolveGuardSessionsToken()
	if token == "" {
		return
	}
	go func() {
		ob, err := openOutbox()
		if err != nil {
			return
		}
		wire, err := outboxWire(token, "")
		if err != nil {
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_, _ = ob.Drain(ctx, wire, slackoutbox.DrainOpts{Root: "."})
	}()
}
