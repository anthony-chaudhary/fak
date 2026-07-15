package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/slackoutbox"
)

const (
	guardStopsSlackSchema       = "fak.guard-stops-slack/1"
	guardStopsSlackSource       = "guard-stops-scoreboard"
	guardStopsSlackStateFile    = "guard-stops-scoreboard.json"
	guardStopsSlackInterval     = 5 * time.Minute
	guardStopsSlackRecentWindow = 20
)

type guardStopsSlackTally struct {
	Schema           string `json:"schema"`
	Total            int    `json:"total"`
	StandDown        int    `json:"stand_down"`
	FailOpen         int    `json:"fail_open"`
	OperatorDirected int    `json:"operator_directed"`
	RecentTotal      int    `json:"recent_total"`
	RecentStandDown  int    `json:"recent_stand_down"`
	RecentFailOpen   int    `json:"recent_fail_open"`
	Status           string `json:"status"`
}

type guardStopsSlackState struct {
	Schema string `json:"schema"`
	Nonce  string `json:"nonce"`
	TS     string `json:"ts,omitempty"`
}

func cmdGuardStopsSlack(argv []string) { os.Exit(runGuardStopsSlack(os.Stdout, os.Stderr, argv)) }

func runGuardStopsSlack(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("guard-stops-slack", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledgerFlag := fs.String("ledger", "", "guard Stop JSONL ledger")
	outboxFlag := fs.String("outbox-dir", "", "durable Slack outbox directory")
	channelFlag := fs.String("channel", "", "scoreboard channel id")
	dryRun := fs.Bool("dry-run", false, "render without enqueueing or network")
	watch := fs.Bool("watch", false, "refresh forever at --interval")
	interval := fs.Duration("interval", guardStopsSlackInterval, "watch refresh interval")
	recent := fs.Int("recent", guardStopsSlackRecentWindow, "recent-row trend window")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *recent < 1 || *interval <= 0 {
		fmt.Fprintln(stderr, "fak guard-stops-slack: --recent and --interval must be positive")
		return 2
	}
	ledger := strings.TrimSpace(*ledgerFlag)
	if ledger == "" {
		ledger = guardStopsLedgerResolved()
	}
	outboxDir := strings.TrimSpace(*outboxFlag)
	if outboxDir == "" {
		outboxDir = resolveOutboxDir()
	}
	channel := strings.TrimSpace(*channelFlag)
	if channel == "" {
		channel = scoreboard.ResolveChannel()
	}
	if !*dryRun && channel == "" {
		fmt.Fprintln(stderr, "fak guard-stops-slack: no scoreboard channel; set FAK_SCOREBOARD_CHANNEL or pass --channel")
		return 1
	}
	tick := func() int {
		content, err := readGuardStopsLedger(ledger)
		if err != nil {
			fmt.Fprintf(stderr, "fak guard-stops-slack: read %s: %v\n", ledger, err)
			return 1
		}
		tally := summarizeGuardStopsSlack(content, *recent)
		text := renderGuardStopsSlack(tally)
		if *dryRun {
			fmt.Fprintln(stdout, text)
			return 0
		}
		if err := upsertGuardStopsSlack(outboxDir, channel, text); err != nil {
			fmt.Fprintf(stderr, "fak guard-stops-slack: %v\n", err)
			return 1
		}
		return 0
	}
	if !*watch {
		return tick()
	}
	for {
		if rc := tick(); rc != 0 {
			return rc
		}
		time.Sleep(*interval)
	}
}

func summarizeGuardStopsSlack(content string, recentN int) guardStopsSlackTally {
	rows := jsonlledger.Parse(content, func(r guardStopRecord) bool { return r.Schema == guardStopRecordSchema })
	sum := summarizeGuardStops(content, 0)
	start := len(rows) - recentN
	if start < 0 {
		start = 0
	}
	tally := guardStopsSlackTally{
		Schema:           guardStopsSlackSchema,
		Total:            sum.Total,
		StandDown:        sum.StandDown,
		FailOpen:         sum.FailOpen,
		OperatorDirected: sum.OperatorDirected,
		RecentTotal:      len(rows) - start,
		Status:           "green",
	}
	for _, row := range rows[start:] {
		switch recordKind(row) {
		case stopKindStandDown:
			tally.RecentStandDown++
		case stopKindFailOpen:
			tally.RecentFailOpen++
		}
	}
	// Thresholds are deliberately trend-only: lifetime counters never decay and must
	// not pin a recovered fleet red forever.
	switch {
	case tally.RecentFailOpen >= 2 || tally.RecentStandDown >= 3:
		tally.Status = "red"
	case tally.RecentFailOpen >= 1 || tally.RecentStandDown >= 1:
		tally.Status = "yellow"
	}
	return tally
}

func renderGuardStopsSlack(t guardStopsSlackTally) string {
	icon := map[string]string{"green": ":large_green_circle:", "yellow": ":large_yellow_circle:", "red": ":red_circle:"}[t.Status]
	return fmt.Sprintf("%s *Guard Stop health* · %s · total %d · stand-down %d · fail-open %d · operator-directed %d · recent %d: stand-down %d / fail-open %d",
		icon, strings.ToUpper(t.Status), t.Total, t.StandDown, t.FailOpen, t.OperatorDirected, t.RecentTotal, t.RecentStandDown, t.RecentFailOpen)
}

func upsertGuardStopsSlack(outboxDir, channel, text string) error {
	ob, err := slackoutbox.Open(outboxDir)
	if err != nil {
		return err
	}
	statePath := filepath.Join(outboxDir, guardStopsSlackStateFile)
	state := loadGuardStopsSlackState(statePath)
	if state.Nonce == "" {
		state = guardStopsSlackState{Schema: guardStopsSlackSchema, Nonce: slackoutbox.NewNonce()}
		if err := saveGuardStopsSlackState(statePath, state); err != nil {
			return err
		}
		_, err = ob.Enqueue(slackoutbox.Row{Nonce: state.Nonce, Channel: channel, Text: text, Source: guardStopsSlackSource})
		return err
	}
	if state.TS == "" {
		snap, err := ob.Load()
		if err != nil {
			return err
		}
		state.TS = snap.PostedTS(state.Nonce)
		if state.TS == "" {
			// Root is still durable/pending. Do not enqueue duplicates while delivery catches up.
			return nil
		}
		if err := saveGuardStopsSlackState(statePath, state); err != nil {
			return err
		}
	}
	_, err = ob.Enqueue(slackoutbox.Row{Channel: channel, Text: text, UpdateTS: state.TS, Source: guardStopsSlackSource})
	return err
}

func loadGuardStopsSlackState(path string) guardStopsSlackState {
	b, err := os.ReadFile(path)
	if err != nil {
		return guardStopsSlackState{}
	}
	var state guardStopsSlackState
	if json.Unmarshal(b, &state) != nil || state.Schema != guardStopsSlackSchema {
		return guardStopsSlackState{}
	}
	return state
}

func saveGuardStopsSlackState(path string, state guardStopsSlackState) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, append(b, '\n'), 0o600); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

var _ = context.Background
var _ = errors.Is
