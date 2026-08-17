package devcmd

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/devcheckpoint"
)

var operatorLiveCommand = func(ctx context.Context, name string, args ...string) ([]byte, error) {
	return exec.CommandContext(ctx, name, args...).Output()
}

type operatorLiveRunner func(context.Context, string, ...string) ([]byte, error)

type operatorWorkerSnapshot struct {
	Lanes []operatorWorkerRow `json:"lanes"`
}

type operatorWorkerRow struct {
	Lane           string `json:"lane"`
	Chip           string `json:"chip"`
	LoopTS         string `json:"loop_ts"`
	Holder         string `json:"holder"`
	HeartbeatAgeMS *int64 `json:"heartbeat_age_ms"`
	LivenessReason string `json:"liveness_reason"`
}

type liveSignalRow struct {
	Attention string
	Outcome   string
	Move      string
	Next      string
	Lane      string
}

var gracePattern = regexp.MustCompile(`grace ([0-9]+) ms`)

func runOperatorLiveSignals(stdout, stderr io.Writer, command operatorLiveRunner, now time.Time, full bool) int {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	payload, err := command(ctx, "dos", "top", "--json")
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev sessiondiag --live-signals: dos top: %v\n", err)
		return 1
	}
	var top operatorWorkerSnapshot
	if err := json.Unmarshal(payload, &top); err != nil {
		fmt.Fprintf(stderr, "fak-dev sessiondiag --live-signals: decode dos top: %v\n", err)
		return 1
	}
	checkpoints, err := readLatestDevCheckpoints(filepathJoin(".fak", "dev-status.jsonl"))
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev sessiondiag --live-signals: checkpoints: %v\n", err)
		return 1
	}
	rows := make([]liveSignalRow, 0, len(top.Lanes))
	for _, lane := range top.Lanes {
		if strings.TrimSpace(lane.Holder) == "" {
			continue
		}
		rows = append(rows, projectLiveSignal(lane, checkpoints[lane.Holder], now))
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if exceptionOrder(rows[i].Attention) != exceptionOrder(rows[j].Attention) {
			return exceptionOrder(rows[i].Attention) < exceptionOrder(rows[j].Attention)
		}
		return rows[i].Lane < rows[j].Lane
	})
	fmt.Fprintln(stdout, "ATTENTION | OUTCOME + AGE | CURRENT MOVE + AGE | NEXT CHECK")
	if full {
		for _, row := range rows {
			writeWorkerProjection(stdout, row)
		}
	} else {
		writeExceptionFirstProjection(stdout, rows)
	}
	if len(rows) == 0 {
		fmt.Fprintln(stdout, "none | no live lanes | idle | check on next lease")
	}
	return 0
}

// filepathJoin is a seam for the fixed repository-local checkpoint log.
var filepathJoin = func(parts ...string) string { return strings.Join(parts, string(os.PathSeparator)) }

func writeExceptionFirstProjection(stdout io.Writer, rows []liveSignalRow) {
	unknown := make([]liveSignalRow, 0)
	healthy := make([]liveSignalRow, 0)
	for _, row := range rows {
		switch row.Attention {
		case "needs-human", "watch":
			writeWorkerProjection(stdout, row)
		case "unknown":
			unknown = append(unknown, row)
		default:
			healthy = append(healthy, row)
		}
	}
	if len(unknown) > 0 {
		fmt.Fprintf(stdout, "unknown x%d | no witnessed outcome | %d live workers | emit durable checkpoints; --full lists workers\n", len(unknown), len(unknown))
	}
	if len(healthy) > 0 {
		fmt.Fprintf(stdout, "none x%d | witnessed outcomes present | %d live workers | bounded next checks; --full lists workers\n", len(healthy), len(healthy))
	}
}

func writeWorkerProjection(stdout io.Writer, row liveSignalRow) {
	fmt.Fprintf(stdout, "%s | %s | %s | %s\n", row.Attention, row.Outcome, row.Move, row.Next)
}

func readLatestDevCheckpoints(path string) (map[string]devcheckpoint.Record, error) {
	out := map[string]devcheckpoint.Record{}
	f, err := os.Open(path)
	if os.IsNotExist(err) {
		return out, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		var record devcheckpoint.Record
		if err := json.Unmarshal(bytes.TrimSpace(scanner.Bytes()), &record); err != nil {
			return nil, err
		}
		if prior, ok := out[record.Actor]; !ok || record.Timestamp.After(prior.Timestamp) {
			out[record.Actor] = record
		}
	}
	return out, scanner.Err()
}

func projectLiveSignal(lane operatorWorkerRow, checkpoint devcheckpoint.Record, now time.Time) liveSignalRow {
	leaseAge := ageSince(parseLiveStamp(lane.LoopTS), now)
	moveAge := pointerDurationText(lane.HeartbeatAgeMS)
	row := liveSignalRow{
		Attention: "unknown",
		Outcome:   "no checkpoint for " + leaseAge,
		Move:      lane.Lane + " lease heartbeat " + moveAge + " ago",
		Next:      "emit a durable checkpoint",
		Lane:      lane.Lane,
	}
	chip := strings.ToUpper(lane.Chip)
	if strings.Contains(chip, "STALLED") {
		row.Attention = "watch"
		row.Next = "inspect stalled lease now"
	}
	if checkpoint.Actor == "" {
		if strings.Contains(chip, "STALLED") {
			row.Attention = "watch"
			row.Outcome = "no witnessed outcome since lease start " + leaseAge + " ago"
			row.Move = lane.Lane + " lease heartbeat " + moveAge + " ago"
		}
		return row
	}
	checkpointAge := ageSince(checkpoint.Timestamp, now)
	if len(checkpoint.Evidence) > 0 {
		row.Outcome = "checkpoint evidence " + checkpoint.Evidence[0] + " " + checkpointAge + " ago"
	} else {
		row.Outcome = "no witnessed outcome; checkpoint " + checkpointAge + " ago"
	}
	move := ""
	if checkpoint.Stage != nil {
		move = strings.TrimSpace(checkpoint.Stage.Name)
	}
	if move == "" {
		move = strings.TrimSpace(checkpoint.Summary)
	}
	if move == "" {
		move = lane.Lane
	}
	row.Move = lane.Lane + ": " + move + " " + checkpointAge + " ago"
	if strings.TrimSpace(checkpoint.Next) != "" {
		row.Next = checkpoint.Next
	} else if remaining := graceRemaining(lane, now); remaining != "" {
		row.Next = "check at grace in " + remaining
	} else {
		row.Next = "check on next checkpoint"
	}
	switch checkpoint.State {
	case devcheckpoint.StateBlocked:
		if checkpointNeedsHuman(checkpoint) {
			row.Attention = "needs-human"
		} else {
			row.Attention = "watch"
		}
	default:
		// A checkpoint identifies the move, but only evidence proves an outcome.
		// Keep evidence-free intent out of the healthy fold.
		lastHeartbeat := now.Add(-time.Duration(valueOrZero(lane.HeartbeatAgeMS)) * time.Millisecond)
		if strings.Contains(chip, "STALLED") && !checkpoint.Timestamp.After(lastHeartbeat) {
			row.Attention = "watch"
		} else if len(checkpoint.Evidence) == 0 {
			row.Attention = "unknown"
		} else {
			row.Attention = "none"
		}
	}
	return row
}

func checkpointNeedsHuman(checkpoint devcheckpoint.Record) bool {
	for _, blocker := range checkpoint.Blockers {
		if strings.HasPrefix(strings.ToLower(strings.TrimSpace(blocker)), "operator:") {
			return true
		}
	}
	return false
}

func exceptionOrder(attention string) int {
	switch attention {
	case "needs-human":
		return 0
	case "watch":
		return 1
	case "unknown":
		return 2
	default:
		return 3
	}
}

func graceRemaining(lane operatorWorkerRow, _ time.Time) string {
	match := gracePattern.FindStringSubmatch(lane.LivenessReason)
	if len(match) != 2 || lane.HeartbeatAgeMS == nil {
		return ""
	}
	graceMS, err := strconv.ParseInt(match[1], 10, 64)
	if err != nil || graceMS <= *lane.HeartbeatAgeMS {
		return ""
	}
	return durationText(time.Duration(graceMS-*lane.HeartbeatAgeMS) * time.Millisecond)
}

func parseLiveStamp(raw string) time.Time {
	t, _ := time.Parse(time.RFC3339Nano, raw)
	return t
}

func ageSince(then, now time.Time) string {
	if then.IsZero() || now.Before(then) {
		return "unknown age"
	}
	return durationText(now.Sub(then))
}

func durationText(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	if d < time.Minute {
		return fmt.Sprintf("%ds", int64(d/time.Second))
	}
	if d < time.Hour {
		return fmt.Sprintf("%dm", int64(d/time.Minute))
	}
	return fmt.Sprintf("%dh", int64(d/time.Hour))
}

func pointerDurationText(value *int64) string {
	if value == nil {
		return "unknown age"
	}
	return durationText(time.Duration(*value) * time.Millisecond)
}

func valueOrZero(value *int64) int64 {
	if value == nil {
		return 0
	}
	return *value
}
