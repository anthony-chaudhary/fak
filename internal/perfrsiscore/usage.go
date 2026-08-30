package perfrsiscore

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
)

const (
	UsageSchema       = "fak-performance-rsi-usage/1"
	UsageFoldSchema   = "fak-performance-rsi-usage-fold/1"
	UsageLedgerEnv    = "FAK_PERFORMANCE_RSI_USAGE_LEDGER"
	DefaultUsagePath  = ".fak/performance-rsi/usage.jsonl"
	usageLedgerMaxLen = jsonlledger.DefaultActiveBytes
)

// UsageRow is the public-safe adoption record for one performance-RSI loop
// invocation. It deliberately excludes input paths, diagnostics, hostnames,
// prompts, and provider data.
type UsageRow struct {
	Schema             string        `json:"schema"`
	At                 time.Time     `json:"at"`
	Status             string        `json:"status"`
	Reason             string        `json:"reason"`
	Snapshot           string        `json:"snapshot,omitempty"`
	InvocationOutcomes OutcomeCounts `json:"invocation_outcomes"`
}

// UsageWeek is one ISO-week adoption bucket.
type UsageWeek struct {
	Week               string        `json:"week"`
	Invocations        int           `json:"invocations"`
	Scored             int           `json:"scored"`
	Unavailable        int           `json:"unavailable"`
	InvocationOutcomes OutcomeCounts `json:"invocation_outcomes"`
}

// UsageFold surfaces performance-RSI adoption or neglect by ISO week.
type UsageFold struct {
	Schema string      `json:"schema"`
	Weeks  []UsageWeek `json:"weeks"`
}

// UsageLedgerPath returns the configured ledger path, falling back to the
// repository-local gitignored runtime ledger.
func UsageLedgerPath() string {
	if path := strings.TrimSpace(os.Getenv(UsageLedgerEnv)); path != "" {
		return path
	}
	return DefaultUsagePath
}

// AppendUsage writes one complete, bounded JSONL row for a loop invocation.
func AppendUsage(path string, at time.Time, receipt LoopTurnReceipt) error {
	path = strings.TrimSpace(path)
	if path == "" {
		return errors.New("performance RSI usage ledger path is required")
	}
	if at.IsZero() {
		at = time.Now().UTC()
	} else {
		at = at.UTC()
	}
	row := UsageRow{
		Schema:             UsageSchema,
		At:                 at,
		Status:             receipt.Status,
		Reason:             receipt.Reason,
		Snapshot:           receipt.Snapshot,
		InvocationOutcomes: receipt.InvocationOutcomes,
	}
	line, err := json.Marshal(row)
	if err != nil {
		return fmt.Errorf("encode performance RSI usage: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create performance RSI usage ledger directory: %w", err)
	}
	if err := jsonlledger.AppendBounded(path, line, usageLedgerMaxLen); err != nil {
		return fmt.Errorf("append performance RSI usage: %w", err)
	}
	return nil
}

// RecordLoopTurnUsage records one invocation using the configured ledger. It
// returns an error so callers can surface telemetry failure without replacing
// the loop's own outcome.
func RecordLoopTurnUsage(receipt LoopTurnReceipt) error {
	return AppendUsage(UsageLedgerPath(), time.Now().UTC(), receipt)
}

// FoldUsage reads valid JSONL rows and folds them by ISO week. A malformed or
// truncated tail is ignored so an interrupted final append cannot erase the
// durable rows before it.
func FoldUsage(path string) (UsageFold, error) {
	f, err := os.Open(path)
	if err != nil {
		return UsageFold{}, err
	}
	defer f.Close()
	return FoldUsageReader(f)
}

func FoldUsageReader(r io.Reader) (UsageFold, error) {
	weeks := make(map[string]*UsageWeek)
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		var row UsageRow
		dec := json.NewDecoder(bytes.NewReader(line))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&row); err != nil || row.Schema != UsageSchema || row.At.IsZero() {
			continue
		}
		if err := dec.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
			continue
		}
		year, week := row.At.ISOWeek()
		key := fmt.Sprintf("%04d-W%02d", year, week)
		bucket := weeks[key]
		if bucket == nil {
			bucket = &UsageWeek{Week: key}
			weeks[key] = bucket
		}
		bucket.Invocations++
		switch row.Status {
		case LoopTurnScored:
			bucket.Scored++
		case LoopTurnUnavailable:
			bucket.Unavailable++
		}
		bucket.InvocationOutcomes.Success += row.InvocationOutcomes.Success
		bucket.InvocationOutcomes.Refusal += row.InvocationOutcomes.Refusal
		bucket.InvocationOutcomes.Error += row.InvocationOutcomes.Error
	}
	if err := scanner.Err(); err != nil {
		return UsageFold{}, err
	}
	keys := make([]string, 0, len(weeks))
	for key := range weeks {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	fold := UsageFold{Schema: UsageFoldSchema, Weeks: make([]UsageWeek, 0, len(keys))}
	for _, key := range keys {
		fold.Weeks = append(fold.Weeks, *weeks[key])
	}
	return fold, nil
}

func FormatUsageFold(fold UsageFold) string {
	var b strings.Builder
	for _, week := range fold.Weeks {
		fmt.Fprintf(&b, "%s invocations=%d scored=%d unavailable=%d success=%d refusal=%d error=%d\n",
			week.Week, week.Invocations, week.Scored, week.Unavailable,
			week.InvocationOutcomes.Success, week.InvocationOutcomes.Refusal, week.InvocationOutcomes.Error)
	}
	return strings.TrimSuffix(b.String(), "\n")
}
