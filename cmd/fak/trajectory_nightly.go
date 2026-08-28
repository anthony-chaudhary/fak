package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

var appendPreparedTrajectoryNightlyReceipt = trajectory.AppendPreparedAttributionReceiptWithRollback

func runTrajectoryNightly(stdout, stderr io.Writer, args []string) int {
	flags := flag.NewFlagSet("fak trajectory nightly", flag.ContinueOnError)
	flags.SetOutput(stderr)
	budgetPath := flags.String("budget", "", "versioned fak-trajectory-attribution-budget/1 JSON file")
	historyPath := flags.String("history", trajectory.DefaultAttributionHistoryPath(), "append-only scrubbed receipt history")
	receiptPath := flags.String("receipt", "", "write the latest scrubbed receipt atomically (stdout when omitted)")
	corpus := flags.String("corpus", "local", "privacy-safe corpus label, for example local or fleet")
	claudeRoot := flags.String("claude-root", "", "override Claude projects root")
	codexRoot := flags.String("codex-root", "", "override Codex sessions root")
	atText := flags.String("at", "", "evaluate at this RFC3339 timestamp (tests/replay; default now)")
	if err := flags.Parse(args); err != nil {
		return 2
	}
	if flags.NArg() != 0 {
		fmt.Fprintf(stderr, "fak trajectory nightly: unexpected arguments: %s\n", strings.Join(flags.Args(), " "))
		return 2
	}
	if strings.TrimSpace(*budgetPath) == "" {
		fmt.Fprintln(stderr, "fak trajectory nightly: --budget is required")
		return 2
	}
	if strings.TrimSpace(*historyPath) == "" {
		fmt.Fprintln(stderr, "fak trajectory nightly: --history must not be empty")
		return 2
	}
	if strings.TrimSpace(*receiptPath) != "" {
		same, err := sameTrajectoryNightlyOutputPath(*historyPath, *receiptPath)
		if err != nil {
			fmt.Fprintln(stderr, "fak trajectory nightly: compare --history and --receipt:", err)
			return 2
		}
		if same {
			fmt.Fprintln(stderr, "fak trajectory nightly: --history and --receipt must name different files")
			return 2
		}
	}
	budget, err := trajectory.ReadAttributionBudget(*budgetPath)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	var at time.Time
	if strings.TrimSpace(*atText) != "" {
		at, err = time.Parse(time.RFC3339Nano, *atText)
		if err != nil {
			fmt.Fprintln(stderr, "fak trajectory nightly: --at must be RFC3339:", err)
			return 2
		}
	}
	sources := trajectory.DefaultAuditSources()
	for i := range sources {
		switch sources[i].Name {
		case trajectory.AuditSourceClaude:
			if strings.TrimSpace(*claudeRoot) != "" {
				sources[i].Root = *claudeRoot
			}
		case trajectory.AuditSourceCodex:
			if strings.TrimSpace(*codexRoot) != "" {
				sources[i].Root = *codexRoot
			}
		}
	}
	receipt := trajectory.RunAttributionNightly(trajectory.AttributionNightlyOptions{
		Sources: sources, Budget: budget, Now: at, Corpus: strings.TrimSpace(*corpus),
	})
	if err := trajectory.PrepareAttributionReceiptTrend(*historyPath, &receipt); err != nil {
		return failTrajectoryNightlyPublication(stdout, stderr, *receiptPath, &receipt, "history_read_failed", err)
	}
	var staged *stagedTrajectoryNightlyReceipt
	if strings.TrimSpace(*receiptPath) != "" {
		staged, err = stageTrajectoryNightlyReceipt(*receiptPath, receipt)
		if err != nil {
			fmt.Fprintln(stderr, "trajectory nightly: receipt_publish_failed:", err)
			return 1
		}
		defer staged.discard()
	}
	rollbackHistory, err := appendPreparedTrajectoryNightlyReceipt(*historyPath, &receipt)
	if err != nil {
		return failTrajectoryNightlyPublication(stdout, stderr, *receiptPath, &receipt, "history_append_failed", err)
	}
	if staged != nil {
		if err := staged.commit(); err != nil {
			if rollbackErr := rollbackHistory(); rollbackErr != nil {
				fmt.Fprintln(stderr, "trajectory nightly: history rollback failed:", rollbackErr)
			}
			fmt.Fprintln(stderr, "trajectory nightly: receipt_publish_failed:", err)
			return 1
		}
	} else {
		if err := writeTrajectoryNightlyReceipt(stdout, "", receipt); err != nil {
			if rollbackErr := rollbackHistory(); rollbackErr != nil {
				fmt.Fprintln(stderr, "trajectory nightly: history rollback failed:", rollbackErr)
			}
			fmt.Fprintln(stderr, "trajectory nightly: receipt_publish_failed:", err)
			return 1
		}
	}
	fmt.Fprintf(stderr, "trajectory nightly: corpus=%s status=%s records=%d breaches=%d\n", receipt.Corpus, receipt.Status, attributionReceiptRecords(receipt), len(receipt.Breaches))
	switch receipt.Status {
	case trajectory.AttributionStatusPass, trajectory.AttributionStatusNoData:
		return 0
	case trajectory.AttributionStatusBudgetFailed:
		return 3
	default:
		return 1
	}
}

type stagedTrajectoryNightlyReceipt struct {
	tempPath  string
	target    string
	committed bool
}

func stageTrajectoryNightlyReceipt(path string, receipt trajectory.AttributionReceipt) (*stagedTrajectoryNightlyReceipt, error) {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("trajectory nightly: create receipt directory: %w", err)
	}
	temp, err := os.CreateTemp(dir, ".trajectory-attribution-*.tmp")
	if err != nil {
		return nil, fmt.Errorf("trajectory nightly: create receipt temp file: %w", err)
	}
	staged := &stagedTrajectoryNightlyReceipt{tempPath: temp.Name(), target: path}
	if err := trajectory.WriteAttributionReceipt(temp, receipt); err != nil {
		temp.Close()
		staged.discard()
		return nil, err
	}
	if err := temp.Sync(); err != nil {
		temp.Close()
		staged.discard()
		return nil, fmt.Errorf("trajectory nightly: sync receipt temp file: %w", err)
	}
	if err := temp.Close(); err != nil {
		staged.discard()
		return nil, fmt.Errorf("trajectory nightly: close receipt temp file: %w", err)
	}
	return staged, nil
}

func (s *stagedTrajectoryNightlyReceipt) commit() error {
	if err := os.Rename(s.tempPath, s.target); err != nil {
		return fmt.Errorf("trajectory nightly: publish receipt: %w", err)
	}
	s.committed = true
	return nil
}

func (s *stagedTrajectoryNightlyReceipt) discard() {
	if s != nil && !s.committed {
		_ = os.Remove(s.tempPath)
	}
}

func failTrajectoryNightlyPublication(stdout, stderr io.Writer, receiptPath string, receipt *trajectory.AttributionReceipt, code string, cause error) int {
	receipt.Status = trajectory.AttributionStatusPublicationFailed
	receipt.PublicationError = code
	if err := writeTrajectoryNightlyReceipt(stdout, receiptPath, *receipt); err != nil {
		fmt.Fprintln(stderr, "trajectory nightly: publish failure receipt:", err)
	}
	fmt.Fprintln(stderr, cause)
	return 1
}

func sameTrajectoryNightlyOutputPath(left, right string) (bool, error) {
	leftAbs, err := filepath.Abs(left)
	if err != nil {
		return false, err
	}
	rightAbs, err := filepath.Abs(right)
	if err != nil {
		return false, err
	}
	leftClean, rightClean := filepath.Clean(leftAbs), filepath.Clean(rightAbs)
	if leftClean == rightClean || runtime.GOOS == "windows" && strings.EqualFold(leftClean, rightClean) {
		return true, nil
	}
	leftInfo, leftErr := os.Stat(leftAbs)
	rightInfo, rightErr := os.Stat(rightAbs)
	if leftErr == nil && rightErr == nil && os.SameFile(leftInfo, rightInfo) {
		return true, nil
	}
	if leftErr != nil && !os.IsNotExist(leftErr) {
		return false, leftErr
	}
	if rightErr != nil && !os.IsNotExist(rightErr) {
		return false, rightErr
	}
	return false, nil
}

func writeTrajectoryNightlyReceipt(stdout io.Writer, path string, receipt trajectory.AttributionReceipt) error {
	if strings.TrimSpace(path) == "" {
		return trajectory.WriteAttributionReceipt(stdout, receipt)
	}
	staged, err := stageTrajectoryNightlyReceipt(path, receipt)
	if err != nil {
		return err
	}
	defer staged.discard()
	return staged.commit()
}

func attributionReceiptRecords(receipt trajectory.AttributionReceipt) int {
	var records int
	for _, coverage := range receipt.Coverage {
		records += coverage.Records
	}
	return records
}
