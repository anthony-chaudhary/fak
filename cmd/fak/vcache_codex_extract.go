package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/vcacheextract"
	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
	"github.com/anthony-chaudhary/fak/internal/vcachescore"
	"github.com/anthony-chaudhary/fak/internal/vcachesnapshot"
)

func runVCacheCodexSessionExtract(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("vcache codex-session-extract", flag.ContinueOnError)
	fs.SetOutput(stderr)
	session := fs.String("session", "", "raw Codex session or codex exec --json JSONL path")
	threadID := fs.String("thread-id", "", "Codex thread id; defaults to CODEX_THREAD_ID")
	codexHome := fs.String("codex-home", "", "Codex home; defaults to CODEX_HOME or ~/.codex")
	out := fs.String("out", "", "sanitized JSONL output path, or '-' for stdout")
	snapshotOut := fs.String("snapshot-out", "", "optional vCache observed-window snapshot path; use 'default' for the path read by `fak vcache score`")
	scoreOut := fs.String("score-out", "", "optional vCache score JSON artifact path computed from the sanitized Codex token counters")
	family := fs.String("family", "codex", "prefix-family label for snapshot rows")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	if *out == "" {
		fmt.Fprintln(stderr, "fak vcache codex-session-extract: --out is required")
		return 2
	}
	targetSession := *session
	if targetSession == "" {
		env := map[string]string{"CODEX_HOME": os.Getenv("CODEX_HOME")}
		tid := *threadID
		if tid == "" {
			tid = os.Getenv("CODEX_THREAD_ID")
		}
		home := *codexHome
		if home == "" {
			home = vcacheextract.CodexHome(env)
		}
		found, err := vcacheextract.FindSession(home, tid)
		if err != nil {
			fmt.Fprintf(stderr, "vcache_codex_session_extract: %v\n", err)
			return 2
		}
		targetSession = found
	}
	if _, err := os.Stat(targetSession); err != nil {
		fmt.Fprintf(stderr, "vcache_codex_session_extract: session not found: %s\n", targetSession)
		return 2
	}
	rows, err := vcacheextract.ExtractRows(targetSession)
	if err != nil {
		fmt.Fprintf(stderr, "vcache_codex_session_extract: %v\n", err)
		return 2
	}
	if len(rows) == 0 {
		fmt.Fprintf(stderr, "vcache_codex_session_extract: no token usage rows found in %s\n", targetSession)
		return 1
	}
	targetOut := *out
	if targetOut != "-" {
		if abs, err := filepath.Abs(targetOut); err == nil {
			targetOut = abs
		}
	}
	if err := vcacheextract.WriteRows(targetOut, rows, stdout); err != nil {
		fmt.Fprintf(stderr, "vcache_codex_session_extract: %v\n", err)
		return 2
	}
	if strings.TrimSpace(*snapshotOut) != "" || strings.TrimSpace(*scoreOut) != "" {
		turns := vcacheextract.TurnsFromRows(rows, *family)
		if len(turns) == 0 {
			fmt.Fprintf(stderr, "vcache_codex_session_extract: no snapshot turns could be derived from %s\n", targetSession)
			return 1
		}
		if strings.TrimSpace(*snapshotOut) != "" {
			targetSnapshot := strings.TrimSpace(*snapshotOut)
			if strings.EqualFold(targetSnapshot, "default") {
				targetSnapshot = vcachesnapshot.DefaultPath()
			}
			if targetSnapshot != "-" {
				if abs, err := filepath.Abs(targetSnapshot); err == nil {
					targetSnapshot = abs
				}
			}
			if targetSnapshot == "-" {
				fmt.Fprintln(stderr, "vcache_codex_session_extract: --snapshot-out does not support stdout; use a file path or 'default'")
				return 2
			}
			if err := vcachesnapshot.Write(targetSnapshot, turns); err != nil {
				fmt.Fprintf(stderr, "vcache_codex_session_extract: snapshot: %v\n", err)
				return 2
			}
			fmt.Fprintf(stderr, "wrote %d vcache snapshot turns to %s\n", len(turns), targetSnapshot)
		}
		if strings.TrimSpace(*scoreOut) != "" {
			targetScore := strings.TrimSpace(*scoreOut)
			if targetScore == "-" {
				fmt.Fprintln(stderr, "vcache_codex_session_extract: --score-out does not support stdout; use a file path")
				return 2
			}
			if abs, err := filepath.Abs(targetScore); err == nil {
				targetScore = abs
			}
			score := codexVCacheScore(turns)
			if err := writeJSONFile(targetScore, score); err != nil {
				fmt.Fprintf(stderr, "vcache_codex_session_extract: score: %v\n", err)
				return 2
			}
			fmt.Fprintf(stderr, "wrote vcache score %s (%.2fx) to %s\n", score.Status, score.ActiveMultiplier, targetScore)
		}
	}
	label := targetOut
	if *out == "-" {
		label = "stdout"
	}
	fmt.Fprintf(stderr, "wrote %d sanitized token rows from %s to %s\n", len(rows), targetSession, label)
	return 0
}

func codexVCacheScore(turns []vcacheobserve.Turn) vcachescore.Report {
	in := vcachescore.DefaultInput()
	in.TelemetryRows = vcacheobserve.Rows(turns)
	in.Ranked = vcacheobserve.RankedWorkload(turns)
	in.AnchorSource = vcachescore.AnchorSourceMeasured
	in.TurnsObserved = len(turns)
	return vcachescore.Score(in)
}
