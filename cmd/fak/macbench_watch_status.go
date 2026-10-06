package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/macbench"
)

const macBenchWatchStatusSchema = "fak.macbench.watch.status.v1"

type macBenchWatchEvent struct {
	Schema      string `json:"schema"`
	GeneratedAt string `json:"generated_at"`
	Phase       string `json:"phase"`
	Error       string `json:"error"`
}

type macBenchWatchStatus struct {
	Schema          string              `json:"schema"`
	GeneratedAt     string              `json:"generated_at"`
	LogPath         string              `json:"log_path,omitempty"`
	ResultPath      string              `json:"result_path,omitempty"`
	LogPresent      bool                `json:"log_present"`
	ResultPresent   bool                `json:"result_present"`
	Reports         int                 `json:"reports"`
	Events          int                 `json:"events"`
	State           string              `json:"state"`
	LastGeneratedAt string              `json:"last_generated_at,omitempty"`
	LastError       string              `json:"last_error,omitempty"`
	NextAction      string              `json:"next_action"`
	LatestReport    *macbench.Report    `json:"latest_report,omitempty"`
	LatestEvent     *macBenchWatchEvent `json:"latest_event,omitempty"`
	Result          *macbench.Report    `json:"result,omitempty"`
}

func runMacBenchWatchStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("macbench watch-status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	logPath := fs.String("log", "", "macbench watch append-only log path")
	resultPath := fs.String("result", "", "macbench watch full result JSON path")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if strings.TrimSpace(*logPath) == "" && strings.TrimSpace(*resultPath) == "" {
		fmt.Fprintln(stderr, "fak macbench watch-status: pass --log and/or --result")
		return 2
	}
	status, err := loadMacBenchWatchStatus(*logPath, *resultPath, time.Now().UTC())
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench watch-status: %v\n", err)
		return 1
	}
	if *asJSON {
		_ = writeIndentedJSONNoEscape(stdout, status)
		return 0
	}
	renderMacBenchWatchStatus(stdout, status)
	return 0
}

func loadMacBenchWatchStatus(logPath, resultPath string, now time.Time) (macBenchWatchStatus, error) {
	s := macBenchWatchStatus{
		Schema:      macBenchWatchStatusSchema,
		GeneratedAt: now.UTC().Format(time.RFC3339),
		LogPath:     filepath.ToSlash(strings.TrimSpace(logPath)),
		ResultPath:  filepath.ToSlash(strings.TrimSpace(resultPath)),
		State:       "no_reports",
		NextAction:  "wait for the first macbench watch poll",
	}
	if strings.TrimSpace(logPath) != "" {
		if err := readMacBenchWatchLog(logPath, &s); err != nil {
			return s, err
		}
	}
	if strings.TrimSpace(resultPath) != "" {
		result, ok, err := readMacBenchResult(resultPath)
		if err != nil {
			return s, err
		}
		s.ResultPresent = ok
		if ok {
			s.Result = &result
		}
	}
	classifyMacBenchWatchStatus(&s)
	return s, nil
}

func readMacBenchWatchLog(path string, s *macBenchWatchStatus) error {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("read --log: %w", err)
	}
	defer f.Close()
	s.LogPresent = true
	dec := json.NewDecoder(f)
	for {
		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("parse --log: %w", err)
		}
		var hdr struct {
			Schema      string `json:"schema"`
			GeneratedAt string `json:"generated_at"`
		}
		if err := json.Unmarshal(raw, &hdr); err != nil {
			return fmt.Errorf("parse --log header: %w", err)
		}
		switch hdr.Schema {
		case macbench.Schema:
			var rep macbench.Report
			if err := json.Unmarshal(raw, &rep); err != nil {
				return fmt.Errorf("parse macbench report: %w", err)
			}
			s.Reports++
			repCopy := rep
			s.LatestReport = &repCopy
			s.LastGeneratedAt = nonEmptyString(rep.GeneratedAt, s.LastGeneratedAt)
		case "fak.macbench.watch.event.v1":
			var ev macBenchWatchEvent
			if err := json.Unmarshal(raw, &ev); err != nil {
				return fmt.Errorf("parse watch event: %w", err)
			}
			s.Events++
			evCopy := ev
			s.LatestEvent = &evCopy
			s.LastGeneratedAt = nonEmptyString(ev.GeneratedAt, s.LastGeneratedAt)
		default:
			return fmt.Errorf("parse --log: unknown schema %q", hdr.Schema)
		}
	}
	return nil
}

func readMacBenchResult(path string) (macbench.Report, bool, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return macbench.Report{}, false, nil
		}
		return macbench.Report{}, false, fmt.Errorf("read --result: %w", err)
	}
	var rep macbench.Report
	if err := json.Unmarshal(b, &rep); err != nil {
		return macbench.Report{}, true, fmt.Errorf("parse --result: %w", err)
	}
	return rep, true, nil
}

func classifyMacBenchWatchStatus(s *macBenchWatchStatus) {
	if s.Result != nil {
		s.LastGeneratedAt = nonEmptyString(s.Result.GeneratedAt, s.LastGeneratedAt)
		if s.Result.HasErrors() {
			s.State = "completed_with_errors"
			s.LastError = firstMacBenchError(*s.Result)
			s.NextAction = "inspect the full macbench result and fix the failing suite"
			return
		}
		s.State = "completed"
		s.NextAction = "record or publish the full macbench result"
		return
	}
	if s.LatestReport != nil {
		rep := *s.LatestReport
		s.LastGeneratedAt = nonEmptyString(rep.GeneratedAt, s.LastGeneratedAt)
		if rep.Suite == macbench.SuiteAll {
			if rep.HasErrors() {
				s.State = "completed_with_errors"
				s.LastError = firstMacBenchError(rep)
				s.NextAction = "inspect the full macbench report in the watch log"
				return
			}
			s.State = "completed"
			s.NextAction = "persist the full macbench report with --result or fold it into nightrun"
			return
		}
		if rep.Health.OK {
			s.State = "healthy_waiting_for_full_run"
			s.NextAction = "wait for the full macbench suite to finish"
			return
		}
		s.State = "waiting_for_gateway"
		s.LastError = firstMacBenchError(rep)
		s.NextAction = "keep the watcher running; gateway health is still false"
		return
	}
	if s.LatestEvent != nil {
		s.State = "watch_error"
		s.LastError = s.LatestEvent.Error
		s.NextAction = "inspect the watch event and restart after fixing the phase"
		return
	}
	if !s.LogPresent && s.LogPath != "" {
		s.State = "missing_log"
		s.NextAction = "confirm the watch process started and wrote its first poll"
	}
}

func firstMacBenchError(rep macbench.Report) string {
	if rep.Health.Error != "" {
		return rep.Health.Error
	}
	if len(rep.Errors) > 0 {
		return rep.Errors[0]
	}
	for _, row := range rep.Rows {
		if row.Error != "" {
			return row.Error
		}
	}
	return ""
}

func renderMacBenchWatchStatus(w io.Writer, s macBenchWatchStatus) {
	fmt.Fprintf(w, "macbench watch-status: %s\n", s.State)
	if s.LastGeneratedAt != "" {
		fmt.Fprintf(w, "last: %s\n", s.LastGeneratedAt)
	}
	fmt.Fprintf(w, "log: present=%v reports=%d events=%d\n", s.LogPresent, s.Reports, s.Events)
	fmt.Fprintf(w, "result: present=%v\n", s.ResultPresent)
	if s.LatestReport != nil {
		fmt.Fprintf(w, "latest: suite=%s gateway=%s model=%s health=%v\n",
			s.LatestReport.Suite, s.LatestReport.Gateway, s.LatestReport.Model, s.LatestReport.Health.OK)
	}
	if s.LastError != "" {
		fmt.Fprintf(w, "error: %s\n", s.LastError)
	}
	fmt.Fprintf(w, "next: %s\n", s.NextAction)
}
