package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
	"github.com/anthony-chaudhary/fak/internal/sessionrecovery"
)

type threadFlags map[string]bool

func (v threadFlags) String() string { return "" }
func (v threadFlags) Set(s string) error {
	s = strings.TrimSpace(s)
	if s == "" {
		return fmt.Errorf("empty thread")
	}
	v[s] = true
	return nil
}

var recoveryInventory = func(since time.Duration) (sessionrecovery.InventoryReport, error) {
	var out bytes.Buffer
	cmd := exec.Command("fak-dev", "sessiondiag", "--inventory", "--json", "--since", since.String())
	cmd.Stdout = &out
	cmd.Stderr = io.Discard
	if err := cmd.Run(); err != nil {
		return sessionrecovery.InventoryReport{}, fmt.Errorf("sessiondiag inventory: %w", err)
	}
	var report sessionrecovery.InventoryReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		return report, fmt.Errorf("decode sessiondiag inventory: %w", err)
	}
	return report, nil
}
var recoveryJournalCrashes = func(path string, now time.Time) ([]sessionjournal.Classified, error) {
	boot, _ := sessionjournal.BootTime(now)
	sessions := sessionjournal.FoldEvents(sessionjournal.LoadFile(path))
	return sessionjournal.Classify(sessions, sessionjournal.ClassifyConfig{Now: now, BootTime: boot}), nil
}
var recoveryLaunch sessionrecovery.Launcher = sessionrecovery.VisibleLauncher{}
var recoveryNow = time.Now
var recoverySleep = time.Sleep

func cmdSessionRecover(args []string) { os.Exit(runSessionRecover(os.Stdout, os.Stderr, args)) }

func runSessionRecover(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("session recover", flag.ContinueOnError)
	fs.SetOutput(stderr)
	since := fs.Duration("since", 24*time.Hour, "candidate evidence window")
	limit := fs.Int("limit", 1, "maximum launches per wave")
	apply := fs.Bool("apply", false, "write receipts and launch visible resumes")
	cwd := fs.String("cwd", "", "explicit override for cwd_unknown candidates")
	prompt := fs.String("prompt", "", "optional resume prompt passed as one exact argv element")
	journal := fs.Bool("journal", true, "include crash candidates from the session journal")
	journalPath := fs.String("journal-path", "", "session journal path (default machine journal)")
	settle := fs.Duration("settle", 5*time.Second, "delay before post-launch witness")
	receipts := fs.String("receipts", "", "receipt directory")
	threads := threadFlags{}
	fs.Var(threads, "thread", "thread ID to recover (repeatable)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || *since <= 0 || *limit <= 0 || *settle < 0 {
		fmt.Fprintln(stderr, "usage: fak session recover [--thread ID] [--since 24h] [--limit 1] [--cwd DIR] [--prompt TEXT] [--apply] [--settle 5s]")
		return 2
	}
	if *receipts == "" {
		base, err := os.UserConfigDir()
		if err != nil {
			fmt.Fprintln(stderr, "fak session recover: config directory unavailable")
			return 2
		}
		*receipts = filepath.Join(base, "fak", "session-recovery")
	}
	before, err := recoveryInventory(*since)
	if err != nil {
		fmt.Fprintln(stderr, "fak session recover:", err)
		return 2
	}
	managerPath, exeErr := os.Executable()
	if exeErr != nil {
		fmt.Fprintln(stderr, "fak session recover: resolve current executable:", exeErr)
		return 2
	}
	requests := sessionrecovery.Select(before, sessionrecovery.Options{ManagerBin: managerPath, Threads: threads, Limit: *limit, CWDOverride: *cwd, Prompt: *prompt, ReceiptDir: *receipts})
	if *journal {
		classified, journalErr := recoveryJournalCrashes(*journalPath, recoveryNow())
		if journalErr != nil {
			fmt.Fprintln(stderr, "fak session recover: session journal:", journalErr)
			return 2
		}
		requests = sessionrecovery.MergeJournalCrashes(requests, classified, sessionrecovery.Options{ManagerBin: managerPath, Threads: threads, Limit: *limit, CWDOverride: *cwd, Prompt: *prompt, ReceiptDir: *receipts})
	}
	if *apply {
		for i := range requests {
			if requests[i].Status != "candidate" {
				continue
			}
			wrote, err := sessionrecovery.WriteReceipt(requests[i], recoveryNow())
			if err != nil {
				requests[i].Status = "receipt_failed"
				requests[i].Reason = err.Error()
				continue
			}
			if !wrote {
				requests[i].Status = "already_receipted"
				continue
			}
			if err := recoveryLaunch.Launch(requests[i]); err != nil {
				requests[i].Status = "launch_failed"
				requests[i].Reason = err.Error()
				continue
			}
			requests[i].Status = "launched_unproven"
			recoverySleep(*settle)
			after, err := recoveryInventory(*since)
			if err == nil {
				requests[i].Status = sessionrecovery.Witness(before, after, requests[i].ThreadID)
				before = after
			}
		}
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(requests); err != nil {
		return 1
	}
	return 0
}
