package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

var resumeSessionID = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{5,127}$`)

type resumeRearmRow struct {
	Ts             string `json:"ts"`
	Phase          string `json:"phase"`
	Session        string `json:"session"`
	Reason         string `json:"reason,omitempty"`
	ManualOverride bool   `json:"manual_override"`
}

func runResumeRearm(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak resume rearm", flag.ContinueOnError)
	fs.SetOutput(stderr)
	regDir := fs.String("registry-dir", "", "fleet registry directory")
	reason := fs.String("reason", "", "operator reason")
	list := fs.Bool("list", false, "list currently re-armed session ids")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	path := filepath.Join(resolveSweepRegDir(*regDir), "resume_ledger.jsonl")
	if *list {
		return listResumeRearms(stdout, stderr, path)
	}
	if fs.NArg() == 0 {
		fmt.Fprintln(stderr, "fak resume rearm: provide at least one session id or --list")
		return 2
	}
	for _, sid := range fs.Args() {
		if !resumeSessionID.MatchString(sid) {
			fmt.Fprintf(stderr, "fak resume rearm: invalid session id %q\n", sid)
			return 2
		}
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, sid := range fs.Args() {
		row := resumeRearmRow{time.Now().UTC().Format("2006-01-02T15:04:05Z"), "rearm", sid, strings.TrimSpace(*reason), true}
		if err := enc.Encode(row); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		fmt.Fprintf(stdout, "rearmed %s\n", sid)
	}
	return 0
}
func listResumeRearms(stdout, stderr io.Writer, path string) int {
	f, err := os.Open(path)
	if os.IsNotExist(err) {
		return 0
	}
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	defer f.Close()
	latest := map[string]string{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var r resumeRearmRow
		if json.Unmarshal(sc.Bytes(), &r) == nil && r.Phase == "rearm" && r.Session != "" {
			latest[r.Session] = r.Reason
		}
	}
	for sid, reason := range latest {
		fmt.Fprintf(stdout, "%s\t%s\n", sid, reason)
	}
	return 0
}
