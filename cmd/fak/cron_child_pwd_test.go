package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// cron_child_pwd_test.go — witness that every cron launch site hands its child a PWD
// naming the child's workdir and no OLDPWD. Setting exec.Cmd.Dir moves only the
// child's cwd: os/exec rewrites PWD only off Windows and only when Cmd.Env is nil, so
// a launcher started from Git Bash that passes its environment through leaves PWD and
// OLDPWD naming the LAUNCHER's directory. opencode trusts PWD over its cwd, boots a
// second instance for that stale directory and never exits after its turn.
//
// Each test plays that launcher and runs this test binary as the child; this file's
// init() turns it into the stand-in child before TestMain or test flag parsing runs.

// cronChildPWDRecordEnv names the file the stand-in child writes what it saw to. Each
// site delivers it the way that site delivers any environment, so a record also
// proves the site's environment reached the child.
const cronChildPWDRecordEnv = "FAK_CRON_CHILD_PWD_RECORD"

type cronChildPWDRecord struct {
	PWD       string `json:"pwd"`
	OLDPWD    string `json:"oldpwd"`
	OLDPWDSet bool   `json:"oldpwd_set"`
	Cwd       string `json:"cwd"`
}

// The stand-in child dispatches from package init, which runs before TestMain and test
// flag parsing, so it can also answer the version probe's `<bin> --version`.
func init() { cronChildPWDWitnessChild() }

// cronChildPWDWitnessChild turns the test binary into the stand-in cron child when
// cronChildPWDRecordEnv is set; in a normal run it returns at once. The session line
// is what a real opencode child prints, so RunScheduledOpenCode sees a started run.
func cronChildPWDWitnessChild() {
	path := os.Getenv(cronChildPWDRecordEnv)
	if path == "" {
		return
	}
	oldpwd, oldpwdSet := os.LookupEnv("OLDPWD")
	cwd, _ := os.Getwd()
	b, err := json.Marshal(cronChildPWDRecord{PWD: os.Getenv("PWD"), OLDPWD: oldpwd, OLDPWDSet: oldpwdSet, Cwd: cwd})
	// Write-then-rename: a child killed at the probe bound must leave no partial record.
	tmp := path + ".tmp"
	if err != nil || os.WriteFile(tmp, b, 0o644) != nil || os.Rename(tmp, path) != nil {
		os.Exit(1)
	}
	fmt.Println(`{"session_id": "ses_child_pwd"}`)
	os.Exit(0)
}

// cronChildPWDGitBashLauncher makes this process look like a Git Bash launcher whose
// PWD and OLDPWD name two directories other than the child's workdir, and returns
// that workdir plus the record path the child writes.
func cronChildPWDGitBashLauncher(t *testing.T) (workdir, record string) {
	t.Helper()
	t.Setenv("PWD", t.TempDir())
	t.Setenv("OLDPWD", t.TempDir())
	return t.TempDir(), filepath.Join(t.TempDir(), "child-env.json")
}

// assertCronChildPWD fails unless the child that site launched in workdir saw PWD
// naming workdir, no OLDPWD, and workdir as its cwd. Directories are compared with
// os.SameFile because Windows spells one directory many ways (short names,
// separators, case).
func assertCronChildPWD(t *testing.T, site, record, workdir string) {
	t.Helper()
	b, err := os.ReadFile(record)
	if err != nil {
		t.Fatalf("%s: child never recorded its environment: %v (workdir=%q)", site, err, workdir)
	}
	var got cronChildPWDRecord
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("%s: child record %q is not JSON: %v", site, b, err)
	}
	isWorkdir := func(p string) bool {
		pi, perr := os.Stat(p)
		wi, werr := os.Stat(workdir)
		return perr == nil && werr == nil && os.SameFile(pi, wi)
	}
	seen := fmt.Sprintf("child saw PWD=%q OLDPWD=%q (set=%t) cwd=%q; workdir=%q", got.PWD, got.OLDPWD, got.OLDPWDSet, got.Cwd, workdir)
	if !filepath.IsAbs(got.PWD) || !isWorkdir(got.PWD) {
		t.Errorf("%s: child PWD does not name its workdir (the launcher's PWD leaked through): %s", site, seen)
	}
	if got.OLDPWDSet {
		t.Errorf("%s: child inherited the launcher's OLDPWD: %s", site, seen)
	}
	if !isWorkdir(got.Cwd) {
		t.Errorf("%s: child cwd is not its workdir: %s", site, seen)
	}
}

func TestCronChildPWDRunScheduledOpenCode(t *testing.T) {
	workdir, record := cronChildPWDGitBashLauncher(t)
	var stdout, stderr bytes.Buffer
	receipt, err := RunScheduledOpenCode(ScheduledOpenCodeOptions{
		Job:     "job-child-pwd",
		Ledger:  filepath.Join(t.TempDir(), "opencode_child_pwd.jsonl"),
		Timeout: 60 * time.Second,
		Workdir: workdir,
		// A non-empty Env is the branch that hands the launcher's environment through.
		Env: []string{cronChildPWDRecordEnv + "=" + record},
		// -test.run=^$ keeps a child that missed the record env from re-running this suite.
		Command: []string{os.Args[0], "-test.run=^$"},
		Stdout:  &stdout,
		Stderr:  &stderr,
	})
	if err != nil || receipt.Outcome != "succeeded" || receipt.ExitCode != 0 {
		t.Errorf("RunScheduledOpenCode: child run err=%v outcome=%q exit=%d start_error=%q stderr=%q; want succeeded/0",
			err, receipt.Outcome, receipt.ExitCode, receipt.StartError, stderr.String())
	}
	assertCronChildPWD(t, "RunScheduledOpenCode", record, workdir)
}

func TestCronChildPWDVersionProbe(t *testing.T) {
	workdir, record := cronChildPWDGitBashLauncher(t)
	env := []string{cronChildPWDRecordEnv + "=" + record}
	// The probe's fixed 5s bound can lapse before a loaded host even starts the child;
	// retrying tells host slowness apart from the PWD defect under test.
	var out string
	var statErr error
	for attempt := 0; attempt < 3; attempt++ {
		out = cronProbeOpenCodeVersion(os.Args[0], workdir, env)
		if _, statErr = os.Stat(record); statErr == nil {
			break
		}
	}
	if statErr != nil {
		t.Fatalf("cronProbeOpenCodeVersion: child never recorded (probe bound) after 3 probes: %v; last probe output=%q workdir=%q", statErr, out, workdir)
	}
	assertCronChildPWD(t, "cronProbeOpenCodeVersion", record, workdir)
}

func TestCronChildPWDCronRun(t *testing.T) {
	workdir, record := cronChildPWDGitBashLauncher(t)
	// `fak cron run` has no env flag: its child inherits this process's environment.
	t.Setenv(cronChildPWDRecordEnv, record)
	var stdout, stderr bytes.Buffer
	code := runCron(&stdout, &stderr, []string{
		"run", "--job", "job-child-pwd", "--ledger", filepath.Join(t.TempDir(), "cron_run_child_pwd.jsonl"),
		"--interval", "1h", "--slot", "slot-child-pwd", "--timeout", "60s", "--workdir", workdir, "--",
		os.Args[0], "-test.run=^$",
	})
	if code != 0 {
		t.Errorf("fak cron run: exit code %d, want 0: stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	assertCronChildPWD(t, "fak cron run", record, workdir)
}
