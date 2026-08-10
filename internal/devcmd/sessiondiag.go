package devcmd

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessiondiag"
)

type sessionDiagQuery func(string, time.Duration) (sessiondiag.Evidence, error)

func RunSessionDiag(stdout, stderr io.Writer, args []string, query sessionDiagQuery) int {
	if query == nil {
		query = queryCodexLogReadOnly
	}
	fs := flag.NewFlagSet("sessiondiag", flag.ContinueOnError)
	fs.SetOutput(stderr)
	db := fs.String("db", defaultCodexLogDB(), "Codex structured-log SQLite path")
	since := fs.Duration("since", 24*time.Hour, "bounded evidence window")
	jsonOut := fs.Bool("json", false, "emit JSON")
	incidentOut := fs.String("incident-out", "", "write one redacted incident envelope (requires --process-id)")
	processID := fs.Int("process-id", 0, "observed Codex process id")
	processUUID := fs.String("process-uuid", "", "opaque process UUID")
	threadID := fs.String("thread-id", "", "opaque thread id")
	exitKind := fs.String("exit-kind", "none", "none, failure, or intentional")
	exitCode := fs.String("exit-code", "", "process exit code when observed")
	osFailure := fs.Bool("os-failure-event", false, "OS process-failure event observed")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || *since <= 0 {
		fmt.Fprintln(stderr, "usage: fak sessiondiag [--db PATH] [--since 24h] [--json]")
		return 2
	}
	e, err := query(*db, *since)
	if err != nil {
		fmt.Fprintf(stderr, "fak sessiondiag: evidence unavailable (%s)\n", "reader error")
		return 2
	}
	e.ProcessCount = countCodexProcesses()
	if *incidentOut != "" {
		if *processID <= 0 {
			fmt.Fprintln(stderr, "fak sessiondiag: --incident-out requires --process-id")
			return 2
		}
		var code *int
		if *exitCode != "" {
			n, err := strconv.Atoi(*exitCode)
			if err != nil {
				fmt.Fprintln(stderr, "fak sessiondiag: invalid --exit-code")
				return 2
			}
			code = &n
		}
		kind := sessiondiag.ExitKind(*exitKind)
		if kind != sessiondiag.ExitNone && kind != sessiondiag.ExitFailure && kind != sessiondiag.ExitIntentional {
			fmt.Fprintln(stderr, "fak sessiondiag: --exit-kind must be none, failure, or intentional")
			return 2
		}
		exitAt := time.Time{}
		if kind != sessiondiag.ExitNone {
			exitAt = time.Now()
		}
		inc := sessiondiag.CaptureIncident(sessiondiag.IncidentInput{CapturedAt: time.Now(), ProcessID: *processID, ProcessUUID: *processUUID, ThreadID: *threadID, ExitAt: exitAt, ExitKind: kind, ExitCode: code, OSFailureEvent: *osFailure, QueueDropDelta: e.QueueDrops, SlowWriteDelta: e.SlowWrites, WriterCount: e.ProcessCount, DBBytes: e.DBBytes, WALBytes: e.WALBytes, FreelistPages: e.FreelistPages, ProcessObserved: kind == sessiondiag.ExitNone})
		b, err := json.MarshalIndent(inc, "", "  ")
		if err != nil {
			fmt.Fprintln(stderr, "fak sessiondiag: encode incident")
			return 1
		}
		if err := os.WriteFile(*incidentOut, append(b, '\n'), 0600); err != nil {
			fmt.Fprintln(stderr, "fak sessiondiag: write incident failed")
			return 1
		}
		if !*jsonOut {
			fmt.Fprintf(stdout, "incident=%s verdict=%s\n", filepath.Base(*incidentOut), inc.Verdict)
		}
	}
	r := sessiondiag.Classify(e, time.Now())
	if *jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(r)
	} else {
		fmt.Fprintf(stdout, "%s causality=%s read_only=%t\n", r.Verdict, r.Causality, r.ReadOnly)
		fmt.Fprintf(stdout, "store=%s db=%d wal=%d reclaimable=%d/%d pages processes=%d window=%ds rows=%d\n", e.DBBasename, e.DBBytes, e.WALBytes, e.FreelistPages, e.PageCount, e.ProcessCount, e.WindowSeconds, e.RecentRows)
		for _, f := range r.Findings {
			fmt.Fprintf(stdout, "- %s [%s] count=%d: %s; next: %s\n", f.Code, f.Severity, f.Count, f.Detail, f.Action)
		}
		fmt.Fprintln(stdout, "causal claim: pressure/event loss is correlation unless EXPLICIT_PROCESS_FAILURE is present")
	}
	if r.Verdict == "NO_FAULT_EVIDENCE" {
		return 0
	}
	return 1
}

func defaultCodexLogDB() string {
	if h, err := os.UserHomeDir(); err == nil {
		return filepath.Join(h, ".codex", "logs_2.sqlite")
	}
	return "logs_2.sqlite"
}

func queryCodexLogReadOnly(path string, since time.Duration) (sessiondiag.Evidence, error) {
	if _, err := os.Stat(path); err != nil {
		return sessiondiag.Evidence{}, fmt.Errorf("read Codex log store: %w", err)
	}
	python, err := exec.LookPath("python")
	if err != nil {
		if runtime.GOOS != "windows" {
			python, err = exec.LookPath("python3")
		}
		if err != nil {
			return sessiondiag.Evidence{}, errors.New("Python sqlite reader not found; install Python or pass captured evidence to the library")
		}
	}
	script := `import json,os,sqlite3,sys,time
p=sys.argv[1]; cutoff=int(time.time()-float(sys.argv[2])); uri='file:'+p.replace('\\','/')+'?mode=ro&immutable=0'
c=sqlite3.connect(uri,uri=True,timeout=5); c.execute('pragma query_only=on')
def one(q,a=()): return int(c.execute(q,a).fetchone()[0] or 0)
page=one('pragma page_count'); free=one('pragma freelist_count'); psz=one('pragma page_size')
recent=one('select count(*) from logs where ts>=?',(cutoff,))
drops=one("select count(*) from logs where ts>=? and target='codex_app_server_client' and feedback_log_body like '%consumer queue is full%'",(cutoff,))
slow=one("select count(*) from logs where ts>=? and target='sqlx::query' and feedback_log_body like 'slow statement:%INSERT INTO logs%'",(cutoff,))
fail=one("select count(*) from logs where ts>=? and (level='ERROR' or level='FATAL') and (target in ('codex_core::spawn','codex_core::process','codex_core::panic') or target like 'fak%') and (lower(feedback_log_body) like '%panicked at%' or lower(feedback_log_body) like '%fatal runtime error%' or lower(feedback_log_body) like '%process exited with%' or lower(feedback_log_body) like '%child exited with%')",(cutoff,))
c.execute('select name from sqlite_master limit 1').fetchone(); integrity='not_checked'
print(json.dumps(dict(db_basename=os.path.basename(p),db_bytes=os.path.getsize(p),wal_bytes=os.path.getsize(p+'-wal') if os.path.exists(p+'-wal') else 0,page_size=psz,page_count=page,freelist_pages=free,recent_rows=recent,queue_drops=drops,slow_writes=slow,explicit_failures=fail,window_seconds=int(float(sys.argv[2])),integrity=integrity)))`
	cmd := exec.Command(python, "-c", script, path, fmt.Sprintf("%.0f", since.Seconds()))
	var out, er bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &er
	if err := cmd.Run(); err != nil {
		return sessiondiag.Evidence{}, fmt.Errorf("read-only Codex log query failed: %s", redactDiagError(er.String()))
	}
	var e sessiondiag.Evidence
	if err := json.Unmarshal(out.Bytes(), &e); err != nil {
		return e, fmt.Errorf("decode bounded query: %w", err)
	}
	return e, nil
}

func countCodexProcesses() int64 {
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.Command("tasklist", "/FI", "IMAGENAME eq codex.exe", "/FO", "CSV", "/NH")
	} else {
		cmd = exec.Command("ps", "-A", "-o", "comm=")
	}
	out, err := cmd.Output()
	if err != nil {
		return 0
	}
	var count int64
	for _, line := range strings.Split(string(out), "\n") {
		name := strings.ToLower(strings.TrimSpace(strings.Trim(line, `"`)))
		if runtime.GOOS == "windows" {
			name = strings.ToLower(strings.TrimSpace(strings.Split(name, `","`)[0]))
		}
		if name == "codex" || name == "codex.exe" {
			count++
		}
	}
	return count
}

func redactDiagError(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "reader exited without detail"
	}
	if len(s) > 160 {
		s = s[:160]
	}
	return filepath.Base(strings.ReplaceAll(s, "\\", "/"))
}
