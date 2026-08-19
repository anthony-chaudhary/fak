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

	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/sessiondiag"
	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
)

type sessionDiagQuery func(string, time.Duration) (sessiondiag.Evidence, error)
type sessionDiagInventoryQuery func(string, time.Duration, time.Time) (sessiondiag.InventoryInput, error)

func RunSessionDiag(stdout, stderr io.Writer, args []string, query sessionDiagQuery) int {
	return runSessionDiagWith(stdout, stderr, args, query, queryCodexInventoryReadOnly, time.Now)
}

func runSessionDiagWith(stdout, stderr io.Writer, args []string, query sessionDiagQuery, inventoryQuery sessionDiagInventoryQuery, now func() time.Time) int {
	if query == nil {
		query = queryCodexLogReadOnly
	}
	if inventoryQuery == nil {
		inventoryQuery = queryCodexInventoryReadOnly
	}
	if now == nil {
		now = time.Now
	}
	fs := flag.NewFlagSet("sessiondiag", flag.ContinueOnError)
	fs.SetOutput(stderr)
	db := fs.String("db", defaultCodexLogDB(), "Codex structured-log SQLite path")
	codexHome := fs.String("codex-home", "", "Codex state root (default: $CODEX_HOME or ~/.codex)")
	since := fs.Duration("since", 24*time.Hour, "bounded evidence window")
	staleAfter := fs.Duration("stale-after", 10*time.Minute, "writer-lock/current-state staleness threshold")
	jsonOut := fs.Bool("json", false, "emit JSON")
	inventory := fs.Bool("inventory", false, "join current Codex threads, turns, writer locks, processes, guard launch receipts, and spawn edges")
	watchdogCandidates := fs.Bool("watchdog-candidates", false, "project typed watchdog candidates from inventory evidence")
	hookProfile := fs.Bool("hook-profile", false, "diagnose the effective Codex hook profile through app-server hooks/list")
	codexBin := fs.String("codex-bin", "", "Codex executable for --hook-profile (auto-detected by default)")
	repoRoot := fs.String("repo", ".", "repository whose HEAD identifies the expected fak hook build")
	recentLogRows := fs.Int("recent-log-rows", 20_000, "bounded trailing Codex log rows inspected by --hook-profile")
	liveSignals := fs.Bool("live-signals", false, "render the minimal operator projection for live DOS lanes (attention, outcome, move, next check)")
	fullLiveSignals := fs.Bool("full", false, "with --live-signals, expand unknown and healthy workers instead of folding them")
	registryPath := fs.String("registry", sessionregistry.DefaultPath(), "child registration JSONL path (missing is allowed)")
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
	if fs.NArg() != 0 || *since <= 0 || *staleAfter <= 0 || *recentLogRows <= 0 {
		fmt.Fprintln(stderr, "usage: fak dev sessiondiag [--inventory|--hook-profile] [--registry FILE] [--codex-home DIR] [--codex-bin FILE] [--db PATH] [--since 24h] [--stale-after 10m] [--json]")
		return 2
	}
	if *fullLiveSignals && !*liveSignals {
		fmt.Fprintln(stderr, "fak sessiondiag: --full requires --live-signals")
		return 2
	}
	if *hookProfile {
		if *inventory || *liveSignals || *incidentOut != "" {
			fmt.Fprintln(stderr, "fak sessiondiag: --hook-profile cannot be combined with --inventory, --live-signals, or --incident-out")
			return 2
		}
		cwd, err := os.Getwd()
		if err != nil {
			fmt.Fprintln(stderr, "fak sessiondiag: working directory unavailable")
			return 2
		}
		if !flagWasSet(fs, "db") && strings.TrimSpace(*codexHome) != "" {
			*db = filepath.Join(*codexHome, "logs_2.sqlite")
		}
		return runCodexHookProfileWith(stdout, stderr, codexHookProfileQueryInput{
			CodexHome:        *codexHome,
			CodexBin:         *codexBin,
			WorkingDirectory: cwd,
			LogDBPath:        *db,
			RepoRoot:         *repoRoot,
			RecentLogRows:    *recentLogRows,
			ObservedAt:       now().UTC(),
		}, *jsonOut, queryCodexHookProfile)
	}
	if *liveSignals {
		if *jsonOut || *inventory || *incidentOut != "" {
			fmt.Fprintln(stderr, "fak sessiondiag: --live-signals cannot be combined with --json, --inventory, or --incident-out")
			return 2
		}
		return runOperatorLiveSignals(stdout, stderr, operatorLiveCommand, now(), *fullLiveSignals)
	}
	if *inventory {
		if *incidentOut != "" {
			fmt.Fprintln(stderr, "fak sessiondiag: --inventory cannot be combined with --incident-out")
			return 2
		}
		observedAt := now().UTC()
		input, err := inventoryQuery(*codexHome, *since, observedAt)
		if err != nil {
			fmt.Fprintf(stderr, "fak sessiondiag: inventory unavailable (%s)\n", "reader error")
			return 2
		}
		input.Window = *since
		input.StaleAfter = *staleAfter
		rows, regErr := (sessionregistry.Store{Path: *registryPath}).ReadAll()
		if regErr == nil {
			input.Registrations = rows
		} else if !errors.Is(regErr, os.ErrNotExist) {
			input.SourceErrors = append(input.SourceErrors, sessiondiag.SourceError{Source: "child_registrations", Code: "READ_FAILED"})
		}
		report := sessiondiag.ReconcileInventory(input, observedAt)
		if *watchdogCandidates {
			if !*jsonOut {
				fmt.Fprintln(stderr, "fak sessiondiag: --watchdog-candidates requires --json")
				return 2
			}
			projected := sessiondiag.ProjectWatchdogCandidates(report)
			enc := json.NewEncoder(stdout)
			enc.SetIndent("", "  ")
			if err := enc.Encode(projected); err != nil {
				fmt.Fprintln(stderr, "fak sessiondiag: encode watchdog candidates")
				return 1
			}
			return 0
		}
		if *jsonOut {
			enc := json.NewEncoder(stdout)
			enc.SetIndent("", "  ")
			if err := enc.Encode(report); err != nil {
				fmt.Fprintln(stderr, "fak sessiondiag: encode inventory")
				return 1
			}
		} else {
			sessiondiag.RenderInventory(stdout, report)
		}
		return 0
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
			exitAt = now()
		}
		inc := sessiondiag.CaptureIncident(sessiondiag.IncidentInput{CapturedAt: now(), ProcessID: *processID, ProcessUUID: *processUUID, ThreadID: *threadID, ExitAt: exitAt, ExitKind: kind, ExitCode: code, OSFailureEvent: *osFailure, QueueDropDelta: e.QueueDrops, SlowWriteDelta: e.SlowWrites, WriterCount: e.ProcessCount, DBBytes: e.DBBytes, WALBytes: e.WALBytes, FreelistPages: e.FreelistPages, ProcessObserved: kind == sessiondiag.ExitNone})
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
	r := sessiondiag.Classify(e, now())
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

type codexInventorySnapshot struct {
	Threads []struct {
		ThreadID      string `json:"thread_id"`
		Source        string `json:"source"`
		ThreadSource  string `json:"thread_source"`
		CreatedAtMS   int64  `json:"created_at_ms"`
		UpdatedAtMS   int64  `json:"updated_at_ms"`
		Archived      bool   `json:"archived"`
		AgentNickname string `json:"agent_nickname"`
		AgentRole     string `json:"agent_role"`
		CWD           string `json:"cwd"`
	} `json:"threads"`
	Turns []struct {
		ThreadID       string `json:"thread_id"`
		TurnID         string `json:"turn_id"`
		Status         string `json:"status"`
		ErrorJSON      string `json:"error_json"`
		StartedAt      int64  `json:"started_at"`
		CompletedAt    int64  `json:"completed_at"`
		DurationMS     int64  `json:"duration_ms"`
		RolloutOrdinal int64  `json:"rollout_ordinal"`
	} `json:"turns"`
	SpawnEdges []struct {
		ParentThreadID string `json:"parent_thread_id"`
		ChildThreadID  string `json:"child_thread_id"`
		Status         string `json:"status"`
	} `json:"spawn_edges"`
	SourceErrors []sessiondiag.SourceError `json:"source_errors"`
}

func queryCodexInventoryReadOnly(codexHome string, _ time.Duration, now time.Time) (sessiondiag.InventoryInput, error) {
	home, err := resolveSessionDiagCodexHome(codexHome)
	if err != nil {
		return sessiondiag.InventoryInput{}, err
	}
	python, err := sessionDiagPython()
	if err != nil {
		return sessiondiag.InventoryInput{}, err
	}
	script := `import json,pathlib,sqlite3,sys
home=pathlib.Path(sys.argv[1]); out={"threads":[],"turns":[],"spawn_edges":[],"source_errors":[]}
def db(name):
    p=home/name
    try:
        c=sqlite3.connect("file:"+p.as_posix()+"?mode=ro",uri=True,timeout=5)
        c.execute("pragma query_only=on")
        return c
    except Exception:
        out["source_errors"].append({"source":name,"code":"READ_FAILED"})
        return None
state=db("state_5.sqlite")
if state is not None:
    try:
        for r in state.execute("select id,source,coalesce(thread_source,''),coalesce(created_at_ms,created_at*1000),coalesce(updated_at_ms,updated_at*1000),archived,coalesce(agent_nickname,''),coalesce(agent_role,''),coalesce(cwd,'') from threads"):
            out["threads"].append(dict(thread_id=r[0],source=r[1],thread_source=r[2],created_at_ms=r[3] or 0,updated_at_ms=r[4] or 0,archived=bool(r[5]),agent_nickname=r[6],agent_role=r[7],cwd=r[8]))
        for r in state.execute("select parent_thread_id,child_thread_id,status from thread_spawn_edges"):
            out["spawn_edges"].append(dict(parent_thread_id=r[0],child_thread_id=r[1],status=r[2]))
    except Exception:
        out["source_errors"].append({"source":"state_5.sqlite","code":"QUERY_FAILED"})
    state.close()
history=db("thread_history_1.sqlite")
if history is not None:
    try:
        for r in history.execute("select thread_id,turn_id,status,coalesce(error_json,''),coalesce(started_at,0),coalesce(completed_at,0),coalesce(duration_ms,0),rollout_ordinal from thread_turns"):
            out["turns"].append(dict(thread_id=r[0],turn_id=r[1],status=r[2],error_json=r[3],started_at=r[4],completed_at=r[5],duration_ms=r[6],rollout_ordinal=r[7]))
    except Exception:
        out["source_errors"].append({"source":"thread_history_1.sqlite","code":"QUERY_FAILED"})
    history.close()
print(json.dumps(out,separators=(",",":")))`
	cmd := exec.Command(python, "-c", script, home)
	var out, er bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &er
	if err := cmd.Run(); err != nil {
		return sessiondiag.InventoryInput{}, fmt.Errorf("read-only Codex inventory query failed: %s", redactDiagError(er.String()))
	}
	var snapshot codexInventorySnapshot
	if err := json.Unmarshal(out.Bytes(), &snapshot); err != nil {
		return sessiondiag.InventoryInput{}, fmt.Errorf("decode Codex inventory: %w", err)
	}
	input := sessiondiag.InventoryInput{SourceErrors: append([]sessiondiag.SourceError(nil), snapshot.SourceErrors...)}
	for _, row := range snapshot.Threads {
		input.Threads = append(input.Threads, sessiondiag.ThreadEvidence{
			ThreadID:      row.ThreadID,
			Source:        row.Source,
			ThreadSource:  row.ThreadSource,
			CreatedAt:     unixMillis(row.CreatedAtMS),
			UpdatedAt:     unixMillis(row.UpdatedAtMS),
			Archived:      row.Archived,
			AgentNickname: row.AgentNickname,
			AgentRole:     row.AgentRole,
			CWD:           row.CWD,
		})
	}
	for _, row := range snapshot.Turns {
		input.Turns = append(input.Turns, sessiondiag.TurnEvidence{
			ThreadID:       row.ThreadID,
			TurnID:         row.TurnID,
			Status:         row.Status,
			ErrorJSON:      row.ErrorJSON,
			StartedAt:      unixFlexible(row.StartedAt),
			CompletedAt:    unixFlexible(row.CompletedAt),
			DurationMS:     row.DurationMS,
			RolloutOrdinal: row.RolloutOrdinal,
		})
	}
	for _, row := range snapshot.SpawnEdges {
		input.SpawnEdges = append(input.SpawnEdges, sessiondiag.SpawnEdgeEvidence{
			ParentThreadID: row.ParentThreadID,
			ChildThreadID:  row.ChildThreadID,
			Status:         row.Status,
		})
	}
	locks, lockErr := collectSessionDiagWriterLocks(home)
	input.WriterLocks = locks
	if lockErr != nil {
		input.SourceErrors = append(input.SourceErrors, *lockErr)
	}
	receipts, receiptErr := collectSessionDiagGuardReceipts(home)
	input.GuardReceipts = receipts
	if receiptErr != nil {
		input.SourceErrors = append(input.SourceErrors, *receiptErr)
	}
	processes, processErr := procguard.CollectRelations()
	if processErr != "" {
		input.SourceErrors = append(input.SourceErrors, sessiondiag.SourceError{Source: "os_process_table", Code: "READ_FAILED"})
	} else {
		for _, process := range processes {
			startedAt, _ := time.Parse(time.RFC3339Nano, process.Start)
			age := int64(0)
			if process.AgeSec != nil {
				age = int64(*process.AgeSec)
				if startedAt.IsZero() && age >= 0 {
					startedAt = now.Add(-time.Duration(age) * time.Second)
				}
			}
			parentPID := 0
			if process.PPID != nil {
				parentPID = *process.PPID
			}
			input.Processes = append(input.Processes, sessiondiag.ProcessEvidence{
				PID:         process.PID,
				ParentPID:   parentPID,
				Name:        process.Name,
				CommandLine: process.Cmdline,
				StartedAt:   startedAt,
				AgeSeconds:  age,
			})
		}
	}
	return input, nil
}

func resolveSessionDiagCodexHome(value string) (string, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		value = strings.TrimSpace(os.Getenv("CODEX_HOME"))
	}
	if value == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("resolve Codex home: %w", err)
		}
		value = filepath.Join(home, ".codex")
	}
	value = filepath.Clean(value)
	if info, err := os.Stat(value); err != nil || !info.IsDir() {
		return "", fmt.Errorf("Codex home unavailable")
	}
	return value, nil
}

func sessionDiagPython() (string, error) {
	python, err := exec.LookPath("python")
	if err == nil {
		return python, nil
	}
	if runtime.GOOS != "windows" {
		if python, err = exec.LookPath("python3"); err == nil {
			return python, nil
		}
	}
	return "", errors.New("Python sqlite reader not found; install Python")
}

func collectSessionDiagWriterLocks(home string) ([]sessiondiag.WriterLockEvidence, *sessiondiag.SourceError) {
	dir := filepath.Join(home, "thread-writer-locks")
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, &sessiondiag.SourceError{Source: "thread_writer_locks", Code: "READ_FAILED"}
	}
	out := []sessiondiag.WriterLockEvidence{}
	for _, entry := range entries {
		if entry.IsDir() || entry.Name() == ".coordination.lock" || !strings.HasSuffix(entry.Name(), ".lock") {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		out = append(out, sessiondiag.WriterLockEvidence{
			ThreadID:   strings.TrimSuffix(entry.Name(), ".lock"),
			ModifiedAt: info.ModTime().UTC(),
		})
	}
	return out, nil
}

func collectSessionDiagGuardReceipts(home string) ([]sessiondiag.GuardReceiptEvidence, *sessiondiag.SourceError) {
	dir := filepath.Join(home, "fak-guarded-sessions")
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, &sessiondiag.SourceError{Source: "guard_launch_receipts", Code: "READ_FAILED"}
	}
	out := []sessiondiag.GuardReceiptEvidence{}
	invalid := false
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(dir, entry.Name()))
		if err != nil {
			invalid = true
			continue
		}
		var receipt struct {
			Schema    string `json:"schema"`
			SessionID string `json:"session_id"`
			GuardedAt string `json:"guarded_at"`
		}
		if json.Unmarshal(raw, &receipt) != nil || receipt.Schema != "fak.codex_guard_witness.v1" ||
			receipt.SessionID == "" || receipt.SessionID != strings.TrimSuffix(entry.Name(), ".json") {
			invalid = true
			continue
		}
		recordedAt, err := time.Parse(time.RFC3339Nano, receipt.GuardedAt)
		if err != nil {
			invalid = true
			continue
		}
		out = append(out, sessiondiag.GuardReceiptEvidence{ThreadID: receipt.SessionID, RecordedAt: recordedAt.UTC()})
	}
	if invalid {
		return out, &sessiondiag.SourceError{Source: "guard_launch_receipts", Code: "INVALID_RECORD_SKIPPED"}
	}
	return out, nil
}

func unixMillis(value int64) time.Time {
	if value <= 0 {
		return time.Time{}
	}
	return time.UnixMilli(value).UTC()
}

func unixFlexible(value int64) time.Time {
	if value <= 0 {
		return time.Time{}
	}
	if value >= 1_000_000_000_000 {
		return time.UnixMilli(value).UTC()
	}
	return time.Unix(value, 0).UTC()
}

func defaultCodexLogDB() string {
	if h := strings.TrimSpace(os.Getenv("CODEX_HOME")); h != "" {
		return filepath.Join(h, "logs_2.sqlite")
	}
	if h, err := os.UserHomeDir(); err == nil {
		return filepath.Join(h, ".codex", "logs_2.sqlite")
	}
	return "logs_2.sqlite"
}

func flagWasSet(fs *flag.FlagSet, name string) bool {
	found := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == name {
			found = true
		}
	})
	return found
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
