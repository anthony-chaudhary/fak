package sessiondiag

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

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
	SourceErrors []SourceError `json:"source_errors"`
}

// ReadCodexInventorySources reads the runtime-safe, read-only evidence used by
// both the development diagnostic command and session recovery.
func ReadCodexInventorySources(codexHome string, _ time.Duration, now time.Time) (InventoryInput, error) {
	home, err := resolveCodexHome(codexHome)
	if err != nil {
		return InventoryInput{}, err
	}
	python, err := inventoryPython()
	if err != nil {
		return InventoryInput{}, err
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
	var out, stderr bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return InventoryInput{}, fmt.Errorf("read-only Codex inventory query failed: %s", redactReadError(stderr.String()))
	}
	var snapshot codexInventorySnapshot
	if err := json.Unmarshal(out.Bytes(), &snapshot); err != nil {
		return InventoryInput{}, fmt.Errorf("decode Codex inventory: %w", err)
	}
	input := InventoryInput{SourceErrors: append([]SourceError(nil), snapshot.SourceErrors...)}
	for _, row := range snapshot.Threads {
		input.Threads = append(input.Threads, ThreadEvidence{
			ThreadID: row.ThreadID, Source: row.Source, ThreadSource: row.ThreadSource,
			CreatedAt: unixMillis(row.CreatedAtMS), UpdatedAt: unixMillis(row.UpdatedAtMS),
			Archived: row.Archived, AgentNickname: row.AgentNickname, AgentRole: row.AgentRole, CWD: row.CWD,
		})
	}
	for _, row := range snapshot.Turns {
		input.Turns = append(input.Turns, TurnEvidence{
			ThreadID: row.ThreadID, TurnID: row.TurnID, Status: row.Status, ErrorJSON: row.ErrorJSON,
			StartedAt: unixFlexible(row.StartedAt), CompletedAt: unixFlexible(row.CompletedAt),
			DurationMS: row.DurationMS, RolloutOrdinal: row.RolloutOrdinal,
		})
	}
	for _, row := range snapshot.SpawnEdges {
		input.SpawnEdges = append(input.SpawnEdges, SpawnEdgeEvidence{
			ParentThreadID: row.ParentThreadID, ChildThreadID: row.ChildThreadID, Status: row.Status,
		})
	}
	locks, lockErr := readWriterLocks(home)
	input.WriterLocks = locks
	if lockErr != nil {
		input.SourceErrors = append(input.SourceErrors, *lockErr)
	}
	receipts, receiptErr := readLaunchReceipts(home)
	input.GuardReceipts = receipts
	if receiptErr != nil {
		input.SourceErrors = append(input.SourceErrors, *receiptErr)
	}
	processes, processErr := procguard.CollectRelations()
	if processErr != "" {
		input.SourceErrors = append(input.SourceErrors, SourceError{Source: "os_process_table", Code: "READ_FAILED"})
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
			input.Processes = append(input.Processes, ProcessEvidence{
				PID: process.PID, ParentPID: parentPID, Name: process.Name, CommandLine: process.Cmdline,
				StartedAt: startedAt, AgeSeconds: age,
			})
		}
	}
	return input, nil
}

func resolveCodexHome(value string) (string, error) {
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

func inventoryPython() (string, error) {
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

func readWriterLocks(home string) ([]WriterLockEvidence, *SourceError) {
	dir := filepath.Join(home, "thread-writer-locks")
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, &SourceError{Source: "thread_writer_locks", Code: "READ_FAILED"}
	}
	out := []WriterLockEvidence{}
	for _, entry := range entries {
		if entry.IsDir() || entry.Name() == ".coordination.lock" || !strings.HasSuffix(entry.Name(), ".lock") {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		out = append(out, WriterLockEvidence{ThreadID: strings.TrimSuffix(entry.Name(), ".lock"), ModifiedAt: info.ModTime().UTC()})
	}
	return out, nil
}

func readLaunchReceipts(home string) ([]GuardReceiptEvidence, *SourceError) {
	dir := filepath.Join(home, "fak-guarded-sessions")
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, &SourceError{Source: "guard_launch_receipts", Code: "READ_FAILED"}
	}
	out := []GuardReceiptEvidence{}
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
		out = append(out, GuardReceiptEvidence{ThreadID: receipt.SessionID, RecordedAt: recordedAt.UTC()})
	}
	if invalid {
		return out, &SourceError{Source: "guard_launch_receipts", Code: "INVALID_RECORD_SKIPPED"}
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

func redactReadError(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return "reader exited without detail"
	}
	if len(value) > 160 {
		value = value[:160]
	}
	return filepath.Base(strings.ReplaceAll(value, "\\", "/"))
}
