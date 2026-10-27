package debtlane

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/processlock"
)

func journalCompatWrite(t *testing.T, root string, entries ...map[string]any) string {
	t.Helper()
	dir := filepath.Join(root, ".dos")
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "lane-journal.jsonl")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	encoder := json.NewEncoder(f)
	for _, entry := range entries {
		if err := encoder.Encode(entry); err != nil {
			f.Close()
			t.Fatal(err)
		}
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

func journalCompatAlive(t *testing.T) {
	t.Helper()
	original := pidLivenessCheck
	pidLivenessCheck = func(int) bool { return true }
	t.Cleanup(func() { pidLivenessCheck = original })
}

func journalCompatProjection(t *testing.T, root string) []string {
	t.Helper()
	held, err := DiscoverHeldLeases(root)
	if err != nil {
		t.Fatal(err)
	}
	result := make([]string, 0, len(held))
	for _, lease := range held {
		data, err := json.Marshal(lease)
		if err != nil {
			t.Fatal(err)
		}
		result = append(result, string(data))
	}
	sort.Strings(result)
	return result
}

// fak-test:justify why=invariant when=changed:**/debtlane/wave.go
// fak-test:runtime fast est=50ms
func TestJournalCompatCheckpointOwnershipAndTail(t *testing.T) {
	journalCompatAlive(t)
	root := t.TempDir()
	now := time.Now().UTC().Format(time.RFC3339Nano)
	lease := func(loop, lane, tree string, pid int) map[string]any {
		return map[string]any{"loop_ts": loop, "lane": lane, "tree": []string{tree}, "pid": pid, "mode": "exclusive", "acquired_at": now}
	}
	journalCompatWrite(t, root,
		map[string]any{"op": "ACQUIRE", "lane": "discarded", "pid": 0, "tree": []string{"discarded"}},
		map[string]any{"op": "CHECKPOINT", "seq_watermark": 42, "leases": []any{lease("one", "same", "a", 1101), lease("two", "same", "b", 1102), lease("blank", "", "unknown/**", 0)}},
		map[string]any{"op": "HEARTBEAT", "loop_ts": "one", "lane": "same", "pid": 9999, "tree": []string{"wrong"}, "heartbeat_at": now},
		map[string]any{"op": "ADOPT", "loop_ts": "one", "lane": "same", "pid": 2101, "tree": []string{"wrong"}, "mode": "shared", "heartbeat_at": now},
		map[string]any{"op": "RELEASE", "loop_ts": "two", "lane": "same"},
		map[string]any{"op": "RECONCILE", "loop_ts": "three", "lane": "other", "lease": lease("three", "other", "c", 1103)},
		map[string]any{"op": "SCAVENGE", "loop_ts": "three", "lane": "other"},
		map[string]any{"op": "ADOPT", "loop_ts": "absent", "lane": "missing", "pid": 9101},
		map[string]any{"op": "HEARTBEAT", "loop_ts": "absent", "lane": "missing", "heartbeat_at": now},
		map[string]any{"op": "_CORRUPT", "_raw": "preserved observation"},
	)
	held, err := DiscoverHeldLeases(root)
	if err != nil {
		t.Fatal(err)
	}
	expected := []HeldLease{{Lane: "", Tree: []string{"unknown/**"}, Mode: "exclusive"}, {Lane: "same", Tree: []string{"a"}, PID: 2101, Mode: "exclusive"}}
	if !reflect.DeepEqual(held, expected) {
		t.Fatalf("held projection=%v, want %v", held, expected)
	}
	trees, err := DiscoverHeldTrees(root)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(trees[""], []string{"unknown/**"}) {
		t.Fatalf("blank-lane tree lost: %v", trees)
	}
}

// fak-test:justify why=invariant when=changed:**/debtlane/wave.go
// fak-test:runtime fast est=50ms
func TestJournalCompatAppendMutexNoEffects(t *testing.T) {
	root := t.TempDir()
	path := journalCompatWrite(t, root, map[string]any{"op": "ACQUIRE", "lane": "held", "pid": 0})
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	release, err := processlock.Try(filepath.Dir(path), filepath.Base(path)+".append.lock")
	if err != nil {
		t.Fatal(err)
	}
	for _, operation := range []func() error{
		func() error { return AcquireTreeLease(root, "new", []string{"tree"}, 11) },
		func() error { return ReleaseTreeLease(root, "held", nil, 0) },
		func() error { return TouchLaneLease(root, "held", 0) },
	} {
		if err := operation(); !errors.Is(err, processlock.ErrBusy) {
			release()
			t.Fatalf("append mutex not respected: %v", err)
		}
	}
	after, err := os.ReadFile(path)
	release()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Fatal("busy append changed journal")
	}
	if _, err := os.Stat(path + ".append.lock"); err != nil {
		t.Fatal("persistent mutex was removed")
	}
}

// fak-test:justify why=invariant when=changed:**/debtlane/wave.go
// fak-test:runtime fast est=100ms
func TestJournalCompatFutureAndLegacyIdentities(t *testing.T) {
	journalCompatAlive(t)
	root := t.TempDir()
	for _, item := range []struct {
		tree []string
		pid  int
	}{{[]string{"b", "a"}, 101}, {[]string{"a", "b"}, 101}, {[]string{"c"}, 102}} {
		if err := AcquireTreeLease(root, "same", item.tree, item.pid); err != nil {
			t.Fatal(err)
		}
	}
	held, err := DiscoverHeldLeases(root)
	if err != nil || len(held) != 2 {
		t.Fatalf("logical identities not stable: %v %v", held, err)
	}
	if err := TouchLaneLease(root, "same", 101); err != nil {
		t.Fatal(err)
	}
	if err := ReleaseTreeLease(root, "same", []string{"b", "a"}, 101); err != nil {
		t.Fatal(err)
	}
	held, err = DiscoverHeldLeases(root)
	if err != nil || len(held) != 1 || held[0].PID != 102 {
		t.Fatalf("release crossed owner identity: %v %v", held, err)
	}
	if err := ReleaseTreeLease(root, "same", nil, 0); err != nil {
		t.Fatal(err)
	}
	held, err = DiscoverHeldLeases(root)
	if err != nil || len(held) != 0 {
		t.Fatalf("explicit wildcard release failed: %v %v", held, err)
	}
	journalCompatWrite(t, root, map[string]any{"op": "ACQUIRE", "lane": "legacy", "tree": []string{"a"}, "pid": 101}, map[string]any{"op": "ACQUIRE", "lane": "legacy", "tree": []string{"b"}, "pid": 102})
	if err := TouchLaneLease(root, "legacy", 101); err != nil {
		t.Fatal(err)
	}
	if err := ReleaseTreeLease(root, "legacy", nil, 101); err != nil {
		t.Fatal(err)
	}
	held, err = DiscoverHeldLeases(root)
	if err != nil || len(held) != 1 || held[0].PID != 102 {
		t.Fatalf("legacy owner release changed: %v %v", held, err)
	}
	if err := ReleaseTreeLease(root, "legacy", nil, 0); err != nil {
		t.Fatal(err)
	}
}

// fak-test:justify why=invariant when=changed:**/debtlane/wave.go
// fak-test:runtime fast est=20ms
func TestJournalCompatHeartbeatNeverGrants(t *testing.T) {
	root := t.TempDir()
	path := journalCompatWrite(t, root)
	if err := TouchLaneLease(root, "absent", 123); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("absent heartbeat must fail closed: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != 0 {
		t.Fatal("absent heartbeat fabricated a journal hold")
	}
}

// fak-test:justify why=invariant when=changed:**/debtlane/wave.go
// fak-test:runtime fast est=200ms
func TestJournalCompatReadErrorsAreUnknown(t *testing.T) {
	root := t.TempDir()
	if _, err := DiscoverHeldLeases(root); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("missing journal returned a known empty world: %v", err)
	}
	path := journalCompatWrite(t, root, map[string]any{"op": "CHECKPOINT", "leases": nil})
	if _, err := DiscoverHeldLeases(root); err == nil {
		t.Fatal("invalid checkpoint returned a known empty world")
	}
	journalCompatWrite(t, root, map[string]any{"op": "CHECKPOINT", "leases": []any{nil}})
	if _, err := DiscoverHeldLeases(root); !errors.Is(err, os.ErrInvalid) {
		t.Fatalf("malformed snapshot lease became free: %v", err)
	}
	if err := os.WriteFile(path, []byte("{"+strings.Repeat("x", 17*1024*1024)), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := DiscoverHeldLeases(root); err == nil {
		t.Fatal("scanner failure returned a known empty world")
	}
}

func journalCompatPython(t *testing.T, root string, code string) *exec.Cmd {
	t.Helper()
	source := os.Getenv("FAK_DOS_SOURCE")
	if source == "" {
		t.Skip("FAK_DOS_SOURCE not set; cross-runtime witness unrun")
	}
	python, err := exec.LookPath("python")
	if err != nil {
		t.Skip("Python unavailable")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	t.Cleanup(cancel)
	cmd := exec.CommandContext(ctx, python, "-u", "-c", code, filepath.Join(source, "src"), root)
	cmd.Dir = root
	journal := filepath.Join(root, ".dos", "lane-journal.jsonl")
	cmd.Env = append(os.Environ(), "DOS_HOME="+filepath.Join(root, ".dos-home"), "PYTHONPATH="+filepath.Join(source, "src"), "DISPATCH_WORKSPACE="+root, "DISPATCH_LANE_JOURNAL_PATH="+journal, "JOB_LANE_JOURNAL_PATH="+journal, "DISPATCH_LANE_LEASE_LOCK_PATH="+filepath.Join(root, ".dos", ".lane-lease.lock"))
	return cmd
}

const journalCompatCompactPython = `
import hashlib,json,pathlib,sys
sys.path.insert(0,sys.argv[1])
import dos
assert pathlib.Path(dos.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]).resolve())
from dos.config import default_config
from dos.lane_lease import compact_journal, _journal_path, _lock_path
root=pathlib.Path(sys.argv[2]).resolve()
config=default_config(root,gather_env=False)
assert _journal_path(config).resolve()==root/".dos"/"lane-journal.jsonl"
assert _lock_path(config).resolve().parent==root/".dos"
from dos.lane_journal import read_all
raw_fragments=[]
raw_lines=_journal_path(config).read_text(encoding="utf-8",errors="replace").splitlines()
for line_index,line in enumerate(raw_lines):
 text=line.strip()
 if not text: continue
 try: json.loads(text)
 except json.JSONDecodeError:
  if line_index<len(raw_lines)-1: raw_fragments.append({"op":"_CORRUPT","_raw":text,"_line":line_index})
def corruption_receipt():
 records=[e for e in read_all(_journal_path(config)) if e.get("op")=="_CORRUPT"]
 encoded=sorted(json.dumps(e,sort_keys=True,separators=(",",":")) for e in records)
 return len(records),hashlib.sha256("\n".join(encoded).encode()).hexdigest()
corrupt_before=corruption_receipt()
compact_journal(config)
assert corruption_receipt()==corrupt_before,"canonical corruption evidence changed"
retained=read_all(_journal_path(config))
assert all(fragment in retained for fragment in raw_fragments),"raw fragments lost their canonical sentinel"
print("CORRUPTION_PRESERVED count=%d digest=%s"%corrupt_before)
print("RAW_FRAGMENTS_PRESERVED count=%d"%len(raw_fragments))
`

// fak-test:justify why=invariant when=changed:**/debtlane/wave.go
// fak-test:runtime fast est=3s
func TestJournalCompatCrossRuntimeCompaction(t *testing.T) {
	journalCompatAlive(t)
	root := t.TempDir()
	if err := AcquireTreeLease(root, "same", []string{"a"}, 101); err != nil {
		t.Fatal(err)
	}
	if err := AcquireTreeLease(root, "same", []string{"b"}, 102); err != nil {
		t.Fatal(err)
	}
	before := journalCompatProjection(t, root)
	code := strings.Replace(journalCompatCompactPython, "compact_journal(config)", `from dos._journal_mutex import locked
with locked(_journal_path(config)):
 print("READY",flush=True)
 sys.stdin.readline()
compact_journal(config)`, 1)
	cmd := journalCompatPython(t, root, code)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	ready, err := bufio.NewReader(stdout).ReadString('\n')
	if err != nil || strings.TrimSpace(ready) != "READY" {
		stdin.Close()
		cmd.Wait()
		t.Fatalf("reference lock handshake failed: %v %s", err, stderr.String())
	}
	appendErr := AcquireTreeLease(root, "blocked", []string{"must-not-appear"}, 103)
	_, writeErr := io.WriteString(stdin, "release\n")
	stdin.Close()
	waitErr := cmd.Wait()
	if !errors.Is(appendErr, processlock.ErrBusy) {
		t.Fatalf("native append crossed Python compactor mutex: %v", appendErr)
	}
	if writeErr != nil || waitErr != nil {
		t.Fatalf("reference compaction failed: %v %v %s", writeErr, waitErr, stderr.String())
	}
	if after := journalCompatProjection(t, root); !reflect.DeepEqual(before, after) {
		t.Fatalf("compaction changed held projection: before=%v after=%v", before, after)
	}
	if err := TouchLaneLease(root, "same", 101); err != nil {
		t.Fatal(err)
	}
	if err := ReleaseTreeLease(root, "same", []string{"a"}, 101); err != nil {
		t.Fatal(err)
	}
	held, err := DiscoverHeldLeases(root)
	if err != nil || len(held) != 1 || held[0].PID != 102 {
		t.Fatalf("checkpoint tail identity lost: %v %v", held, err)
	}
}

// fak-test:justify why=invariant when=changed:**/debtlane/wave.go
// fak-test:runtime slow est=15s
func TestJournalCompatActualCopyProjection(t *testing.T) {
	input := os.Getenv("FAK_WAL_COMPAT_INPUT")
	if input == "" {
		t.Skip("FAK_WAL_COMPAT_INPUT not set; actual-copy acceptance unrun")
	}
	root := t.TempDir()
	path := journalCompatWrite(t, root)
	src, err := os.Open(input)
	if err != nil {
		t.Fatal(err)
	}
	dst, err := os.Create(path)
	if err != nil {
		src.Close()
		t.Fatal(err)
	}
	_, copyErr := io.Copy(dst, src)
	src.Close()
	closeErr := dst.Close()
	if copyErr != nil || closeErr != nil {
		t.Fatalf("copy failed: %v %v", copyErr, closeErr)
	}
	original := pidLivenessCheck
	observed := map[int]bool{}
	pidLivenessCheck = func(pid int) bool {
		if live, ok := observed[pid]; ok {
			return live
		}
		live := original(pid)
		observed[pid] = live
		return live
	}
	t.Cleanup(func() { pidLivenessCheck = original })
	savedLog := log.Writer()
	log.SetOutput(io.Discard)
	t.Cleanup(func() { log.SetOutput(savedLog) })
	before := journalCompatProjection(t, root)
	cmd := journalCompatPython(t, root, journalCompatCompactPython)
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("copy compaction failed: %v %s", err, output)
	} else {
		t.Log(strings.TrimSpace(string(output)))
	}
	after := journalCompatProjection(t, root)
	if !reflect.DeepEqual(before, after) {
		t.Fatalf("HOLD_LEGACY_PROJECTION_MISMATCH: before=%d after=%d", len(before), len(after))
	}
	t.Logf("COPY_PROJECTION_EQUIVALENT: held=%d observed_pid_count=%d", len(before), len(observed))
}

// fak-test:justify why=invariant when=changed:**/debtlane/wave.go
// fak-test:runtime fast est=20ms
func TestJournalCompatUnknownScopeCannotLookFree(t *testing.T) {
	root := t.TempDir()
	journalCompatWrite(t, root, map[string]any{"op": "CHECKPOINT", "leases": []any{map[string]any{"loop_ts": "unknown-scope", "lane": "", "mode": "exclusive", "pid": 0}}})
	if _, err := DiscoverHeldTrees(root); !errors.Is(err, os.ErrInvalid) {
		t.Fatalf("unknown exclusive scope must hold: %v", err)
	}
	journalCompatWrite(t, root, map[string]any{"op": "CHECKPOINT", "leases": []any{map[string]any{"loop_ts": "advisory", "lane": "", "mode": "advisory", "pid": 0}}})
	if _, err := DiscoverHeldTrees(root); err != nil {
		t.Fatalf("native advisory policy changed: %v", err)
	}
}

// fak-test:justify why=invariant when=changed:**/debtlane/wave.go
// fak-test:runtime fast est=50ms
func TestJournalCompatForeignOwnershipHasNoEffects(t *testing.T) {
	root := t.TempDir()
	path := journalCompatWrite(t, root, map[string]any{"op": "ACQUIRE", "loop_ts": "foreign", "lane": "same", "tree": []string{"a"}, "pid": 101})
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, act := range []func() error{
		func() error { return ReleaseTreeLease(root, "same", []string{"a"}, 101) },
		func() error { return TouchLaneLease(root, "same", 101) },
	} {
		if err := act(); !errors.Is(err, os.ErrInvalid) {
			t.Fatalf("foreign owner must hold: %v", err)
		}
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Fatal("foreign owner journal mutated")
	}
}

// fak-test:justify why=invariant when=changed:**/debtlane/wave.go
// fak-test:runtime fast est=50ms
func TestJournalCompatNativeCaseIdentity(t *testing.T) {
	journalCompatAlive(t)
	root := t.TempDir()
	for _, lane := range []string{"Core", "core"} {
		if err := AcquireTreeLease(root, lane, []string{"A"}, 101); err != nil {
			t.Fatal(err)
		}
	}
	held, err := DiscoverHeldLeases(root)
	if err != nil || len(held) != 1 {
		t.Fatalf("lane case forked identity: %v %v", held, err)
	}
	if err := AcquireTreeLease(root, "core", []string{"a"}, 101); err != nil {
		t.Fatal(err)
	}
	held, err = DiscoverHeldLeases(root)
	want := 2
	if runtime.GOOS == "windows" {
		want = 1
	}
	if err != nil || len(held) != want {
		t.Fatalf("platform path identity collapsed: %v %v", held, err)
	}
	if err := ReleaseTreeLease(root, "core", []string{"A"}, 101); err != nil {
		t.Fatal(err)
	}
	held, err = DiscoverHeldLeases(root)
	want = 1
	if runtime.GOOS == "windows" {
		want = 0
	}
	if err != nil || len(held) != want {
		t.Fatalf("release crossed case-sensitive tree identity: %v %v", held, err)
	}
}

// fak-test:justify why=invariant when=changed:**/debtlane/wave.go
// fak-test:runtime fast est=50ms
func TestJournalCompatCanonicalPayloadAndLargeCheckpoint(t *testing.T) {
	journalCompatAlive(t)
	root := t.TempDir()
	lease := map[string]any{"loop_ts": "nested", "lane": "real", "tree": []string{"real/**"}, "pid": 101, "mode": "exclusive"}
	journalCompatWrite(t, root, map[string]any{"op": "ACQUIRE", "loop_ts": "outer", "lane": "wrong", "tree": []string{"wrong/**"}, "pid": 102, "lease": lease})
	held, err := DiscoverHeldLeases(root)
	if err != nil || len(held) != 1 || held[0].Lane != "real" || held[0].PID != 101 || !reflect.DeepEqual(held[0].Tree, []string{"real/**"}) {
		t.Fatalf("canonical nested ownership changed: %v %v", held, err)
	}
	journalCompatWrite(t, root, map[string]any{"op": "CHECKPOINT", "leases": []any{lease}, "evidence_records": []any{map[string]any{"op": "RELEASE", "loop_ts": "nested", "lane": "real", "payload": strings.Repeat("x", 2<<20)}}})
	held, err = DiscoverHeldLeases(root)
	if err != nil || len(held) != 1 {
		t.Fatalf("valid large checkpoint lost holds: %v %v", held, err)
	}
}

// fak-test:justify why=invariant when=changed:internal/debtlane/wave.go
// fak-test:runtime fast est=20ms
func TestJournalCompatPlanReaderUnavailableBlocksDispatch(t *testing.T) {
	plan := PlanWaves(Report{Workspace: t.TempDir()}, WavePlanOptions{AutoDetectHeld: true, OpencodeCommands: true})
	encoded, err := json.Marshal(plan)
	if err != nil {
		t.Fatal(err)
	}
	var observed struct {
		BlockedCode      string   `json:"blocked_code"`
		OpencodeCommands []string `json:"opencode_commands"`
		Waves            []Wave   `json:"waves"`
	}
	if err := json.Unmarshal(encoded, &observed); err != nil {
		t.Fatal(err)
	}
	if observed.BlockedCode != "LEASE_READER_UNAVAILABLE" || len(observed.OpencodeCommands) != 0 || len(observed.Waves) != 0 {
		t.Fatalf("unknown lease state admitted dispatch: %s", encoded)
	}
}
