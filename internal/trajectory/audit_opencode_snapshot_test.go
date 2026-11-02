package trajectory

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime medium est=10s lane=default
func TestOpenCodeSnapshotRunAudit(t *testing.T) {
	source, image := openCodeSnapshotFixture(t, "valid")
	opts := AuditOptions{Sources: []AuditSource{source}, Now: time.Unix(10, 0), SchemaBaseline: &AuditSchemaBaseline{}}
	result, err := RunAudit(opts)
	if err != nil {
		t.Fatal(err)
	}
	want := AuditTokens{InputTokens: 10, OutputTokens: 5, CacheReadTokens: 5, CacheCreateTokens: 1}
	if len(result.Refusals) != 0 || result.ConclusionStatus.BroadEfficiencySupported || result.Summary.Tokens != want || len(result.Transcripts) != 1 {
		t.Fatalf("not exact synthetic accounting: tokens=%+v refusals=%v rows=%d", result.Summary.Tokens, result.Refusals, len(result.Transcripts))
	}
	d := result.Denominators[0]
	if d.FilesDiscovered != 1 || d.FilesScanned != 1 || d.Records != 1 || d.UsageRecordsSeen != 1 || d.UsageRecordsExact != 1 || d.UsageRecordsApplied != 1 {
		t.Fatalf("denominators=%+v", d)
	}
	row := result.Transcripts[0]
	if row.ToolCalls != 1 || row.ToolErrors != 1 || row.OpenCodeSnapshot == nil || row.OpenCodeSnapshot.ReasoningTokens != 2 || row.OpenCodeSnapshot.ModelAttribution != "not_exported" {
		t.Fatalf("snapshot evidence missing: %+v", row)
	}
	if len(row.OpenCodeSnapshot.Tools) != 1 || row.OpenCodeSnapshot.Tools[0].Executed == nil || *row.OpenCodeSnapshot.Tools[0].Executed {
		t.Fatal("explicit executed=false was not preserved")
	}
	if result.Summary.OpenCodeSnapshotCoverage != "exact_usage_and_tool_status_only" || result.Summary.RepeatedFailuresPerSession != nil || result.Summary.RepeatedFailureSemantics != "unknown" || result.Summary.RepeatedFailureNormalization != "unknown" {
		t.Fatal("unobserved summary metrics were promoted to exact zero")
	}
	var out bytes.Buffer
	if err := WriteAuditJSONL(&out, result); err != nil {
		t.Fatal(err)
	}
	for _, secret := range []string{"PRIVATE_PROMPT_SENTINEL", "PRIVATE_TOOL_SECRET", "PRIVATE_ERROR_SECRET", "private-session-id", "private-message-id", "private-tool-id", "private-provider", "private-model", source.Root} {
		if strings.Contains(out.String(), secret) {
			t.Fatal("raw input escaped the aggregate projection")
		}
	}
	again, err := RunAudit(opts)
	if err != nil {
		t.Fatal(err)
	}
	var second bytes.Buffer
	if err := WriteAuditJSONL(&second, again); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out.Bytes(), second.Bytes()) {
		t.Fatal("immutable image is not deterministic")
	}
	retained, err := os.ReadFile(source.Root)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(retained, image) {
		t.Fatal("reader changed the input")
	}
	clone := row.clone()
	*clone.OpenCodeSnapshot.Tools[0].Executed = true
	if *row.OpenCodeSnapshot.Tools[0].Executed {
		t.Fatal("snapshot metadata clone aliases the original")
	}
}

// fak-test:runtime medium est=20s lane=default
func TestOpenCodeSnapshotRefusesUnknownCoverage(t *testing.T) {
	for _, mode := range []string{
		"missing-usage", "fractional", "boolean", "negative", "overflow", "unfinished", "unfinished-tool",
		"unknown-content", "duplicate-tool", "executed-null", "unknown-kind", "identity", "duplicate-json", "deep-json",
		"oversize-json", "aggregate", "fork", "duplicate-seq", "row-limit", "session-limit", "view", "generated", "legacy", "wrong-type",
	} {
		t.Run(mode, func(t *testing.T) {
			source, _ := openCodeSnapshotFixture(t, mode)
			r, err := RunAudit(AuditOptions{Sources: []AuditSource{source}, Now: time.Unix(10, 0), SchemaBaseline: &AuditSchemaBaseline{}})
			if err != nil {
				t.Fatal(err)
			}
			if len(r.Refusals) == 0 || r.ConclusionStatus.BroadEfficiencySupported || r.Summary.Tokens != (AuditTokens{}) || len(r.Transcripts) != 0 {
				t.Fatalf("unknown coverage was promoted: refusals=%v tokens=%+v rows=%d", r.Refusals, r.Summary.Tokens, len(r.Transcripts))
			}
			if len(r.Denominators) != 1 || r.Denominators[0].RefusedRecords == 0 {
				t.Fatal("refusal denominator absent")
			}
		})
	}
	t.Run("compaction-usage", func(t *testing.T) {
		source, _ := openCodeSnapshotFixture(t, "compaction")
		r, err := RunAudit(AuditOptions{Sources: []AuditSource{source}, Now: time.Unix(10, 0), SchemaBaseline: &AuditSchemaBaseline{}})
		if err != nil || len(r.Refusals) != 0 || r.Summary.Tokens.OutputTokens != 5 || r.Summary.Tokens.InputTokens != 10 {
			t.Fatalf("compaction's own usage lost: error=%v result=%+v", err, r.Summary)
		}
	})
}

// fak-test:runtime medium est=5s lane=default
func TestOpenCodeSnapshotInputBinding(t *testing.T) {
	source, image := openCodeSnapshotFixture(t, "valid")
	for _, tc := range []struct {
		name   string
		mutate func(*AuditSource)
	}{
		{"digest", func(s *AuditSource) { s.OpenCodeSnapshotSHA256 = strings.Repeat("0", 64) }},
		{"profile", func(s *AuditSource) { s.OpenCodeSnapshotProfile = "unknown" }},
		{"missing", func(s *AuditSource) { s.Root = filepath.Join(t.TempDir(), "absent") }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := source
			tc.mutate(&s)
			r, err := RunAudit(AuditOptions{Sources: []AuditSource{s}, SchemaBaseline: &AuditSchemaBaseline{}})
			if err != nil || len(r.Refusals) == 0 || len(r.Transcripts) != 0 {
				t.Fatalf("unbound source accepted: %v %+v", err, r.Refusals)
			}
		})
	}
	t.Run("wal-header-no-edit", func(t *testing.T) {
		data := append([]byte(nil), image...)
		data[18] = 2
		data[19] = 2
		path := filepath.Join(t.TempDir(), "standalone.sqlite")
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
		hash := sha256.Sum256(data)
		s, err := NewOpenCodeSnapshotSource(path, hex.EncodeToString(hash[:]), OpenCodeSnapshotProfile)
		if err != nil {
			t.Fatal(err)
		}
		r, err := RunAudit(AuditOptions{Sources: []AuditSource{s}, SchemaBaseline: &AuditSchemaBaseline{}})
		if err != nil || len(r.Refusals) != 1 || r.Refusals[0].Code != "opencode_snapshot_unsupported_image" {
			t.Fatalf("WAL image not refused: %v %+v", err, r.Refusals)
		}
		after, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(data, after) {
			t.Fatal("WAL header was edited")
		}
	})
	t.Run("symlink", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "alias.sqlite")
		if err := os.Symlink(source.Root, path); err != nil {
			t.Skip("host cannot create test symlink")
		}
		s := source
		s.Root = path
		r, err := RunAudit(AuditOptions{Sources: []AuditSource{s}, SchemaBaseline: &AuditSchemaBaseline{}})
		if err != nil || len(r.Refusals) == 0 {
			t.Fatal("symlink was accepted")
		}
	})
	t.Run("cancel", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		r, err := RunAuditWithContext(ctx, AuditOptions{Sources: []AuditSource{source}, SchemaBaseline: &AuditSchemaBaseline{}})
		if !errors.Is(err, context.Canceled) || !r.Partial || len(r.Transcripts) != 0 {
			t.Fatalf("cancel lost: %v", err)
		}
	})
	t.Run("no-multiple-images", func(t *testing.T) {
		_, err := RunAudit(AuditOptions{Sources: []AuditSource{source, source}})
		if err == nil || err.Error() != "OPENCODE_SNAPSHOT_MULTIPLE_IMAGES_UNSUPPORTED" {
			t.Fatal("ambiguous image union allowed")
		}
	})
}

// fak-test:runtime fast est=100ms lane=default
func TestOpenCodeSnapshotPureBounds(t *testing.T) {
	for _, tc := range []struct{ path, pin, profile string }{
		{"", strings.Repeat("a", 64), OpenCodeSnapshotProfile}, {"x", "", OpenCodeSnapshotProfile},
		{"x", strings.Repeat("A", 64), OpenCodeSnapshotProfile}, {"x", strings.Repeat("a", 64), "legacy"},
	} {
		if _, err := NewOpenCodeSnapshotSource(tc.path, tc.pin, tc.profile); err == nil {
			t.Fatal("invalid input accepted")
		}
	}
	for _, v := range [][]int64{{1, 2, 3, 4}, {-1, 0, 0, 0, 0}, {openCodeSnapshotMaxTokens, 1, 0, 0, 0}} {
		if _, ok := openCodeSnapshotTokens(v); ok {
			t.Fatal("invalid token vector")
		}
	}
	writer := &openCodeSnapshotBuffer{limit: 8}
	if _, err := io.Copy(writer, strings.NewReader("0123456789")); err == nil || !writer.overflow || writer.Len() > 8 {
		t.Fatal("output cap bypass")
	}
	header := make([]byte, 512)
	copy(header, "SQLite format 3\x00")
	header[16] = 2
	header[18] = 1
	header[19] = 1
	header[31] = 1
	if !openCodeSnapshotHeader(header) {
		t.Fatal("valid synthetic header rejected")
	}
	bad := append([]byte(nil), header...)
	bad[31] = 2
	if openCodeSnapshotHeader(bad) {
		t.Fatal("header size mismatch accepted")
	}
	if reflect.DeepEqual(AuditSource{Root: "x"}, AuditSource{Root: "x", OpenCodeSnapshotProfile: OpenCodeSnapshotProfile}) {
		t.Fatal("source binding collapsed")
	}
}

// fak-test:runtime medium est=5s lane=default
func TestOpenCodeSnapshotPersistedCoverageAndComposition(t *testing.T) {
	source, _ := openCodeSnapshotFixture(t, "valid")
	r, err := RunAudit(AuditOptions{Sources: []AuditSource{source}, Now: time.Unix(10, 0), SchemaBaseline: &AuditSchemaBaseline{}})
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := WriteAuditJSONL(&out, r); err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(out.Bytes(), []byte(`"opencode_snapshot_coverage":"exact_usage_and_tool_status_only"`)) || !bytes.Contains(out.Bytes(), []byte(`"repeated_failures_per_session":null`)) {
		t.Fatal("persisted summary lost unknown coverage")
	}
	path := filepath.Join(t.TempDir(), "limited.jsonl")
	if err := os.WriteFile(path, out.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadAuditBaseline(path); err == nil || err.Error() != "OPENCODE_SNAPSHOT_BASELINE_UNSUPPORTED" {
		t.Fatal("limited snapshot became a baseline")
	}
	for _, sources := range [][]AuditSource{{source}, {source, {Name: AuditSourceClaude, Root: "must-not-read"}}} {
		_, err := RunAudit(AuditOptions{Sources: sources, Baseline: &AuditSummaryRow{}, SchemaBaseline: &AuditSchemaBaseline{}})
		if err == nil {
			t.Fatal("snapshot API accepted baseline or mixed sources")
		}
	}
	if _, err := RunAudit(AuditOptions{Sources: []AuditSource{source, {Name: AuditSourceClaude, Root: "must-not-read"}}}); err == nil || err.Error() != "OPENCODE_SNAPSHOT_MIXED_SOURCES_UNSUPPORTED" {
		t.Fatal("mixed-source API was not refused before scan")
	}
	if _, err := RunAudit(AuditOptions{Sources: []AuditSource{{Name: AuditSourceClaude, Root: "must-not-read"}}, Baseline: &r.Summary}); err == nil || err.Error() != "OPENCODE_SNAPSHOT_BASELINE_UNSUPPORTED" {
		t.Fatal("limited baseline entered ordinary source audit")
	}
	ordinary := AuditSummaryRow{Schema: AuditSchema, Kind: "summary", Transcripts: 1, RepeatedFailureSemantics: auditRepeatedFailureSemantics, RepeatedFailureNormalization: auditRepeatedFailureNormalization}
	for _, pair := range [][2]AuditSummaryRow{{r.Summary, ordinary}, {ordinary, r.Summary}, {r.Summary, r.Summary}} {
		d := newAuditRepeatedFailureDelta(pair[0], pair[1])
		if d.Comparable || d.Current != nil || d.Baseline != nil || d.NormalizedCurrent != nil || d.NormalizedBaseline != nil || d.ComparabilityStatus != "unsupported_coverage" {
			t.Fatal("unknown repeated failures became comparable")
		}
	}
	ordinaryPath := filepath.Join(t.TempDir(), "ordinary.jsonl")
	var original bytes.Buffer
	if err := WriteAuditJSONL(&original, AuditResult{Summary: ordinary}); err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(original.Bytes(), []byte("opencode_snapshot_coverage")) {
		t.Fatal("ordinary artifact bytes gained a snapshot field")
	}
	if err := os.WriteFile(ordinaryPath, original.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	prior, err := ReadAuditBaseline(ordinaryPath)
	if err != nil {
		t.Fatal(err)
	}
	if d := newAuditRepeatedFailureDelta(ordinary, *prior); !d.Comparable {
		t.Fatal("ordinary qualified baseline comparison changed")
	}
}

func openCodeSnapshotFixture(t *testing.T, mode string) (AuditSource, []byte) {
	t.Helper()
	python, err := exec.LookPath("python3")
	if err != nil {
		python, err = exec.LookPath("python")
	}
	if err != nil {
		t.Skip("Python SQLite snapshot capability unavailable")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, python, "-I", "-S", "-B", "-c", openCodeSnapshotFixturePython, mode)
	image, err := cmd.Output()
	if ee, ok := err.(*exec.ExitError); ok && ee.ExitCode() == 77 {
		t.Skip("Python SQLite serialize/deserialize/setlimit unavailable")
	}
	if err != nil {
		t.Fatalf("synthetic fixture creation failed (%T)", err)
	}
	if len(image) > openCodeSnapshotMaxBytes {
		t.Fatal("synthetic fixture exceeded image budget")
	}
	path := filepath.Join(t.TempDir(), "pinned.sqlite")
	if err := os.WriteFile(path, image, 0600); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(image)
	source, err := NewOpenCodeSnapshotSource(path, hex.EncodeToString(digest[:]), OpenCodeSnapshotProfile)
	if err != nil {
		t.Fatal(err)
	}
	return source, image
}

const openCodeSnapshotFixturePython = `import sys,sqlite3,json
if not hasattr(sqlite3.Connection,"serialize") or not hasattr(sqlite3.Connection,"deserialize") or not hasattr(sqlite3.Connection,"setlimit"):
    sys.exit(77)
mode=sys.argv[1]
con=sqlite3.connect(":memory:")
con.executescript("""
CREATE TABLE session_v2 (
 id TEXT PRIMARY KEY, project_id TEXT, workspace_id TEXT, parent_id TEXT, fork_session_id TEXT,
 fork_boundary TEXT, slug TEXT, directory TEXT, path TEXT, title TEXT, version TEXT, share_url TEXT,
 summary_additions INTEGER, summary_deletions INTEGER, summary_files INTEGER, summary_diffs TEXT,
 metadata TEXT, cost REAL, tokens_input INTEGER, tokens_output INTEGER, tokens_reasoning INTEGER,
 tokens_cache_read INTEGER, tokens_cache_write INTEGER, revert TEXT, permission TEXT, agent TEXT,
 model TEXT, time_created INTEGER, time_updated INTEGER, time_idle INTEGER, time_viewed INTEGER,
 idle_outcome TEXT, time_compacting INTEGER, time_archived INTEGER, time_suspended INTEGER,
 resume_attempts INTEGER
);
CREATE TABLE session_message (
 id TEXT PRIMARY KEY, session_id TEXT, type TEXT, seq INTEGER, time_created INTEGER,
 time_updated INTEGER, data TEXT
);
""")
sid="private-session-id"; mid="private-message-id"
totals=[10,3,2,5,1]
con.execute("INSERT INTO session_v2(id,time_created,time_updated,tokens_input,tokens_output,tokens_reasoning,tokens_cache_read,tokens_cache_write) VALUES (?,?,?,?,?,?,?,?)",(sid,1000,2000,*totals))
data={"agent":"private-agent","model":{"providerID":"private-provider","modelID":"private-model"},
      "tokens":{"input":10,"output":3,"reasoning":2,"cache":{"read":5,"write":1}},
      "time":{"created":1000,"completed":2000},
      "content":[{"type":"text","text":"PRIVATE_PROMPT_SENTINEL"},
                 {"type":"tool","id":"private-tool-id","name":"private-tool",
                  "executed":False,"time":{"created":1000,"completed":2000},
                  "state":{"status":"error","input":{"secret":"PRIVATE_TOOL_SECRET"},
                           "error":{"type":"private-error","message":"PRIVATE_ERROR_SECRET"}}}]}
kind="assistant"
if mode=="missing-usage": del data["tokens"]
elif mode=="fractional": data["tokens"]["input"]=1.5
elif mode=="boolean": data["tokens"]["input"]=True
elif mode=="negative": data["tokens"]["input"]=-1
elif mode=="overflow": data["tokens"]["input"]=9007199254740992
elif mode=="unfinished": del data["time"]["completed"]
elif mode=="unfinished-tool": data["content"][1]["state"]["status"]="running"
elif mode=="unknown-content": data["content"].append({"type":"opaque"})
elif mode=="executed-null": data["content"][1]["executed"]=None
elif mode=="duplicate-tool": data["content"].append(data["content"][1])
elif mode=="unknown-kind": kind="new-schema-kind"
elif mode=="identity": data["id"]="ambiguous-id"
elif mode=="compaction":
    kind="compaction"; data={"tokens":data["tokens"],"status":"completed","summary":"PRIVATE_PROMPT_SENTINEL"}
raw=json.dumps(data)
if mode=="duplicate-json": raw='{"tokens":null,"tokens":{}}'
elif mode=="deep-json": raw='{"padding":'+'['*33+'0'+']'*33+'}'
elif mode=="oversize-json": raw=json.dumps({"padding":"x"*65536})
con.execute("INSERT INTO session_message VALUES(?,?,?,?,?,?,?)",(mid,sid,kind,1,1000,2000,raw))
if mode=="aggregate": con.execute("UPDATE session_v2 SET tokens_input=11")
elif mode=="fork": con.execute("UPDATE session_v2 SET fork_session_id='private-fork'")
elif mode=="duplicate-seq": con.execute("INSERT INTO session_message VALUES(?,?,?,?,?,?,?)",("second",sid,"user",1,1000,2000,"{}"))
elif mode=="row-limit":
    con.executemany("INSERT INTO session_message VALUES(?,?,?,?,?,?,?)",[(str(n),sid,"user",n+2,1000,2000,"{}") for n in range(2048)])
elif mode=="session-limit":
    con.executemany("INSERT INTO session_v2(id,time_updated,tokens_input,tokens_output,tokens_reasoning,tokens_cache_read,tokens_cache_write) VALUES(?,?,?,?,?,?,?)",[(str(n),2000,0,0,0,0,0) for n in range(32)])
elif mode=="view":
    con.execute("ALTER TABLE session_message RENAME TO original_messages")
    con.execute("CREATE VIEW session_message AS SELECT * FROM original_messages")
elif mode=="generated":
    con.execute("ALTER TABLE session_message ADD COLUMN injected TEXT GENERATED ALWAYS AS (id) VIRTUAL")
elif mode=="legacy":
    con.execute("ALTER TABLE session_message RENAME TO message")
elif mode=="wrong-type":
    con.execute("ALTER TABLE session_message RENAME TO original_messages")
    con.execute("CREATE TABLE session_message(id TEXT PRIMARY KEY,session_id TEXT,type TEXT,seq REAL,time_created INTEGER,time_updated INTEGER,data TEXT)")
    con.execute("INSERT INTO session_message SELECT * FROM original_messages")
con.commit()
sys.stdout.buffer.write(con.serialize())
con.close()
`
