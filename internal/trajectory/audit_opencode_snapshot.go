// Copyright (c) 2025 opencode; schema/projection adapted under MIT.
// See OPENCODE-AUDIT-LICENSE.txt and audit_opencode_snapshot.README.md.
package trajectory

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"time"
)

// OpenCodeSnapshotProfile admits one inspected upstream schema, not a claim
// that a caller's image was produced by that commit or contains every session.
const OpenCodeSnapshotProfile = "opencode-2.0.15-session-v2"
const openCodeSnapshotMaxBytes = 16 << 20
const openCodeSnapshotMaxResult = 2 << 20
const openCodeSnapshotMaxTokens int64 = 1<<53 - 1

// OpenCodeSnapshotEvidence preserves the fifth disjoint donor token bucket.
// Canonical OutputTokens is output+reasoning; no token is counted twice.
// IDs are digests, never raw prompts, tool inputs/results, or database paths.
type OpenCodeSnapshotEvidence struct {
	Profile          string                 `json:"profile"`
	ImageSHA256      string                 `json:"image_sha256"`
	ReasoningTokens  int64                  `json:"reasoning_tokens"`
	ModelAttribution string                 `json:"model_attribution"`
	Coverage         string                 `json:"coverage"`
	Tools            []OpenCodeSnapshotTool `json:"tools,omitempty"`
}
type OpenCodeSnapshotTool struct {
	ID       string `json:"call_id_sha256"`
	Status   string `json:"status"`
	Executed *bool  `json:"executed,omitempty"`
}
type openCodeSnapshotUsage struct {
	ID     string  `json:"id"`
	Tokens []int64 `json:"tokens"`
}
type openCodeSnapshotSession struct {
	ID       string                  `json:"id"`
	Records  int                     `json:"records"`
	Types    map[string]int          `json:"types"`
	Seen     int                     `json:"seen"`
	Exact    int                     `json:"exact"`
	Usage    []openCodeSnapshotUsage `json:"usage"`
	Tools    []OpenCodeSnapshotTool  `json:"tools"`
	Refusals map[string]int          `json:"refusals"`
	Tokens   []int64                 `json:"tokens"`
}
type openCodeSnapshotReply struct {
	Error    string                    `json:"error"`
	Profile  string                    `json:"profile"`
	Sessions []openCodeSnapshotSession `json:"sessions"`
}

// NewOpenCodeSnapshotSource selects ONLY the supplied image. Empty input,
// unsupported profiles and absent pins are refusals, never live-home discovery.
func NewOpenCodeSnapshotSource(path, pin, profile string) (AuditSource, error) {
	if path == "" || profile != OpenCodeSnapshotProfile || !openCodeSnapshotDigest(pin) {
		return AuditSource{}, errors.New("OPENCODE_SNAPSHOT_INPUT_REFUSED")
	}
	return AuditSource{Name: AuditSourceOpencode, Root: path, RootLabel: "opencode/pinned-snapshot", OpenCodeSnapshotSHA256: pin, OpenCodeSnapshotProfile: profile}, nil
}
func openCodeSnapshotDigest(s string) bool {
	if len(s) != 64 || s != strings.ToLower(s) {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}
func openCodeSnapshotRefusal(res auditSourceResult, code string) (auditSourceResult, error) {
	res.denominator.RefusedRecords++
	res.refusals = append(res.refusals, AuditRefusalRow{Schema: AuditSchema, Kind: "refusal", Source: AuditSourceOpencode, SourcePath: ".", Code: "opencode_snapshot_" + code, Detail: "Explicit snapshot coverage is incomplete; no live database fallback was attempted"})
	return res, nil
}

func auditOpenCodeSnapshot(ctx context.Context, source AuditSource, opts AuditOptions, res auditSourceResult) (auditSourceResult, error) {
	res.denominator.Root = "opencode/pinned-snapshot"
	res.denominator.OpenCodeSnapshotProfile = OpenCodeSnapshotProfile
	if openCodeSnapshotDigest(source.OpenCodeSnapshotSHA256) {
		res.denominator.OpenCodeSnapshotSHA256 = source.OpenCodeSnapshotSHA256
	}
	res.denominator.TokenSemantics = "pinned OpenCode v2 image; input/cache-read/cache-write are disjoint; canonical output is donor output+reasoning; incomplete sessions are not applied; model/latency/repeated-failure coverage is unknown"
	if source.Name != AuditSourceOpencode || source.OpenCodeSnapshotProfile != OpenCodeSnapshotProfile || !openCodeSnapshotDigest(source.OpenCodeSnapshotSHA256) || opts.UserContains != "" {
		return openCodeSnapshotRefusal(res, "input_refused")
	}
	if err := ctx.Err(); err != nil {
		return res, err
	}
	info, err := os.Lstat(source.Root)
	if err != nil || !info.Mode().IsRegular() || info.Size() < 100 || info.Size() > openCodeSnapshotMaxBytes {
		return openCodeSnapshotRefusal(res, "image_unavailable")
	}
	res.denominator.RootPresent = true
	f, err := os.Open(source.Root)
	if err != nil {
		return openCodeSnapshotRefusal(res, "image_unavailable")
	}
	opened, statErr := f.Stat()
	if statErr != nil || !os.SameFile(info, opened) || !opened.Mode().IsRegular() {
		f.Close()
		return openCodeSnapshotRefusal(res, "image_changed")
	}
	image, readErr := io.ReadAll(io.LimitReader(f, openCodeSnapshotMaxBytes+1))
	after, afterErr := f.Stat()
	closeErr := f.Close()
	if readErr != nil || afterErr != nil || closeErr != nil || len(image) > openCodeSnapshotMaxBytes || int64(len(image)) != info.Size() || !os.SameFile(opened, after) || after.Size() != opened.Size() || !after.ModTime().Equal(opened.ModTime()) {
		return openCodeSnapshotRefusal(res, "image_changed")
	}
	if err := ctx.Err(); err != nil {
		return res, err
	}
	sum := sha256.Sum256(image)
	if hex.EncodeToString(sum[:]) != source.OpenCodeSnapshotSHA256 {
		return openCodeSnapshotRefusal(res, "digest_mismatch")
	}
	if !openCodeSnapshotHeader(image) {
		return openCodeSnapshotRefusal(res, "unsupported_image")
	}
	// No SQLite connection ever sees the source pathname, journal, or WAL. Python
	// receives the pinned bytes on stdin and deserializes only into :memory:.
	python, err := exec.LookPath("python3")
	if err != nil {
		python, err = exec.LookPath("python")
	}
	if err != nil {
		return openCodeSnapshotRefusal(res, "sqlite_unavailable")
	}
	cutoff := int64(0)
	if opts.Since > 0 {
		cutoff = opts.Now.Add(-opts.Since).UnixMilli()
		if cutoff < 0 {
			cutoff = 0
		}
	}
	reply, err := runOpenCodeSnapshot(ctx, python, image, cutoff)
	if err != nil {
		if ctx.Err() != nil {
			return res, ctx.Err()
		}
		return openCodeSnapshotRefusal(res, "reader_refused")
	}
	if reply.Error != "" {
		// The child has only fixed reason codes, but do not accept arbitrary data as
		// a public diagnostic if a different interpreter is selected by the host.
		if openCodeSnapshotReason(reply.Error) {
			return openCodeSnapshotRefusal(res, reply.Error)
		}
		return openCodeSnapshotRefusal(res, "reader_contract")
	}
	if reply.Profile != OpenCodeSnapshotProfile || len(reply.Sessions) > 32 {
		return openCodeSnapshotRefusal(res, "reader_contract")
	}
	var total AuditTokens
	totalRecords, totalTools := 0, 0
	for _, s := range reply.Sessions {
		if !openCodeSnapshotDigest(s.ID) || s.Records < 0 || s.Seen < 0 || s.Exact < 0 || s.Exact > s.Seen || s.Seen > s.Records || len(s.Usage) != s.Exact || len(s.Tools) > 4096 {
			return openCodeSnapshotRefusal(res, "reader_contract")
		}
		totalRecords += s.Records
		totalTools += len(s.Tools)
		if totalRecords > 2048 || totalTools > 4096 {
			return openCodeSnapshotRefusal(res, "reader_contract")
		}
		res.denominator.FilesDiscovered++
		res.denominator.FilesScanned++
		res.denominator.Records += s.Records
		res.denominator.UsageRecordsSeen += s.Seen
		res.denominator.UsageRecordsExact += s.Exact
		typeCount := 0
		for kind, n := range s.Types {
			if !openCodeSnapshotRecordType(kind) || n < 0 {
				return openCodeSnapshotRefusal(res, "reader_contract")
			}
			typeCount += n
			res.denominator.RecordTypes[kind] += n
		}
		if typeCount > s.Records {
			return openCodeSnapshotRefusal(res, "reader_contract")
		}
		if len(s.Refusals) > 0 {
			reasons := make([]string, 0, len(s.Refusals))
			for reason, n := range s.Refusals {
				if !openCodeSnapshotReason(reason) || n < 1 || n > 4096 {
					return openCodeSnapshotRefusal(res, "reader_contract")
				}
				reasons = append(reasons, reason)
			}
			sort.Strings(reasons)
			res.denominator.RefusedRecords++
			res.refusals = append(res.refusals, AuditRefusalRow{Schema: AuditSchema, Kind: "refusal", Source: AuditSourceOpencode, SourcePath: "snapshot#" + s.ID, Code: "opencode_snapshot_session_incomplete", Detail: "Entire session excluded; reasons=" + strings.Join(reasons, ",")})
			continue
		}
		row := AuditTranscriptRow{Schema: AuditSchema, Kind: "transcript", Source: AuditSourceOpencode, TranscriptID: s.ID, SourcePath: "snapshot#" + s.ID, Models: []string{}, BuildIdentities: []AuditBuildIdentity{}, usageByID: map[string]AuditTokens{}, failureCounts: map[string]int{}, OpenCodeSnapshot: &OpenCodeSnapshotEvidence{Profile: OpenCodeSnapshotProfile, ImageSHA256: source.OpenCodeSnapshotSHA256, ModelAttribution: "not_exported", Coverage: "exact_usage_and_tool_status_only"}, fragmentDigest: source.OpenCodeSnapshotSHA256}
		expected, ok := openCodeSnapshotTokens(s.Tokens)
		if !ok {
			return openCodeSnapshotRefusal(res, "reader_contract")
		}
		for _, u := range s.Usage {
			tok, ok := openCodeSnapshotTokens(u.Tokens)
			if !ok || !openCodeSnapshotDigest(u.ID) {
				return openCodeSnapshotRefusal(res, "reader_contract")
			}
			if _, exists := row.usageByID[u.ID]; exists {
				return openCodeSnapshotRefusal(res, "reader_contract")
			}
			row.usageByID[u.ID] = tok
			row.Tokens.add(tok)
			row.UsageRecords++
			if row.Tokens.accountedTotal() > openCodeSnapshotMaxTokens {
				return openCodeSnapshotRefusal(res, "usage_overflow")
			}
		}
		if row.Tokens != expected {
			return openCodeSnapshotRefusal(res, "reader_contract")
		}
		row.OpenCodeSnapshot.ReasoningTokens = s.Tokens[2]
		toolIDs := map[string]bool{}
		for _, tool := range s.Tools {
			if !openCodeSnapshotDigest(tool.ID) || toolIDs[tool.ID] || (tool.Status != "completed" && tool.Status != "error") {
				return openCodeSnapshotRefusal(res, "reader_contract")
			}
			toolIDs[tool.ID] = true
			row.ToolCalls++
			if tool.Status == "error" {
				row.ToolErrors++
			}
			row.OpenCodeSnapshot.Tools = append(row.OpenCodeSnapshot.Tools, tool)
		}
		total.add(row.Tokens)
		if total.accountedTotal() > openCodeSnapshotMaxTokens {
			return openCodeSnapshotRefusal(res, "usage_overflow")
		}
		res.denominator.UsageRecordsApplied += row.UsageRecords
		res.transcripts = append(res.transcripts, row)
	}
	sort.Slice(res.transcripts, func(i, j int) bool { return res.transcripts[i].TranscriptID < res.transcripts[j].TranscriptID })
	return res, nil
}
func openCodeSnapshotReason(s string) bool {
	switch s {
	case "invalid_integer", "usage_shape", "usage_overflow", "duplicate_json_key", "invalid_number", "json_size", "json_depth", "json_nodes", "json_shape", "identity_shape", "sqlite_capability_unavailable", "image_limit", "sqlite_limit_unavailable", "unsupported_schema", "session_limit", "record_limit", "fork_accounting_unsupported", "query_deadline", "duplicate_message", "unknown_record", "ambiguous_identity", "content_shape", "unknown_content", "duplicate_tool", "tool_state", "tool_limit", "unfinished_tool", "unfinished_record", "invalid_record", "usage_aggregate_mismatch", "snapshot_query_refused", "result_limit":
		return true
	}
	return false
}
func openCodeSnapshotRecordType(s string) bool {
	switch s {
	case "agent-switched", "model-switched", "location-switched", "user", "synthetic", "system", "skill", "shell", "assistant", "compaction", "idle", "unknown":
		return true
	}
	return false
}
func openCodeSnapshotTokens(v []int64) (AuditTokens, bool) {
	if len(v) != 5 {
		return AuditTokens{}, false
	}
	var sum int64
	for _, n := range v {
		if n < 0 || n > openCodeSnapshotMaxTokens-sum {
			return AuditTokens{}, false
		}
		sum += n
	}
	return AuditTokens{InputTokens: v[0], OutputTokens: v[1] + v[2], CacheReadTokens: v[3], CacheCreateTokens: v[4]}, true
}
func openCodeSnapshotHeader(b []byte) bool {
	if len(b) < 100 || !bytes.Equal(b[:16], []byte("SQLite format 3\x00")) || b[18] != 1 || b[19] != 1 || b[20] != 0 {
		return false
	}
	page := uint64(binary.BigEndian.Uint16(b[16:18]))
	if page == 1 {
		page = 65536
	}
	if page < 512 || page > 65536 || page&(page-1) != 0 {
		return false
	}
	count := uint64(binary.BigEndian.Uint32(b[28:32]))
	return count > 0 && count*page == uint64(len(b))
}

type openCodeSnapshotBuffer struct {
	buf      bytes.Buffer
	limit    int
	overflow bool
}

func (b *openCodeSnapshotBuffer) Len() int      { return b.buf.Len() }
func (b *openCodeSnapshotBuffer) Bytes() []byte { return b.buf.Bytes() }
func (b *openCodeSnapshotBuffer) Write(p []byte) (int, error) {
	if len(p) > b.limit-b.Len() {
		b.overflow = true
		return 0, errors.New("snapshot output limit")
	}
	return b.buf.Write(p)
}
func runOpenCodeSnapshot(ctx context.Context, python string, image []byte, cutoff int64) (openCodeSnapshotReply, error) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	// Strictly static SQL/script; no image path, shell, extension or credentials
	// are placed in argv. -I/-S/-B disable user-site/env-driven Python startup.
	cmd := exec.CommandContext(ctx, python, "-I", "-S", "-B", "-c", openCodeSnapshotPython, strconv.FormatInt(cutoff, 10))
	cmd.Stdin = bytes.NewReader(image)
	out := &openCodeSnapshotBuffer{limit: openCodeSnapshotMaxResult}
	stderr := &openCodeSnapshotBuffer{limit: 4096}
	cmd.Stdout = out
	cmd.Stderr = stderr
	cmd.WaitDelay = time.Second
	if err := cmd.Run(); err != nil || out.overflow || stderr.overflow {
		return openCodeSnapshotReply{}, errors.New("snapshot reader refused")
	}
	var reply openCodeSnapshotReply
	dec := json.NewDecoder(bytes.NewReader(out.Bytes()))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&reply); err != nil {
		return reply, errors.New("snapshot reader contract")
	}
	if dec.Decode(new(any)) != io.EOF {
		return reply, errors.New("snapshot reader trailing data")
	}
	return reply, nil
}

const openCodeSnapshotPython = `import sys,json,sqlite3,time,hashlib,re
PROFILE="opencode-2.0.15-session-v2"
MAX_ROWS=2048
MAX_SESSIONS=32
MAX_TEXT=65536
MAX_TEXT_TOTAL=8388608
MAX_TOTAL=9007199254740991
class Refused(Exception):
    pass
def fail(code):
    raise Refused(code)
def digest(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
def number(x):
    if type(x) is not int or x<0 or x>MAX_TOTAL:
        fail("invalid_integer")
    return x
def tokens(x):
    if not isinstance(x,dict) or set(x)!={"input","output","reasoning","cache"}:
        fail("usage_shape")
    c=x["cache"]
    if not isinstance(c,dict) or set(c)!={"read","write"}:
        fail("usage_shape")
    return [number(x["input"]),number(x["output"]),number(x["reasoning"]),number(c["read"]),number(c["write"])]
def add(a,b):
    z=[x+y for x,y in zip(a,b)]
    if sum(z)>MAX_TOTAL:
        fail("usage_overflow")
    return z
def pairs(p):
    d={}
    for k,v in p:
        if k in d: fail("duplicate_json_key")
        d[k]=v
    return d
def decode(raw):
    if not isinstance(raw,str) or len(raw.encode("utf-8"))>MAX_TEXT:
        fail("json_size")
    depth=0; nodes=0; quoted=False; escaped=False
    for c in raw:
        if quoted:
            if escaped: escaped=False
            elif c=="\\": escaped=True
            elif c=='"': quoted=False
        elif c=='"': quoted=True; nodes+=1
        elif c in "[{":
            depth+=1; nodes+=1
            if depth>32: fail("json_depth")
        elif c in "]}": depth-=1
        elif c in ",:": nodes+=1
        if nodes>8192: fail("json_nodes")
    if depth!=0 or quoted: fail("json_shape")
    x=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:fail("invalid_number"))
    if not isinstance(x,dict): fail("json_shape")
    return x
def valid_id(x):
    if not isinstance(x,str) or not 1<=len(x)<=256: fail("identity_shape")
    return x
SESSION={
"id":"TEXT","project_id":"TEXT","workspace_id":"TEXT","parent_id":"TEXT","fork_session_id":"TEXT",
"fork_boundary":"TEXT","slug":"TEXT","directory":"TEXT","path":"TEXT","title":"TEXT","version":"TEXT",
"share_url":"TEXT","summary_additions":"INTEGER","summary_deletions":"INTEGER","summary_files":"INTEGER",
"summary_diffs":"TEXT","metadata":"TEXT","cost":"REAL","tokens_input":"INTEGER","tokens_output":"INTEGER",
"tokens_reasoning":"INTEGER","tokens_cache_read":"INTEGER","tokens_cache_write":"INTEGER","revert":"TEXT",
"permission":"TEXT","agent":"TEXT","model":"TEXT","time_created":"INTEGER","time_updated":"INTEGER",
"time_idle":"INTEGER","time_viewed":"INTEGER","idle_outcome":"TEXT","time_compacting":"INTEGER",
"time_archived":"INTEGER","time_suspended":"INTEGER","resume_attempts":"INTEGER"}
MESSAGE={"id":"TEXT","session_id":"TEXT","type":"TEXT","seq":"INTEGER","time_created":"INTEGER","time_updated":"INTEGER","data":"TEXT"}
KNOWN={"agent-switched","model-switched","location-switched","user","synthetic","system","skill","shell","assistant","compaction","idle"}
def main():
    if not hasattr(sqlite3.Connection,"deserialize") or not hasattr(sqlite3.Connection,"setlimit"):
        fail("sqlite_capability_unavailable")
    image=sys.stdin.buffer.read(16777217)
    if len(image)>16777216: fail("image_limit")
    cutoff=int(sys.argv[1])
    deadline=time.monotonic()+3.0
    con=sqlite3.connect(":memory:",isolation_level=None)
    try:
        for k,v in [(sqlite3.SQLITE_LIMIT_ATTACHED,0),(sqlite3.SQLITE_LIMIT_LENGTH,131072),
                    (sqlite3.SQLITE_LIMIT_SQL_LENGTH,4096),(sqlite3.SQLITE_LIMIT_COLUMN,128),
                    (sqlite3.SQLITE_LIMIT_EXPR_DEPTH,32),(sqlite3.SQLITE_LIMIT_COMPOUND_SELECT,1),
                    (sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER,64)]:
            con.setlimit(k,v)
            if con.getlimit(k)>v: fail("sqlite_limit_unavailable")
        con.deserialize(image)
        con.execute("PRAGMA trusted_schema=OFF")
        con.execute("PRAGMA query_only=ON")
        con.execute("PRAGMA temp_store=MEMORY")
        con.execute("PRAGMA cache_size=-2048")
        if con.execute("PRAGMA trusted_schema").fetchone()!=(0,) or con.execute("PRAGMA query_only").fetchone()!=(1,):
            fail("sqlite_capability_unavailable")
        if hasattr(con,"enable_load_extension"): con.enable_load_extension(False)
        steps=0
        def progress():
            nonlocal steps
            steps+=1000
            return int(steps>4000000 or time.monotonic()>deadline)
        con.set_progress_handler(progress,1000)
        def authorize(action,a,b,db,trigger):
            if trigger is not None: return sqlite3.SQLITE_DENY
            if action==sqlite3.SQLITE_SELECT: return sqlite3.SQLITE_OK
            if action==sqlite3.SQLITE_READ and db=="main" and a in {"sqlite_master","sqlite_schema","session_v2","session_message"}:
                return sqlite3.SQLITE_OK
            if action==sqlite3.SQLITE_PRAGMA and a=="table_xinfo" and b in {"session_v2","session_message"}:
                return sqlite3.SQLITE_OK
            if action==sqlite3.SQLITE_FUNCTION and b in {"count","max","sum","length"}:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        con.set_authorizer(authorize)
        for table,expected in [("session_v2",SESSION),("session_message",MESSAGE)]:
            entry=con.execute("SELECT type,sql FROM sqlite_master WHERE name=?",(table,)).fetchall()
            if len(entry)!=1 or entry[0][0]!="table" or not isinstance(entry[0][1],str):
                fail("unsupported_schema")
            ddl=entry[0][1]
            if not re.match(r"^\s*CREATE\s+TABLE\b",ddl,re.I) or re.search(r"\b(?:VIRTUAL|GENERATED)\b",ddl,re.I):
                fail("unsupported_schema")
            cols=con.execute('PRAGMA table_xinfo("'+table+'")').fetchall()
            if len(cols)!=len(expected) or {c[1]:c[2].upper() for c in cols}!=expected:
                fail("unsupported_schema")
            if any(c[6]!=0 for c in cols) or [c[1] for c in cols if c[5]]!=["id"]:
                fail("unsupported_schema")
        sessions=con.execute("SELECT id,fork_session_id,time_updated,tokens_input,tokens_output,tokens_reasoning,tokens_cache_read,tokens_cache_write FROM session_v2 WHERE time_updated>=? ORDER BY id LIMIT 33",(cutoff,)).fetchall()
        if len(sessions)>MAX_SESSIONS: fail("session_limit")
        rows=0; total_text=0
        for sess in sessions:
            sid=valid_id(sess[0]); number(sess[2])
            n,m,total=con.execute("SELECT count(*),max(length(CAST(data AS BLOB))),sum(length(CAST(data AS BLOB))) FROM session_message WHERE session_id=?",(sid,)).fetchone()
            rows+=number(n); total_text+=number(total or 0)
            if rows>MAX_ROWS or number(m or 0)>MAX_TEXT or total_text>MAX_TEXT_TOTAL: fail("record_limit")
        result=[]; total_tools=0; actual_text=0; global_usage=[0]*5
        for sess in sessions:
            sid=sess[0]; expected=[number(x) for x in sess[3:8]]
            out={"id":digest(sid),"records":0,"types":{},"seen":0,"exact":0,"usage":[],"tools":[],"refusals":{}}
            def refuse(code):
                out["refusals"][code]=out["refusals"].get(code,0)+1
            if sess[1] is not None: refuse("fork_accounting_unsupported")
            sums=[0]*5; last_seq=-1; ids=set(); tool_ids=set()
            for mid,seq,kind,raw in con.execute("SELECT id,seq,type,data FROM session_message WHERE session_id=? ORDER BY seq,id LIMIT 2049",(sid,)):
                out["records"]+=1
                if time.monotonic()>deadline: fail("query_deadline")
                if isinstance(raw,str):
                    actual_text+=len(raw.encode("utf-8"))
                    if actual_text>MAX_TEXT_TOTAL: fail("record_limit")
                try:
                    valid_id(mid); number(seq)
                    if seq<=last_seq or mid in ids: fail("duplicate_message")
                    last_seq=seq; ids.add(mid)
                    label=kind if kind in KNOWN else "unknown"
                    out["types"][label]=out["types"].get(label,0)+1
                    if kind not in KNOWN: fail("unknown_record")
                    data=decode(raw)
                    if "id" in data or "type" in data: fail("ambiguous_identity")
                    if kind not in {"assistant","compaction"}: continue
                    out["seen"]+=1
                    if kind=="assistant":
                        stamp=data.get("time")
                        if not isinstance(stamp,dict) or "completed" not in stamp: fail("unfinished_record")
                        number(stamp["completed"])
                    elif data.get("status") not in {"completed","failed"}:
                        fail("unfinished_record")
                    ts=tokens(data.get("tokens"))
                    sums=add(sums,ts)
                    out["exact"]+=1
                    out["usage"].append({"id":digest(json.dumps([sid,mid],separators=(",",":"))),"tokens":ts})
                    if kind=="compaction": continue
                    content=data.get("content")
                    if not isinstance(content,list) or len(content)>128: fail("content_shape")
                    for item in content:
                        if not isinstance(item,dict): fail("content_shape")
                        if item.get("type") in {"text","reasoning"}: continue
                        if item.get("type")!="tool": fail("unknown_content")
                        tid=valid_id(item.get("id"))
                        if tid in tool_ids: fail("duplicate_tool")
                        tool_ids.add(tid)
                        st=item.get("state")
                        if not isinstance(st,dict) or st.get("status") not in {"streaming","running","completed","error"}:
                            fail("tool_state")
                        executed=item.get("executed")
                        if "executed" in item and type(executed) is not bool: fail("tool_state")
                        total_tools+=1
                        if total_tools>4096: fail("tool_limit")
                        out["tools"].append({"call_id_sha256":digest(json.dumps([sid,mid,tid],separators=(",",":"))),"status":st["status"],"executed":executed})
                        if st["status"] in {"streaming","running"}: refuse("unfinished_tool")
                except Refused as e: refuse(str(e))
                except (ValueError,TypeError,KeyError,RecursionError,OverflowError): refuse("invalid_record")
            if sums!=expected: refuse("usage_aggregate_mismatch")
            if not out["refusals"]: global_usage=add(global_usage,sums)
            out["tokens"]=sums
            result.append(out)
        return {"profile":PROFILE,"sessions":result}
    finally:
        con.close()
try:
    obj=main()
except Refused as e:
    obj={"error":str(e)}
except BaseException:
    obj={"error":"snapshot_query_refused"}
encoded=json.dumps(obj,separators=(",",":"),ensure_ascii=True)
if len(encoded)>2097152: encoded='{"error":"result_limit"}'
sys.stdout.write(encoded)
`
