package dgxbridge

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Typed readback failures, so SelfTest and orchestrators can tell *why* a run
// could not be read back rather than reporting a generic timeout. See SelfTest.
var (
	// ErrNoSessionTranscript means !dump never produced a transcript.jsonl bound to
	// this control thread (the session isn't uploading, or only wrong-session
	// transcripts are present). This is the operator-side bridge symptom.
	ErrNoSessionTranscript = errors.New("dgxbridge: no transcript bound to this control thread")
	// ErrSentinelMissing means a transcript for *this* thread was found, but the
	// command's completion sentinel never appeared in it (shell wedged / command
	// not executed / output not captured within the timeout).
	ErrSentinelMissing = errors.New("dgxbridge: completion sentinel not found in this thread's transcript")
)

// Bridge is the high-level RPC over a single control-session thread. It issues
// commands through the PTY shell and reads results back via the !dump transcript
// file, which is immune to the PTY mirror's msg_too_long wedge.
type Bridge struct {
	Client   *Client
	Channel  string
	ThreadTS string

	// SessionID names a hub session (e.g. "default-1"). When set, the bridge speaks
	// the MULTI-session control_hub protocol: stdin still goes to ThreadTS, but the
	// hub control verbs (!dump/!clear) take the id and MUST be posted at channel
	// top-level (the hub ignores commands inside a live session thread), and the
	// uploaded transcript is named "<SessionID>-transcript.jsonl". When empty, the
	// bridge speaks the legacy single-session protocol (bare !dump in-thread).
	SessionID string

	// Tunables (zero values get sane defaults in normalize()).
	Settle   time.Duration // wait after posting a command before the first !dump
	DumpWait time.Duration // wait after !dump for the upload to land
	Timeout  time.Duration // overall per-Exec ceiling

	// ScratchDir is where downloaded transcripts are cached (for debugging).
	ScratchDir string

	now func() time.Time // injectable clock for tests

	// classified caches fileID -> thread_ts for transcripts we've already downloaded
	// and found to belong to a *different* control session, so the Exec poll loop
	// never re-downloads the same wrong-session transcript on every iteration.
	classified map[string]string
}

func (b *Bridge) normalize() {
	if b.Channel == "" {
		b.Channel = DefaultChannel
	}
	if b.Settle == 0 {
		b.Settle = 6 * time.Second
	}
	if b.DumpWait == 0 {
		b.DumpWait = 9 * time.Second
	}
	if b.Timeout == 0 {
		b.Timeout = 240 * time.Second
	}
	if b.now == nil {
		b.now = time.Now
	}
	if b.classified == nil {
		b.classified = map[string]string{}
	}
}

// ansi strips terminal escape sequences and bare control bytes so PTY-rendered
// output collapses to readable text.
var ansi = regexp.MustCompile(`\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b[=>P]|\x1b\(B|[\r\x00-\x08\x0b\x0c\x0e-\x1f]`)

// transcriptEvent is the subset of the bridge's JSONL we care about.
type transcriptEvent struct {
	Event    string `json:"event"`
	Text     string `json:"text"`
	ThreadTS string `json:"thread_ts"`
}

// transcriptThreadTS returns the control-session thread the transcript belongs to.
// Every bridge event carries "thread_ts"; the bridge_start event is the authoritative
// session identity, so it is preferred, falling back to the first event that has one.
// Returns "" if the JSONL carries no thread_ts at all (an old-format transcript).
func transcriptThreadTS(jsonl []byte) string {
	sc := bufio.NewScanner(bytes.NewReader(jsonl))
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	var fallback string
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var ev transcriptEvent
		if err := json.Unmarshal(line, &ev); err != nil {
			continue
		}
		if ev.ThreadTS == "" {
			continue
		}
		if ev.Event == "bridge_start" {
			return ev.ThreadTS
		}
		if fallback == "" {
			fallback = ev.ThreadTS
		}
	}
	return fallback
}

// transcriptStdout reconstructs the shell's combined stdout from a raw transcript
// JSONL by concatenating only process_output text (NOT stdin echoes), then stripping
// ANSI. Reading only process_output is what cleanly separates real output from the
// command-echo that defeated the line-scrape approach.
func transcriptStdout(jsonl []byte) string {
	var sb strings.Builder
	sc := bufio.NewScanner(bytes.NewReader(jsonl))
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var ev transcriptEvent
		if err := json.Unmarshal(line, &ev); err != nil {
			continue
		}
		if ev.Event == "process_output" && ev.Text != "" {
			sb.WriteString(ev.Text)
		}
	}
	clean := ansi.ReplaceAllString(sb.String(), "")
	clean = htmlUnescape(clean)
	return clean
}

// htmlUnescape reverses the &gt; &lt; &amp; that Slack applies to message text.
// (process_output is raw, but file-bytes round-trips can carry these.)
func htmlUnescape(s string) string {
	r := strings.NewReplacer("&gt;", ">", "&lt;", "<", "&amp;", "&")
	return r.Replace(s)
}

// extractBlock returns the text the shell printed between the START sentinel
// (a line that is exactly `<nonce>`) and the END sentinel (a line that is
// exactly `<nonce>_DONE`). Returns ok=false if the DONE sentinel line is absent
// (command not finished / transcript predates completion).
//
// Sentinels are matched as WHOLE LINES, not substrings. The PTY echoes the
// command line itself (which contains both `echo <nonce>` and `echo
// <nonce>_DONE` inline), and — critically — the command's OUTPUT may itself
// contain the nonce as a substring (e.g. SelfTest echoes `SELFTEST_<nonce>`).
// A substring scan latches onto those occurrences and drops the real output;
// requiring the sentinel to BE the line is immune to both.
func extractBlock(stdout, nonce string) (string, bool) {
	done := nonce + "_DONE"
	lines := strings.Split(stdout, "\n")
	// Anchor on the LAST line equal to nonce_DONE (the output sentinel, printed
	// after the command echo), then the last bare-nonce line before it.
	doneLine := -1
	for i := len(lines) - 1; i >= 0; i-- {
		if strings.TrimSpace(lines[i]) == done {
			doneLine = i
			break
		}
	}
	if doneLine < 0 {
		return "", false
	}
	startLine := -1
	for i := doneLine - 1; i >= 0; i-- {
		if strings.TrimSpace(lines[i]) == nonce {
			startLine = i
			break
		}
	}
	if startLine < 0 {
		return "", false
	}
	return strings.TrimSpace(strings.Join(lines[startLine+1:doneLine], "\n")), true
}

// fileMarkers wrap file-byte readbacks; chosen to never collide with shell tokens.
func fileStart(nonce string) string { return nonce + "_FS" }
func fileEnd(nonce string) string   { return nonce + "_FE" }

// Exec runs a command on the DGX shell and returns its combined stdout/stderr,
// reading the result from the !dump transcript (raceless: it re-dumps until the
// downloaded transcript actually contains the completion sentinel).
func (b *Bridge) Exec(ctx context.Context, command string) (string, error) {
	b.normalize()
	nonce := b.nonce()

	// Hub (multi-session) vs legacy (single-session) control protocol. In hub mode
	// the !dump verb takes the session id and is posted at channel top-level (the hub
	// ignores commands inside a live session thread); the PTY-buffer !clear is a
	// legacy-only, pty-mode concern and would risk truncating the transcript we are
	// about to read, so we skip it for pipe-mode hub sessions.
	dumpText := "!dump"
	dumpThread := b.ThreadTS // legacy: dump in-thread
	if b.SessionID != "" {
		dumpText = "!dump " + b.SessionID
		dumpThread = "" // hub: control verbs only run at top level
	} else {
		_, _ = b.Client.Post(ctx, b.Channel, b.ThreadTS, "!clear")
		time.Sleep(2 * time.Second)
	}

	full := fmt.Sprintf("echo %s; { %s ; } ; echo %s_DONE", nonce, command, nonce)
	if _, err := b.Client.Post(ctx, b.Channel, b.ThreadTS, full); err != nil {
		return "", fmt.Errorf("post command: %w", err)
	}

	deadline := b.now().Add(b.Timeout)
	var lastErr error = ErrNoSessionTranscript // until we've at least seen our thread's transcript
	time.Sleep(b.Settle)
	for b.now().Before(deadline) {
		if _, err := b.Client.Post(ctx, b.Channel, dumpThread, dumpText); err != nil {
			lastErr = fmt.Errorf("post !dump: %w", err)
			time.Sleep(b.DumpWait)
			continue
		}
		time.Sleep(b.DumpWait)
		// Read back the transcript bound to *our* control thread, not merely the newest
		// channel transcript (which may be a different session's — the root cause of the
		// "completion sentinel not found" wedge).
		jsonl, seen, err := b.transcriptForThread(ctx)
		if err != nil {
			lastErr = err
			continue
		}
		if jsonl == nil {
			lastErr = fmt.Errorf("%w (saw threads %v)", ErrNoSessionTranscript, seen)
			continue
		}
		out := transcriptStdout(jsonl)
		if block, ok := extractBlock(out, nonce); ok {
			return block, nil
		}
		// Our transcript is present but the command hasn't completed in it yet.
		lastErr = ErrSentinelMissing
	}
	return "", fmt.Errorf("exec: %w", lastErr)
}

// ReadFile returns the bytes of a file on the DGX by having the shell base64 it on
// one line between markers, then reassembling — robust to PTY chunking because the
// payload is pure base64 (any interleaved redraw bytes are stripped by the alphabet
// filter). For large/binary artifacts prefer PullArtifact (bridge file upload).
func (b *Bridge) ReadFile(ctx context.Context, remotePath string) ([]byte, error) {
	b.normalize()
	nonce := b.nonce()
	cmd := fmt.Sprintf("echo %s; base64 -w0 %s; echo; echo %s",
		fileStart(nonce), shellQuote(remotePath), fileEnd(nonce))
	out, err := b.Exec(ctx, cmd)
	if err != nil {
		return nil, err
	}
	return decodeFileBlock(out, nonce)
}

func decodeFileBlock(out, nonce string) ([]byte, error) {
	s := strings.Index(out, fileStart(nonce))
	e := strings.Index(out, fileEnd(nonce))
	if s < 0 || e < 0 || e <= s {
		return nil, fmt.Errorf("file markers not found in output")
	}
	blob := out[s+len(fileStart(nonce)) : e]
	// keep only base64 alphabet
	var bb strings.Builder
	for _, r := range blob {
		if (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '+' || r == '/' || r == '=' {
			bb.WriteRune(r)
		}
	}
	b64 := bb.String()
	if pad := len(b64) % 4; pad != 0 {
		b64 += strings.Repeat("=", 4-pad)
	}
	dec, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return nil, fmt.Errorf("base64 decode (len=%d): %w", len(b64), err)
	}
	return dec, nil
}

// PullArtifact has the DGX upload a file to the channel via the remote slack CLI is
// not assumed here; instead we read it via ReadFile (base64) which needs no remote
// tooling. Kept as a named helper for the orchestrator's intent.
func (b *Bridge) PullArtifact(ctx context.Context, remotePath, localPath string) error {
	data, err := b.ReadFile(ctx, remotePath)
	if err != nil {
		return err
	}
	return os.WriteFile(localPath, data, 0o644)
}

// Ship writes localPath's bytes onto the DGX at remotePath.
func (b *Bridge) Ship(ctx context.Context, localPath, remotePath string) error {
	data, err := os.ReadFile(localPath)
	if err != nil {
		return err
	}
	return b.ShipBytes(ctx, data, remotePath)
}

// ShipBytes writes data onto the DGX at remotePath by streaming base64 in chunks
// (each well under any per-message limit), then decoding remotely. This sidesteps
// heredoc/quoting hazards entirely — useful for shipping a generated script.
func (b *Bridge) ShipBytes(ctx context.Context, data []byte, remotePath string) error {
	b.normalize()
	b64 := base64.StdEncoding.EncodeToString(data)
	// truncate remote, then append chunks, then decode
	if _, err := b.Exec(ctx, fmt.Sprintf(": > %s.b64", shellQuote(remotePath))); err != nil {
		return err
	}
	const chunk = 2000
	for i := 0; i < len(b64); i += chunk {
		end := i + chunk
		if end > len(b64) {
			end = len(b64)
		}
		piece := b64[i:end]
		if _, err := b.Exec(ctx, fmt.Sprintf("printf %%s %s >> %s.b64", piece, shellQuote(remotePath))); err != nil {
			return err
		}
	}
	_, err := b.Exec(ctx, fmt.Sprintf("base64 -d %s.b64 > %s && rm -f %s.b64",
		shellQuote(remotePath), shellQuote(remotePath), shellQuote(remotePath)))
	return err
}

// transcriptForThread downloads the transcript.jsonl that belongs to *this* control
// thread, identified by each candidate's thread_ts. It returns (jsonl, seenThreads, err):
//
//   - jsonl != nil: the newest transcript bound to our thread.
//   - jsonl == nil, err == nil: no transcript bound to our thread is present yet;
//     seenThreads lists the other sessions' threads found (for diagnostics).
//   - err != nil: a hard files.list / download failure.
//
// Files already known to belong to a different session are cached (b.classified) and
// skipped on later polls, so a repeated poll only downloads transcripts it has not yet
// classified — usually just the one our own !dump created. When NO candidate carries any
// thread_ts (an old-format bridge), it falls back to the newest candidate so legacy
// single-session use does not regress.
func (b *Bridge) transcriptForThread(ctx context.Context) ([]byte, []string, error) {
	files, err := b.Client.ListFiles(ctx, b.Channel, 16)
	if err != nil {
		return nil, nil, err
	}
	var cands []File
	for _, f := range files {
		// Legacy single-session uploads are named "transcript.jsonl"; the hub names
		// them "<session-id>-transcript.jsonl". Match both by suffix.
		if strings.HasSuffix(f.Name, "transcript.jsonl") {
			cands = append(cands, f)
		}
	}
	if len(cands) == 0 {
		return nil, nil, fmt.Errorf("no transcript.jsonl in channel")
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].Created > cands[j].Created })

	var seen []string
	var newest []byte // legacy fallback: newest candidate we actually downloaded
	anyThreadTS := false
	downloaded := 0
	for _, f := range cands {
		if tts, ok := b.classified[f.ID]; ok { // known other-session transcript — skip
			anyThreadTS = true
			seen = append(seen, tts)
			continue
		}
		if downloaded >= 8 {
			break
		}
		downloaded++
		url := f.URLDownload
		if url == "" {
			url = f.URLPrivate
		}
		data, err := b.Client.Download(ctx, url)
		if err != nil {
			continue
		}
		if b.ScratchDir != "" {
			_ = os.MkdirAll(b.ScratchDir, 0o755)
			_ = os.WriteFile(fmt.Sprintf("%s/transcript-%d.jsonl", b.ScratchDir, f.Created), data, 0o644)
		}
		if newest == nil {
			newest = data
		}
		tts := transcriptThreadTS(data)
		if tts == "" {
			continue
		}
		anyThreadTS = true
		if tts == b.ThreadTS {
			return data, seen, nil
		}
		seen = append(seen, tts)
		b.classified[f.ID] = tts // cache the wrong-session file so we never re-fetch it
	}
	if !anyThreadTS && newest != nil { // old-format bridge: no thread_ts anywhere
		return newest, seen, nil
	}
	return nil, seen, nil
}

// Alive reports whether the bridge session answers a control verb within wait.
// A banner can exist for a session whose shell has died or whose bridge is
// detached; Alive distinguishes a driveable thread from a stale one by posting
// !status and watching for the bridge's reply.
func (b *Bridge) Alive(ctx context.Context, wait time.Duration) (bool, string, error) {
	b.normalize()
	sentTS, err := b.Client.Post(ctx, b.Channel, b.ThreadTS, "!status")
	if err != nil {
		return false, "", err
	}
	sent, _ := strconv.ParseFloat(sentTS, 64)
	deadline := b.now().Add(wait)
	for b.now().Before(deadline) {
		time.Sleep(3 * time.Second)
		msgs, err := b.Client.Replies(ctx, b.Channel, b.ThreadTS, strconv.FormatFloat(sent, 'f', 6, 64), 10)
		if err != nil {
			return false, "", err
		}
		for _, m := range msgs {
			txt := ansi.ReplaceAllString(m.Text, "")
			// The bridge answers !status with a "Slack control status" block.
			if strings.Contains(txt, "control status") || strings.Contains(txt, "reader_alive") {
				return true, txt, nil
			}
			// A dead shell answers with an explicit not-running message.
			if strings.Contains(txt, "process is not running") || strings.Contains(txt, "process_not_running") {
				return false, txt, nil
			}
		}
	}
	return false, "no control reply within timeout", nil
}

// SelfTestResult classifies a readback round-trip probe of the control session.
type SelfTestResult struct {
	OK     bool   // true iff an echoed nonce round-tripped through the read path
	Reason string // "" when OK; else a typed reason (see Reason* constants)
	Detail string // the underlying error, or the echoed text on success
}

// Typed SelfTest failure reasons.
const (
	ReasonNoSessionTranscript = "no_session_transcript" // !dump never uploaded a transcript bound to this thread
	ReasonSentinelMissing     = "sentinel_missing"      // our transcript exists but the echo never appeared (shell wedged)
	ReasonEchoMismatch        = "echo_mismatch"         // a round-trip returned, but not our nonce
	ReasonExecError           = "exec_error"            // some other exec/transport failure
)

// SelfTest verifies the *readback* path works, not merely that the session answers a
// control verb. A session can be Alive (answers !status) yet have a broken result
// readback: the PTY stdout mirror wedges on msg_too_long, and !dump does not always
// upload a transcript bound to this thread. SelfTest round-trips a unique echo through
// the same path Exec uses and returns a typed reason on failure, so an orchestrator can
// fail fast with the real cause instead of a multi-minute generic timeout.
func (b *Bridge) SelfTest(ctx context.Context) (*SelfTestResult, error) {
	b.normalize()
	payload := "SELFTEST_" + b.nonce()
	out, err := b.Exec(ctx, "echo "+payload)
	if err != nil {
		res := &SelfTestResult{Detail: err.Error()}
		switch {
		case errors.Is(err, ErrNoSessionTranscript):
			res.Reason = ReasonNoSessionTranscript
		case errors.Is(err, ErrSentinelMissing):
			res.Reason = ReasonSentinelMissing
		default:
			res.Reason = ReasonExecError
		}
		return res, nil
	}
	if !strings.Contains(out, payload) {
		return &SelfTestResult{Reason: ReasonEchoMismatch, Detail: out}, nil
	}
	return &SelfTestResult{OK: true, Detail: payload}, nil
}

// nonce returns a unique-per-call sentinel token.
func (b *Bridge) nonce() string {
	return "RPC" + strconv.FormatInt(b.now().UnixNano()%1_000_000_000, 10)
}

// shellQuote single-quotes a path for safe shell interpolation.
func shellQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}
