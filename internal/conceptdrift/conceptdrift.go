package conceptdrift

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"time"
	"unicode"
	"unicode/utf16"

	"github.com/anthony-chaudhary/fak/pkg/sysproc"
)

// The durable-file contract, preserved byte-for-byte from the Python original so
// existing recorded baselines stay readable.
const (
	BaselineSchema = "fak-concept-coverage-drift-baseline/1"
	ReceiptSchema  = "fak-concept-coverage-drift-receipt/1"

	ScorecardName     = "concept_disambiguation_scorecard.py"
	DefaultDataDirRel = "tools/concept_disambiguation_scorecard.data"
	DefaultBasename   = "drift-baseline.json"
	ReceiptPrefix     = "drift-receipt-"

	TicketLane = "tools"

	MaxTicketTokens = 20

	scorecardTimeout = 1800 * time.Second
	gitTimeout       = 30 * time.Second
	ticketTimeout    = 300 * time.Second
)

// Exit codes: 0 clean / recorded / simulated OK - 2 operational error - 3 debt
// re-opened - 4 simulate failed to detect its own drift.
const (
	ExitOK             = 0
	ExitOperational    = 2
	ExitDrift          = 3
	ExitSimulateFailed = 4
)

var normRE = regexp.MustCompile(`[^a-z0-9]`)

// Options is one watcher invocation: the resolved inputs plus the mode selects.
type Options struct {
	// Workspace is the tree root the scorecard scans (absolute).
	Workspace string
	// DataDir is the catalog rows directory; empty means the scorecard's own
	// default (<workspace>/tools/concept_disambiguation_scorecard.data).
	DataDir string
	// Baseline is the baseline JSON path; empty means <data-dir>/drift-baseline.json
	// (with data dir falling back to the workspace default above).
	Baseline string
	// Scorecard is the path to the grandfathered scorecard script; empty means
	// <workspace>/tools/concept_disambiguation_scorecard.py.
	Scorecard string
	// RecordBaseline runs the scorecard once and records the baseline.
	RecordBaseline bool
	// Simulate runs the self-witness (no baseline involved).
	Simulate bool
	// AutoTicket best-effort creates a positioning ticket on a real drift.
	AutoTicket bool
}

// UncoveredToken is one unpositioned tree token: the family that owns it plus the
// normalized token itself, as emitted by the scorecard's coverage.uncovered list.
type UncoveredToken struct {
	Family string
	Token  string
}

// Snapshot is the drift-relevant slice of a scorecard payload: the coverage
// counters, the per-verdict standing, and the unpositioned token list.
type Snapshot struct {
	Discovered   int
	Covered      int
	CoverageDebt int
	Standing     map[string]any
	Uncovered    []UncoveredToken
}

// Run dispatches one watcher invocation and returns its exit code. stdout/stderr
// carry the operator-facing lines exactly as the Python tool printed them.
func Run(stdout, stderr io.Writer, o Options) int {
	ws, err := filepath.Abs(o.Workspace)
	if err != nil || ws == "" {
		fmt.Fprintf(stderr, "drift-watcher: cannot resolve workspace %q\n", o.Workspace)
		return ExitOperational
	}
	dataDir := ""
	if o.DataDir != "" {
		dataDir, _ = filepath.Abs(o.DataDir)
	}
	defaultData := filepath.Join(ws, DefaultDataDirRel)
	baselinePath := o.Baseline
	if baselinePath != "" {
		baselinePath, _ = filepath.Abs(baselinePath)
	} else {
		baselinePath = filepath.Join(dataDirOr(defaultData, dataDir), DefaultBasename)
	}

	switch {
	case o.RecordBaseline:
		return recordBaseline(stdout, stderr, ws, dataDir, baselinePath, o.Scorecard)
	case o.Simulate:
		return simulate(stdout, stderr, ws, dataDir, o.Scorecard)
	default:
		return checkMode(stdout, stderr, ws, dataDir, baselinePath, o.AutoTicket, o.Scorecard)
	}
}

func dataDirOr(def, explicit string) string {
	if explicit != "" {
		return explicit
	}
	return def
}

// ---------------------------------------------------------------------------
// Small pure helpers.
// ---------------------------------------------------------------------------

// NormToken collapses a name to its comparable token: lowercase, keep only
// [a-z0-9] - the same normalization the scorecard uses, so a watcher-side count
// agrees with the scorecard's coverage bookkeeping.
func NormToken(s string) string {
	return normRE.ReplaceAllString(strings.ToLower(s), "")
}

// RowTokens returns every normalized name one catalog row answers to: canonical +
// grounding + aliases. Non-string values contribute nothing, mirroring the
// Python norm_token(non-str) -> "" rule.
func RowTokens(r map[string]any) map[string]bool {
	toks := map[string]bool{}
	add := func(v any) {
		s, ok := v.(string)
		if !ok {
			return
		}
		if t := NormToken(s); t != "" {
			toks[t] = true
		}
	}
	add(r["canonical"])
	add(r["grounding"])
	if al, ok := r["aliases"].([]any); ok {
		for _, a := range al {
			add(a)
		}
	}
	return toks
}

// utcNow is the single clock seam: UTC now, second precision.
var utcNow = func() time.Time { return time.Now().UTC() }

// StampISO renders the receipt/baseline timestamp: 2006-01-02T15:04:05Z in UTC.
func StampISO(dt time.Time) string { return dt.Format("2006-01-02T15:04:05Z") }

// StampCompact renders the ticket-id stamp: 20060102-1504 in UTC.
func StampCompact(dt time.Time) string { return dt.Format("20060102-1504") }

// DecodeJSONBytes parses the scorecard's JSON payload from raw stdout bytes.
// The pipe path is clean UTF-8; the BOM / UTF-16 branches cover manually
// redirected output (PowerShell 5 `>` writes UTF-16 with a BOM).
func DecodeJSONBytes(raw []byte) (any, error) {
	var text string
	switch {
	case bytes.HasPrefix(raw, []byte{0xff, 0xfe}):
		u, err := decodeUTF16(raw[len("\xff\xfe"):], binary.LittleEndian)
		if err != nil {
			return nil, err
		}
		text = u
	case bytes.HasPrefix(raw, []byte{0xfe, 0xff}):
		u, err := decodeUTF16(raw[len("\xfe\xff"):], binary.BigEndian)
		if err != nil {
			return nil, err
		}
		text = u
	case bytes.HasPrefix(raw, []byte{0xef, 0xbb, 0xbf}):
		text = string(raw[len("\xef\xbb\xbf"):])
	default:
		text = string(raw)
	}
	text = strings.TrimLeftFunc(text, func(r rune) bool { return r == '\ufeff' || unicode.IsSpace(r) })
	var v any
	if err := json.Unmarshal([]byte(text), &v); err != nil {
		return nil, err
	}
	return v, nil
}

// decodeUTF16 decodes big/little-endian UTF-16 units, rejecting an odd trailing
// byte the way Python's utf-16 codec does.
func decodeUTF16(b []byte, order binary.ByteOrder) (string, error) {
	if len(b)%2 != 0 {
		return "", errors.New("truncated UTF-16 payload")
	}
	units := make([]uint16, len(b)/2)
	for i := range units {
		units[i] = order.Uint16(b[2*i:])
	}
	return string(utf16.Decode(units)), nil
}

// GitHead returns the workspace HEAD sha, best-effort (the empty string when
// not a git checkout or the probe fails).
func GitHead(workspace string) string {
	ctx, cancel := context.WithTimeout(context.Background(), gitTimeout)
	defer cancel()
	cmd := sysproc.CommandContext(ctx, "git", "rev-parse", "HEAD")
	cmd.Dir = workspace
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// ---------------------------------------------------------------------------
// Scorecard subprocess + snapshot extraction.
// ---------------------------------------------------------------------------

// resolvePython finds an available Python interpreter ("python" on Windows,
// "python3" elsewhere first, then the other spelling).
func resolvePython() (string, error) {
	primary, fallback := "python3", "python"
	if runtime.GOOS == "windows" {
		primary, fallback = "python", "python3"
	}
	if p, err := exec.LookPath(primary); err == nil {
		return p, nil
	}
	if p, err := exec.LookPath(fallback); err == nil {
		return p, nil
	}
	return "", fmt.Errorf("Python interpreter not found (tried %q and %q)", primary, fallback)
}

// RunScorecard runs the sibling scorecard as a subprocess and parses its --json
// payload. A non-zero exit is NOT an error here: the scorecard exits 1 whenever
// the namespace is not ok - which is exactly the drift state this watcher
// measures. Only a missing / undecodable payload is an operational failure.
func RunScorecard(workspace, dataDir, scorecard string) (map[string]any, error) {
	python, err := resolvePython()
	if err != nil {
		return nil, fmt.Errorf("cannot launch scorecard: %v", err)
	}
	if scorecard == "" {
		scorecard = filepath.Join(workspace, "tools", ScorecardName)
	}
	args := []string{scorecard, "--json", "--workspace", workspace}
	if dataDir != "" {
		args = append(args, "--data", dataDir)
	}
	ctx, cancel := context.WithTimeout(context.Background(), scorecardTimeout)
	defer cancel()
	cmd := sysproc.CommandContext(ctx, python, args...)
	cmd.Dir = workspace
	var stderrBuf bytes.Buffer
	cmd.Stderr = &stderrBuf
	out, runErr := cmd.Output()
	if runErr != nil {
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return nil, fmt.Errorf("scorecard timed out after %s", scorecardTimeout)
		}
		var exitErr *exec.ExitError
		if !errors.As(runErr, &exitErr) {
			return nil, fmt.Errorf("cannot launch scorecard: %v", runErr)
		}
		// A non-zero exit still carries a parseable payload: fall through.
	}
	payload, err := DecodeJSONBytes(out)
	if err != nil {
		tail := tailOf(stderrBuf.Bytes(), 400)
		exitCode := 0
		var exitErr *exec.ExitError
		if errors.As(runErr, &exitErr) {
			exitCode = exitErr.ExitCode()
		}
		return nil, fmt.Errorf("cannot parse scorecard JSON (exit %d): %v; stderr: %s", exitCode, err, tail)
	}
	doc, ok := payload.(map[string]any)
	if !ok {
		return nil, errors.New("scorecard payload is not a JSON object")
	}
	return doc, nil
}

// tailOf returns the last n bytes of b as trimmed ASCII-safe text.
func tailOf(b []byte, n int) string {
	if len(b) > n {
		b = b[len(b)-n:]
	}
	return strings.TrimSpace(string(b))
}

// ExtractSnapshot pulls the drift-relevant slice out of a scorecard payload: the
// coverage counters, the per-verdict standing, and the unpositioned token list
// (family + token).
func ExtractSnapshot(payload map[string]any) Snapshot {
	c, _ := payload["corpus"].(map[string]any)
	cov, _ := c["coverage"].(map[string]any)
	snap := Snapshot{
		Discovered:   toInt(cov["discovered"]),
		Covered:      toInt(cov["covered"]),
		CoverageDebt: toInt(cov["coverage_debt"]),
		Standing:     map[string]any{},
		Uncovered:    []UncoveredToken{},
	}
	if st, ok := c["standing"].(map[string]any); ok && len(st) > 0 {
		snap.Standing = st
	}
	if ul, ok := cov["uncovered"].([]any); ok {
		for _, u := range ul {
			m, ok := u.(map[string]any)
			if !ok {
				continue
			}
			fam, _ := m["family"].(string)
			tok, _ := m["token"].(string)
			snap.Uncovered = append(snap.Uncovered, UncoveredToken{Family: fam, Token: tok})
		}
	}
	return snap
}

// toInt coerces a JSON number (float64) to int, 0 for absent/non-numeric - the
// Python int(x.get(k, 0)) semantics.
func toInt(v any) int {
	switch n := v.(type) {
	case float64:
		return int(n)
	case int:
		return n
	case int64:
		return int(n)
	}
	return 0
}

// UncoveredByFamily groups the unpositioned tokens by owning family,
// deterministically sorted (families ascending, tokens ascending within).
func UncoveredByFamily(uncovered []UncoveredToken) map[string][]string {
	out := map[string][]string{}
	for _, u := range uncovered {
		fam := u.Family
		if fam == "" {
			fam = "?"
		}
		if u.Token != "" {
			out[fam] = append(out[fam], u.Token)
		}
	}
	for fam := range out {
		sort.Strings(out[fam])
	}
	return out
}

// AuditError returns the scorecard's own AUDIT_ERROR reason, when the payload
// says the read failed.
func AuditError(payload map[string]any) string {
	if payload != nil && payload["verdict"] == "AUDIT_ERROR" {
		reason, _ := payload["reason"].(string)
		return fmt.Sprintf("scorecard audit error: %s", reason)
	}
	return ""
}

// ---------------------------------------------------------------------------
// Baseline + receipt IO.
// ---------------------------------------------------------------------------

// LoadBaseline reads the recorded baseline JSON (BOM-tolerant).
func LoadBaseline(path string) (map[string]any, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("no baseline at %s - run --record-baseline first", path)
		}
		return nil, fmt.Errorf("cannot read baseline %s: %v", filepath.Base(path), err)
	}
	raw = bytes.TrimPrefix(raw, []byte{0xef, 0xbb, 0xbf})
	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, fmt.Errorf("cannot read baseline %s: %v", filepath.Base(path), err)
	}
	if doc == nil {
		return nil, fmt.Errorf("baseline %s is not a JSON object", filepath.Base(path))
	}
	return doc, nil
}

// WriteReceipt lands the drift receipt NEXT TO the baseline it diffs against,
// named drift-receipt-<recorded_at without colons>.json.
func WriteReceipt(receipt map[string]any, baselinePath string) (string, error) {
	ts, _ := receipt["recorded_at"].(string)
	ts = strings.ReplaceAll(ts, ":", "")
	out := filepath.Join(filepath.Dir(baselinePath), ReceiptPrefix+ts+".json")
	if err := writeJSONFile(out, receipt); err != nil {
		return "", err
	}
	return out, nil
}

// writeJSONFile writes v as 2-space-indented JSON with sorted keys and a
// trailing newline - byte-compatible with the Python json.dumps(indent=2,
// sort_keys=True) + "\n" the original wrote.
func writeJSONFile(path string, v any) error {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		return err
	}
	return os.WriteFile(path, buf.Bytes(), 0o644)
}

// ---------------------------------------------------------------------------
// Modes.
// ---------------------------------------------------------------------------

func recordBaseline(stdout, stderr io.Writer, workspace, dataDir, baselinePath, scorecard string) int {
	payload, err := RunScorecard(workspace, dataDir, scorecard)
	if err != nil {
		fmt.Fprintf(stderr, "drift-watcher: %v\n", err)
		return ExitOperational
	}
	if reason := AuditError(payload); reason != "" {
		fmt.Fprintf(stderr, "drift-watcher: %s\n", reason)
		return ExitOperational
	}
	snap := ExtractSnapshot(payload)
	head := GitHead(workspace)
	baseline := map[string]any{
		"schema":        BaselineSchema,
		"recorded_at":   StampISO(utcNow()),
		"git_head":      head,
		"discovered":    snap.Discovered,
		"covered":       snap.Covered,
		"coverage_debt": snap.CoverageDebt,
		"standing":      snap.Standing,
	}
	if err := os.MkdirAll(filepath.Dir(baselinePath), 0o755); err != nil {
		fmt.Fprintf(stderr, "drift-watcher: %v\n", err)
		return ExitOperational
	}
	if err := writeJSONFile(baselinePath, baseline); err != nil {
		fmt.Fprintf(stderr, "drift-watcher: %v\n", err)
		return ExitOperational
	}
	fmt.Fprintf(stdout, "baseline recorded -> %s\n", baselinePath)
	headShown := head
	if headShown == "" {
		headShown = "(no git head)"
	}
	fmt.Fprintf(stdout, "  git_head=%s discovered=%d covered=%d coverage_debt=%d\n",
		headShown, snap.Discovered, snap.Covered, snap.CoverageDebt)
	return ExitOK
}

func checkMode(stdout, stderr io.Writer, workspace, dataDir, baselinePath string, autoTicket bool, scorecard string) int {
	baseline, err := LoadBaseline(baselinePath)
	if err != nil {
		fmt.Fprintf(stderr, "drift-watcher: %v\n", err)
		return ExitOperational
	}
	payload, err := RunScorecard(workspace, dataDir, scorecard)
	if err != nil {
		fmt.Fprintf(stderr, "drift-watcher: %v\n", err)
		return ExitOperational
	}
	if reason := AuditError(payload); reason != "" {
		fmt.Fprintf(stderr, "drift-watcher: %s\n", reason)
		return ExitOperational
	}
	snap := ExtractSnapshot(payload)
	bDebt := toInt(baseline["coverage_debt"])
	bDisc := toInt(baseline["discovered"])
	debtDelta := snap.CoverageDebt - bDebt
	discDelta := snap.Discovered - bDisc
	drifted := debtDelta > 0
	ubf := UncoveredByFamily(snap.Uncovered)
	head := GitHead(workspace)
	nUncovered := 0
	for _, v := range ubf {
		nUncovered += len(v)
	}

	var why string
	if drifted {
		baselineAt, _ := baseline["recorded_at"].(string)
		if baselineAt == "" {
			baselineAt = "?"
		}
		baselineHead, _ := baseline["git_head"].(string)
		if baselineHead == "" {
			baselineHead = "unknown head"
		}
		why = fmt.Sprintf("coverage-debt re-opened %d -> %d (+%d) since baseline recorded %s at %s; %d token(s) unpositioned",
			bDebt, snap.CoverageDebt, debtDelta, baselineAt, baselineHead, nUncovered)
	} else {
		why = fmt.Sprintf("coverage-debt held at %d (baseline %d); discovered %d (%+d); no re-opening",
			snap.CoverageDebt, bDebt, snap.Discovered, discDelta)
	}

	receipt := map[string]any{
		"schema":              ReceiptSchema,
		"kind":                "check",
		"recorded_at":         StampISO(utcNow()),
		"workspace":           workspace,
		"git_head":            head,
		"baseline_file":       baselinePath,
		"baseline":            pickKeys(baseline, "recorded_at", "git_head", "discovered", "covered", "coverage_debt", "standing"),
		"current":             pickKeys(map[string]any{"discovered": snap.Discovered, "covered": snap.Covered, "coverage_debt": snap.CoverageDebt, "standing": snap.Standing}, "discovered", "covered", "coverage_debt", "standing"),
		"delta":               map[string]any{"coverage_debt": debtDelta, "discovered": discDelta},
		"drift":               drifted,
		"verdict":             "OK",
		"uncovered_by_family": ubf,
		"reason":              why,
	}
	if drifted {
		receipt["verdict"] = "DRIFT"
	}
	rp, err := WriteReceipt(receipt, baselinePath)
	if err != nil {
		fmt.Fprintf(stderr, "drift-watcher: %v\n", err)
		return ExitOperational
	}
	if drifted {
		headShown := head
		if headShown == "" {
			headShown = "(no git head)"
		}
		fmt.Fprintf(stdout, "DRIFT: coverage-debt re-opened %d -> %d (+%d); discovered %d -> %d (%+d) at %s\n",
			bDebt, snap.CoverageDebt, debtDelta, bDisc, snap.Discovered, discDelta, headShown)
		for _, fam := range sortedKeys(ubf) {
			fmt.Fprintf(stdout, "  %s: %s\n", fam, showTokens(ubf[fam]))
		}
		fmt.Fprintf(stdout, "receipt: %s\n", rp)
		if autoTicket {
			createDriftTicket(stdout, workspace, debtDelta, flattenFamilies(ubf))
		}
		return ExitDrift
	}
	fmt.Fprintf(stdout, "OK: coverage-debt held at %d (baseline %d); discovered %d (%+d); no re-opening\n",
		snap.CoverageDebt, bDebt, snap.Discovered, discDelta)
	fmt.Fprintf(stdout, "receipt: %s\n", rp)
	return ExitOK
}

// pickKeys mirrors Python {k: doc.get(k) for k in keys}: every key is present,
// null when the source lacks it.
func pickKeys(doc map[string]any, keys ...string) map[string]any {
	out := make(map[string]any, len(keys))
	for _, k := range keys {
		out[k] = doc[k]
	}
	return out
}

func sortedKeys(m map[string][]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// showTokens renders a family's token list capped at MaxTicketTokens, the way
// the Python printer did: "a, b (+N more)".
func showTokens(toks []string) string {
	shown := strings.Join(toks[:min(len(toks), MaxTicketTokens)], ", ")
	if len(toks) > MaxTicketTokens {
		shown += fmt.Sprintf(" (+%d more)", len(toks)-MaxTicketTokens)
	}
	return shown
}

func flattenFamilies(ubf map[string][]string) []string {
	var out []string
	for _, fam := range sortedKeys(ubf) {
		out = append(out, ubf[fam]...)
	}
	return out
}

// createDriftTicket is the best-effort positioning ticket on a real drift. Never
// fatal: a missing fak-run or a refused create is reported and swallowed.
func createDriftTicket(stdout io.Writer, workspace string, nReopened int, tokens []string) {
	exe, err := exec.LookPath("fak-run")
	if err != nil {
		fmt.Fprintln(stdout, "auto-ticket: fak-run not on PATH - ticket skipped (best-effort)")
		return
	}
	intent := fmt.Sprintf("position %d re-opened tokens: %s", nReopened, showTokens(tokens))
	tid := "concept-coverage-drift-" + StampCompact(utcNow())
	ctx, cancel := context.WithTimeout(context.Background(), ticketTimeout)
	defer cancel()
	cmd := sysproc.CommandContext(ctx, exe, "fak-ticket", "create", tid, "--lane", TicketLane, "--intent", intent)
	cmd.Dir = workspace
	var stderrBuf bytes.Buffer
	cmd.Stderr = &stderrBuf
	stdoutBuf, runErr := captureStdout(cmd)
	if runErr != nil {
		var exitErr *exec.ExitError
		if errors.As(runErr, &exitErr) {
			code := exitErr.ExitCode()
			tail := tailOf(stderrBuf.Bytes(), 300)
			if tail == "" {
				tail = tailOf(stdoutBuf, 300)
			}
			fmt.Fprintf(stdout, "auto-ticket: fak-run exited %d (%s) - ticket not created\n", code, tail)
			return
		}
		fmt.Fprintf(stdout, "auto-ticket: skipped (%v)\n", runErr)
		return
	}
	fmt.Fprintf(stdout, "auto-ticket: created %s\n", tid)
}

// captureStdout runs cmd with stdout captured and the pre-wired stderr buffer
// retained.
func captureStdout(cmd *exec.Cmd) ([]byte, error) {
	var out bytes.Buffer
	cmd.Stdout = &out
	if err := cmd.Run(); err != nil {
		return out.Bytes(), err
	}
	return out.Bytes(), nil
}

// FileTokens returns the normalized tokens one rows file covers (row_tokens,
// summed over its rows).
func FileTokens(rowsFile string) map[string]bool {
	out := map[string]bool{}
	raw, err := os.ReadFile(rowsFile)
	if err != nil {
		return out
	}
	raw = bytes.TrimPrefix(raw, []byte{0xef, 0xbb, 0xbf})
	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		return out
	}
	rows, _ := doc["rows"].([]any)
	for _, r := range rows {
		m, ok := r.(map[string]any)
		if !ok {
			continue
		}
		for t := range RowTokens(m) {
			out[t] = true
		}
	}
	return out
}

// PickVictim returns the rows file whose removal uncovers the most tokens:
// argmax over |row_tokens|, ties broken by file name (deterministic). Files
// starting with "_" (catalog metadata) are excluded, matching the Python glob
// filter.
func PickVictim(dataDir string) (string, map[string]bool, error) {
	entries, err := os.ReadDir(dataDir)
	if err != nil {
		return "", nil, err
	}
	type candidate struct {
		tokens map[string]bool
		name   string
	}
	var candidates []candidate
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") || strings.HasPrefix(e.Name(), "_") {
			continue
		}
		candidates = append(candidates, candidate{tokens: FileTokens(filepath.Join(dataDir, e.Name())), name: e.Name()})
	}
	if len(candidates) == 0 {
		return "", nil, fmt.Errorf("no rows-*.json files in %s", dataDir)
	}
	sort.Slice(candidates, func(i, j int) bool {
		if len(candidates[i].tokens) != len(candidates[j].tokens) {
			return len(candidates[i].tokens) > len(candidates[j].tokens)
		}
		return candidates[i].name < candidates[j].name
	})
	top := candidates[0]
	return filepath.Join(dataDir, top.name), top.tokens, nil
}

func simulate(stdout, stderr io.Writer, workspace, dataDir, scorecard string) int {
	src := dataDir
	if src == "" {
		src = filepath.Join(workspace, DefaultDataDirRel)
	}
	if fi, err := os.Stat(src); err != nil || !fi.IsDir() {
		fmt.Fprintf(stderr, "drift-watcher: no data directory at %s\n", src)
		return ExitOperational
	}
	payload0, err := RunScorecard(workspace, src, scorecard)
	if err != nil {
		fmt.Fprintf(stderr, "drift-watcher: %v\n", err)
		return ExitOperational
	}
	if reason := AuditError(payload0); reason != "" {
		fmt.Fprintf(stderr, "drift-watcher: %s\n", reason)
		return ExitOperational
	}
	snap0 := ExtractSnapshot(payload0)

	td, err := os.MkdirTemp("", "concept-drift-sim-")
	if err != nil {
		fmt.Fprintf(stderr, "drift-watcher: %v\n", err)
		return ExitOperational
	}
	defer os.RemoveAll(td)
	dst := filepath.Join(td, "data")
	if err := copyTree(src, dst); err != nil {
		fmt.Fprintf(stderr, "drift-watcher: %v\n", err)
		return ExitOperational
	}
	victim, removed, err := PickVictim(dst)
	if err != nil {
		fmt.Fprintf(stderr, "drift-watcher: cannot simulate: %v\n", err)
		return ExitOperational
	}
	if err := os.Remove(victim); err != nil {
		fmt.Fprintf(stderr, "drift-watcher: cannot simulate: %v\n", err)
		return ExitOperational
	}
	payload1, err := RunScorecard(workspace, dst, scorecard)
	if err != nil {
		fmt.Fprintf(stderr, "drift-watcher: %v\n", err)
		return ExitOperational
	}
	if reason := AuditError(payload1); reason != "" {
		fmt.Fprintf(stderr, "drift-watcher: %s\n", reason)
		return ExitOperational
	}
	snap1 := ExtractSnapshot(payload1)

	u0 := tokenSet(snap0.Uncovered)
	u1 := tokenSet(snap1.Uncovered)
	newUncovered := difference(u1, u0)
	removedSet := map[string]bool{}
	for t := range removed {
		removedSet[t] = true
	}
	var foreign []string
	for _, t := range newUncovered {
		if !removedSet[t] {
			foreign = append(foreign, t)
		}
	}
	debtDelta := snap1.CoverageDebt - snap0.CoverageDebt
	detected := debtDelta > 0 && len(newUncovered) > 0 && len(foreign) == 0

	reason := fmt.Sprintf("self-witness holds: removing %s (%d tokens) re-opened coverage-debt %d -> %d (+%d); all %d newly-uncovered token(s) belong to the removed file",
		filepath.Base(victim), len(removed), snap0.CoverageDebt, snap1.CoverageDebt, debtDelta, len(newUncovered))
	if !detected {
		reason = fmt.Sprintf("self-witness FAILED: removing %s moved debt %d -> %d (delta %+d); newly-uncovered=%d; foreign tokens (not in removed file): %s",
			filepath.Base(victim), snap0.CoverageDebt, snap1.CoverageDebt, debtDelta, len(newUncovered), pyList(foreign[:min(len(foreign), 20)]))
	}
	receipt := map[string]any{
		"schema":          ReceiptSchema,
		"kind":            "simulate",
		"recorded_at":     StampISO(utcNow()),
		"workspace":       workspace,
		"git_head":        GitHead(workspace),
		"removed_file":    filepath.Base(victim),
		"removed_tokens":  sortedSet(removed),
		"pre":             map[string]any{"discovered": snap0.Discovered, "covered": snap0.Covered, "coverage_debt": snap0.CoverageDebt},
		"post":            map[string]any{"discovered": snap1.Discovered, "covered": snap1.Covered, "coverage_debt": snap1.CoverageDebt},
		"delta":           map[string]any{"coverage_debt": debtDelta},
		"newly_uncovered": newUncovered,
		"drift_detected":  detected,
		"verdict":         "NO-DRIFT",
		"reason":          reason,
	}
	if detected {
		receipt["verdict"] = "DRIFT-DETECTED"
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(receipt); err != nil {
		fmt.Fprintf(stderr, "drift-watcher: %v\n", err)
		return ExitOperational
	}
	io.Copy(stdout, &buf)
	if detected {
		return ExitOK
	}
	return ExitSimulateFailed
}

func tokenSet(uncovered []UncoveredToken) map[string]bool {
	out := map[string]bool{}
	for _, u := range uncovered {
		if u.Token != "" {
			out[u.Token] = true
		}
	}
	return out
}

func difference(a, b map[string]bool) []string {
	var out []string
	for t := range a {
		if !b[t] {
			out = append(out, t)
		}
	}
	sort.Strings(out)
	return out
}

func sortedSet(s map[string]bool) []string {
	out := make([]string, 0, len(s))
	for t := range s {
		out = append(out, t)
	}
	sort.Strings(out)
	return out
}

// pyList renders a string slice in Python-list repr form (single quotes), the
// shape the original embedded in its failure reason.
func pyList(ss []string) string {
	if len(ss) == 0 {
		return "[]"
	}
	quoted := make([]string, len(ss))
	for i, s := range ss {
		quoted[i] = "'" + strings.ReplaceAll(s, "'", "''") + "'"
	}
	return "[" + strings.Join(quoted, ", ") + "]"
}

// copyTree copies src into dst (both directories), mirroring shutil.copytree.
func copyTree(src, dst string) error {
	return filepath.Walk(src, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		target := filepath.Join(dst, rel)
		if info.IsDir() {
			return os.MkdirAll(target, info.Mode().Perm())
		}
		body, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(target, body, info.Mode().Perm())
	})
}
