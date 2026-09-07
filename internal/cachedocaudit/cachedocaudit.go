// Package cachedocaudit implements the doc-numbers gate: verifying that every
// headline number in operational cachevalue documents traces to a committed
// snapshot field, renders to the right rounding, and stays internally consistent.
//
// Ported from tools/cachedoc_numbers_audit.py.
package cachedocaudit

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	// DefaultManifestGlob is the repository glob for docnumbers manifests.
	DefaultManifestGlob = "tools/docnumbers/*.json"

	// Schema is the expected manifest schema token.
	Schema = "fak.docnumbers.v1"
)

var (
	// ProvenanceTokens are the permitted provenance classifications.
	ProvenanceTokens = []string{"WITNESSED", "OBSERVED", "ESTIMATED"}

	// LiveEqualityWindows are doubly-bounded settled windows where live equality checking is enforced.
	LiveEqualityWindows = []string{"bounded"}

	// LiveNoTouchWindows are live snapshot windows that grow continuously and must not be touched by live equality probes.
	LiveNoTouchWindows = []string{"live_snapshot"}

	numCoreRe  = regexp.MustCompile(`-?[\d,]*\.?\d+`)
	safeExprRe = regexp.MustCompile(`^[\d\s.+\-*/()]+$`)
)

// Manifest represents a tools/docnumbers/<stem>.json manifest.
type Manifest struct {
	Schema         string            `json:"schema"`
	Doc            string            `json:"doc"`
	Title          string            `json:"title,omitempty"`
	SnapshotDir    string            `json:"snapshot_dir"`
	SnapshotDate   string            `json:"snapshot_date,omitempty"`
	StaleAfterDays *int              `json:"stale_after_days,omitempty"`
	Note           string            `json:"note,omitempty"`
	Sources        map[string]Source `json:"sources,omitempty"`
	Claims         []Claim           `json:"claims,omitempty"`
	Invariants     []Invariant       `json:"invariants,omitempty"`

	ManifestPath string `json:"_manifest_path,omitempty"`
}

// Source represents a data source in a manifest.
type Source struct {
	Cmd    []string `json:"cmd"`
	Window string   `json:"window"`
}

// Claim represents a verified claim within a guarded document.
type Claim struct {
	ID               string        `json:"id"`
	AppearsAs        string        `json:"appears_as"`
	Source           string        `json:"source"`
	Provenance       string        `json:"provenance,omitempty"`
	ProvenanceInline *bool         `json:"provenance_inline,omitempty"`
	Numbers          []ClaimNumber `json:"numbers,omitempty"`
}

// ClaimNumber represents an individual number asserted within a claim.
type ClaimNumber struct {
	Display     string   `json:"display"`
	Field       string   `json:"field,omitempty"`
	Expected    float64  `json:"expected"`
	Scale       *float64 `json:"scale,omitempty"`
	DerivedFrom string   `json:"derived_from,omitempty"`
}

// Invariant represents an arithmetic constraint that must hold in the document.
type Invariant struct {
	Kind   string    `json:"kind"`
	Label  string    `json:"label,omitempty"`
	Total  *float64  `json:"total,omitempty"`
	Parts  []float64 `json:"parts,omitempty"`
	Expr   string    `json:"expr,omitempty"`
	Expect *float64  `json:"expect,omitempty"`
	Tol    *float64  `json:"tol,omitempty"`
}

// Finding represents a single audit issue (failure or warning).
type Finding struct {
	Manifest string `json:"manifest"`
	Check    string `json:"check"`
	Msg      string `json:"msg"`
}

// AuditResult is the JSON output structure for the audit.
type AuditResult struct {
	Fails []Finding `json:"fails"`
	Warns []Finding `json:"warns"`
	Live  []string  `json:"live"`
	OK    bool      `json:"ok"`
}

// ParsedDisplay holds the parsed parts of a rendered number string.
type ParsedDisplay struct {
	Value  float64 `json:"value"`
	Place  float64 `json:"place"`
	IsPct  bool    `json:"is_pct"`
	Approx bool    `json:"approx"`
}

// AuditOptions configures an audit execution run.
type AuditOptions struct {
	Root     string
	Manifest string
	Live     bool
	Refresh  bool
	AsJSON   bool
	Today    time.Time
	FakBin   string
}

// Dotted traverses a dotted path on nested map[string]any objects.
func Dotted(obj any, path string) (any, error) {
	cur := obj
	parts := strings.Split(path, ".")
	for _, part := range parts {
		curMap, ok := cur.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("cannot traverse %q: not an object", part)
		}
		val, exists := curMap[part]
		if !exists {
			return nil, fmt.Errorf("key %q not found in dotted path %q", part, path)
		}
		cur = val
	}
	return cur, nil
}

// ParseDisplay parses a rendered number string (e.g. "≈$1,181.37", "62.6M", "89.3%") into a comparable value.
func ParseDisplay(s string) (ParsedDisplay, error) {
	raw := strings.TrimSpace(s)
	runes := []rune(raw)
	approx := len(runes) > 0 && (runes[0] == '≈' || runes[0] == '~' || strings.HasPrefix(strings.ToLower(raw), "about"))
	t := strings.TrimLeft(raw, "≈~ ")
	t = strings.TrimSpace(t)
	isPct := strings.HasSuffix(t, "%")
	t = strings.TrimRight(t, "%")
	t = strings.TrimSpace(t)
	t = strings.ReplaceAll(t, "$", "")
	t = strings.TrimSpace(t)

	mult := 1.0
	if len(t) > 0 {
		last := t[len(t)-1]
		switch last {
		case 'k', 'K':
			mult = 1e3
			t = t[:len(t)-1]
		case 'm', 'M':
			mult = 1e6
			t = t[:len(t)-1]
		case 'b', 'B':
			mult = 1e9
			t = t[:len(t)-1]
		}
	}
	core := strings.TrimSpace(strings.ReplaceAll(t, ",", ""))
	m := numCoreRe.FindString(core)
	if m == "" {
		return ParsedDisplay{}, fmt.Errorf("cannot parse a number from %q", s)
	}
	core = strings.ReplaceAll(m, ",", "")
	val, err := strconv.ParseFloat(core, 64)
	if err != nil {
		return ParsedDisplay{}, fmt.Errorf("cannot parse a number from %q: %w", s, err)
	}
	value := val * mult
	decimals := 0
	if strings.Contains(core, ".") {
		parts := strings.Split(core, ".")
		decimals = len(parts[1])
	}
	place := math.Pow(10.0, float64(-decimals)) * mult
	return ParsedDisplay{
		Value:  value,
		Place:  place,
		IsPct:  isPct,
		Approx: approx,
	}, nil
}

// RoundingOK checks if display is a correct rounding of trueValue.
func RoundingOK(display string, trueValue float64) (bool, string) {
	d, err := ParseDisplay(display)
	if err != nil {
		return false, fmt.Sprintf("parse error: %v", err)
	}
	if d.Approx {
		denom := math.Abs(trueValue)
		if denom <= 1e-9 {
			denom = 1.0
		}
		rel := math.Abs(trueValue-d.Value) / denom
		ok := rel <= 0.03
		detail := fmt.Sprintf("approx |%.6g-%.6g| rel=%.3f%% (≤3%%)", trueValue, d.Value, rel*100.0)
		return ok, detail
	}
	tol := 0.5*d.Place + 1e-9*math.Max(1.0, math.Abs(trueValue))
	diff := math.Abs(trueValue - d.Value)
	ok := diff <= tol
	detail := fmt.Sprintf("|%.6g-%.6g|=%.4g (≤%.4g)", trueValue, d.Value, diff, tol)
	return ok, detail
}

// SafeEval safely evaluates an arithmetic expression over numbers and + - * / ( ) only.
func SafeEval(expr string) (float64, error) {
	if !safeExprRe.MatchString(expr) {
		return 0, fmt.Errorf("unsafe expression: %q", expr)
	}
	p := &arithmeticParser{input: expr}
	val, err := p.parseExpr()
	if err != nil {
		return 0, err
	}
	p.skipWhitespace()
	if p.pos < len(p.input) {
		return 0, fmt.Errorf("unexpected character at position %d: %q", p.pos, p.input[p.pos])
	}
	return val, nil
}

type arithmeticParser struct {
	input string
	pos   int
}

func (p *arithmeticParser) skipWhitespace() {
	for p.pos < len(p.input) && (p.input[p.pos] == ' ' || p.input[p.pos] == '\t' || p.input[p.pos] == '\r' || p.input[p.pos] == '\n') {
		p.pos++
	}
}

func (p *arithmeticParser) parseExpr() (float64, error) {
	left, err := p.parseTerm()
	if err != nil {
		return 0, err
	}
	for {
		p.skipWhitespace()
		if p.pos >= len(p.input) {
			break
		}
		ch := p.input[p.pos]
		if ch == '+' {
			p.pos++
			right, err := p.parseTerm()
			if err != nil {
				return 0, err
			}
			left += right
		} else if ch == '-' {
			p.pos++
			right, err := p.parseTerm()
			if err != nil {
				return 0, err
			}
			left -= right
		} else {
			break
		}
	}
	return left, nil
}

func (p *arithmeticParser) parseTerm() (float64, error) {
	left, err := p.parseFactor()
	if err != nil {
		return 0, err
	}
	for {
		p.skipWhitespace()
		if p.pos >= len(p.input) {
			break
		}
		ch := p.input[p.pos]
		if ch == '*' {
			p.pos++
			right, err := p.parseFactor()
			if err != nil {
				return 0, err
			}
			left *= right
		} else if ch == '/' {
			p.pos++
			right, err := p.parseFactor()
			if err != nil {
				return 0, err
			}
			if right == 0 {
				return 0, errors.New("division by zero")
			}
			left /= right
		} else {
			break
		}
	}
	return left, nil
}

func (p *arithmeticParser) parseFactor() (float64, error) {
	p.skipWhitespace()
	if p.pos >= len(p.input) {
		return 0, errors.New("unexpected end of expression")
	}
	ch := p.input[p.pos]
	if ch == '+' {
		p.pos++
		return p.parseFactor()
	}
	if ch == '-' {
		p.pos++
		val, err := p.parseFactor()
		if err != nil {
			return 0, err
		}
		return -val, nil
	}
	return p.parsePrimary()
}

func (p *arithmeticParser) parsePrimary() (float64, error) {
	p.skipWhitespace()
	if p.pos >= len(p.input) {
		return 0, errors.New("unexpected end of expression")
	}
	ch := p.input[p.pos]
	if ch == '(' {
		p.pos++
		val, err := p.parseExpr()
		if err != nil {
			return 0, err
		}
		p.skipWhitespace()
		if p.pos >= len(p.input) || p.input[p.pos] != ')' {
			return 0, errors.New("missing closing parenthesis ')'")
		}
		p.pos++
		return val, nil
	}
	start := p.pos
	for p.pos < len(p.input) && ((p.input[p.pos] >= '0' && p.input[p.pos] <= '9') || p.input[p.pos] == '.') {
		p.pos++
	}
	if start == p.pos {
		return 0, fmt.Errorf("expected number at position %d, got %q", p.pos, p.input[p.pos])
	}
	numStr := p.input[start:p.pos]
	val, err := strconv.ParseFloat(numStr, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid number %q: %w", numStr, err)
	}
	return val, nil
}

func formatNum(v float64) string {
	return strconv.FormatFloat(v, 'f', -1, 64)
}

func isLiveNoTouch(window string) bool {
	for _, w := range LiveNoTouchWindows {
		if w == window {
			return true
		}
	}
	return false
}

func isLiveEquality(window string) bool {
	for _, w := range LiveEqualityWindows {
		if w == window {
			return true
		}
	}
	return false
}

// ReadDoc reads the document relative to root.
func ReadDoc(root, rel string) (string, error) {
	data, err := os.ReadFile(filepath.Join(root, rel))
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// ReadSnapshot reads a snapshot JSON file relative to root and snapshotDir.
func ReadSnapshot(root, snapDir, source string) (map[string]any, error) {
	data, err := os.ReadFile(filepath.Join(root, snapDir, source))
	if err != nil {
		return nil, err
	}
	var snap map[string]any
	if err := json.Unmarshal(data, &snap); err != nil {
		return nil, err
	}
	return snap, nil
}

// LoadManifests loads all docnumbers manifests under tools/docnumbers/*.json.
func LoadManifests(root string, only string) ([]*Manifest, error) {
	pattern := filepath.Join(root, "tools", "docnumbers", "*.json")
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}
	sort.Strings(matches)
	var manifests []*Manifest
	for _, path := range matches {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		var m Manifest
		if err := json.Unmarshal(data, &m); err != nil {
			continue
		}
		if m.Schema != Schema {
			continue
		}
		if only != "" && filepath.Base(path) != filepath.Base(only) {
			continue
		}
		m.ManifestPath = path
		manifests = append(manifests, &m)
	}
	return manifests, nil
}

// AuditManifest audits a single manifest against its document, snapshot, and invariant constraints.
func AuditManifest(root string, m *Manifest, today time.Time) ([]Finding, []Finding) {
	var fails []Finding
	var warns []Finding
	mid := filepath.Base(m.ManifestPath)
	if mid == "" || mid == "." {
		mid = filepath.Base(m.Doc)
	}

	fail := func(check, msg string) {
		fails = append(fails, Finding{Manifest: mid, Check: check, Msg: msg})
	}
	warn := func(check, msg string) {
		warns = append(warns, Finding{Manifest: mid, Check: check, Msg: msg})
	}

	doc, err := ReadDoc(root, m.Doc)
	if err != nil {
		fail("binding", fmt.Sprintf("doc not found: %s", m.Doc))
		return fails, warns
	}
	docLines := strings.Split(strings.ReplaceAll(doc, "\r\n", "\n"), "\n")
	snapDir := m.SnapshotDir

	for _, c := range m.Claims {
		cid := c.ID
		appears := c.AppearsAs

		// 1. binding: the rendered string is present in the doc
		if !strings.Contains(doc, appears) {
			fail("binding", fmt.Sprintf("[%s] appears_as not found in doc: %q", cid, appears))
			continue
		}

		snap, err := ReadSnapshot(root, snapDir, c.Source)
		if err != nil {
			fail("snapshot", fmt.Sprintf("[%s] snapshot unreadable: %s", cid, c.Source))
			continue
		}

		for _, n := range c.Numbers {
			disp := n.Display
			scale := 1.0
			if n.Scale != nil {
				scale = *n.Scale
			}
			expected := n.Expected

			// 2. snapshot binding: expected == committed snapshot field
			if n.Field != "" {
				got, err := Dotted(snap, n.Field)
				if err != nil {
					fail("snapshot", fmt.Sprintf("[%s] field %s missing in %s", cid, n.Field, c.Source))
					continue
				}
				var gotFloat float64
				switch v := got.(type) {
				case float64:
					gotFloat = v
				case int:
					gotFloat = float64(v)
				case int64:
					gotFloat = float64(v)
				default:
					fail("snapshot", fmt.Sprintf("[%s] field %s not numeric: %v", cid, n.Field, got))
					continue
				}
				if math.Abs(gotFloat-expected) > 1e-6 {
					fail("snapshot", fmt.Sprintf("[%s] expected %s != snapshot %s::%s=%s", cid, formatNum(expected), c.Source, n.Field, formatNum(gotFloat)))
					continue
				}
			}

			// 3. rounding: the doc render is a correct rounding of expected*scale
			ok, detail := RoundingOK(disp, expected*scale)
			if !ok {
				fail("rounding", fmt.Sprintf("[%s] %q does not render %s×%s: %s", cid, disp, formatNum(expected), formatNum(scale), detail))
			}
		}

		// 5. provenance co-location (table-row claims only)
		prov := c.Provenance
		provInline := true
		if c.ProvenanceInline != nil {
			provInline = *c.ProvenanceInline
		}
		if prov != "" && provInline {
			var hosts []string
			for _, ln := range docLines {
				if strings.Contains(ln, appears) {
					hosts = append(hosts, ln)
				}
			}
			if len(hosts) > 0 {
				hasProv := false
				for _, ln := range hosts {
					if strings.Contains(ln, prov) {
						hasProv = true
						break
					}
				}
				if !hasProv {
					prefix := appears
					if len(prefix) > 32 {
						prefix = prefix[:32]
					}
					warn("provenance", fmt.Sprintf("[%s] line with %q lacks its %s label", cid, prefix, prov))
				}
			}
		}
	}

	// 4. invariants
	for _, inv := range m.Invariants {
		label := inv.Label
		if label == "" {
			label = inv.Kind
		}
		switch inv.Kind {
		case "sum":
			total := 0.0
			if inv.Total != nil {
				total = *inv.Total
			}
			tol := 0.0
			if inv.Tol != nil {
				tol = *inv.Tol
			}
			sumParts := 0.0
			partStrs := make([]string, len(inv.Parts))
			for i, p := range inv.Parts {
				sumParts += p
				partStrs[i] = formatNum(p)
			}
			if math.Abs(sumParts-total) > tol {
				fail("invariants", fmt.Sprintf("sum '%s': %s = %s != %s", label, strings.Join(partStrs, " + "), formatNum(sumParts), formatNum(total)))
			}
		case "formula":
			expect := 0.0
			if inv.Expect != nil {
				expect = *inv.Expect
			}
			tol := 1e-6
			if inv.Tol != nil {
				tol = *inv.Tol
			}
			got, err := SafeEval(inv.Expr)
			if err != nil {
				fail("invariants", fmt.Sprintf("formula '%s': %v", label, err))
				continue
			}
			if math.Abs(got-expect) > tol {
				fail("invariants", fmt.Sprintf("formula '%s': %s = %.6g != %s", label, inv.Expr, got, formatNum(expect)))
			}
		default:
			warn("invariants", fmt.Sprintf("unknown invariant kind: %q", inv.Kind))
		}
	}

	// 6. staleness (advisory)
	if m.SnapshotDate != "" && m.StaleAfterDays != nil {
		parsedDate, err := time.Parse("2006-01-02", m.SnapshotDate)
		if err != nil {
			warn("staleness", fmt.Sprintf("bad snapshot_date: %q", m.SnapshotDate))
		} else {
			todayDate := time.Date(today.Year(), today.Month(), today.Day(), 0, 0, 0, 0, time.UTC)
			snapDate := time.Date(parsedDate.Year(), parsedDate.Month(), parsedDate.Day(), 0, 0, 0, 0, time.UTC)
			age := int(todayDate.Sub(snapDate).Hours() / 24)
			if age > *m.StaleAfterDays {
				warn("staleness", fmt.Sprintf("snapshot %s is %dd old (> %dd) — run the refresh pass", m.SnapshotDate, age, *m.StaleAfterDays))
			}
		}
	}

	return fails, warns
}

func findFakBin(root string) string {
	if bin, err := exec.LookPath("fak"); err == nil {
		return bin
	}
	if bin, err := exec.LookPath("fak.exe"); err == nil {
		return bin
	}
	if root != "" {
		localFak := filepath.Join(root, "fak")
		if info, err := os.Stat(localFak); err == nil && !info.IsDir() {
			return localFak
		}
		localFakExe := filepath.Join(root, "fak.exe")
		if info, err := os.Stat(localFakExe); err == nil && !info.IsDir() {
			return localFakExe
		}
	}
	return ""
}

// RunLive re-probes or equality-checks cited commands against live fak.
func RunLive(root string, m *Manifest, fakBin string) ([]Finding, []string) {
	var fails []Finding
	var notes []string
	mid := filepath.Base(m.ManifestPath)
	if mid == "" || mid == "." {
		mid = filepath.Base(m.Doc)
	}

	bin := fakBin
	if bin == "" {
		bin = findFakBin(root)
	}
	if bin == "" {
		return fails, []string{fmt.Sprintf("%s: SKIP live (fak not on PATH)", mid)}
	}

	probed := 0
	equality := 0

	for srcName, src := range m.Sources {
		if len(src.Cmd) == 0 || isLiveNoTouch(src.Window) {
			continue
		}
		ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
		cmd := exec.CommandContext(ctx, bin, src.Cmd...)
		cmd.Dir = root
		var stdout, stderr bytes.Buffer
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr
		err := cmd.Run()
		cancel()

		if err != nil {
			if exitErr, ok := err.(*exec.ExitError); ok {
				notes = append(notes, fmt.Sprintf("%s:%s: SKIP live (fak exit %d)", mid, srcName, exitErr.ExitCode()))
			} else {
				notes = append(notes, fmt.Sprintf("%s:%s: SKIP live (%v)", mid, srcName, err))
			}
			continue
		}

		var live map[string]any
		if err := json.Unmarshal(stdout.Bytes(), &live); err != nil {
			notes = append(notes, fmt.Sprintf("%s:%s: SKIP live (non-JSON output)", mid, srcName))
			continue
		}

		equalityWindow := isLiveEquality(src.Window)
		for _, c := range m.Claims {
			if c.Source != srcName {
				continue
			}
			for _, n := range c.Numbers {
				if n.Field == "" {
					continue
				}
				got, err := Dotted(live, n.Field)
				if err != nil {
					fails = append(fails, Finding{
						Manifest: mid,
						Check:    "live",
						Msg:      fmt.Sprintf("[%s] cited field %s no longer emitted by `fak %s`", c.ID, n.Field, strings.Join(src.Cmd, " ")),
					})
					continue
				}
				var gotFloat float64
				switch v := got.(type) {
				case float64:
					gotFloat = v
				case int:
					gotFloat = float64(v)
				case int64:
					gotFloat = float64(v)
				default:
					fails = append(fails, Finding{
						Manifest: mid,
						Check:    "live",
						Msg:      fmt.Sprintf("[%s] cited field %s not numeric in `fak %s`", c.ID, n.Field, strings.Join(src.Cmd, " ")),
					})
					continue
				}

				if equalityWindow {
					equality++
					exp := n.Expected
					denom := math.Abs(exp)
					if denom <= 1e-9 {
						denom = 1.0
					}
					if math.Abs(gotFloat-exp)/denom > 0.01 {
						fails = append(fails, Finding{
							Manifest: mid,
							Check:    "live",
							Msg:      fmt.Sprintf("[%s] %s live=%.6g vs frozen %.6g (>1%% drift, bounded window)", c.ID, n.Field, gotFloat, exp),
						})
					}
				} else {
					probed++
				}
			}
		}
	}

	if probed > 0 || equality > 0 {
		notes = append(notes, fmt.Sprintf("%s: live probed %d open-window field(s), equality-checked %d bounded field(s)", mid, probed, equality))
	}
	return fails, notes
}

func nestFields(fields []string, src map[string]any) map[string]any {
	out := make(map[string]any)
	for _, path := range fields {
		parts := strings.Split(path, ".")
		curSrc := src
		curOut := out
		ok := true
		for _, p := range parts[:len(parts)-1] {
			child, exists := curSrc[p]
			if !exists {
				ok = false
				break
			}
			childMap, isMap := child.(map[string]any)
			if !isMap {
				ok = false
				break
			}
			curSrc = childMap

			existingOut, hasOut := curOut[p]
			if !hasOut {
				newMap := make(map[string]any)
				curOut[p] = newMap
				curOut = newMap
			} else if existingMap, isOutMap := existingOut.(map[string]any); isOutMap {
				curOut = existingMap
			} else {
				newMap := make(map[string]any)
				curOut[p] = newMap
				curOut = newMap
			}
		}
		if ok {
			lastPart := parts[len(parts)-1]
			if val, exists := curSrc[lastPart]; exists {
				curOut[lastPart] = val
			}
		}
	}
	return out
}

// RunRefresh regenerates the trimmed snapshots for a manifest from live fak output.
func RunRefresh(root string, m *Manifest, fakBin string, stdout, stderr io.Writer) int {
	mid := filepath.Base(m.ManifestPath)
	if mid == "" || mid == "." {
		mid = filepath.Base(m.Doc)
	}

	bin := fakBin
	if bin == "" {
		bin = findFakBin(root)
	}
	if bin == "" {
		fmt.Fprintf(stderr, "refresh %s: cannot run — fak not on PATH\n", mid)
		return 2
	}

	snapDir := filepath.Join(root, m.SnapshotDir)
	if err := os.MkdirAll(snapDir, 0755); err != nil {
		fmt.Fprintf(stderr, "refresh %s: cannot create snapshot dir %s: %v\n", mid, m.SnapshotDir, err)
		return 2
	}

	bySource := make(map[string][]string)
	for _, c := range m.Claims {
		for _, n := range c.Numbers {
			if n.Field != "" {
				bySource[c.Source] = append(bySource[c.Source], n.Field)
			}
		}
	}

	wrote := 0
	srcNames := make([]string, 0, len(m.Sources))
	for name := range m.Sources {
		srcNames = append(srcNames, name)
	}
	sort.Strings(srcNames)

	for _, srcName := range srcNames {
		src := m.Sources[srcName]
		if len(src.Cmd) == 0 {
			fmt.Fprintf(stdout, "  %s: SKIP (no cmd — machine-specific path)\n", srcName)
			continue
		}

		ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
		cmd := exec.CommandContext(ctx, bin, src.Cmd...)
		cmd.Dir = root
		var cmdStdout, cmdStderr bytes.Buffer
		cmd.Stdout = &cmdStdout
		cmd.Stderr = &cmdStderr
		err := cmd.Run()
		cancel()

		if err != nil {
			if exitErr, ok := err.(*exec.ExitError); ok {
				fmt.Fprintf(stderr, "  %s: SKIP (fak exit %d)\n", srcName, exitErr.ExitCode())
			} else {
				fmt.Fprintf(stderr, "  %s: SKIP (%v)\n", srcName, err)
			}
			continue
		}

		var live map[string]any
		if err := json.Unmarshal(cmdStdout.Bytes(), &live); err != nil {
			fmt.Fprintf(stderr, "  %s: SKIP (non-JSON output)\n", srcName)
			continue
		}

		captured := ""
		if genAt, ok := live["generated_at"].(string); ok {
			captured = genAt
		}

		trimmed := map[string]any{
			"_source":   "fak " + strings.Join(src.Cmd, " "),
			"_window":   src.Window,
			"_captured": captured,
		}

		fields := bySource[srcName]
		nested := nestFields(fields, live)
		for k, v := range nested {
			trimmed[k] = v
		}

		snapData, err := json.MarshalIndent(trimmed, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "  %s: marshal error: %v\n", srcName, err)
			continue
		}
		snapData = append(snapData, '\n')

		snapPath := filepath.Join(snapDir, srcName)
		if err := os.WriteFile(snapPath, snapData, 0644); err != nil {
			fmt.Fprintf(stderr, "  %s: write error: %v\n", srcName, err)
			continue
		}
		wrote++
		fmt.Fprintf(stdout, "  %s: refreshed (%d fields)\n", srcName, len(fields))
	}

	fmt.Fprintf(stdout, "refresh %s: wrote %d snapshot(s) to %s\n", mid, wrote, m.SnapshotDir)
	fmt.Fprintln(stdout, "Next: update the doc's numbers, bump snapshot_date, re-run the audit.")
	return 0
}

// Run executes the complete audit workflow according to options.
func Run(stdout, stderr io.Writer, opts AuditOptions) int {
	root := opts.Root
	if root == "" {
		root = "."
	}

	manifests, err := LoadManifests(root, opts.Manifest)
	if err != nil || len(manifests) == 0 {
		fmt.Fprintln(stderr, "doc-numbers: no manifests found under tools/docnumbers/*.json")
		return 2
	}

	if opts.Refresh {
		rc := 0
		for _, m := range manifests {
			if r := RunRefresh(root, m, opts.FakBin, stdout, stderr); r != 0 {
				rc = r
			}
		}
		return rc
	}

	today := opts.Today
	if today.IsZero() {
		today = time.Now().UTC()
	}

	var allFails []Finding
	var allWarns []Finding
	for _, m := range manifests {
		f, w := AuditManifest(root, m, today)
		allFails = append(allFails, f...)
		allWarns = append(allWarns, w...)
	}

	var liveSkips []string
	if opts.Live {
		for _, m := range manifests {
			lf, ls := RunLive(root, m, opts.FakBin)
			allFails = append(allFails, lf...)
			liveSkips = append(liveSkips, ls...)
		}
	}

	if opts.AsJSON {
		res := AuditResult{
			Fails: allFails,
			Warns: allWarns,
			Live:  liveSkips,
			OK:    len(allFails) == 0,
		}
		if res.Fails == nil {
			res.Fails = []Finding{}
		}
		if res.Warns == nil {
			res.Warns = []Finding{}
		}
		if res.Live == nil {
			res.Live = []string{}
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(res)
		if len(allFails) > 0 {
			return 1
		}
		return 0
	}

	nClaims := 0
	for _, m := range manifests {
		nClaims += len(m.Claims)
	}

	for _, w := range allWarns {
		fmt.Fprintf(stdout, "  WARN [%s] %s: %s\n", w.Check, w.Manifest, w.Msg)
	}
	for _, s := range liveSkips {
		fmt.Fprintf(stdout, "  live: %s\n", s)
	}

	if len(allFails) == 0 {
		fmt.Fprintf(stdout, "doc-numbers: clean — %d claims across %d doc(s) trace to their snapshots (%d warn).\n", nClaims, len(manifests), len(allWarns))
		return 0
	}

	fmt.Fprintf(stderr, "DOC_NUMBERS: %d FAIL(s):\n", len(allFails))
	for _, fl := range allFails {
		fmt.Fprintf(stderr, "  FAIL [%s] %s: %s\n", fl.Check, fl.Manifest, fl.Msg)
	}
	fmt.Fprintln(stderr, "  fix: correct the doc render, the manifest expected, or re-run --refresh; see tools/docnumbers/README.md.")
	return 1
}
