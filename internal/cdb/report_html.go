package cdb

// report_html.go is the "context debugger as a product" surface (#574): it
// renders an attached core image as a SELF-CONTAINED, static HTML inspection
// report — the shareable artifact a teammate opens in a browser (no fak
// install, no model, no JS, no external CSS). It is the polished standalone
// form of the inspection capabilities the package already ships (Info /
// Backtrace / Examine / WorkingSet / Grep), assembled into one document.
//
// Trust boundary (load-bearing): HTMLReport routes every byte it prints through
// the SAME Examine gate the CLI uses. A sealed page appears ONLY as its safe
// sealed-metadata descriptor (the same Frame.Descriptor Backtrace already
// returns — the gate's reason code + the sealed-bytes length, never the bytes);
// it is never paged in, never previewed, never echoed. A benign page is
// demand-paged through Examine (byte-identical + re-screened), then truncated to
// a text preview that html/template auto-escapes. So the report can never leak
// poison the same way the page table cannot.
//
// Honesty (load-bearing, per the issue's "honest fences"): the seal/quarantine
// panel renders each sealed page with its reason code AND carries the verbatim
// SealPanelNote — the decision cdb surfaces is INHERITED from an evadable
// detector; cdb makes it durable and queryable, not more correct. The known
// false-positive base64-image seals on a real session are surfaced as such,
// not hidden.

import (
	"context"
	"fmt"
	"html/template"
	"io"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
)

// SealPanelNote is the verbatim honest disclosure #574 requires the UI to carry
// next to every sealed page. Surfaced, not buried in a footnote: the seal
// decision is INHERITED from an ~100% evadable detector; cdb makes it durable
// and queryable, not more correct. A sealed page in this report is a frozen,
// inspectable record of what the gate decided at WRITE time — not an improved
// verdict, and never a re-screen.
const SealPanelNote = "The seal decision is inherited from an ~100% evadable " +
	"detector (obfuscation, pure-semantic paraphrase, and base64/high-entropy " +
	"false positives all leak through). cdb makes the decision durable and " +
	"queryable; it does not make it more correct. A sealed page here is a " +
	"frozen record of what the gate decided at write time, surfaced — not hidden."

// htmlExamineRow is one row of the examine panel: a benign page previews its
// leading text (truncated, html-escaped); a sealed page is REFUSED at the gate
// and shows no bytes. The descriptor is the safe sealed-metadata one.
type htmlExamineRow struct {
	Step    int
	Role    string
	Digest  string // short content address from the page table (Backtrace.Frame)
	Len     int64
	Sealed  bool
	Outcome string // "RESOLVED" or "REFUSED"
	Preview string // leading text of a benign page; an honest note for a sealed one
	Reason  string // reason code, only for sealed rows
}

// htmlReportData is the full template binding. Every field is either already
// gate-safe (Info, Frame, WorkingSet residency counters) or routed through
// Examine (htmlExamineRow.Preview on benign pages only).
type htmlReportData struct {
	GeneratedAt  string
	AppVersion   string
	SessionID    string
	ImageDir     string
	Source       string
	Query        string
	Info         Info
	WorkingSet   WorkingSet
	Frames       []Frame
	SealedFrames []Frame
	ExamineRows  []htmlExamineRow
	SealNote     string
}

// HTMLReport writes a self-contained static HTML inspection report to w.
//
// query is the follow-up question the working-set residency view demand-pages
// for. imageDir/source are recorded into the report header so a teammate
// receiving the artifact can re-attach (the core image is the shareable form
// too — `fak debug --dir <dir>` reopens it).
//
// The report is idempotent over a frozen image: an attached image never writes,
// so two HTMLReport calls on the same image produce byte-identical inspection
// rows (the GeneratedAt timestamp is the only non-deterministic field).
func (im *Image) HTMLReport(ctx context.Context, query, imageDir, source string, w io.Writer) error {
	info := im.Info()
	frames := im.Backtrace()
	ws := im.WorkingSet(ctx, query, 0)

	sealed := make([]Frame, 0)
	for _, f := range frames {
		if f.Sealed {
			sealed = append(sealed, f)
		}
	}

	// The examine panel: route every page through the SAME gate the CLI uses.
	// A sealed page MUST stay REFUSED with no bytes paged in — this is the only
	// place the report could leak poison, and it does not, by construction.
	exRows := make([]htmlExamineRow, 0, len(frames))
	for _, f := range frames {
		row := htmlExamineRow{Step: f.Step, Role: f.Role, Digest: f.Digest, Len: f.Len, Sealed: f.Sealed}
		if f.Sealed {
			row.Outcome = "REFUSED"
			row.Reason = f.Reason
			row.Preview = "(sealed — bytes withheld at the gate; the descriptor above is sealed-metadata only, never the content)"
			exRows = append(exRows, row)
			continue
		}
		b, err := im.Examine(ctx, f.Step) // gated page-in: byte-identical + re-screened
		if err != nil {
			// A benign page that fails page-in (e.g. a tombstoned page after a
			// context-control change) is also honestly REFUSED — no preview.
			row.Outcome = "REFUSED"
			row.Preview = "(" + err.Error() + ")"
			exRows = append(exRows, row)
			continue
		}
		row.Outcome = "RESOLVED"
		row.Preview = previewText(b, 480)
		exRows = append(exRows, row)
	}

	data := htmlReportData{
		GeneratedAt:  time.Now().UTC().Format(time.RFC3339),
		AppVersion:   appversion.Current(),
		SessionID:    info.SessionID,
		ImageDir:     imageDir,
		Source:       source,
		Query:        query,
		Info:         info,
		WorkingSet:   ws,
		Frames:       frames,
		SealedFrames: sealed,
		ExamineRows:  exRows,
		SealNote:     SealPanelNote,
	}
	tmpl, err := template.New("cdb-report").Parse(reportHTMLTemplate)
	if err != nil {
		return fmt.Errorf("cdb html report: template parse: %w", err)
	}
	if err := tmpl.Execute(w, data); err != nil {
		return fmt.Errorf("cdb html report: render: %w", err)
	}
	return nil
}

// previewText returns a single-line, length-bounded preview of a paged-in
// benign page. Newlines/tabs collapse to spaces; non-printable bytes are
// dropped, so a 6.5KB web-search wall or a base64-image page does not blow up
// the report. The html/template engine escapes the result again at render
// time, so the only transform here is shape, not safety.
func previewText(b []byte, max int) string {
	var sb strings.Builder
	sb.Grow(max)
	n := 0
	for _, r := range string(b) {
		if r == '\n' || r == '\t' || r == '\r' {
			r = ' '
		}
		if r < 0x20 {
			continue
		}
		if n >= max {
			break
		}
		sb.WriteRune(r)
		n++
	}
	s := strings.TrimSpace(sb.String())
	if len(b) > max {
		// +N B more on disk than the preview shows.
		s = s + fmt.Sprintf(" … (+%d B)", len(b)-len(s))
	}
	return s
}

const reportHTMLTemplate = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>fak cdb — {{.SessionID}} · context inspection report</title>
<style>
:root {
  --bg: #0d1117; --panel: #161b22; --panel2: #21262d; --border: #30363d;
  --fg: #e6edf3; --muted: #8b949e; --accent: #58a6ff; --good: #3fb950;
  --warn: #d29922; --bad: #f85149; --seal: #bc8cff;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--fg); margin: 0; padding: 0;
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 24px 20px 80px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 28px 0 10px; padding-bottom: 6px; border-bottom: 1px solid var(--border); color: var(--accent); }
h3 { font-size: 13px; margin: 16px 0 8px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
.sub { color: var(--muted); font-size: 13px; margin-bottom: 16px; }
.sub code { background: var(--panel2); padding: 1px 6px; border-radius: 4px; }
.kv { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; background: var(--panel);
  border: 1px solid var(--border); border-radius: 8px; padding: 14px 18px; }
.kv dt { color: var(--muted); }
.kv dd { margin: 0; word-break: break-word; }
.kv .num { font-variant-numeric: tabular-nums; }
table { width: 100%; border-collapse: collapse; background: var(--panel);
  border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }
th { background: var(--panel2); color: var(--muted); font-weight: 600; font-size: 12px;
  text-transform: uppercase; letter-spacing: 0.05em; }
tr:last-child td { border-bottom: none; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
td.mono, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
.tag { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 11px;
  font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; }
.tag-seal  { background: rgba(188,140,255,0.15); color: var(--seal); border: 1px solid var(--seal); }
.tag-tomb  { background: rgba(248,81,73,0.12); color: var(--bad); border: 1px solid var(--bad); }
.tag-heavy { background: rgba(210,153,34,0.12); color: var(--warn); border: 1px solid var(--warn); }
.tag-ok    { background: rgba(63,185,80,0.12); color: var(--good); border: 1px solid var(--good); }
.note { background: var(--panel); border: 1px solid var(--border); border-left: 3px solid var(--warn);
  border-radius: 6px; padding: 12px 16px; color: var(--fg); }
.note strong { color: var(--warn); }
.fence { background: var(--panel); border: 1px solid var(--seal); border-radius: 6px;
  padding: 12px 16px; color: var(--fg); }
.fence strong { color: var(--seal); }
.preview { color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px; word-break: break-word; }
.refused { color: var(--bad); font-weight: 600; }
.resolved { color: var(--good); font-weight: 600; }
.residency { font-size: 28px; font-weight: 700; color: var(--accent); font-variant-numeric: tabular-nums; }
.muted { color: var(--muted); }
hr { border: none; border-top: 1px solid var(--border); margin: 24px 0; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
footer { color: var(--muted); font-size: 12px; margin-top: 40px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>fak cdb — context inspection report</h1>
  <p class="sub">session <code>{{.SessionID}}</code> · generated {{.GeneratedAt}} · fak {{.AppVersion}}
    {{if .ImageDir}}· core image <code>{{.ImageDir}}</code>{{end}}{{if .Source}} · source <code>{{.Source}}</code>{{end}}</p>

  <p class="muted">A time-travel inspector over a finished agent session, attached as a
    <strong>core image</strong>: a small page table over a content-addressed swap device. The
    session is frozen (no more writes will ever happen), so this report is a durable snapshot
    of what the agent's context looked like — the bytes each tool result produced, what the
    trust gate let through, and what it sealed.</p>

  <h2>1 · Core image (the decomposition)</h2>
  <dl class="kv">
    <dt>Session</dt>           <dd class="mono">{{.Info.SessionID}} <span class="muted">(world version {{.Info.WorldVer}}, frozen)</span></dd>
    <dt>Pages</dt>             <dd class="num">{{.Info.Pages}} <span class="muted">=</span> {{.Info.Benign}} benign <span class="muted">+</span> {{.Info.Sealed}} sealed <span class="muted">+</span> {{.Info.Tombstoned}} tombstoned <span class="muted">(cleared: {{.Info.Cleared}})</span></dd>
    <dt>Heavy / paged out</dt> <dd class="num">{{.Info.Heavy}} <span class="muted">page(s) oversize at write time → swapped to the CAS device</span></dd>
    <dt>Raw tool-result bytes</dt> <dd class="num">{{.Info.RawBytes}} B</dd>
    <dt>Swap device (dedup'd CAS)</dt> <dd class="num">{{.Info.CASBytes}} B <span class="muted">across {{.Info.DistinctBlobs}} distinct blob(s); dedup saved {{.Info.DedupSaved}} B</span></dd>
    <dt>Resident (benign) bytes</dt> <dd class="num">{{.Info.ResidentBytes}} B <span class="muted">— the demand-pageable universe</span></dd>
    <dt>Page table on disk</dt> <dd class="num">{{.Info.ManifestFileBytes}} B <span class="muted">— the map you always carry</span></dd>
    <dt>Swap file on disk</dt>  <dd class="num">{{.Info.CASFileBytes}} B <span class="muted">(base64-inflated)</span></dd>
  </dl>

  <h2>2 · Page table — the backtrace timeline (bt)</h2>
  <p class="muted">One row per tool result, in step order. The map carries NO bytes: a sealed
    page's row is its safe sealed-metadata descriptor only, so reading this table can never
    surface poison.</p>
  <table>
    <thead><tr>
      <th class="num">step</th><th>state</th><th>role</th><th>descriptor</th>
      <th class="num">bytes</th><th>digest</th>{{if .Frames}}<th>seal reason</th>{{end}}
    </tr></thead>
    <tbody>
    {{range .Frames}}
      <tr>
        <td class="num">{{.Step}}</td>
        <td>
          {{if .Tombstoned}}<span class="tag tag-tomb">tomb</span>
          {{else if .Sealed}}<span class="tag tag-seal">sealed</span>
          {{else if .Heavy}}<span class="tag tag-heavy">heavy</span>
          {{else}}<span class="tag tag-ok">benign</span>{{end}}
        </td>
        <td class="mono">{{.Role}}</td>
        <td>{{.Descriptor}}</td>
        <td class="num mono">{{.Len}}</td>
        <td class="mono">{{.Digest}}{{if .QID}} · {{.QID}}{{end}}</td>
        <td class="mono muted">{{if .Sealed}}{{.Reason}}{{else}}—{{end}}</td>
      </tr>
    {{end}}
    </tbody>
  </table>

  <h2>3 · Quarantine / seal panel — what got sealed and why</h2>
  {{if .SealedFrames}}
  <table>
    <thead><tr>
      <th class="num">step</th><th>role</th><th>seal reason</th><th>qid</th>
      <th class="num">bytes</th><th>descriptor (safe sealed-metadata)</th>
    </tr></thead>
    <tbody>
    {{range .SealedFrames}}
      <tr>
        <td class="num">{{.Step}}</td>
        <td class="mono">{{.Role}}</td>
        <td class="mono"><strong>{{.Reason}}</strong></td>
        <td class="mono">{{.QID}}</td>
        <td class="num mono">{{.Len}}</td>
        <td>{{.Descriptor}}</td>
      </tr>
    {{end}}
    </tbody>
  </table>
  {{else}}
  <p class="muted">No pages were sealed on this session.</p>
  {{end}}
  <div class="fence">
    <strong>Honest fence.</strong> {{.SealNote}}
  </div>

  <h2>4 · Working-set residency — what a follow-up actually paged in</h2>
  <p class="muted">The follow-up question demand-pages its working set
    <strong>W(query)</strong> from the cold swap device — the pages whose content words the
    query references, gated on every page-in. The number below is how little of the resident
    image the follow-up actually had to fault in; the rest stayed cold.</p>
  <dl class="kv">
    <dt>Follow-up query</dt>     <dd class="mono">{{.Query}}</dd>
    <dt>Residency</dt>           <dd><span class="residency">{{printf "%.2f" .WorkingSet.ResidencyPct}}%</span>
                                   <span class="muted">= {{.WorkingSet.BytesPagedIn}} B paged in / {{.WorkingSet.ResidentBytes}} B resident</span></dd>
    <dt>Pages touched</dt>       <dd class="num">{{.WorkingSet.PagesTouched}} of {{.WorkingSet.PagesBenign}} benign
                                   <span class="muted">(total {{.WorkingSet.PagesTotal}})</span></dd>
    <dt>Excluded by the gate</dt><dd class="num">{{.WorkingSet.SealedSkipped}} sealed · {{.WorkingSet.TombstonedSkipped}} tombstoned</dd>
    <dt>Faults avoided</dt>      <dd class="num">{{.WorkingSet.FaultsAvoided}} <span class="muted">(benign pages never referenced → never faulted in)</span></dd>
    <dt>Poison in working set</dt><dd class="num {{if .WorkingSet.PoisonInSet}}refused{{else}}resolved{{end}}">{{.WorkingSet.PoisonInSet}}
                                   <span class="muted">(false by construction; re-checked, not assumed)</span></dd>
  </dl>
  {{if .WorkingSet.Slices}}
  <h3>Pages this follow-up faulted in</h3>
  <table>
    <thead><tr><th class="num">step</th><th>role</th><th>descriptor</th></tr></thead>
    <tbody>
    {{range .WorkingSet.Slices}}
      <tr><td class="num">{{.Step}}</td><td class="mono">{{.Role}}</td><td>{{.Descriptor}}</td></tr>
    {{end}}
    </tbody>
  </table>
  {{end}}

  <h2>5 · Examine — demand-paged previews (the gate on every page-in)</h2>
  <p class="muted">Each benign page demand-paged through the gate and previewed below
    (truncated, html-escaped). Sealed pages are <span class="refused">REFUSED</span> at the
    gate — their bytes are withheld, by construction, on this attached image as on the live
    one. A clearance does not launder poison: <code>Examine</code> re-screens on page-in.</p>
  <table>
    <thead><tr>
      <th class="num">step</th><th>verdict</th><th>role</th>
      <th class="num">bytes</th><th>digest</th><th>preview (benign) / note (sealed)</th>
    </tr></thead>
    <tbody>
    {{range .ExamineRows}}
      <tr>
        <td class="num">{{.Step}}</td>
        <td>{{if .Sealed}}<span class="refused">REFUSED</span>{{else if eq .Outcome "RESOLVED"}}<span class="resolved">RESOLVED</span>{{else}}<span class="refused">REFUSED</span>{{end}}</td>
        <td class="mono">{{.Role}}</td>
        <td class="num mono">{{.Len}}</td>
        <td class="mono">{{.Digest}}{{if .Reason}} · {{.Reason}}{{end}}</td>
        <td class="preview">{{.Preview}}</td>
      </tr>
    {{end}}
    </tbody>
  </table>

  <footer>
    <p>Generated by <code>fak debug --cmd html</code> — a static, self-contained artifact.
       The core image itself is the shareable form too: a teammate can re-attach with
       <code>fak debug --dir &lt;image-dir&gt;</code> and re-run any of these panels. The
       trust gate still stands on every page-in from the reloaded image.</p>
  </footer>
</div>
</body>
</html>
`
