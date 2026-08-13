package amoprofpub

import (
	"archive/tar"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"html"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type File struct {
	Path       string `json:"path"`
	Attachment string `json:"attachment"`
	Bytes      int64  `json:"bytes"`
	SHA256     string `json:"sha256"`
	MediaType  string `json:"media_type"`
}
type Manifest struct {
	Schema      string   `json:"schema"`
	Source      string   `json:"source"`
	Generated   string   `json:"generated"`
	DefaultHTML string   `json:"default_html,omitempty"`
	Files       []File   `json:"files"`
	Pages       []string `json:"pages"`
}
type Options struct{ Input, Out, Title, Space, ParentID string }

func Generate(o Options) (Manifest, error) {
	if o.Input == "" || o.Out == "" {
		return Manifest{}, fmt.Errorf("input and out are required")
	}
	if o.Title == "" {
		o.Title = "AMOProf report"
	}
	if o.Space == "" {
		o.Space = "MPL"
	}
	stage, cleanup, err := materialize(o.Input)
	if err != nil {
		return Manifest{}, err
	}
	defer cleanup()
	files, err := inventory(stage)
	if err != nil {
		return Manifest{}, err
	}
	if len(files) == 0 {
		return Manifest{}, fmt.Errorf("AMOProf input contains no files")
	}
	defaultHTML := pickHTML(files)
	if err = os.MkdirAll(filepath.Join(o.Out, "attachments"), 0755); err != nil {
		return Manifest{}, err
	}
	for i := range files {
		src := filepath.Join(stage, filepath.FromSlash(files[i].Path))
		dst := filepath.Join(o.Out, "attachments", files[i].Attachment)
		if err = copyFile(src, dst); err != nil {
			return Manifest{}, err
		}
	}
	if st, statErr := os.Stat(o.Input); statErr == nil && !st.IsDir() {
		dst := filepath.Join(o.Out, "attachments", filepath.Base(o.Input))
		if err = copyFile(o.Input, dst); err != nil {
			return Manifest{}, err
		}
		af, fileErr := describeFile(dst, filepath.Base(o.Input), filepath.Base(o.Input))
		if fileErr != nil {
			return Manifest{}, fileErr
		}
		files = append(files, af)
	}
	sort.Slice(files, func(i, j int) bool { return files[i].Path < files[j].Path })
	tsvPath := filepath.Join(o.Out, "attachments", "ATTACHMENT-MANIFEST.tsv")
	if err = os.WriteFile(tsvPath, []byte(attachmentManifest(files)), 0644); err != nil {
		return Manifest{}, err
	}
	mf, fileErr := describeFile(tsvPath, "ATTACHMENT-MANIFEST.tsv", "ATTACHMENT-MANIFEST.tsv")
	if fileErr != nil {
		return Manifest{}, fileErr
	}
	files = append(files, mf)
	sort.Slice(files, func(i, j int) bool { return files[i].Path < files[j].Path })
	pages := []string{"index.confluence.xhtml", "source-report.confluence.xhtml", "analysis.confluence.xhtml", "usefulness.confluence.xhtml"}
	m := Manifest{Schema: "fak-amoprof-confluence/1", Source: filepath.Base(o.Input), DefaultHTML: defaultHTML, Files: files, Pages: pages}
	if err = writePage(filepath.Join(o.Out, pages[0]), meta(o.Title, o.Space, o.ParentID)+parentBody(o.Title, m)); err != nil {
		return Manifest{}, err
	}
	sourceTitle := o.Title + " â€” Original AMOProf report"
	body := sourceBody(sourceTitle, stage, defaultHTML, files)
	if err = writePage(filepath.Join(o.Out, pages[1]), meta(sourceTitle, o.Space, "PARENT_PAGE_ID")+body); err != nil {
		return Manifest{}, err
	}
	usefulTitle := o.Title + " â€” What this report is useful for"
	analysisTitle := o.Title + " — AMOProf analysis"
	if err = writePage(filepath.Join(o.Out, pages[2]), meta(analysisTitle, o.Space, "PARENT_PAGE_ID")+analysisBody(analysisTitle, stage, files)); err != nil {
		return Manifest{}, err
	}
	if err = writePage(filepath.Join(o.Out, pages[3]), meta(usefulTitle, o.Space, "PARENT_PAGE_ID")+usefulnessBody(usefulTitle, stage, files)); err != nil {
		return Manifest{}, err
	}
	b, _ := json.MarshalIndent(m, "", "  ")
	b = append(b, '\n')
	if err = os.WriteFile(filepath.Join(o.Out, "manifest.json"), b, 0644); err != nil {
		return Manifest{}, err
	}
	return m, nil
}
func materialize(input string) (string, func(), error) {
	st, e := os.Stat(input)
	if e != nil {
		return "", func() {}, e
	}
	if st.IsDir() {
		return input, func() {}, nil
	}
	f, e := os.Open(input)
	if e != nil {
		return "", func() {}, e
	}
	defer f.Close()
	gz, e := gzip.NewReader(f)
	if e != nil {
		return "", func() {}, fmt.Errorf("input must be a directory or .tgz: %w", e)
	}
	defer gz.Close()
	d, e := os.MkdirTemp("", "fak-amoprof-")
	if e != nil {
		return "", func() {}, e
	}
	clean := func() { os.RemoveAll(d) }
	tr := tar.NewReader(gz)
	for {
		h, e := tr.Next()
		if e == io.EOF {
			break
		}
		if e != nil {
			clean()
			return "", func() {}, e
		}
		p := filepath.Clean(filepath.Join(d, filepath.FromSlash(h.Name)))
		if !strings.HasPrefix(p, d+string(os.PathSeparator)) {
			clean()
			return "", func() {}, fmt.Errorf("unsafe archive path %q", h.Name)
		}
		if h.Typeflag == tar.TypeDir {
			os.MkdirAll(p, 0755)
			continue
		}
		if h.Typeflag != tar.TypeReg {
			continue
		}
		os.MkdirAll(filepath.Dir(p), 0755)
		w, e := os.Create(p)
		if e != nil {
			clean()
			return "", func() {}, e
		}
		_, e = io.Copy(w, tr)
		w.Close()
		if e != nil {
			clean()
			return "", func() {}, e
		}
	}
	return d, clean, nil
}
func inventory(root string) ([]File, error) {
	var out []File
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, e error) error {
		if e != nil {
			return e
		}
		if d.IsDir() {
			return nil
		}
		r, _ := filepath.Rel(root, p)
		r = filepath.ToSlash(r)
		f, e := os.Open(p)
		if e != nil {
			return e
		}
		h := sha256.New()
		n, e := io.Copy(h, f)
		f.Close()
		if e != nil {
			return e
		}
		out = append(out, File{Path: r, Attachment: attachmentName(r), Bytes: n, SHA256: hex.EncodeToString(h.Sum(nil)), MediaType: media(r)})
		return nil
	})
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out, err
}
func attachmentName(p string) string {
	return strings.ReplaceAll(strings.ReplaceAll(filepath.ToSlash(p), "/", "__"), " ", "_")
}
func media(p string) string {
	switch strings.ToLower(filepath.Ext(p)) {
	case ".html", ".htm":
		return "text/html"
	case ".json":
		return "application/json"
	case ".csv":
		return "text/csv"
	case ".png":
		return "image/png"
	case ".svg":
		return "image/svg+xml"
	case ".pdf":
		return "application/pdf"
	default:
		return "application/octet-stream"
	}
}
func pickHTML(f []File) string {
	best := ""
	score := -1
	for _, x := range f {
		if x.MediaType != "text/html" {
			continue
		}
		s := 0
		b := strings.ToLower(filepath.Base(x.Path))
		if b == "index.html" {
			s += 100
		}
		if strings.Contains(b, "report") {
			s += 70
		}
		if strings.Contains(b, "amoprof") {
			s += 40
		}
		s -= strings.Count(x.Path, "/") * 5
		if s > score {
			score = s
			best = x.Path
		}
	}
	return best
}
func meta(title, space, parent string) string {
	p := ""
	if parent != "" {
		p = "parent_id: " + parent + "\n"
	}
	return fmt.Sprintf("<!-- confluence-meta\ntitle: %s\nspace: %s\n%slabels: amoprof, performance, generated\n-->\n", title, space, p)
}
func parentBody(title string, m Manifest) string {
	var b strings.Builder
	b.WriteString("<h1>" + html.EscapeString(title) + "</h1>")
	b.WriteString("<p>This page is the simple entry point for the captured AMOProf run. Start with the original report, then use the analysis and practical-use guide for interpretation.</p>")
	childTitles := []string{title + " — Original AMOProf report", title + " — AMOProf analysis", title + " — What this report is useful for"}
	b.WriteString("<h2>Report pages</h2><ul>")
	for _, child := range childTitles {
		b.WriteString("<li><ac:link><ri:page ri:content-title=\"" + xmlAttr(child) + "\"/><ac:plain-text-link-body><![CDATA[" + child + "]]></ac:plain-text-link-body></ac:link></li>")
	}
	b.WriteString("</ul>")
	if m.DefaultHTML != "" {
		b.WriteString("<p><strong>Default report:</strong> " + html.EscapeString(m.DefaultHTML) + ". It is reproduced first on the Original AMOProf report child page.</p>")
	}
	b.WriteString(fmt.Sprintf("<h2>All source files (%d)</h2>", len(m.Files)))
	b.WriteString("<p>Every captured file is listed below with its size and digest. Select a file name to download the exact attachment.</p>")
	b.WriteString("<table><tbody><tr><th>Source path</th><th>Attachment</th><th>Bytes</th><th>SHA-256</th></tr>")
	for _, f := range m.Files {
		b.WriteString("<tr><td><code>" + html.EscapeString(f.Path) + "</code></td><td><ac:link><ri:attachment ri:filename=\"" + xmlAttr(f.Attachment) + "\"/><ac:plain-text-link-body><![CDATA[" + f.Attachment + "]]></ac:plain-text-link-body></ac:link></td><td>" + fmt.Sprint(f.Bytes) + "</td><td><code>" + f.SHA256 + "</code></td></tr>")
	}
	b.WriteString("</tbody></table>")
	return b.String()
}
func sourceBody(title, root, def string, files []File) string {
	var b strings.Builder
	b.WriteString("<h1>" + html.EscapeString(title) + "</h1>\n")
	if def == "" {
		b.WriteString("<p>No HTML entry page was present. The source files remain linked from the parent page.</p>")
		return b.String()
	}
	var att string
	for _, f := range files {
		if f.Path == def {
			att = f.Attachment
		}
	}
	fmt.Fprintf(&b, "<p><strong>Source:</strong> <ac:link><ri:attachment ri:filename=\"%s\"/><ac:plain-text-link-body><![CDATA[%s (open original HTML)]]></ac:plain-text-link-body></ac:link></p>\n", xmlEscape(att), def)
	raw, _ := os.ReadFile(filepath.Join(root, filepath.FromSlash(def)))
	blocks := visibleBlocks(string(raw))
	if len(blocks) == 0 {
		b.WriteString("<p>The HTML report is interactive or script-only; use the original attachment above.</p>")
		return b.String()
	}
	b.WriteString("<p>The sections below reproduce the human-visible content in source order. Interactive JavaScript is intentionally not executed by Confluence.</p>\n")
	for _, x := range blocks {
		switch x.Tag {
		case "h1", "h2", "h3", "h4":
			fmt.Fprintf(&b, "<%s>%s</%s>\n", x.Tag, html.EscapeString(x.Text), x.Tag)
		default:
			fmt.Fprintf(&b, "<p>%s</p>\n", html.EscapeString(x.Text))
		}
	}
	return b.String()
}

type block struct{ Tag, Text string }

func visibleBlocks(s string) []block {
	dec := xml.NewDecoder(strings.NewReader(s))
	dec.Strict = false
	var out []block
	skip := 0
	tag := ""
	var text strings.Builder
	flush := func() {
		v := strings.Join(strings.Fields(text.String()), " ")
		if v != "" {
			out = append(out, block{tag, v})
		}
		text.Reset()
	}
	for {
		t, e := dec.Token()
		if e != nil {
			break
		}
		switch v := t.(type) {
		case xml.StartElement:
			n := strings.ToLower(v.Name.Local)
			if n == "script" || n == "style" {
				skip++
			}
			if skip == 0 && (n == "h1" || n == "h2" || n == "h3" || n == "h4" || n == "p" || n == "li" || n == "th" || n == "td") {
				flush()
				tag = n
			}
		case xml.EndElement:
			n := strings.ToLower(v.Name.Local)
			if skip == 0 && n == tag {
				flush()
				tag = ""
			}
			if (n == "script" || n == "style") && skip > 0 {
				skip--
			}
		case xml.CharData:
			if skip == 0 && tag != "" {
				text.Write(v)
			}
		}
	}
	flush()
	return out
}
func usefulnessBody(title, root string, files []File) string {
	jsonN, csvN, htmlN := 0, 0, 0
	for _, f := range files {
		switch f.MediaType {
		case "application/json":
			jsonN++
		case "text/csv":
			csvN++
		case "text/html":
			htmlN++
		}
	}
	return fmt.Sprintf("<h1>%s</h1><ac:structured-macro ac:name=\"info\"><ac:rich-text-body><p><strong>Bottom line:</strong> useful for understanding the observed resource envelope and finding follow-up questions; not sufficient by itself to prove model quality, causality, or a production-wide performance gain.</p></ac:rich-text-body></ac:structured-macro><h2>What is here</h2><p>%d files: %d HTML report(s), %d JSON file(s), and %d CSV file(s). Every file is linked from the parent page.</p><h2>Good uses</h2><ul><li>Locate CPU, memory, disk, network, or accelerator pressure during the captured window.</li><li>Correlate profiler observations with the workload timeline when timestamps overlap.</li><li>Identify a smaller experiment or metric that should be measured next.</li><li>Preserve raw evidence for independent re-analysis.</li></ul><h2>Do not infer</h2><ul><li>Request completion is not code correctness or benchmark quality.</li><li>Correlation in one run is not causation.</li><li>A single machine and workload do not establish a general production gain.</li><li>Missing samples are not proof that a resource had no activity.</li></ul><h2>Usefulness rating</h2><table><tbody><tr><th>Question</th><th>Usefulness</th><th>Why</th></tr><tr><td>What happened on this host during this run?</td><td>High</td><td>This is the report's direct evidence boundary.</td></tr><tr><td>Where should we investigate next?</td><td>High</td><td>Resource peaks and timing narrow the search.</td></tr><tr><td>Did AMOProf or fak cause a latency change?</td><td>Low without a controlled baseline</td><td>The report alone does not isolate causality.</td></tr><tr><td>Is this model or system better overall?</td><td>Low</td><td>That needs correctness and tuned comparative benchmarks.</td></tr></tbody></table>", html.EscapeString(title), len(files), htmlN, jsonN, csvN)
}
func xmlEscape(s string) string {
	var b strings.Builder
	xml.EscapeText(&b, []byte(s))
	return b.String()
}
func writePage(p, s string) error { return os.WriteFile(p, []byte(s), 0644) }
func copyFile(a, b string) error {
	r, e := os.Open(a)
	if e != nil {
		return e
	}
	defer r.Close()
	w, e := os.Create(b)
	if e != nil {
		return e
	}
	defer w.Close()
	_, e = io.Copy(w, r)
	return e
}

func xmlAttr(s string) string {
	var b strings.Builder
	xml.EscapeText(&b, []byte(s))
	return b.String()
}

func describeFile(path, source, attachment string) (File, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return File{}, err
	}
	s := sha256.Sum256(b)
	return File{Path: source, Attachment: attachment, Bytes: int64(len(b)), SHA256: hex.EncodeToString(s[:]), MediaType: media(source)}, nil
}
func attachmentManifest(files []File) string {
	var b strings.Builder
	b.WriteString("attachment_name\trelative_path\tbytes\tsha256\n")
	for _, f := range files {
		fmt.Fprintf(&b, "%s\t%s\t%d\t%s\n", f.Attachment, f.Path, f.Bytes, f.SHA256)
	}
	return b.String()
}
