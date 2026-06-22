// Command lensviz is a visual next-token debugger for fak's own in-kernel engine.
// It runs ONE forward pass over a prompt and then applies the "logit lens": every
// layer's residual stream is projected through the model's final-norm + LM head, so
// you can watch what token the model would predict at each depth — and at what
// confidence the final answer crystallizes as you ascend the stack.
//
// Two views, from the same forward pass:
//
//   - Terminal: for one position (default: the last prompt token, i.e. the next-token
//     prediction), a per-layer table of the top-k candidates with probability bars,
//     plus a p(final-token) column showing where in depth the model "decides".
//   - HTML (-html out.html): the canonical logit-lens grid — rows = layers,
//     columns = positions — each cell the top-1 predicted token there, shaded by its
//     probability. This is the layer×position "belief map" of the whole prefill.
//
// Example:
//
//	go build -o lensviz ./cmd/lensviz
//	./lensviz -hf ~/.cache/fak-models/qwen2.5-1.5b-instruct \
//	          -tok ~/.cache/fak-models/tokenizers/qwen2.5 \
//	          -p "The capital of France is" -raw -k 6 -html lens.html
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"html"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/demoui"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

func main() {
	hf := flag.String("hf", "", "HuggingFace model dir (config.json + model.safetensors[.index.json])")
	gguf := flag.String("gguf", "", "GGUF checkpoint path; loads through the memory-lean quant path")
	tokDir := flag.String("tok", "", "tokenizer dir containing tokenizer.json (default: -hf dir, or GGUF sidecar)")
	sys := flag.String("sys", "You are a helpful assistant.", "system prompt (ChatML mode)")
	prompt := flag.String("p", "", "prompt text — REQUIRED")
	raw := flag.Bool("raw", false, "feed the prompt verbatim (no ChatML wrapping) — best for completion-style logit-lens")
	k := flag.Int("k", 5, "top-k candidates to show per layer")
	pos := flag.Int("pos", -1, "position to inspect (token index); -1 = last token (next-token prediction)")
	htmlOut := flag.String("html", "", "also write the full layer×position heatmap to this HTML file")
	flag.Parse()

	// Expand a leading ~ in the path flags (Go/PowerShell don't), so `-hf ~/...`,
	// `-gguf ~/...`, `-tok ~/...`, and `-html ~/out.html` resolve as intended.
	*hf = pathutil.ExpandTilde(*hf)
	*gguf = pathutil.ExpandTilde(*gguf)
	*tokDir = pathutil.ExpandTilde(*tokDir)
	*htmlOut = pathutil.ExpandTilde(*htmlOut)

	if (*hf == "") == (*gguf == "") || *prompt == "" {
		fmt.Fprintln(os.Stderr, "usage: lensviz (-hf <model-dir> | -gguf <model.gguf>) -p <prompt> [-tok <dir>] [-raw] [-k N] [-pos P] [-html out.html]")
		os.Exit(2)
	}

	cfg, err := readModelConfig(*hf, *gguf)
	check("config", err)

	// Surface the real compute surface up front (this box is CPU-only — the summary
	// says so plainly rather than implying a GPU that isn't there).
	fmt.Fprintf(os.Stderr, "hardware: %s\n", demoui.Probe().Summary)

	// Model load + quantize is the longest silent phase (tens of seconds on the big
	// rungs); spin a live stderr counter so the terminal isn't dead air while it loads.
	stopLoad := demoui.Spinner(os.Stderr, "Loading model")
	m, err := loadModel(*hf, *gguf, cfg)
	stopLoad()
	check("load", err)

	td := resolveTokDir(*tokDir, *hf, *gguf)
	if td == "" {
		fmt.Fprintln(os.Stderr, "tokenizer: could not locate tokenizer.json; pass -tok <dir>")
		os.Exit(2)
	}
	tok, err := tokenizer.LoadJSON(filepath.Join(td, "tokenizer.json"))
	check("tokenizer", err)

	text := *prompt
	if !*raw {
		text = "<|im_start|>system\n" + *sys + "<|im_end|>\n" +
			"<|im_start|>user\n" + *prompt + "<|im_end|>\n" +
			"<|im_start|>assistant\n"
	}
	ids, err := tok.Encode(text)
	check("encode", err)
	if len(ids) == 0 {
		fmt.Fprintln(os.Stderr, "encode: empty prompt")
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "model=%s  layers=%d  vocab=%d  prompt_tokens=%d\n",
		cfg.ModelType, cfg.NumLayers, cfg.VocabSize, len(ids))

	// The forward pass is the other silent stretch; spin it too so the screen keeps moving.
	stopFwd := demoui.Spinner(os.Stderr, "Running forward pass")
	act := m.Forward(ids)
	stopFwd()

	inspect := *pos
	if inspect < 0 {
		inspect = act.Seq - 1
	}
	if inspect >= act.Seq {
		fmt.Fprintf(os.Stderr, "pos %d out of range (seq=%d)\n", inspect, act.Seq)
		os.Exit(1)
	}

	renderTerminal(m, tok, act, ids, inspect, *k)

	if *htmlOut != "" {
		if err := writeHTML(*htmlOut, m, tok, act, ids); err != nil {
			check("html", err)
		}
		fmt.Fprintf(os.Stderr, "\nwrote layer×position heatmap → %s\n", *htmlOut)
	}
}

// renderTerminal prints the per-layer top-k table for one position, plus a column
// tracking the probability the FINAL predicted token already carries at each depth.
func renderTerminal(m *model.Model, tok *tokenizer.Tokenizer, act *model.Activations, ids []int, pos, k int) {
	lens := m.LayerLogits(act, pos)
	if len(lens) == 0 {
		fmt.Fprintln(os.Stderr, "no activations")
		return
	}
	final := lens[len(lens)-1]
	finalTop := model.TopK(final, 1)
	finalID := finalTop[0].ID
	finalStr := tokDisplay(tok, finalID)

	fmt.Printf("\nPosition %d = %q  →  predicting the NEXT token\n", pos, tokDisplay(tok, ids[pos]))
	fmt.Printf("Final prediction: %s  (p=%.3f)\n\n", colorTok(finalStr, true), finalTop[0].Prob)
	fmt.Printf("Logit lens — top-%d per layer.  p(final) tracks %q crystallizing through depth.\n\n", k, finalStr)
	fmt.Printf(" %-5s | %-13s | top-%d predictions\n", "layer", "p(final)", k)
	fmt.Printf(" %s+%s+%s\n", strings.Repeat("-", 6), strings.Repeat("-", 15), strings.Repeat("-", 44))

	for l, lg := range lens {
		top := model.TopK(lg, k)
		pFinal := softmaxProbOf(lg, finalID)
		var sb strings.Builder
		for i, tp := range top {
			if i > 0 {
				sb.WriteString("  ")
			}
			sb.WriteString(colorTok(tokDisplay(tok, tp.ID), tp.ID == finalID))
			sb.WriteString(fmt.Sprintf("(%.2f)", tp.Prob))
		}
		fmt.Printf(" %-5s | %s %-6.4f | %s\n", layerLabel(l, len(lens)), bar(pFinal), pFinal, sb.String())
	}
	fmt.Println()
}

// softmaxProbOf returns the full-vocab softmax probability of token id within logits.
func softmaxProbOf(logits []float32, id int) float32 {
	if id < 0 || id >= len(logits) {
		return 0
	}
	var max float32 = logits[0]
	for _, v := range logits {
		if v > max {
			max = v
		}
	}
	var sum float64
	for _, v := range logits {
		sum += math.Exp(float64(v - max))
	}
	if sum == 0 {
		return 0
	}
	return float32(math.Exp(float64(logits[id]-max)) / sum)
}

// layerLabel names a lens row: L0 is the raw embedding output, the last index is the
// post-final-block stream (the model's real prediction).
func layerLabel(l, n int) string {
	switch l {
	case 0:
		return "emb"
	case n - 1:
		return "fin"
	default:
		return "L" + itoa(l)
	}
}

// bar renders an 8-step unicode block proportional to p in [0,1].
func bar(p float32) string {
	blocks := []rune("▁▂▃▄▅▆▇█")
	if p < 0 {
		p = 0
	}
	if p > 1 {
		p = 1
	}
	idx := int(p * float32(len(blocks)))
	if idx >= len(blocks) {
		idx = len(blocks) - 1
	}
	return string(blocks[idx])
}

// colorTok renders a token piece, made printable and optionally green (the final tok).
func colorTok(s string, highlight bool) string {
	s = printable(s)
	if highlight {
		return "\x1b[1;32m" + s + "\x1b[0m"
	}
	return s
}

// tokDisplay decodes one token id to its surface piece, falling back to <id> when the
// tokenizer can't render it standalone (e.g. a byte-fragment).
func tokDisplay(tok *tokenizer.Tokenizer, id int) string {
	if piece, err := tok.Decode([]int{id}); err == nil && piece != "" {
		return piece
	}
	return "<" + itoa(id) + ">"
}

// printable makes a token piece safe and visible on one line: spaces become "·",
// newlines/tabs become escapes, and it is bracket-trimmed to a sane width.
func printable(s string) string {
	r := strings.NewReplacer(" ", "·", "\n", "\\n", "\t", "\\t", "\r", "\\r")
	s = r.Replace(s)
	if len(s) > 14 {
		s = s[:14] + "…"
	}
	return s
}

func itoa(n int) string { return fmt.Sprintf("%d", n) }

func check(stage string, err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", stage, err)
		os.Exit(1)
	}
}

// ---- loading (mirrors cmd/fakchat, minus Metal/decode) ----------------------------

func readModelConfig(hf, gguf string) (model.Config, error) {
	if gguf != "" {
		f, err := ggufload.Open(gguf)
		if err != nil {
			return model.Config{}, fmt.Errorf("gguf: %w", err)
		}
		return f.Config()
	}
	var cfg model.Config
	cb, err := os.ReadFile(filepath.Join(hf, "config.json"))
	if err != nil {
		return cfg, fmt.Errorf("config.json: %w", err)
	}
	if err := json.Unmarshal(cb, &cfg); err != nil {
		return cfg, fmt.Errorf("config.json parse: %w", err)
	}
	if cfg.HeadDim == 0 && cfg.NumHeads != 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumHeads
	}
	return cfg, nil
}

// loadModel loads weights for a cacheless Forward pass. Llama family: quantize-at-load
// (Q8); GGUF streams through the quant loader; the Qwen3.5 hybrid uses the f32 GDN path.
func loadModel(hf, gguf string, cfg model.Config) (*model.Model, error) {
	if gguf != "" {
		return ggufload.LoadModelQuant(gguf)
	}
	if cfg.IsQwen35Hybrid() {
		return model.LoadSafetensorsDir(hf, cfg)
	}
	if _, err := os.Stat(filepath.Join(hf, "model.safetensors.index.json")); err == nil {
		return model.LoadSafetensorsQuantDir(hf, cfg)
	}
	return model.LoadSafetensorsQuant(filepath.Join(hf, "model.safetensors"), cfg)
}

func resolveTokDir(tokDir, hf, gguf string) string {
	if tokDir != "" {
		return tokDir
	}
	if hf != "" {
		if _, err := os.Stat(filepath.Join(hf, "tokenizer.json")); err == nil {
			return hf
		}
		if home, err := os.UserHomeDir(); err == nil {
			cand := filepath.Join(home, ".cache", "fak-models", "tokenizers", "qwen2.5")
			if _, err := os.Stat(filepath.Join(cand, "tokenizer.json")); err == nil {
				return cand
			}
		}
		return ""
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(gguf), "tokenizer.json")); err == nil {
		return filepath.Dir(gguf)
	}
	return ""
}

// ---- HTML heatmap -----------------------------------------------------------------

// writeHTML emits the canonical logit-lens grid: one row per layer (embedding at the
// bottom, final block at top), one column per prompt position. Each cell shows the
// top-1 token the lens predicts there, shaded by that token's probability — so the
// "belief" sweeping up and across the residual stream is visible at a glance.
func writeHTML(path string, m *model.Model, tok *tokenizer.Tokenizer, act *model.Activations, ids []int) error {
	seq := act.Seq
	nLayer := len(act.Hidden)
	// cell[layer][pos] = top-1 TokenProb
	cell := make([][]model.TokenProb, nLayer)
	for p := 0; p < seq; p++ {
		lens := m.LayerLogits(act, p)
		for l := 0; l < nLayer; l++ {
			if cell[l] == nil {
				cell[l] = make([]model.TokenProb, seq)
			}
			top := model.TopK(lens[l], 1)
			if len(top) > 0 {
				cell[l][p] = top[0]
			}
		}
	}

	var b strings.Builder
	b.WriteString("<!doctype html><html><head><meta charset=\"utf-8\"><title>fak logit lens</title>\n")
	b.WriteString("<style>")
	b.WriteString("body{background:#0b0e14;color:#cbd5e1;font:12px/1.3 ui-monospace,Menlo,Consolas,monospace;margin:20px}")
	b.WriteString("h1{font-size:15px;font-weight:600}.sub{color:#64748b;margin-bottom:14px}")
	b.WriteString("table{border-collapse:collapse}td,th{padding:2px 5px;text-align:center;white-space:nowrap}")
	b.WriteString("th{color:#94a3b8;font-weight:500;position:sticky}")
	b.WriteString(".rowlab{color:#94a3b8;text-align:right;padding-right:8px}")
	b.WriteString(".cell{border:1px solid #0b0e14;border-radius:3px;color:#0b0e14;font-weight:600}")
	b.WriteString(".final{outline:2px solid #22c55e}")
	b.WriteString("</style></head><body>\n")
	b.WriteString("<h1>fak logit lens — layer × position belief map</h1>\n")

	// prompt strip
	var promptStr strings.Builder
	for _, id := range ids {
		promptStr.WriteString(printable(tokDisplay(tok, id)))
		promptStr.WriteString(" ")
	}
	fmt.Fprintf(&b, "<div class=\"sub\">model=%s &nbsp; layers=%d &nbsp; %d tokens &nbsp;|&nbsp; prompt: %s</div>\n",
		html.EscapeString(string(m.Cfg.ModelType)), m.Cfg.NumLayers, seq, html.EscapeString(promptStr.String()))

	b.WriteString("<table>\n")
	// header: position tokens
	b.WriteString("<tr><th></th>")
	for p := 0; p < seq; p++ {
		fmt.Fprintf(&b, "<th title=\"pos %d\">%s</th>", p, html.EscapeString(printable(tokDisplay(tok, ids[p]))))
	}
	b.WriteString("</tr>\n")

	// rows: final layer at top, embedding at bottom (read like the residual stream rising)
	for l := nLayer - 1; l >= 0; l-- {
		b.WriteString("<tr>")
		fmt.Fprintf(&b, "<td class=\"rowlab\">%s</td>", layerLabel(l, nLayer))
		for p := 0; p < seq; p++ {
			c := cell[l][p]
			cls := "cell"
			if l == nLayer-1 {
				cls += " final"
			}
			fmt.Fprintf(&b, "<td class=\"%s\" style=\"background:%s\" title=\"p=%.3f logit=%.2f\">%s</td>",
				cls, heat(c.Prob), c.Prob, c.Logit, html.EscapeString(printable(tokDisplay(tok, c.ID))))
		}
		b.WriteString("</tr>\n")
	}
	b.WriteString("</table>\n")
	b.WriteString("<div class=\"sub\" style=\"margin-top:14px\">Cell = top-1 token the logit lens predicts at that (layer, position); brightness ∝ probability. Top row (outlined) is the model's real prediction.</div>\n")
	b.WriteString("</body></html>\n")
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

// heat maps a probability in [0,1] to a CSS color on a dark→bright cyan/amber ramp.
func heat(p float32) string {
	if p < 0 {
		p = 0
	}
	if p > 1 {
		p = 1
	}
	// low prob -> dim slate, high prob -> bright amber. Interpolate in RGB.
	lr, lg, lb := 30, 41, 59   // #1e293b
	hr, hg, hb := 251, 191, 36 // #fbbf24
	r := lr + int(float32(hr-lr)*p)
	g := lg + int(float32(hg-lg)*p)
	bl := lb + int(float32(hb-lb)*p)
	return fmt.Sprintf("#%02x%02x%02x", r, g, bl)
}
