// Command simpledemo is the friendliest way to chat with a local AI model.
//
// It runs entirely on your computer - no API keys, no cloud, no cost.
// Perfect for trying out local AI or when you want privacy.
//
// Quick start:
//
//	go run ./cmd/simpledemo
//
// It will auto-detect downloaded models or download one automatically.
//
// Recommended models for CPU-only:
// - Qwen2.5-0.5B-Instruct-Q8_0.gguf (~500MB) - Fastest, good for testing
// - Qwen2.5-1.5B-Instruct-Q8_0.gguf (~1.6GB) - Best balance of speed/quality
// - Qwen2.5-3B-Instruct-Q4_K_M.gguf (~2GB) - Better quality, still usable
//
// Get models from: https://huggingface.co/models?search=gguf qwen2.5 instruct
package main

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/demoui"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

func main() {
	// Command-line flags
	gguf := flag.String("gguf", "", "Path to .gguf model file (auto-detected if empty)")
	tokDir := flag.String("tok", "", "Tokenizer dir (auto-detected if empty)")
	sys := flag.String("sys", "You are a helpful assistant. Keep answers short and clear.", "System prompt (keep it short for small models)")
	maxNew := flag.Int("n", 128, "Max tokens per response (smaller = faster)")
	temp := flag.Float64("temp", 0.5, "Temperature (0=focused, 1=creative)")
	seed := flag.Int64("seed", time.Now().UnixNano(), "Random seed (for reproducibility)")
	quiet := flag.Bool("quiet", false, "Skip welcome banner")
	autoDownload := flag.Bool("download", false, "Auto-download default model if not found")
	backend := flag.String("backend", "", "Compute backend to run through, e.g. cuda (default: the pure-Go Q8 CPU path). Use to prove GPU usage on a build that registered an accelerator.")
	flag.Parse()

	// Expand a leading ~ at the parse boundary so every downstream path — and the
	// boundary lint that audits this convention — sees it done here, where the flags
	// are declared. ExpandTilde is idempotent, so resolveModelPath/loadModel
	// re-expanding it is a safe no-op.
	*gguf = pathutil.ExpandTilde(*gguf)
	*tokDir = pathutil.ExpandTilde(*tokDir)

	// Resolve the model path: auto-detect a local model, auto-download or print help
	// when none is found, and fetch an explicit -gguf that isn't on disk.
	resolveModelPath(gguf, tokDir, *autoDownload, *quiet)

	// Load model
	if !*quiet {
		fmt.Fprintln(os.Stderr, "📦 Loading model...")
	}
	t0 := time.Now()
	// The GGUF load + quantize is the longest silent phase (tens of seconds on the
	// bigger rungs). Spin a live "Loading model… 12.3s" line on stderr so the terminal
	// never freezes; stop it the instant load returns, before the "Loaded …" line.
	var stopLoad func()
	if !*quiet {
		stopLoad = demoui.Spinner(os.Stderr, "Loading model")
	}
	m, tok, modelName, tokSource, err := loadModel(*gguf, *tokDir, *autoDownload, *quiet)
	if stopLoad != nil {
		stopLoad()
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ Error loading model: %v\n", err)
		fmt.Fprintln(os.Stderr, "")
		if strings.Contains(err.Error(), "loading GGUF") {
			fmt.Fprintln(os.Stderr, "💡 Tip: the file may be truncated or not a valid .gguf — re-download it,")
			fmt.Fprintln(os.Stderr, "        or pass a known-good model with -gguf <path>.")
		}
		os.Exit(1)
	}
	loadMS := time.Since(t0).Milliseconds()

	// Stop tokens: prefer what the model/tokenizer actually declares over magic ids.
	stops := stopTokenIDs(tok, *gguf)

	// Pick the compute path. The default is the proven pure-Go Q8 CPU lane; -backend
	// routes the forward pass through the compute HAL instead — the path that runs on
	// (and proves usage of) a GPU when this build registered one (e.g. cuda).
	session, device, err := newSession(m, *backend)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ %v\n", err)
		os.Exit(1)
	}
	defer func() { session.Close() }()

	sampling := "greedy (argmax)"
	if *temp > 0 {
		sampling = fmt.Sprintf("temperature %.2f", *temp)
	}

	stats := gatherModelStats(m, modelName, device)
	if !*quiet {
		fmt.Fprintf(os.Stderr, "✅ Loaded %s in %.1fs (tokenizer: %s)\n", modelName, float64(loadMS)/1000, tokSource)
		// Show the real compute surface this run is using (cores / matmul workers /
		// accelerator) so the user sees the hardware, not "whatever". On this CPU-only
		// build the honest summary says pure-Go Q8 CPU with no GPU backend.
		fmt.Fprintf(os.Stderr, "🖥️  %s\n", demoui.Probe().Summary)
		printModelCard(stats, *temp, *maxNew, sampling)
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "💬 Chat with your AI! Type a message and press Enter.")
		fmt.Fprintln(os.Stderr, "   Commands: /clear = new chat, /exit = quit")
		fmt.Fprintln(os.Stderr, "")
	}

	chatLoop(&session, m, tok, stops, stats, chatConfig{
		sys:     *sys,
		maxNew:  *maxNew,
		temp:    *temp,
		seed:    *seed,
		quiet:   *quiet,
		backend: *backend,
	})
}

// chatConfig is the run-invariant set of flag values the incremental chat loop reads
// per turn — bundled so chatLoop keeps a single, stable signature.
type chatConfig struct {
	sys     string
	maxNew  int
	temp    float64
	seed    int64
	quiet   bool
	backend string
}

// chatLoop runs the interactive REPL: it reads a user line, handles the /exit and
// /clear commands, ingests the new ChatML block into the resident KV cache, prefills
// only the new suffix, streams the decoded reply, and prints the per-turn stats. It
// takes session by pointer because /clear rebuilds it; the caller's deferred Close on
// the same variable then closes whatever session the loop last left resident.
//
// Rather than re-tokenize and re-prefill the WHOLE conversation each turn (which also
// duplicates it, at shifted positions, into a never-reset KV cache), it keeps the cache
// resident and feeds only the NEW turn: the system block once, then each user block. The
// KV prefix from prior turns is reused verbatim, so prefill recomputes only the suffix.
// cachedIDs tracks exactly what the cache holds, so the cache-hit % the stats report is
// measured, not estimated.
func chatLoop(session **model.Session, m *model.Model, tok *tokenizer.Tokenizer, stops map[int]bool, stats modelStats, cfg chatConfig) {
	input := bufio.NewReader(os.Stdin)
	out := bufio.NewWriter(os.Stdout)
	rng := rand.New(rand.NewSource(cfg.seed)) // seeded once: turns continue one RNG stream, not N correlated ones
	var cachedIDs []int                       // the exact token sequence resident in the KV cache, in order
	firstTurn := true
	turnNum := 0
	cumReused, cumPrompt := 0, 0

	for {
		// Show prompt
		if cfg.quiet {
			fmt.Fprint(os.Stderr, "> ")
		} else {
			fmt.Fprint(os.Stderr, "You: ")
		}

		userMsg, err := input.ReadString('\n')
		if err != nil {
			break
		}
		userMsg = strings.TrimSpace(userMsg)

		// Handle commands
		if userMsg == "/exit" || userMsg == "/quit" {
			fmt.Fprintln(os.Stderr, "👋 Goodbye!")
			break
		}
		if userMsg == "/clear" {
			(*session).Close()
			*session, _, _ = newSession(m, cfg.backend) // backend already validated at startup
			cachedIDs = nil
			firstTurn = true
			cumReused, cumPrompt = 0, 0
			fmt.Fprintln(os.Stderr, "✨ Chat cleared.")
			continue
		}
		if userMsg == "" {
			continue
		}

		// The new text to ingest this turn: the system block only on the first turn, then
		// the user block. Encoding each block on its own matches encoding the full prompt
		// because ChatML's <|im_start|>/<|im_end|> are atomic special tokens — natural reuse
		// boundaries — so the resident prefix stays bit-identical to a fresh prefill.
		newText := userBlock(userMsg)
		if firstTurn {
			newText = systemBlock(cfg.sys) + newText
		}
		newIDs, err := tok.Encode(newText)
		if err != nil {
			fmt.Fprintf(os.Stderr, "⚠️  Encoding error: %v\n", err)
			continue
		}

		reused := len(cachedIDs)
		promptTokens := reused + len(newIDs)

		// Prefill ONLY the new suffix; the cache already holds the reused prefix. Prefill
		// is silent (no streamed tokens yet) and can run for seconds on a long prompt, so
		// spin a live "Thinking… 1.2s" line on stderr while it runs. Stop it the instant
		// prefill returns — and only THEN print the "AI: " prefix — so the spinner (which
		// clears its own line with a carriage return) never overlaps the streamed reply.
		var stopThink func()
		if !cfg.quiet {
			stopThink = demoui.Spinner(os.Stderr, "Thinking")
		}
		tPrefill := time.Now()
		logits := (*session).Prefill(newIDs)
		prefillS := time.Since(tPrefill).Seconds()
		if stopThink != nil {
			stopThink()
		}
		cachedIDs = append(cachedIDs, newIDs...)

		// Show we're thinking
		if !cfg.quiet {
			fmt.Fprint(os.Stderr, "AI: ")
		}

		decodeStart := time.Now()
		full, genIDs := decodeReply(*session, tok, logits, stops, cfg.maxNew, cfg.temp, rng, out)
		decodeS := time.Since(decodeStart).Seconds()
		cachedIDs = append(cachedIDs, genIDs...)

		// Close the assistant turn in the cache so the NEXT user block continues a
		// well-formed ChatML transcript and is reused as a prefix, not recomputed.
		if closeIDs, e := tok.Encode(assistantTurnClose); e == nil && len(closeIDs) > 0 {
			(*session).PrefillNoLogits(closeIDs)
			cachedIDs = append(cachedIDs, closeIDs...)
		}

		// A coherent model on a matched build does not repeat itself to death. When
		// the model and build disagree the reply collapses into a loop ("2 2 2 …" or
		// ".assistant.assistant…", issue #91); flag it so a first-time user isn't left
		// staring at gibberish wondering whether the demo is broken.
		if !cfg.quiet && looksDegenerate(full) {
			fmt.Fprintln(os.Stderr, "\n⚠️  That reply looks degenerate (stuck repeating). Make sure you're on a build")
			fmt.Fprintln(os.Stderr, "   with the GGUF chat fixes, or try another model — see cmd/simpledemo/README.md.")
		}

		// If decode stopped at the token budget instead of a stop token, the reply was cut
		// off; say so. The turn is still closed in the cache above so the transcript stays
		// well-formed for the next turn.
		if !cfg.quiet && len(genIDs) == cfg.maxNew {
			fmt.Fprintf(os.Stderr, "\n  ✂️  reply truncated at the -n %d token budget (raise -n for longer replies)\n", cfg.maxNew)
		}

		firstTurn = false
		turnNum++
		cumReused += reused
		cumPrompt += promptTokens

		// Stats: prefill vs decode reported separately, the cache hit prefix reuse
		// achieved, the analytic prefill operation count, the decode bandwidth, and the
		// device — everything measured this run or counted from the model shape.
		if !cfg.quiet {
			printTurnStats(stats, turnStats{
				turn:         turnNum,
				promptTokens: promptTokens,
				newTokens:    len(newIDs),
				reusedTokens: reused,
				prefillS:     prefillS,
				genTokens:    len(genIDs),
				decodeS:      decodeS,
				kvPositions:  len(cachedIDs),
				cumReused:    cumReused,
				cumPrompt:    cumPrompt,
			})
		}
	}
}

// resolveModelPath resolves *gguf (and *tokDir) to a usable on-disk model before load.
// It expands a leading ~, auto-detects a local model when none was given, and — when
// still empty — either auto-downloads the default model (-download) or prints the help
// text and exits. Finally it fetches an explicit -gguf that names a model not yet on
// disk. It exits the process on a fatal download or no-model condition, matching the
// inline behavior it replaced.
func resolveModelPath(gguf, tokDir *string, autoDownload, quiet bool) {
	// Expand a leading ~ ourselves: Go's flag parsing and os.Open never do, and a
	// quoted/PowerShell "-gguf ~/Downloads/model.gguf" reaches us as a literal "~"
	// path that can't be opened (issue: "system cannot find the path specified").
	*gguf = pathutil.ExpandTilde(*gguf)
	*tokDir = pathutil.ExpandTilde(*tokDir)

	// Try to auto-find a model if none specified
	if *gguf == "" {
		foundModel, foundTok := findModel()
		if foundModel != "" {
			*gguf = foundModel
			if *tokDir == "" && foundTok != "" {
				*tokDir = foundTok
			}
			if !quiet {
				fmt.Fprintf(os.Stderr, "🤖 Found model: %s\n\n", filepath.Base(foundModel))
			}
		}
	}

	// Still no model? Auto-download or show help
	if *gguf == "" {
		if autoDownload {
			downloadDefaultModel(gguf, tokDir, quiet)
		} else {
			printNoModelHelp(quiet)
			os.Exit(1)
		}
	}

	// An explicit -gguf path that isn't on disk: fetch it instead of failing. The
	// user named the model they want, so "should auto download" means we go get it —
	// no -download flag required. We derive the real HuggingFace URL from the
	// filename; an unrecognizable name yields a friendly, actionable error.
	if *gguf != "" {
		if err := ensureModelFile(*gguf, !quiet); err != nil {
			fmt.Fprintf(os.Stderr, "❌ %v\n", err)
			os.Exit(1)
		}
	}
}

// downloadDefaultModel fetches the default Qwen2.5-0.5B model (and its tokenizer) into
// the fak-models cache, setting *gguf and *tokDir to the cached paths. It exits the
// process if the model download fails; a failed tokenizer fetch is a non-fatal warning
// because virtually every GGUF embeds its own tokenizer.
func downloadDefaultModel(gguf, tokDir *string, quiet bool) {
	home, _ := os.UserHomeDir()
	cacheDir := filepath.Join(home, ".cache", "fak-models")
	ggufDir := filepath.Join(cacheDir, "gguf")
	os.MkdirAll(ggufDir, 0755)

	defaultModel := "Qwen2.5-0.5B-Instruct-Q8_0.gguf"
	*gguf = filepath.Join(ggufDir, defaultModel)
	*tokDir = filepath.Join(cacheDir, "tokenizers", "qwen2.5")

	if !quiet {
		printWelcome()
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "📥 Downloading model (first time only)...")
		fmt.Fprintf(os.Stderr, "   Model: %s\n", defaultModel)
	}

	// Try all URLs (primary + mirrors)
	modelErr := downloadWithMirrors(modelURLs(defaultModel), *gguf, !quiet)
	if modelErr != nil {
		fmt.Fprintf(os.Stderr, "❌ Download failed: %v\n", modelErr)
		os.Exit(1)
	}

	if !quiet {
		fmt.Fprintln(os.Stderr, "   ✅ Download complete!")
		fmt.Fprintln(os.Stderr, "")
	}

	// Download tokenizer
	os.MkdirAll(*tokDir, 0755)
	tokPath := filepath.Join(*tokDir, "tokenizer.json")
	if _, err := os.Stat(tokPath); os.IsNotExist(err) {
		if !quiet {
			fmt.Fprintln(os.Stderr, "📥 Downloading tokenizer...")
		}
		tokErr := downloadWithMirrors(tokenizerURLs(), tokPath, !quiet)
		if tokErr != nil {
			fmt.Fprintf(os.Stderr, "⚠️  Tokenizer download failed: %v\n", tokErr)
		}
	}
}

// printNoModelHelp prints the "no model found" download instructions to stderr (the
// banner first unless quiet). The caller exits after this — it explains both the
// one-line auto-download and the manual curl options.
func printNoModelHelp(quiet bool) {
	if !quiet {
		printWelcome()
	}
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "❌ No model found. You need to download a model first.")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "📥 Quick options:")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "  1. Auto-download (recommended):")
	fmt.Fprintln(os.Stderr, "     go -C fak run ./cmd/simpledemo -download")
	fmt.Fprintln(os.Stderr, "     # or from the fak directory:")
	fmt.Fprintln(os.Stderr, "     go run ./cmd/simpledemo -download")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "  2. Manual download (pick one):")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "     FASTEST (500MB):")
	fmt.Fprintf(os.Stderr, "     curl -L %s -o model.gguf\n", modelURL("Qwen2.5-0.5B-Instruct-Q8_0.gguf"))
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "     RECOMMENDED (1.6GB):")
	fmt.Fprintf(os.Stderr, "     curl -L %s -o model.gguf\n", modelURL("Qwen2.5-1.5B-Instruct-Q8_0.gguf"))
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "  Then run: go -C fak run ./cmd/simpledemo -gguf model.gguf")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "Or browse: https://huggingface.co/models?search=gguf qwen2.5 instruct")
	fmt.Fprintln(os.Stderr, "")
}

// newSession builds the generation session for the chosen compute path plus a human
// label for it. backend=="" is the default: the proven pure-Go Q8 CPU lane. A non-empty
// backend routes through the compute HAL (NewBackendSession) — the path that runs on a
// GPU when this build registered one — and fails loudly (rather than silently on CPU) if
// the named backend is not compiled into this build.
func newSession(m *model.Model, backend string) (*model.Session, string, error) {
	if backend == "" {
		s := m.NewSession()
		s.Quant = true // resident Q8_0 weights — the fast, proven CPU path
		return s, "cpu (pure-Go Q8 reference)", nil
	}
	be, ok := compute.Lookup(backend)
	if !ok {
		return nil, "", fmt.Errorf("compute backend %q is not registered in this build (have: %s)\n"+
			"   → rebuild with its build tag on matching hardware, e.g. `go run -tags cuda ./cmd/simpledemo -backend cuda` on an NVIDIA box",
			backend, strings.Join(compute.Registered(), ", "))
	}
	// The in-kernel device backends stream f32 weights (widened to f16 on the GPU); they
	// have no quantized device GEMM, so they cannot serve a GGUF-quantized model loaded
	// here. Refuse up front with the real GPU witness path instead of panicking deep in
	// the HAL weight lookup (it would otherwise fail on the first f32 weight fetch).
	if rep := m.ResidentReport(); rep.Q8Tensors > 0 || rep.Q4KTensors > 0 {
		if be.Name() == "cpu-ref" {
			// cpu-ref is the CPU reference HAL, not a GPU — steer back to the fast default.
			return nil, "", fmt.Errorf("backend %q can't serve this model: it loaded as quantized GGUF (%d Q8 + %d Q4_K tensors) and the HAL reference path streams f32 weights\n"+
				"   → drop -backend to use the fast pure-Go Q8 CPU path (the demo default)",
				backend, rep.Q8Tensors, rep.Q4KTensors)
		}
		return nil, "", fmt.Errorf("backend %q can't serve this model: it loaded as quantized GGUF (%d Q8 + %d Q4_K tensors) and the device backends run f32 weights\n"+
			"   → the GPU path runs f32 safetensors — prove GPU usage with: go run -tags cuda ./cmd/gpucheck -hf <hf-snapshot-dir> -backend cuda",
			backend, rep.Q8Tensors, rep.Q4KTensors)
	}
	return m.NewBackendSession(be), fmt.Sprintf("%s (%s · %s)", be.Name(), be.Tier(), be.Class()), nil
}

// systemBlock / userBlock / assistantTurnClose are the ChatML pieces the incremental chat
// loop feeds one at a time. systemBlock(sys)+userBlock(msg) is byte-identical to
// buildPrompt(sys, nil, msg) — so turn 1 of the loop equals a fresh single-turn prompt and
// the proven decode path is unchanged; later turns append only a userBlock + a closing tag.
func systemBlock(system string) string {
	return "<|im_start|>system\n" + system + "<|im_end|>\n"
}

func userBlock(userMsg string) string {
	return "<|im_start|>user\n" + userMsg + "<|im_end|>\n<|im_start|>assistant\n"
}

// assistantTurnClose closes the assistant's reply in the cache after decode so the next
// user block continues a well-formed ChatML transcript.
const assistantTurnClose = "<|im_end|>\n"

// printWelcome shows a friendly banner.
func printWelcome() {
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "╔════════════════════════════════════════════════════════════════════════╗")
	fmt.Fprintln(os.Stderr, "║     🤖 Simple Demo - Chat with a Local AI (No API Key Required!)      ║")
	fmt.Fprintln(os.Stderr, "╚════════════════════════════════════════════════════════════════════════╝")
}

// quantSuffixRe splits a GGUF basename into its base model name and quant label.
// mradermacher (the GGUF publisher we pull from) names quants with a DOT —
// "Qwen2.5-1.5B-Instruct.Q8_0.gguf" — but users and our own older docs routinely
// type a dash before the quant. Accept either separator so a hand-typed
// "-gguf Qwen2.5-1.5B-Instruct-Q8_0.gguf" still resolves to the real file.
var quantSuffixRe = regexp.MustCompile(`^(.+?)[.-]((?:IQ|Q)\d[0-9A-Za-z_]*|f16|bf16|fp16)\.gguf$`)

// modelDownload derives the canonical HuggingFace filename and resolve URLs (primary
// + mirror) for a requested GGUF basename. mradermacher publishes one repo per model
// — "mradermacher/<base>-GGUF" — and names the file "<base>.<quant>.gguf". ok is
// false when the name isn't a recognizable "<base>.<quant>.gguf" we can map to a repo,
// in which case callers fall back to a friendly error rather than a bogus URL.
func modelDownload(requested string) (canonical string, urls []string, ok bool) {
	m := quantSuffixRe.FindStringSubmatch(filepath.Base(requested))
	if m == nil {
		return "", nil, false
	}
	base, quant := m[1], m[2]
	canonical = base + "." + quant + ".gguf"
	repo := "mradermacher/" + base + "-GGUF"
	return canonical, []string{
		fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repo, canonical),
		fmt.Sprintf("https://hf-mirror.com/%s/resolve/main/%s", repo, canonical),
	}, true
}

// modelURL returns the primary HuggingFace download URL for a model file, used in the
// copy-paste curl hints. Derived from the filename so the URL matches mradermacher's
// real layout (dot before the quant) instead of a 404-ing guess.
func modelURL(filename string) string {
	if _, urls, ok := modelDownload(filename); ok {
		return urls[0]
	}
	return "https://huggingface.co/mradermacher/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/" + filename
}

// modelURLs returns all possible URLs for a model file (primary + mirrors).
func modelURLs(filename string) []string {
	if _, urls, ok := modelDownload(filename); ok {
		return urls
	}
	return []string{
		"https://huggingface.co/mradermacher/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/" + filename,
		"https://hf-mirror.com/mradermacher/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/" + filename,
	}
}

// tokenizerURLs returns all possible URLs for the tokenizer.
func tokenizerURLs() []string {
	return []string{
		"https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json",
		"https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json",
	}
}

// ensureModelFile makes path exist, downloading the model named by its basename when
// it doesn't. The user passed "-gguf <path>"; if the file is absent we derive the
// HuggingFace URL from the filename and fetch it ("should auto download") instead of
// erroring out. It is a no-op when the file is already present.
func ensureModelFile(path string, showProgress bool) error {
	if _, err := os.Stat(path); err == nil {
		return nil // already on disk
	}
	canonical, urls, ok := modelDownload(path)
	if !ok {
		return fmt.Errorf("model file not found: %s\n"+
			"   → can't derive a download URL from that name; pass -gguf <existing .gguf>,\n"+
			"     or run with -download to fetch the default model", filepath.Base(path))
	}
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("create model directory: %w", err)
		}
	}
	if showProgress {
		fmt.Fprintf(os.Stderr, "📥 Model not found locally — downloading %s (first time only)...\n", canonical)
	}
	if err := downloadWithMirrors(urls, path, showProgress); err != nil {
		return fmt.Errorf("downloading %s: %w", canonical, err)
	}
	if showProgress {
		fmt.Fprintln(os.Stderr, "   ✅ Download complete!")
		fmt.Fprintln(os.Stderr, "")
	}
	return nil
}

// downloadWithMirrors tries downloading from multiple URLs until one succeeds.
// Tries Go HTTP first, then curl.exe on Windows (works around TLS issues).
func downloadWithMirrors(urls []string, dest string, showProgress bool) error {
	var lastErr error
	for i, url := range urls {
		if showProgress && i > 0 {
			fmt.Fprintf(os.Stderr, "   🔄 Trying mirror %d/%d...\n", i+1, len(urls))
		}
		err := downloadFile(url, dest, showProgress)
		if err == nil {
			return nil
		}
		lastErr = err
		if showProgress {
			fmt.Fprintf(os.Stderr, "   ⚠️  URL %d failed: %v\n", i+1, err)
		}
	}
	return fmt.Errorf("all %d URLs failed, last error: %w", len(urls), lastErr)
}

// downloadFile downloads a file with progress reporting.
// Tries Go HTTP first, then falls back to curl on Windows.
func downloadFile(url, dest string, showProgress bool) error {
	// Try Go HTTP client first (fastest when it works)
	err := downloadFileGo(url, dest, showProgress)
	if err == nil {
		return nil
	}

	// On Windows, fall back to curl.exe (uses Schannel, works with HF)
	if runtime.GOOS == "windows" {
		if showProgress {
			fmt.Fprintln(os.Stderr, "   ⚠️  Go HTTP failed, trying curl.exe fallback...")
		}
		err := downloadFileCurl(url, dest, showProgress)
		if err == nil {
			return nil
		}
		if showProgress {
			fmt.Fprintf(os.Stderr, "   ⚠️  curl also failed: %v\n", err)
		}
	}

	return fmt.Errorf("all download methods failed: %w", err)
}

// downloadClient fetches models and tokenizers. It sets connection and
// response-header deadlines so a dead or stalled host fails fast, but deliberately
// leaves Client.Timeout at 0: a multi-GB model legitimately takes minutes to stream,
// and a blanket deadline would abort a large download mid-flight.
var downloadClient = &http.Client{
	Transport: &http.Transport{
		DialContext:           (&net.Dialer{Timeout: 30 * time.Second}).DialContext,
		TLSHandshakeTimeout:   30 * time.Second,
		ResponseHeaderTimeout: 60 * time.Second,
	},
}

// downloadFileGo downloads using Go's HTTP client.
func downloadFileGo(url, dest string, showProgress bool) error {
	resp, err := downloadClient.Get(url)
	if err != nil {
		return fmt.Errorf("fetch failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("server returned %d", resp.StatusCode)
	}

	// Create destination file
	out, err := os.Create(dest)
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer out.Close()

	// Track progress
	total := resp.ContentLength
	written := int64(0)
	start := time.Now()

	if showProgress {
		fmt.Fprintf(os.Stderr, "   Downloading: %.1f MB...\n", float64(total)/(1024*1024))
	}

	// Copy with progress
	writer := &progressWriter{
		w:            out,
		total:        total,
		written:      &written,
		start:        start,
		showProgress: showProgress,
	}

	if _, err := io.Copy(writer, resp.Body); err != nil {
		return fmt.Errorf("download failed: %w", err)
	}

	if showProgress {
		elapsed := time.Since(start).Seconds()
		mb := float64(written) / (1024 * 1024)
		fmt.Fprintf(os.Stderr, "   ✅ %.1f MB in %.1fs (%.1f MB/s)\n", mb, elapsed, mb/elapsed)
	}

	return nil
}

// downloadFileCurl downloads using curl.exe on Windows (uses Schannel, works with HF).
func downloadFileCurl(url, dest string, showProgress bool) error {
	if showProgress {
		fmt.Fprintln(os.Stderr, "   Running curl with resume support...")
	}

	// Run curl with individual arguments (avoids shell escaping issues)
	args := []string{
		"-L",      // Follow redirects
		"-C", "-", // Resume download
		"--retry", "8", // Retry 8 times
		"--retry-delay", "2", // Wait 2s between retries
		"--connect-timeout", "30", // 30s connection timeout
		"-o", dest, // Output file
		url, // URL (last argument)
	}

	cmd := exec.Command("curl.exe", args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("curl failed: %w, output: %s", err, string(output))
	}

	// Verify the file was created
	if _, err := os.Stat(dest); os.IsNotExist(err) {
		return fmt.Errorf("curl completed but file not created")
	}

	if showProgress {
		info, _ := os.Stat(dest)
		mb := float64(info.Size()) / (1024 * 1024)
		fmt.Fprintf(os.Stderr, "   ✅ Downloaded %.1f MB via curl\n", mb)
	}

	return nil
}

// progressWriter wraps an io.Writer to show download progress.
type progressWriter struct {
	w            io.Writer
	total        int64
	written      *int64
	start        time.Time
	showProgress bool
	lastUpdate   time.Time
}

// Write forwards bytes to the wrapped writer, accumulates the running total, and
// throttles a percent/throughput progress line to stderr at most every 500ms.
func (p *progressWriter) Write(b []byte) (int, error) {
	n, err := p.w.Write(b)
	*p.written += int64(n)

	if p.showProgress && time.Since(p.lastUpdate) > 500*time.Millisecond {
		pct := float64(*p.written) / float64(p.total) * 100
		elapsed := time.Since(p.start).Seconds()
		speed := float64(*p.written) / (1024 * 1024) / elapsed
		fmt.Fprintf(os.Stderr, "\r   Progress: %.0f%% (%.1f MB/s)...", pct, speed)
		p.lastUpdate = time.Now()
	}

	return n, err
}

// ggufShardSuffixRe matches the "-NNNNN-of-MMMMM.gguf" shard suffix that
// HuggingFace's GGUF split writer produces for large models (7B+). Auto-detect
// treats shard 1 as the entry point (it carries the model config); later shards
// are skipped so the loader is never handed a config-less fragment.
var ggufShardSuffixRe = regexp.MustCompile(`-(\d+)-of-(\d+)\.gguf$`)

// isLaterGGUFShard reports whether name is a non-first GGUF split shard.
func isLaterGGUFShard(name string) bool {
	m := ggufShardSuffixRe.FindStringSubmatch(name)
	if m == nil {
		return false
	}
	n, _ := strconv.Atoi(m[1])
	return n != 1
}

// findModel looks for .gguf files in standard locations.
// Returns (modelPath, tokenizerDir) or ("", "") if not found.
// Split checkpoints are entered through their shard-1 file; later shards are
// skipped. When several candidates exist, the smallest by file size wins so the
// demo defaults to a model that fits in RAM rather than e.g. a 72B checkpoint.
func findModel() (string, string) {
	home, _ := os.UserHomeDir()

	// Standard cache locations to search
	searchPaths := []string{
		filepath.Join(home, "models"), // Windows: C:\Users\You\models\
		filepath.Join(home, ".cache", "fak-models", "gguf"),
		filepath.Join(home, ".cache", "huggingface", "hub"),
		filepath.Join(home, "Downloads"),
	}

	if runtime.GOOS == "windows" {
		searchPaths = append(searchPaths, filepath.Join(home, "Downloads"))
	}

	// resolveTokenizer finds a tokenizer dir for a candidate model dir, or "".
	resolveTokenizer := func(modelDir string) string {
		if _, err := os.Stat(filepath.Join(modelDir, "tokenizer.json")); err == nil {
			return modelDir
		}
		qwenDir := filepath.Join(home, ".cache", "fak-models", "tokenizers", "qwen2.5")
		if _, err := os.Stat(filepath.Join(qwenDir, "tokenizer.json")); err == nil {
			return qwenDir
		}
		return ""
	}

	type candidate struct {
		path   string
		tokDir string
		size   int64
	}
	var cands []candidate
	addCandidate := func(modelPath, tokDir string) {
		if isLaterGGUFShard(filepath.Base(modelPath)) {
			return
		}
		info, err := os.Stat(modelPath)
		if err != nil {
			return
		}
		cands = append(cands, candidate{path: modelPath, tokDir: tokDir, size: info.Size()})
	}

	for _, searchDir := range searchPaths {
		if _, err := os.Stat(searchDir); os.IsNotExist(err) {
			continue
		}
		files, _ := os.ReadDir(searchDir)
		for _, f := range files {
			if strings.HasSuffix(f.Name(), ".gguf") {
				addCandidate(filepath.Join(searchDir, f.Name()), resolveTokenizer(searchDir))
			}
		}
		// Also search subdirectories (for huggingface hub structure)
		for _, sd := range files {
			if sd.IsDir() {
				subPath := filepath.Join(searchDir, sd.Name())
				subFiles, _ := os.ReadDir(subPath)
				for _, f := range subFiles {
					if strings.HasSuffix(f.Name(), ".gguf") {
						addCandidate(filepath.Join(subPath, f.Name()), subPath)
					}
				}
			}
		}
	}

	if len(cands) == 0 {
		return "", ""
	}
	// Prefer the smallest checkpoint that still ships (or can find) a tokenizer;
	// fall back to the smallest overall so the demo still launches without a
	// co-located tokenizer.json.
	sort.Slice(cands, func(i, j int) bool { return cands[i].size < cands[j].size })
	for _, c := range cands {
		if c.tokDir != "" {
			return c.path, c.tokDir
		}
	}
	return cands[0].path, cands[0].tokDir
}

// loadModel loads a GGUF model and resolves a tokenizer for it. It returns the
// model, the tokenizer, a display name, and a short note on where the tokenizer
// came from (shown to the user). allowDownload permits a one-time tokenizer fetch
// as a last resort.
func loadModel(ggufPath, tokDir string, allowDownload, quiet bool) (*model.Model, *tokenizer.Tokenizer, string, string, error) {
	// Load GGUF weights.
	m, err := ggufload.LoadModelQuant(ggufPath)
	if err != nil {
		return nil, nil, "", "", fmt.Errorf("loading GGUF: %w", err)
	}

	// Resolve a tokenizer — robust to a missing/absent tokenizer.json because the
	// GGUF almost always embeds its own.
	tok, tokSource, err := loadTokenizer(ggufPath, tokDir, allowDownload, quiet)
	if err != nil {
		return nil, nil, "", "", err
	}

	// Display name from the GGUF config (best-effort).
	modelName := filepath.Base(ggufPath)
	if f, ferr := ggufload.Open(ggufPath); ferr == nil {
		if cfg, cerr := f.Config(); cerr == nil && cfg.ModelType != "" {
			modelName = cfg.ModelType
		}
	}

	return m, tok, modelName, tokSource, nil
}

// loadTokenizer resolves a working tokenizer for the GGUF, in order of preference:
//
//  1. an explicit -tok dir's tokenizer.json,
//  2. a tokenizer.json sitting next to the GGUF,
//  3. the tokenizer EMBEDDED in the GGUF itself (offline, always matches the model),
//  4. the cached Qwen2.5 tokenizer.json, and finally
//  5. a one-time download of tokenizer.json (only when allowDownload).
//
// Steps 1/2/4 are best-effort: a missing or unparseable tokenizer.json there does
// not abort — the embedded path (3) means the demo works with nothing but a .gguf.
// Returns the tokenizer and a short human description of where it came from.
func loadTokenizer(ggufPath, tokDir string, allowDownload, quiet bool) (*tokenizer.Tokenizer, string, error) {
	type jsonSrc struct{ dir, label string }
	tries := []jsonSrc{}
	if tokDir != "" {
		tries = append(tries, jsonSrc{tokDir, "tokenizer.json (-tok)"})
	}
	tries = append(tries, jsonSrc{filepath.Dir(ggufPath), "tokenizer.json (next to model)"})

	// 1 + 2: explicit dir, then sidecar.
	for _, t := range tries {
		if tok, err := tokenizer.LoadJSON(filepath.Join(t.dir, "tokenizer.json")); err == nil {
			return tok, t.label, nil
		}
	}

	// 3: embedded in the GGUF — the bulletproof default (no file, no network).
	if tok, err := embeddedTokenizer(ggufPath); err == nil {
		return tok, "embedded in model file", nil
	}

	// 4: cached Qwen2.5 tokenizer.json.
	home, _ := os.UserHomeDir()
	cacheTokDir := filepath.Join(home, ".cache", "fak-models", "tokenizers", "qwen2.5")
	if tok, err := tokenizer.LoadJSON(filepath.Join(cacheTokDir, "tokenizer.json")); err == nil {
		return tok, "tokenizer.json (cache)", nil
	}

	// 5: one-time download, only when explicitly allowed.
	if allowDownload {
		if err := os.MkdirAll(cacheTokDir, 0o755); err == nil {
			dst := filepath.Join(cacheTokDir, "tokenizer.json")
			if !quiet {
				fmt.Fprintln(os.Stderr, "📥 Fetching tokenizer (first time only)...")
			}
			if derr := downloadWithMirrors(tokenizerURLs(), dst, !quiet); derr == nil {
				if tok, err := tokenizer.LoadJSON(dst); err == nil {
					return tok, "tokenizer.json (downloaded)", nil
				}
			}
		}
	}

	return nil, "", errNoTokenizer(allowDownload)
}

// embeddedTokenizer builds a tokenizer straight from the GGUF's own
// tokenizer.ggml.* metadata. This is what makes the demo bulletproof: virtually
// every GGUF carries its full vocab + merges, so no separate tokenizer.json (and
// no network) is required, and the result always matches the model exactly.
func embeddedTokenizer(ggufPath string) (*tokenizer.Tokenizer, error) {
	f, err := ggufload.Open(ggufPath)
	if err != nil {
		return nil, err
	}
	gt, ok := f.GGMLTokenizer()
	if !ok {
		return nil, fmt.Errorf("no embedded BPE tokenizer in %s", filepath.Base(ggufPath))
	}
	return tokenizer.FromGGML(gt.Tokens, gt.Merges, gt.TokenTypes, gt.Pre)
}

// errNoTokenizer builds the friendly, actionable error shown when every tokenizer
// source is exhausted (rare: it means the GGUF embeds none and none is on disk).
func errNoTokenizer(allowDownload bool) error {
	if allowDownload {
		return errors.New("no tokenizer found: the model has no embedded tokenizer and the download failed\n" +
			"   → check your connection, or pass -tok <dir containing tokenizer.json>")
	}
	return errors.New("no tokenizer found: the model has no embedded tokenizer and none is on disk\n" +
		"   → re-run with -download to fetch one, or pass -tok <dir containing tokenizer.json>")
}

// buildPrompt creates a ChatML-formatted prompt with conversation history.
// For small models, keep it short!
func buildPrompt(system string, conversation []string, userMsg string) string {
	var sb strings.Builder

	// System prompt (keep it short for small models!)
	sb.WriteString("<|im_start|>system\n")
	sb.WriteString(system)
	sb.WriteString("<|im_end|>\n")

	// Add conversation history
	for i := 0; i < len(conversation); i += 2 {
		sb.WriteString("<|im_start|>user\n")
		sb.WriteString(conversation[i])
		sb.WriteString("<|im_end|>\n")
		sb.WriteString("<|im_start|>assistant\n")
		sb.WriteString(conversation[i+1])
		sb.WriteString("<|im_end|>\n")
	}

	// Current message
	sb.WriteString("<|im_start|>user\n")
	sb.WriteString(userMsg)
	sb.WriteString("<|im_end|>\n")
	sb.WriteString("<|im_start|>assistant\n")

	return sb.String()
}

// validUTF8Len returns the length in bytes of the longest prefix of s that is
// entirely complete UTF-8 runes. It holds back a trailing byte sequence that is
// only a partial multi-byte rune so streaming never emits a � for a character
// split across two tokens.
func validUTF8Len(s string) int {
	n := 0
	for n < len(s) {
		r, size := utf8.DecodeRuneInString(s[n:])
		if r == utf8.RuneError && size <= 1 {
			break // incomplete/invalid rune at the tail — wait for more bytes
		}
		n += size
	}
	return n
}

// stopTokenIDs collects the ids that end a turn, from what the model actually
// declares rather than magic constants: the GGUF's tokenizer.ggml.eos_token_id,
// and any special token named <|im_end|> or <|endoftext|>. The Qwen ChatML ids
// (151643/151645) are kept only as a harmless fallback. We deliberately do NOT
// hardcode id 2 (SPM </s>): in a byte-level Qwen vocab id 2 is an ordinary token,
// and the real EOS comes from the GGUF eos_token_id instead.
func stopTokenIDs(tok *tokenizer.Tokenizer, ggufPath string) map[int]bool {
	stops := map[int]bool{151643: true, 151645: true}
	for id, content := range tok.SpecialTokens() {
		if content == "<|im_end|>" || content == "<|endoftext|>" {
			stops[id] = true
		}
	}
	if f, err := ggufload.Open(ggufPath); err == nil {
		if v, ok := f.Uint64("tokenizer.ggml.eos_token_id"); ok {
			stops[int(v)] = true
		}
	}
	return stops
}

// sample picks the next token using temperature sampling.
func sample(logits []float32, temp float64, rng *rand.Rand) int {
	if temp <= 0 || len(logits) == 0 {
		// Greedy argmax
		best, idx := float32(-math.MaxFloat32), 0
		for i, v := range logits {
			if v > best {
				best, idx = v, i
			}
		}
		return idx
	}

	// Temperature sampling
	maxL := float32(-math.MaxFloat32)
	for _, v := range logits {
		if v > maxL {
			maxL = v
		}
	}

	sum := float64(0)
	probs := make([]float64, len(logits))
	for i, v := range logits {
		p := math.Exp(float64(v-maxL) / temp)
		probs[i] = p
		sum += p
	}

	// Roulette wheel selection
	r := rng.Float64() * sum
	for i, p := range probs {
		r -= p
		if r <= 0 {
			return i
		}
	}
	return len(logits) - 1
}

// decodeReply runs one reply's autoregressive decode loop: starting from the
// prefill logits it samples a token, stops on EOS, steps the session, and streams
// the reply to out. It decodes the whole id list each step and emits only the
// newly-completed UTF-8 prefix, so a multi-byte rune (CJK, emoji) split across two
// tokens is held back until complete instead of printing a � placeholder. Returns
// the full decoded reply and the generated token ids (the count is len(genIDs); the
// ids let the caller append them to the resident KV-cache token sequence for prefix
// reuse on the next turn).
//
// This is the exact production decode path; main() and the greedy non-degeneracy
// guard test (issue #91) both go through it so the test guards what users run.
func decodeReply(session *model.Session, tok *tokenizer.Tokenizer, logits []float32, stops map[int]bool, maxNew int, temp float64, rng *rand.Rand, out *bufio.Writer) (string, []int) {
	var genIDs []int
	emitted := 0
	full := ""
	for len(genIDs) < maxNew {
		next := sample(logits, temp, rng)
		if stops[next] { // stop on EOS / end-of-turn
			break
		}
		genIDs = append(genIDs, next)
		if decoded, err := tok.Decode(genIDs); err == nil {
			full = decoded
			safe := full[:validUTF8Len(full)]
			if len(safe) > emitted {
				out.WriteString(safe[emitted:])
				out.Flush()
				emitted = len(safe)
			}
		}
		logits = session.Step(next)
	}
	// Flush any remaining bytes (e.g. a final char held back mid-stream).
	if len(full) > emitted {
		out.WriteString(full[emitted:])
	}
	out.Flush()
	return full, genIDs
}

// looksDegenerate reports whether a generated reply has collapsed into the
// repetition failure mode from issue #91: greedy decode on a model/build mismatch
// produced byte-identical loops like "2 2 2 2 …" and ".assistant.assistant…". It
// is deliberately conservative — a coherent answer must never trip it — so it only
// fires on long, overwhelmingly-repetitive output via two complementary signals:
//
//   - whitespace-token repetition ("2 2 2 2 2 2 …"): many tokens but almost no
//     distinct ones, and
//   - short-period tiling (".assistant.assistant…"): a short unit repeated to cover
//     the whole string, which catches loops with no internal whitespace.
//
// It is both a friendly run-time warning and the assertion the guard test makes
// against a real greedy run.
func looksDegenerate(s string) bool {
	s = strings.TrimSpace(s)
	if utf8.RuneCountInString(s) < 24 {
		return false // too short to judge; a curt correct answer must not trip
	}

	// Signal 1: whitespace-separated tokens, lots of them but barely any distinct.
	if fields := strings.Fields(s); len(fields) >= 8 {
		uniq := make(map[string]bool, len(fields))
		for _, f := range fields {
			uniq[f] = true
		}
		if len(uniq) <= len(fields)/8 { // <=12.5% distinct => looping
			return true
		}
	}

	// Signal 2: a short substring tiled across the whole string. Requiring the
	// ENTIRE reply to be p-periodic is a very strong gate — natural language is
	// never exactly periodic — so even a unit as long as a role header ("…assistant
	// \n") is safe to flag. The per-byte p-back check also matches a truncated final
	// repeat, since the tail mirrors the unit's own prefix.
	if unit, repeats := dominantPeriod(s); unit <= 24 && repeats >= 3 {
		return true
	}
	return false
}

// dominantPeriod finds the smallest period p (in bytes, p<=32) such that s is
// s[:p] repeated — i.e. s[i] == s[i-p] for every i>=p, which also admits a partial
// trailing repeat. It returns that unit length and how many whole copies fit, or
// (0,0) when no short period tiles the string.
func dominantPeriod(s string) (unitLen, repeats int) {
	n := len(s)
	for p := 1; p <= 32 && p <= n/2; p++ {
		ok := true
		for i := p; i < n; i++ {
			if s[i] != s[i-p] {
				ok = false
				break
			}
		}
		if ok {
			return p, n / p
		}
	}
	return 0, 0
}
