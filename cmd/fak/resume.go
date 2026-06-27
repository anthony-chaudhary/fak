package main

// resume.go — `fak resume`, the operator front door to the DETERMINISTIC RESUME-CACHE
// decision (internal/resume). It answers the one question an operator asks when bringing a
// dormant agent session back to life: "I am resuming a 250k-token session — what happens to
// the prompt cache, and what should I do about it?"
//
//	fak resume plan --resident-tokens 250000 --idle-seconds 7200
//	fak resume plan --image ./session-img --json
//
// The decision is PURE (internal/resume.Plan): same facts in, same priced verdict out — no
// clock, no model, no network. This shell does only the I/O the pure leaf must not: it reads
// the facts from flags (and, with --image, from a portable session image's trajectory +
// metadata), calls Plan, and renders the report as an aligned table or raw JSON. It is the
// exact split session_cmd.go uses — the decision lives in the leaf, the wire lives here.

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/sessionimage"
)

// cmdResume is the `fak resume` entry point; it maps the testable core's exit code to the
// process exit code, mirroring cmdSession.
func cmdResume(argv []string) { os.Exit(runResume(os.Stdout, os.Stderr, argv)) }

// runResume is the testable core: it returns the process exit code (0 ok, 1 a runtime error,
// 2 a usage error) and takes its streams explicitly so a test drives it without a process.
func runResume(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		resumeUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "plan":
		return runResumePlan(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		resumeUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak resume: unknown subcommand %q (want plan)\n", argv[0])
		resumeUsage(stderr)
		return 2
	}
}

// runResumePlan parses the resume facts, optionally grounds them on a real session image,
// computes the deterministic plan, and renders it.
func runResumePlan(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("resume plan", flag.ContinueOnError)
	fs.SetOutput(stderr)
	residentTokens := fs.Int("resident-tokens", 0, "size of the context that would be re-prefilled on a full resume (the whole transcript)")
	idleSeconds := fs.Int64("idle-seconds", -1, "how long the session was dormant before this resume (-1 = unknown; drives cold-vs-warm against the TTL)")
	ttlStr := fs.String("ttl", "5m", "provider cache TTL tier the session used: 5m (default) or 1h")
	horizon := fs.Int("horizon", 0, "turns expected to remain after resume (0 = default)")
	shedBudget := fs.Int("shed-budget", 0, "CUT target in tokens — what a shed keeps (0 = default ~48k)")
	seedTokens := fs.Int("seed-tokens", 0, "RESET carryover seed size in tokens (0 = default)")
	inputPrice := fs.Float64("input-price", 5, "model base input price per million tokens (default: Opus 4.8 = 5)")
	outputPrice := fs.Float64("output-price", 25, "model base output price per million tokens (default: Opus 4.8 = 25)")
	outputPerTurn := fs.Int("output-per-turn", 0, "completion tokens per modeled turn (0 = default)")
	imageDir := fs.String("image", "", "ground the plan on a portable session image directory: derive resident tokens from its trajectory and idle from its timestamp")
	transcript := fs.String("transcript", "", "ground the plan on a REAL Claude Code session transcript (.jsonl): derive resident tokens from the last assistant turn's prompt size and idle from its timestamp")
	asJSON := fs.Bool("json", false, "emit the raw Report JSON instead of the human table")
	if err := fs.Parse(argv); err != nil {
		return 2 // flag already printed the error
	}

	ttl, ok := parseResumeTTL(*ttlStr)
	if !ok {
		fmt.Fprintf(stderr, "fak resume plan: bad --ttl %q (want 5m or 1h)\n", *ttlStr)
		return 2
	}

	in := resume.Input{
		ResidentTokens:      *residentTokens,
		IdleSeconds:         *idleSeconds,
		TTL:                 ttl,
		Pricing:             resume.Pricing{InputPerMTokUSD: *inputPrice, OutputPerMTokUSD: *outputPrice},
		HorizonTurns:        *horizon,
		ShedBudgetTokens:    *shedBudget,
		SeedTokens:          *seedTokens,
		OutputTokensPerTurn: *outputPerTurn,
	}

	var groundNote string
	if *imageDir != "" {
		note, code := groundOnImage(stderr, *imageDir, &in, fs)
		if code != 0 {
			return code
		}
		groundNote = note
	}
	if *transcript != "" {
		note, code := groundOnTranscript(stderr, *transcript, &in, fs)
		if code != 0 {
			return code
		}
		groundNote = note
	}

	if in.ResidentTokens <= 0 {
		fmt.Fprintln(stderr, "fak resume plan: need --resident-tokens > 0 (or an --image / --transcript that carries token usage)")
		return 2
	}

	rep := resume.Plan(in)
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintf(stderr, "fak resume plan: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	renderResumeReport(stdout, rep, groundNote)
	return 0
}

// groundOnImage loads a portable session image and fills the resident-token and idle facts
// from it (unless the operator set them explicitly). It returns a one-line note about what
// it derived, and an exit code (0 ok, 1 load/parse error). The resident size is the sum of
// the trajectory's per-turn token estimates; idle is now - the image's UpdatedUnix.
func groundOnImage(stderr io.Writer, dir string, in *resume.Input, fs *flag.FlagSet) (string, int) {
	img, err := sessionimage.LoadDir(dir)
	if err != nil {
		fmt.Fprintf(stderr, "fak resume plan: load image %q: %v\n", dir, err)
		return "", 1
	}
	turns, err := img.Trajectory()
	if err != nil {
		fmt.Fprintf(stderr, "fak resume plan: read image trajectory: %v\n", err)
		return "", 1
	}
	sum := 0
	for _, t := range turns {
		if t.TokenEstimate > 0 {
			sum += t.TokenEstimate
		}
	}
	// Only fill what the operator did not pin on the command line.
	if !flagSet(fs, "resident-tokens") && sum > 0 {
		in.ResidentTokens = sum
	}
	if !flagSet(fs, "idle-seconds") && img.Meta.UpdatedUnix > 0 {
		idle := time.Now().Unix() - img.Meta.UpdatedUnix
		if idle < 0 {
			idle = 0
		}
		in.IdleSeconds = idle
	}
	model := img.Meta.Model
	if model == "" {
		model = "unknown"
	}
	return fmt.Sprintf("image %s (model %s, %d turns, resident≈%d tok)", dir, model, len(turns), sum), 0
}

// groundOnTranscript reads a REAL Claude Code session transcript (.jsonl) and fills the
// resident-token and idle facts from it (unless the operator pinned them). The resident
// context that a resume re-prefills is the prompt size of the MOST RECENT assistant turn:
// the provider's reported input_tokens + cache_read_input_tokens + cache_creation_input_tokens
// for that turn (the full prompt the model last had to read). Idle is now minus the last
// record's timestamp. This is the deterministic, observable counterpart to `claude --resume`:
// it answers "this exact session I am about to resume — what happens to the cache?".
func groundOnTranscript(stderr io.Writer, path string, in *resume.Input, fs *flag.FlagSet) (string, int) {
	f, err := os.Open(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak resume plan: open transcript %q: %v\n", path, err)
		return "", 1
	}
	defer f.Close()

	resident, model, lastUnix, turns, ok := scanTranscriptResident(f)
	if !ok {
		fmt.Fprintf(stderr, "fak resume plan: transcript %q has no assistant turn with usage — pass --resident-tokens\n", path)
		return "", 1
	}
	if !flagSet(fs, "resident-tokens") && resident > 0 {
		in.ResidentTokens = resident
	}
	if !flagSet(fs, "idle-seconds") && lastUnix > 0 {
		idle := time.Now().Unix() - lastUnix
		if idle < 0 {
			idle = 0
		}
		in.IdleSeconds = idle
	}
	if model == "" {
		model = "unknown"
	}
	return fmt.Sprintf("transcript %s (model %s, %d turns, resident=%d tok from last assistant prompt)", path, model, turns, resident), 0
}

// scanTranscriptResident scans a Claude Code transcript JSONL and returns the resident
// context size (the last assistant turn's total prompt tokens), the model that turn used,
// the last record's unix timestamp, the number of assistant turns seen, and whether any
// assistant usage was found. It is best-effort over real data: a malformed line is skipped,
// never fatal. Only the fields it needs are typed (forward-compatible by construction).
func scanTranscriptResident(r io.Reader) (resident int, model string, lastUnix int64, turns int, ok bool) {
	type usage struct {
		InputTokens         int `json:"input_tokens"`
		CacheReadTokens     int `json:"cache_read_input_tokens"`
		CacheCreationTokens int `json:"cache_creation_input_tokens"`
	}
	type jrec struct {
		Type      string `json:"type"`
		Timestamp string `json:"timestamp"`
		Message   *struct {
			Role  string `json:"role"`
			Model string `json:"model"`
			Usage *usage `json:"usage"`
		} `json:"message"`
	}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 1<<20), 64<<20) // a single tool-result line can be large
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var jr jrec
		if json.Unmarshal(line, &jr) != nil {
			continue
		}
		if t := parseTranscriptUnix(jr.Timestamp); t > lastUnix {
			lastUnix = t
		}
		if jr.Message == nil || jr.Message.Usage == nil || jr.Message.Role != "assistant" {
			continue
		}
		turns++
		// The most recent assistant turn's prompt size IS the resident context a resume
		// re-prefills: the uncached remainder plus whatever the provider had cached.
		resident = jr.Message.Usage.InputTokens + jr.Message.Usage.CacheReadTokens + jr.Message.Usage.CacheCreationTokens
		if jr.Message.Model != "" {
			model = jr.Message.Model
		}
		ok = true
	}
	return resident, model, lastUnix, turns, ok
}

// parseTranscriptUnix parses a Claude Code transcript timestamp (RFC3339, e.g.
// "2026-06-26T18:31:17.123Z") into unix seconds, returning 0 on any parse failure so a
// missing/odd timestamp simply does not advance the idle clock.
func parseTranscriptUnix(s string) int64 {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0
	}
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t.Unix()
	}
	if t, err := time.Parse(time.RFC3339Nano, s); err == nil {
		return t.Unix()
	}
	return 0
}

// flagSet reports whether a flag was explicitly provided on the command line (vs left at its
// default), so --image / --transcript only fill the facts the operator did not pin.
func flagSet(fs *flag.FlagSet, name string) bool {
	set := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == name {
			set = true
		}
	})
	return set
}

// parseResumeTTL maps the --ttl token to a resume.CacheTTL.
func parseResumeTTL(s string) (resume.CacheTTL, bool) {
	switch s {
	case "5m", "5min", "ephemeral", "":
		return resume.TTL5m, true
	case "1h", "1hr", "hour":
		return resume.TTL1h, true
	}
	return "", false
}

// renderResumeReport prints the deterministic plan as an aligned, scannable table: the
// projected cache posture, the three priced strategies, and the recommendation with its
// closed reason. Every dollar is a projection over the resident-token count at the supplied
// base price, never a witnessed bill (the leaf's fence, surfaced here too).
func renderResumeReport(w io.Writer, r resume.Report, imgNote string) {
	if imgNote != "" {
		fmt.Fprintf(w, "grounded on %s\n", imgNote)
	}
	idle := "unknown"
	if r.IdleSeconds >= 0 {
		idle = fmt.Sprintf("%ds", r.IdleSeconds)
	}
	fmt.Fprintf(w, "resume plan — resident=%d tok  idle=%s  ttl=%s (%ds)  posture=%s (%s)\n",
		r.ResidentTokens, idle, r.TTL, r.TTLSeconds, upper(string(r.Posture)), r.PostureReason)
	fmt.Fprintf(w, "model input=$%.2f/MTok output=$%.2f/MTok  horizon=%d turns  output/turn=%d\n\n",
		r.Pricing.InputPerMTokUSD, r.Pricing.OutputPerMTokUSD, r.HorizonTurns, r.OutputTokensPerTurn)

	fmt.Fprintf(w, "%-12s %10s %6s %16s %14s %16s\n",
		"strategy", "prefill", "keep", "cold-reprefill", "first-turn", fmt.Sprintf("horizon(%d)", r.HorizonTurns))
	for _, c := range r.Strategies {
		mark := "  "
		if c.Strategy == r.Recommended {
			mark = "->"
		}
		fmt.Fprintf(w, "%s%-10s %10d %5.0f%% %16s %14s %16s\n",
			mark, c.Strategy, c.PrefillTokens, c.ContextKeptFraction*100,
			usd(c.ColdReprefillUSD), usd(c.FirstTurnUSD), usd(c.HorizonUSD))
	}

	fmt.Fprintf(w, "\nrecommended: %s  (%s)\n", upper(string(r.Recommended)), r.Reason)
	if r.RecommendedSavingsUSD > 0 {
		fmt.Fprintf(w, "  projected horizon saving vs resume_full: %s over %d turns\n", usd(r.RecommendedSavingsUSD), r.HorizonTurns)
	}
	if r.BreakEvenTurns > 0 {
		fmt.Fprintf(w, "  warm-burst gate: a cut repays its re-prefill after %d turns\n", r.BreakEvenTurns)
	}
	fmt.Fprintln(w, "  (dollars are a projection over the resident-token count, not a witnessed bill)")
}

// usd renders a dollar figure: small values keep enough precision to be meaningful, larger
// ones round to cents. A cold 250k re-prefill is ~$1.56; a reset seed is ~$0.025 — both
// need to read cleanly.
func usd(v float64) string {
	switch {
	case v == 0:
		return "$0"
	case v < 1:
		return fmt.Sprintf("$%.4f", v)
	default:
		return fmt.Sprintf("$%.2f", v)
	}
}

// upper uppercases an ASCII token for the header line (posture/strategy emphasis) without
// pulling in strings just for this.
func upper(s string) string {
	b := []byte(s)
	for i := range b {
		if b[i] >= 'a' && b[i] <= 'z' {
			b[i] -= 'a' - 'A'
		}
	}
	return string(b)
}

func resumeUsage(w io.Writer) {
	fmt.Fprint(w, `fak resume — the deterministic RESUME-CACHE decision

  fak resume plan [--resident-tokens N] [--idle-seconds S] [--ttl 5m|1h]
                  [--horizon N] [--shed-budget N] [--seed-tokens N]
                  [--input-price F] [--output-price F] [--output-per-turn N]
                  [--image DIR] [--transcript FILE.jsonl] [--json]

Answers "I am resuming a long session — what happens to the prompt cache, and what
should I do?" It projects the cache posture (cold if idle exceeds the TTL, warm if not),
prices RESUME_FULL / CUT / RESET, and recommends a cut-by-default re-entry. Pure and
deterministic: same facts in, same priced verdict out.

example (resume a 250k session idle 2h on a 5-minute cache):
  fak resume plan --resident-tokens 250000 --idle-seconds 7200

example (plan the resume of a REAL Claude Code session you are about to --resume):
  fak resume plan --transcript ~/.claude/projects/<ns>/<uuid>.jsonl
`)
}
