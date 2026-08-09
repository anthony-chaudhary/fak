package main

// guard_resume.go — `fak guard resume <id>` / `fak guard --resume <id>`, the cache-safe
// resume PLANNER (issue #1206, epic #1193 Pillar 5 / C10). It makes the guard-wrapped
// session ADDRESSABLE for resume by composing two primitives that already ship:
//
//   - C1 (#1197): the durable session registry (internal/session Registry + FileStore,
//     the descriptor `fak guard` register-on-start persists at defaultSessionRegistryPath()).
//     Each descriptor carries the wrapped Argv, the PCB Run state, the parent_id branch link
//     (C4/#1200), and LastSeen — enough to reconstruct the launch and price the resume.
//   - C9 (#1205): the auto-wired resume-cache preflight rung (preflight.PlanResume), which
//     projects the WARM-SPLICE / WARM / CUT / RESET posture + a priced token estimate.
//
// The planner resolves <id> against the registry, feeds the descriptor's idle/size facts to
// the C9 rung, and emits the cheapest safe relaunch: the recorded `fak guard ... -- <agent>`
// with the #745 --compact-history-budget byte-splice flags for CUT/RESET and the agent's own
// resume flag (--continue for Claude Code) appended — the SAME reattach the --restart-on-budget
// path uses (guardRestartRelaunchCommand), so a budget-restart and an operator-resume share one
// mechanism. Per the epic's fence, fak supplies the cache-safe flags + posture + the guarded
// gateway, NOT the transcript bytes: the provider harness (`claude --continue`) restores the
// conversation; the KV cache is rebuilt on the first turn by design.
//
// It is a read-only, offline peel-off (like `fak guard sessions` / `fak guard restart-audit`):
// a bare leading verb, never a program to wrap, and it never binds a gateway — so it is disjoint
// from the flagship `fak guard -- claude` launch path. Resuming a BRANCH id resolves that
// branch's own descriptor (its own Argv/Trace); the parent descriptor is never read or written,
// so a branch resume leaves the parent untouched by construction.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/preflight"
	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// guardResumePlanSchema tags the machine-readable plan so a --json consumer can bind it.
const guardResumePlanSchema = "fak.guard-resume-plan.v1"

// The resume projection's SUBSTITUTED price, used when the operator supplies neither
// --input-price nor --output-price. Named (rather than inlined at the substitution
// site) for two reasons #5483 names: an unpriced input silently becoming a priced
// number is the defect, so the substitution must be reportable as one; and this is
// the fourth in-tree Opus-class rate card, so guardOpusClassRateCards can read the
// LIVE value instead of transcribing it. Matches `fak resume plan`'s default and
// gateway.ClaudeOpus48*PerMTokUSD so the three agree — not because 5/25 is settled
// (it is one of three in-tree bases), but so a substituted number is at least the
// same substituted number everywhere.
const (
	guardResumeFallbackInputPerMTokUSD  = 5.0
	guardResumeFallbackOutputPerMTokUSD = 25.0
	// guardResumePricingSourceOperator / ...Substituted are the two basis stamps the
	// plan can carry. They are the whole point of the field: without them a reader
	// cannot tell a projection the operator priced from one fak defaulted.
	guardResumePricingSourceOperator    = "operator:--input-price/--output-price"
	guardResumePricingSourceSubstituted = "SUBSTITUTED default:anthropic/claude-opus-4-8 (ESTIMATED — no price supplied)"
	guardResumePricingSourceMixed       = "MIXED operator + SUBSTITUTED default (only one of --input-price/--output-price was supplied)"
)

// guardResumeInput carries the resume facts the planner feeds to the C9 rung. Idle and size
// are resolved by the CLI (from the descriptor + flags) before the pure core runs, so
// planGuardResume stays total and I/O-free — the posture/argv selection is unit-tested
// without a registry file or a live gateway.
type guardResumeInput struct {
	IdleSeconds          int64
	ResidentTokens       int
	HorizonTurns         int
	BudgetTokens         int
	WarmSpliceAvailable  bool
	CompactHistoryBudget int
	Pricing              resume.Pricing
	TTL                  resume.CacheTTL
}

// guardResumePlan is the deterministic, advisory resume decision — the observable record the
// CLI prints or emits as JSON. It names the resolved session, the C9 posture + chosen path,
// the priced projection, and the exact cache-safe `fak guard` relaunch to run.
type guardResumePlan struct {
	Schema string `json:"schema"`
	Query  string `json:"query"`

	ID       string `json:"id"`
	Trace    string `json:"trace,omitempty"`
	Host     string `json:"host,omitempty"`
	PCBState string `json:"pcb_state,omitempty"`
	Agent    string `json:"agent,omitempty"`
	// Branch / ParentID surface a C4 (#1200) branch resume: resolving a branch id brings the
	// branch up from its own checkpoint, and the parent descriptor is left untouched.
	Branch   bool   `json:"branch"`
	ParentID string `json:"parent_id,omitempty"`

	Path        preflight.ResumePath `json:"path"`
	Posture     resume.Posture       `json:"posture"`
	Recommended resume.Strategy      `json:"recommended"`
	Reason      string               `json:"reason"`

	IdleSeconds            int64   `json:"idle_seconds"`
	ResidentTokens         int     `json:"resident_tokens"`
	ProjectedPrefillTokens int     `json:"projected_prefill_tokens"`
	ProjectedPromptCostUSD float64 `json:"projected_prompt_cost_usd"`
	// PricingSource is the BASIS STAMP for ProjectedPromptCostUSD (#5483): which rate
	// card produced the dollars. Never omitempty — a $ figure with no basis is exactly
	// the unfalsifiable number this field exists to eliminate, so it is always present.
	PricingSource string `json:"pricing_source"`
	// PricingInputPerMTokUSD / PricingOutputPerMTokUSD are the rates actually applied,
	// carried alongside the label so a consumer can reproduce the projection.
	PricingInputPerMTokUSD  float64 `json:"pricing_input_per_mtok_usd"`
	PricingOutputPerMTokUSD float64 `json:"pricing_output_per_mtok_usd"`

	CompactHistoryBudget int      `json:"compact_history_budget,omitempty"`
	ContinueFlag         string   `json:"continue_flag,omitempty"`
	RelaunchArgv         []string `json:"relaunch_argv"`

	Verdict preflight.ResumeVerdict `json:"verdict"`
}

// planGuardResume composes the resolved C1 descriptor with the C9 posture rung into the
// cache-safe relaunch decision. It is pure over its inputs (no I/O) so the WARM-SPLICE /
// WARM / CUT / RESET selection and the reconstructed relaunch argv are unit-tested directly.
func planGuardResume(query string, desc session.Descriptor, in guardResumeInput) guardResumePlan {
	if in.CompactHistoryBudget <= 0 {
		in.CompactHistoryBudget = gateway.DefaultCompactHistoryBudget
	}
	// An UNPRICED input must not silently become a PRICED number: when the operator
	// supplies no rate the projection still runs on a substituted default, so record
	// WHICH of the two happened and report it beside the dollars (#5483).
	pricingSource := guardResumePricingSourceOperator
	if in.Pricing.InputPerMTokUSD == 0 && in.Pricing.OutputPerMTokUSD == 0 {
		pricingSource = guardResumePricingSourceSubstituted
		in.Pricing = resume.Pricing{
			InputPerMTokUSD:  guardResumeFallbackInputPerMTokUSD,
			OutputPerMTokUSD: guardResumeFallbackOutputPerMTokUSD,
		}
	}

	verdict := preflight.PlanResume(preflight.ResumeInput{
		Plan: resume.Input{
			ResidentTokens: in.ResidentTokens,
			IdleSeconds:    in.IdleSeconds,
			TTL:            in.TTL,
			Pricing:        in.Pricing,
			HorizonTurns:   in.HorizonTurns,
		},
		WarmSpliceAvailable: in.WarmSpliceAvailable,
		BudgetTokens:        in.BudgetTokens,
	})

	flags, compact := guardResumeCacheSafeFlags(verdict.Path, verdict.ProjectedPrefillTokens, in.CompactHistoryBudget)
	argv, agent, continueFlag := guardResumeRelaunchArgv(desc, flags)

	return guardResumePlan{
		Schema:                 guardResumePlanSchema,
		Query:                  strings.TrimSpace(query),
		ID:                     desc.ID,
		Trace:                  desc.Trace,
		Host:                   desc.Host,
		PCBState:               guardResumePCBState(desc),
		Agent:                  agent,
		Branch:                 strings.TrimSpace(desc.ParentID) != "",
		ParentID:               desc.ParentID,
		Path:                   verdict.Path,
		Posture:                verdict.Posture,
		Recommended:            verdict.Recommended,
		Reason:                 verdict.Reason,
		IdleSeconds:            in.IdleSeconds,
		ResidentTokens:         in.ResidentTokens,
		ProjectedPrefillTokens: verdict.ProjectedPrefillTokens,
		ProjectedPromptCostUSD: verdict.ProjectedPromptCostUSD,

		PricingSource:           pricingSource,
		PricingInputPerMTokUSD:  in.Pricing.InputPerMTokUSD,
		PricingOutputPerMTokUSD: in.Pricing.OutputPerMTokUSD,

		CompactHistoryBudget: compact,
		ContinueFlag:         continueFlag,
		RelaunchArgv:         argv,
		Verdict:              verdict,
	}
}

// guardResumeCacheSafeFlags maps the C9-selected path to the guard flags the relaunch carries.
// WARM-SPLICE (in-process KV reattach) and WARM (byte-identical prefix intact) preserve the
// cached prefix, so guard's default cache-safe forwarding is kept — nothing is spliced. CUT and
// RESET apply epic #745's --compact-history-budget byte-splice at the rung's priced prefill
// target (falling back to guard's default budget when the rung has no size to project), the
// cache-safe relaunch lever the issue names. Returns the flag vector and the resolved budget
// (0 when the prefix is preserved).
func guardResumeCacheSafeFlags(path preflight.ResumePath, prefillTokens, defaultBudget int) ([]string, int) {
	switch path {
	case preflight.ResumePathCut, preflight.ResumePathReset:
		compact := prefillTokens
		if compact <= 0 {
			compact = defaultBudget
		}
		return []string{"--compact-history-budget", strconv.Itoa(compact)}, compact
	default: // WARM-SPLICE / WARM: keep the prefix byte-identical, no override.
		return nil, 0
	}
}

// guardResumeRelaunchArgv reconstructs the `fak guard <flags> -- <recorded command>` relaunch
// from the descriptor's stored Argv, appending the wrapped agent's own resume flag (--continue
// for Claude Code) for a recognized child — the same reattach guardRestartRelaunchCommand uses
// on the budget-restart path, idempotent so a recorded argv that already carried --continue is
// never doubled. An unrecognized binary gets no continue flag (fak cannot guess a foreign
// resume syntax), and a descriptor with no recorded command yields a `<agent>` placeholder so
// the posture is still reported honestly. Returns the argv, the wrapped agent name, and the
// continue flag actually appended (empty when none).
func guardResumeRelaunchArgv(desc session.Descriptor, flags []string) (argv []string, agent, continueFlag string) {
	argv = append([]string{"fak", "guard"}, flags...)
	argv = append(argv, "--")
	if len(desc.Argv) == 0 {
		argv = append(argv, "<agent>")
		return argv, "", ""
	}
	argv = append(argv, desc.Argv...)
	agent = desc.Argv[0]
	if cf, ok := guardContinueFlagForAgent(agent); ok {
		continueFlag = cf
		argv = guardAppendContinueFlag(argv, cf)
	}
	return argv, agent, continueFlag
}

// guardResumePCBState prefers the human PCBState string, falling back to the typed Run enum,
// so a descriptor written by either path reports a state.
func guardResumePCBState(desc session.Descriptor) string {
	if s := strings.TrimSpace(desc.PCBState); s != "" {
		return s
	}
	return strings.ToUpper(desc.Run.String())
}

// guardResumeIdleSeconds resolves the idle time fed to the C9 posture. An explicit override
// (>= 0) wins; otherwise idle is derived from the descriptor's LastSeen (the durable last-drive
// stamp), the cheapest honest proxy for dormancy. An unreadable/zero source yields a negative
// "unknown", which the C9 rung projects to the conservative RESUME_FULL rather than shedding on
// a guess.
func guardResumeIdleSeconds(desc session.Descriptor, override int64, now time.Time) int64 {
	if override >= 0 {
		return override
	}
	if !desc.LastSeen.IsZero() {
		if d := now.Sub(desc.LastSeen); d >= 0 {
			return int64(d.Seconds())
		}
	}
	return -1
}

// resolveGuardResumeDescriptor turns a query into the one session it names, matching the
// guardsessions.Resolve semantics over the durable descriptors: an EXACT id or trace equality
// wins outright, else a case-insensitive PREFIX of the id or trace. Returns the matched
// descriptor, the match count, and the candidates so the caller can report an ambiguity.
func resolveGuardResumeDescriptor(descs []session.Descriptor, query string) (session.Descriptor, int, []session.Descriptor) {
	q := strings.TrimSpace(query)
	if q == "" {
		return session.Descriptor{}, 0, nil
	}
	for _, d := range descs {
		if d.ID == q || d.Trace == q {
			return d, 1, []session.Descriptor{d}
		}
	}
	lq := strings.ToLower(q)
	var hits []session.Descriptor
	for _, d := range descs {
		if strings.HasPrefix(strings.ToLower(d.ID), lq) ||
			(d.Trace != "" && strings.HasPrefix(strings.ToLower(d.Trace), lq)) {
			hits = append(hits, d)
		}
	}
	switch len(hits) {
	case 0:
		return session.Descriptor{}, 0, nil
	case 1:
		return hits[0], 1, hits
	default:
		return session.Descriptor{}, len(hits), hits
	}
}

func cmdGuardResume(argv []string) { os.Exit(runGuardResume(os.Stdout, os.Stderr, argv)) }

// runGuardResume is the testable CLI core: exit 0 a resolved plan, 1 no match, 2 usage, 3 an
// ambiguous resolve (so a script can gate on a clean single-session resolution).
func runGuardResume(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("guard resume", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "guard")
	registryFlag := fs.String("registry", "", "C1 session registry file to resolve <id> against (default: $FAK_SESSION_REGISTRY, else <config>/fak/session-registry.json)")
	asJSON := fs.Bool("json", false, "emit the resume plan as JSON")
	idleSeconds := fs.Int64("idle-seconds", -1, "override the idle time fed to the cache-resume posture (default: derived from the descriptor's LastSeen; -1 = derive)")
	residentTokens := fs.Int("resident-tokens", 0, "resident context size (tokens) the resume would re-prefill, fed to the C9 posture (0 = unknown -> the safe RESUME_FULL/WARM recommendation)")
	horizonTurns := fs.Int("horizon-turns", 0, "expected turns remaining after resume (0 = default), fed to the C9 posture")
	budgetTokens := fs.Int("budget-tokens", 0, "advisory cold-prefill budget; a cold path over this is LABELED, never blocked")
	warmSplice := fs.Bool("warm-splice", false, "assert an in-process warm-KV splice is available (internal/session/warmsplice) so the plan selects WARM-SPLICE (no re-prefill). Default false: a fresh-process resume rebuilds KV on the first turn by design.")
	compactBudget := fs.Int("compact-history-budget", gateway.DefaultCompactHistoryBudget, "the #745 --compact-history-budget floor spliced into a CUT/RESET relaunch when the rung projects no size")
	inputPrice := fs.Float64("input-price", 5, "model base input price per million tokens (default: Opus 4.8 = 5)")
	outputPrice := fs.Float64("output-price", 25, "model base output price per million tokens (default: Opus 4.8 = 25)")
	ttl1h := fs.Bool("ttl-1h", false, "price the posture against the 1h extended cache tier instead of the 5m default")
	if !parseFlags(fs, argv) {
		return 2
	}
	// --input-price / --output-price carry NON-ZERO defaults, so the planner's
	// "did the operator price this?" test cannot see the difference between a supplied
	// 5 and a defaulted 5 — every CLI plan would claim an operator basis it did not
	// have. fs.Visit reports only the flags actually PRESENT on the command line, so an
	// unpriced invocation hands the planner a zero Pricing and is stamped SUBSTITUTED
	// (#5483). The dollars are identical either way; only the honesty of the label changes.
	var inPriceSupplied, outPriceSupplied bool
	fs.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "input-price":
			inPriceSupplied = true
		case "output-price":
			outPriceSupplied = true
		}
	})
	if fs.NArg() == 0 {
		fmt.Fprintln(stderr, "fak guard resume: need a session id (id or trace prefix) — list with `fak session ls --fleet`")
		return 2
	}
	query := fs.Arg(0)

	regPath := strings.TrimSpace(*registryFlag)
	if regPath == "" {
		regPath = defaultSessionRegistryPath()
	}
	reg := session.NewRegistry(session.NewFileStore(pathutil.ExpandTilde(regPath)))
	now := time.Now()
	descs, err := reg.List(now)
	if err != nil {
		fmt.Fprintf(stderr, "fak guard resume: read session registry %s: %v\n", regPath, err)
		return 1
	}
	desc, matched, candidates := resolveGuardResumeDescriptor(descs, query)
	switch {
	case matched == 0:
		fmt.Fprintf(stderr, "fak guard resume: no session matches %q in %s — list with `fak session ls --fleet`\n", query, regPath)
		return 1
	case matched > 1:
		fmt.Fprintf(stderr, "fak guard resume: %q is ambiguous — %d sessions match:\n", query, matched)
		for _, c := range guardResumeSortCandidates(candidates) {
			fmt.Fprintf(stderr, "  %s\t%s\t%s\n", orDash(c.ID), orDash(c.Trace), guardResumePCBState(c))
		}
		fmt.Fprintln(stderr, "  narrow the prefix to name exactly one")
		return 3
	}

	ttl := resume.TTL5m
	if *ttl1h {
		ttl = resume.TTL1h
	}
	pricing := resume.Pricing{InputPerMTokUSD: *inputPrice, OutputPerMTokUSD: *outputPrice}
	if !inPriceSupplied && !outPriceSupplied {
		pricing = resume.Pricing{} // let the planner substitute, and SAY that it did
	}
	plan := planGuardResume(query, desc, guardResumeInput{
		IdleSeconds:          guardResumeIdleSeconds(desc, *idleSeconds, now),
		ResidentTokens:       *residentTokens,
		HorizonTurns:         *horizonTurns,
		BudgetTokens:         *budgetTokens,
		WarmSpliceAvailable:  *warmSplice,
		CompactHistoryBudget: *compactBudget,
		Pricing:              pricing,
		TTL:                  ttl,
	})
	// Half-supplied is its own basis, and the honest one to report: the other half is
	// still fak's default, so calling the whole projection operator-priced would be a
	// claim the operator did not make.
	if inPriceSupplied != outPriceSupplied {
		plan.PricingSource = guardResumePricingSourceMixed
	}

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, plan, "fak guard resume")
	}
	renderGuardResumePlan(stdout, plan)
	return 0
}

// guardResumeSortCandidates orders an ambiguity list by id then trace so the reported tie is
// stable across runs.
func guardResumeSortCandidates(in []session.Descriptor) []session.Descriptor {
	out := append([]session.Descriptor(nil), in...)
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].ID != out[j].ID {
			return out[i].ID < out[j].ID
		}
		return out[i].Trace < out[j].Trace
	})
	return out
}

func renderGuardResumePlan(w io.Writer, p guardResumePlan) {
	fmt.Fprintf(w, "guard resume plan for %s\n", orDash(p.ID))
	if p.Trace != "" && p.Trace != p.ID {
		fmt.Fprintf(w, "  trace:     %s\n", p.Trace)
	}
	if p.Branch {
		fmt.Fprintf(w, "  branch of: %s (parent untouched)\n", orDash(p.ParentID))
	}
	if p.PCBState != "" {
		fmt.Fprintf(w, "  state:     %s\n", p.PCBState)
	}
	fmt.Fprintf(w, "  posture:   %s (%s)\n", p.Posture, p.Reason)
	fmt.Fprintf(w, "  path:      %s  [%s]\n", p.Path, p.Recommended)
	fmt.Fprintf(w, "  idle:      %ds   resident: %d tok\n", p.IdleSeconds, p.ResidentTokens)
	fmt.Fprintf(w, "  projected: %d prefill tok  ~$%.4f prompt\n", p.ProjectedPrefillTokens, p.ProjectedPromptCostUSD)
	// The basis rides directly under the $ it explains, so the dollars can never be
	// read without the card that produced them (#5483).
	fmt.Fprintf(w, "  $ basis:   %s  [$%g/MTok in, $%g/MTok out]\n",
		p.PricingSource, p.PricingInputPerMTokUSD, p.PricingOutputPerMTokUSD)
	if p.CompactHistoryBudget > 0 {
		fmt.Fprintf(w, "  compact:   --compact-history-budget %d  (#745 cache-safe byte-splice)\n", p.CompactHistoryBudget)
	}
	fmt.Fprintf(w, "\ncache-safe relaunch:\n  %s\n", strings.Join(p.RelaunchArgv, " "))
	fmt.Fprintln(w, "\n(fak supplies the cache-safe flags + posture + the guarded gateway; the provider harness restores the transcript, and the KV cache rebuilds on the first turn by design.)")
}
