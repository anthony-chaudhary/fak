package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// cmd/fak/org.go — `fak org status`: the ONE screen that answers "who controls this
// box's capability floor, and which channel set each knob?" (#5322, W4 of epic #5315).
//
// The org-policy plane shipped its plumbing across four children — a `central`
// amendment channel (#5319), a signed envelope (#5320), a pull with a last-good cache
// and a refuse-to-widen offline fallback (#5321), and a pinned trust anchor (#5323) —
// and every one of them is invisible from the outside. An operator handed a fleet with
// central policy on it could see THAT they were enrolled (`fak enroll`) but not what
// their org had actually done to their floor, and an org admin could not tell whether a
// grant they published had landed, been clamped, or been refused.
//
// That gap is not cosmetic. Centralized control that cannot be inspected is
// indistinguishable from centralized control that is not working, and the failure mode
// is the dangerous direction: a box quietly running on the compiled floor because its
// cache went stale looks exactly like a box faithfully applying an org manifest. This
// verb makes the difference legible in one read.
//
// THREE THINGS IT REPORTS, and they are genuinely different questions:
//
//	posture      — is there a central authority here at all, and is what it said still
//	               fresh? (inert / fresh / last_good / floor, with the reason.)
//	movement     — what did central actually change: widened, tightened, refused; and
//	               what did the local overlay ask for that got clamped back?
//	provenance   — per knob, WHICH CHANNEL owns the running value. This is the field an
//	               operator needs to know whether to edit their own overlay or file a
//	               ticket with IT.
//
// READ-ONLY AND OFFLINE BY DEFAULT. A status verb that reached the network on every
// invocation would make `fak org status` fail in exactly the situations an operator
// runs it (the endpoint is down; what is my box doing?). Without --pull the puller is
// built with no URL and no fetcher, which drives policy.OrgPuller.Pull straight to its
// age-out-the-cache path: the box reports the posture it would run on right now.
//
// WHAT THIS DOES NOT DO, stated plainly because the honest gap matters more than the
// shipped half: it does not install the composed floor into a running `fak guard`.
// loadGuardCapabilityFloor still assembles compiled-in → operator with no central stage
// (see guard_startup.go), so this verb reports the floor the lattice WOULD produce, not
// one the guard is currently enforcing. Wiring policy.ComposeOrgFloor into the launch
// path is the next step and is deliberately not smuggled in here — it changes what a
// guarded session admits, and that belongs in its own reviewable change.

func cmdOrg(argv []string) { os.Exit(runOrg(os.Stdout, os.Stderr, argv)) }

func runOrg(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		printOrgUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "status":
		return runOrgStatus(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		printOrgUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak org: unknown subcommand %q\n", argv[0])
		printOrgUsage(stderr)
		return 2
	}
}

func printOrgUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: fak org status [--json] [--pull] [--url <endpoint>] [--policy <manifest>]")
	fmt.Fprintln(w, "  Show which channel controls each capability knob on this box:")
	fmt.Fprintln(w, "  the compiled-in floor, the central org manifest, or the local operator overlay.")
	fmt.Fprintln(w, "  Offline by default; --pull fetches a fresh envelope from the org endpoint.")
	fmt.Fprintln(w, "  See also: fak enroll (pin the org trust anchor).")
}

// orgStatusView is everything one status run resolved — assembled first, rendered
// second, so the text and --json arms cannot drift into disagreeing about the same box.
type orgStatusView struct {
	pull        policy.OrgPullResult
	composition policy.OrgComposition
	enrollment  policy.OrgEnrollment
	enrolled    bool
	floorSource string
	overlayNote string
	maxStale    time.Duration
	granted     int
}

func runOrgStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("org status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the resolved posture and per-knob provenance as JSON")
	pull := fs.Bool("pull", false, "fetch a fresh envelope from the org endpoint (default: read the last-good cache only)")
	url := fs.String("url", "", "org policy endpoint for --pull (default: "+policy.OrgPolicyURLEnv+")")
	policyPath := fs.String("policy", "", "capability floor manifest to treat as compiled-in (default: the built-in guard floor)")
	enrollPath := fs.String("enrollment", "", "enrollment store path (default: "+policy.OrgEnrollmentPathEnv+", else the user config dir)")
	cachePath := fs.String("cache", "", "last-good envelope cache path (default: "+policy.OrgPolicyCachePathEnv+", else the user config dir)")
	if !parseFlags(fs, argv) {
		return 2
	}

	compiled, floorSource, err := orgCompiledFloor(*policyPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak org status: %v\n", err)
		return 1
	}

	store := strings.TrimSpace(*enrollPath)
	if store == "" {
		store = policy.OrgEnrollmentPath()
	}
	// A DAMAGED enrollment store is a refusal, not "not enrolled" — the same rule
	// `fak enroll` follows. Reporting a box as un-enrolled because its anchor will not
	// parse is the fail-open the whole plane is built to close.
	rec, enrolled, err := policy.LoadOrgEnrollment(store)
	if err != nil {
		fmt.Fprintf(stderr, "fak org status: %v\n", err)
		fmt.Fprintf(stderr, "  store: %s\n", store)
		fmt.Fprintf(stderr, "  next:  fak enroll --revoke   (clears a damaged anchor; re-enroll afterwards)\n")
		return 1
	}

	puller := &policy.OrgPuller{
		CachePath:      strings.TrimSpace(*cachePath),
		EnrollmentPath: store,
		// The min_version gate reads this. It is the RUNNING binary's version, so an
		// envelope that requires a newer fak is refused here exactly as it would be at
		// a guard launch — a status verb that reported a posture the real pull would
		// not produce would be worse than no status verb.
		RunningVersion: appversion.Current(),
	}
	if *pull {
		// Only --pull may reach the network. With URL empty and Fetch nil, Pull skips
		// the fetch entirely and ages out the cache — the offline read this verb wants.
		puller.URL = strings.TrimSpace(*url)
		if puller.URL == "" {
			puller.URL = strings.TrimSpace(os.Getenv(policy.OrgPolicyURLEnv))
		}
		if puller.URL == "" {
			fmt.Fprintf(stderr, "fak org status: --pull needs an endpoint: pass --url or set %s\n", policy.OrgPolicyURLEnv)
			return 2
		}
	}
	res := puller.Pull(context.Background())

	var central *adjudicator.Policy
	if res.Widen && res.Runtime != nil {
		c := res.Runtime.Adjudicator
		central = &c
	}

	operator, overlayNote, err := orgOperatorProposal(compiled, central)
	if err != nil {
		fmt.Fprintf(stderr, "fak org status: %v\n", err)
		return 1
	}

	view := orgStatusView{
		pull:        res,
		composition: policy.ComposeOrgFloor(compiled, central, operator),
		enrollment:  rec,
		enrolled:    enrolled,
		floorSource: floorSource,
		overlayNote: overlayNote,
		maxStale:    policy.MaxOrgPolicyStaleness,
	}
	// A FRESH pull is the moment a central grant actually lands on this box, so it is
	// the moment worth witnessing. An offline read changes nothing and journals nothing.
	// journal.Active() is nil outside a session, and AppendCapabilityGrant no-ops on a
	// nil receiver, so an ordinary CLI run stays byte-identical.
	if res.Posture == policy.OrgPostureFresh {
		view.granted = orgEmitCentralGrants(journal.Active(), view.composition.Fold, res.Issuer, res.Version)
	}

	if *asJSON {
		if err := writeIndentedJSON(stdout, orgStatusJSON(view)); err != nil {
			fmt.Fprintf(stderr, "fak org status: %v\n", err)
			return 1
		}
		return 0
	}
	writeOrgStatus(stdout, store, view)
	return 0
}

// orgCompiledFloor resolves what to treat as the compiled-in floor: an explicit
// --policy manifest, or the embedded guard floor `fak guard` runs by default.
//
// It parses rather than assuming an empty policy. A status report whose "compiled-in"
// stage were the zero value would attribute every knob on the box to an overlay,
// which is the one thing this verb exists to get right.
func orgCompiledFloor(path string) (adjudicator.Policy, string, error) {
	if p := strings.TrimSpace(path); p != "" {
		b, err := os.ReadFile(p)
		if err != nil {
			return adjudicator.Policy{}, "", fmt.Errorf("policy: %w", err)
		}
		rt, err := policy.ParseRuntime(b)
		if err != nil {
			return adjudicator.Policy{}, "", fmt.Errorf("policy %s: %w", p, err)
		}
		return rt.Adjudicator, p, nil
	}
	rt, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	if err != nil {
		return adjudicator.Policy{}, "", fmt.Errorf("built-in guard floor: %w", err)
	}
	return rt.Adjudicator, "built-in guard floor (fak guard --dump-policy to see it)", nil
}

// orgOperatorProposal builds what the LOCAL overlays would make of the floor once
// central has had its say — the operator stage's proposal, before the lattice clamps it.
//
// It composes once to learn the central stage, then applies the two operator overlays
// on top of THAT rather than on top of the compiled floor. Layering the overlay onto
// the wrong base would misreport provenance in both directions: a knob central already
// set would look operator-owned, and an operator entry that merely restates a central
// grant would look like a widening past it.
//
// nil means "this box has no operator overlay", which ComposeOrgFloor treats
// differently from an empty one — see its package comment.
func orgOperatorProposal(compiled adjudicator.Policy, central *adjudicator.Policy) (*adjudicator.Policy, string, error) {
	allowOverlay, layers, err := loadGuardAllowOverlayLayers()
	if err != nil {
		return nil, "", err
	}
	denyPath := guardDenyOverlayPath()
	denyOverlay, err := loadGuardDenyOverlay(denyPath)
	if err != nil {
		return nil, "", err
	}
	if len(allowOverlay.Allow) == 0 && len(allowOverlay.AllowPrefix) == 0 && len(denyOverlay.Deny) == 0 {
		return nil, "", nil
	}

	// guardApply*Overlay mutate the maps and slices they are handed IN PLACE, so the
	// central stage has to be cloned before they touch it — otherwise the "before"
	// snapshot the whole diff is measured against would already contain the overlay.
	base := policy.ComposeOrgFloor(compiled, central, nil).Stages.Central
	rt := policy.Runtime{Adjudicator: orgClonePolicyContainers(base)}
	guardApplyAllowOverlay(&rt, allowOverlay)
	guardApplyDenyOverlay(&rt, denyOverlay)

	var sources []string
	for _, layer := range layers {
		if p := strings.TrimSpace(layer.Path); p != "" {
			sources = append(sources, p)
		}
	}
	if len(denyOverlay.Deny) > 0 {
		sources = append(sources, denyPath)
	}
	out := rt.Adjudicator
	return &out, strings.Join(sources, "; "), nil
}

// orgClonePolicyContainers deep-copies exactly the containers guardApplyAllowOverlay,
// guardApplyDenyOverlay and AttachShellDangerRules write into. It is deliberately not a
// general-purpose deep clone: a full one would have to grow with adjudicator.Policy and
// would silently rot, whereas this is scoped to three known mutators and says so.
func orgClonePolicyContainers(p adjudicator.Policy) adjudicator.Policy {
	out := p
	if p.Allow != nil {
		allow := make(map[string]bool, len(p.Allow))
		for k, v := range p.Allow {
			allow[k] = v
		}
		out.Allow = allow
	}
	if p.Deny != nil {
		deny := make(map[string]abi.ReasonCode, len(p.Deny))
		for k, v := range p.Deny {
			deny[k] = v
		}
		out.Deny = deny
	}
	// Slices are copied to their exact length so an append writes into fresh backing
	// storage instead of through spare capacity into the caller's array.
	out.AllowPrefix = append([]string(nil), p.AllowPrefix...)
	out.ArgPredicates = append(p.ArgPredicates[:0:0], p.ArgPredicates...)
	return out
}

// ---------------------------------------------------------------------------
// rendering
// ---------------------------------------------------------------------------

func writeOrgStatus(w io.Writer, store string, v orgStatusView) {
	fmt.Fprintf(w, "org policy plane: %s%s\n", strings.ToUpper(v.pull.Posture), orgPostureGloss(v))
	if !v.enrolled {
		// The DoD's exact requirement, said in words rather than implied by blank
		// fields: an operator must be able to tell "nobody governs this box" from
		// "the org governs it and asked for nothing".
		fmt.Fprintf(w, "  no central authority on this box — every knob below is compiled-in or operator-set\n")
		fmt.Fprintf(w, "  enrollment: %s (absent)\n", store)
		fmt.Fprintf(w, "  next:       fak enroll --org <url> --root-key <base64|file>\n")
	} else {
		fmt.Fprintf(w, "  org:        %s\n", v.enrollment.OrgURL)
		fmt.Fprintf(w, "  issuer:     %s\n", orgFirst(v.pull.Issuer, v.enrollment.Issuer))
		fmt.Fprintf(w, "  device:     %s\n", v.enrollment.DeviceID)
		if v.pull.Version > 0 {
			fmt.Fprintf(w, "  version:    %d\n", v.pull.Version)
		}
		fmt.Fprintf(w, "  freshness:  %s (refuse-to-widen after %s)\n", orgAge(v.pull), v.maxStale)
		if v.pull.Reason != "" {
			fmt.Fprintf(w, "  reason:     %s\n", v.pull.Reason)
		}
		if v.pull.Err != nil {
			fmt.Fprintf(w, "  detail:     %v\n", v.pull.Err)
		}
		fmt.Fprintf(w, "  enrollment: %s\n", store)
	}

	fmt.Fprintf(w, "\ncapability floor: %s\n", v.floorSource)
	if v.overlayNote != "" {
		fmt.Fprintf(w, "  operator overlay: %s\n", v.overlayNote)
	}
	f := v.composition.Fold
	orgWriteChanges(w, "central widened", f.CentralWiden)
	orgWriteChanges(w, "central tightened", f.CentralTighten)
	orgWriteChanges(w, "central REFUSED (no authority over the compiled floor)", v.composition.CentralRefused)
	orgWriteChanges(w, "operator CLAMPED to the central grant", v.composition.OperatorClamped)
	if v.granted > 0 {
		fmt.Fprintf(w, "  journaled %d central capability grant(s)\n", v.granted)
	}
	if !v.composition.CentralAuthority() && len(f.CentralWiden)+len(f.CentralTighten) == 0 {
		fmt.Fprintf(w, "  (no central manifest applied)\n")
	}

	fmt.Fprintf(w, "\nper-knob provenance (%d knobs)\n", len(f.Knobs))
	for _, k := range orgSortedKnobs(f.Knobs) {
		note := ""
		if k.Widened {
			note = "  WIDENED"
		}
		// Trimmed rather than padded to the full width: the overwhelmingly common row
		// has no note, and a table where every line carries invisible trailing spaces
		// is a nuisance to diff and to paste into an issue.
		fmt.Fprintln(w, strings.TrimRight(fmt.Sprintf("  %-26s %-12s %-17s%s", k.Field, k.Class, k.Channel, note), " "))
	}
}

// orgPostureGloss turns the posture token into the sentence an operator needs. The
// tokens are precise and unhelpful on their own: "floor" in particular reads like a
// success to anyone who has not read internal/policy/orgpull.go.
func orgPostureGloss(v orgStatusView) string {
	switch v.pull.Posture {
	case policy.OrgPostureInert:
		return " — not enrolled; the org plane does nothing here"
	case policy.OrgPostureFresh:
		return " — a verified org manifest is in force"
	case policy.OrgPostureLastGood:
		return " — the endpoint did not answer; a cached verified manifest is still in force"
	case policy.OrgPostureFloor:
		return " — REFUSING TO WIDEN: nothing from the org plane applies, the compiled floor stands"
	default:
		return ""
	}
}

func orgAge(res policy.OrgPullResult) string {
	switch res.Posture {
	case policy.OrgPostureFresh:
		return "fresh (verified just now)"
	case policy.OrgPostureLastGood:
		return res.Age.Round(time.Second).String() + " old"
	default:
		return "n/a"
	}
}

func orgWriteChanges(w io.Writer, label string, changes []policy.AmendmentChange) {
	if len(changes) == 0 {
		return
	}
	fmt.Fprintf(w, "  %s: %s\n", label, policy.FormatAmendmentChanges(changes))
}

// orgSortedKnobs orders the provenance table so the knobs someone CHANGED come first —
// a 21-row table where the two interesting rows are in registry-declaration order buries
// the answer. Within each group the order is the registry's, which is stable across runs.
func orgSortedKnobs(knobs []policy.OrgKnobProvenance) []policy.OrgKnobProvenance {
	out := append([]policy.OrgKnobProvenance(nil), knobs...)
	rank := func(k policy.OrgKnobProvenance) int {
		switch k.Channel {
		case policy.ChannelCentral:
			return 0
		case policy.ChannelOperatorOverlay:
			return 1
		default:
			return 2
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return rank(out[i]) < rank(out[j]) })
	return out
}

func orgFirst(vals ...string) string {
	for _, v := range vals {
		if s := strings.TrimSpace(v); s != "" {
			return s
		}
	}
	return ""
}

// orgStatusJSON mirrors the text arm. Every key is ALWAYS present and never null
// (#5299 house rule): a consumer must not have to read an absent key as "no central
// authority", because an absent key is also what a truncated write produces.
func orgStatusJSON(v orgStatusView) map[string]any {
	f := v.composition.Fold
	knobs := make([]map[string]any, 0, len(f.Knobs))
	for _, k := range orgSortedKnobs(f.Knobs) {
		knobs = append(knobs, map[string]any{
			"knob":    k.Field,
			"class":   string(k.Class),
			"channel": k.Channel,
			"widened": k.Widened,
			"changes": orgChangesJSON(k.Changes),
		})
	}
	return map[string]any{
		"posture":           v.pull.Posture,
		"central_authority": v.composition.CentralAuthority(),
		"enrolled":          v.enrolled,
		"org_url":           v.enrollment.OrgURL,
		"issuer":            orgFirst(v.pull.Issuer, v.enrollment.Issuer),
		"device_id":         v.enrollment.DeviceID,
		"version":           v.pull.Version,
		"age_seconds":       int64(v.pull.Age / time.Second),
		"max_staleness_sec": int64(v.maxStale / time.Second),
		"reason":            v.pull.Reason,
		"error":             orgErrString(v.pull.Err),
		"floor_source":      v.floorSource,
		"operator_overlay":  v.overlayNote,
		"central_widen":     orgChangesJSON(f.CentralWiden),
		"central_tighten":   orgChangesJSON(f.CentralTighten),
		"central_refused":   orgChangesJSON(v.composition.CentralRefused),
		"operator_clamped":  orgChangesJSON(v.composition.OperatorClamped),
		"grants_journaled":  v.granted,
		"knobs":             knobs,
	}
}

func orgChangesJSON(changes []policy.AmendmentChange) []map[string]any {
	out := make([]map[string]any, 0, len(changes))
	for _, c := range changes {
		out = append(out, map[string]any{
			"knob":  c.Field,
			"label": c.Label,
			"old":   c.Old,
			"new":   c.New,
		})
	}
	return out
}

func orgErrString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

// ---------------------------------------------------------------------------
// journal: central widenings, with issuer + envelope version
// ---------------------------------------------------------------------------

// orgCentralGrantRows derives one CAPABILITY_GRANT row per knob the central manifest
// WIDENED — the #5322 provenance requirement, and the first producer of
// journal.GrantChannelCentral (the constant has shipped since #5178 with no caller).
//
// Only widenings are recorded, matching journal.GrantDirectionWiden: a central RATCHET
// is the org making the fleet stricter and needs no grant trail to justify it. Issuer
// and envelope version ride the Source field as `<issuer>@v<n>`, which is the pair an
// auditor needs to go find the exact document that handed out the capability — a bare
// issuer would not distinguish two manifests from the same org.
func orgCentralGrantRows(fold policy.OrgFold, issuer string, version uint64) []journal.CapabilityGrantRow {
	source := fmt.Sprintf("%s@v%d", orgFirst(issuer, "unknown-issuer"), version)
	rows := make([]journal.CapabilityGrantRow, 0, len(fold.CentralWiden))
	for _, c := range fold.CentralWiden {
		class := string(policy.AmendGatedWiden)
		if knob, ok := policy.KnobByField(c.Field); ok {
			// Report the knob's DECLARED class rather than assuming GATED_WIDEN. A
			// widening of a RATCHET knob is a real and more alarming event (the org
			// removed a tighten), and flattening it to GATED_WIDEN in the ledger would
			// hide exactly the row an auditor is looking for.
			class = string(knob.Class)
		}
		rows = append(rows, journal.CapabilityGrantRow{
			Knob:    c.Field,
			Class:   class,
			Old:     c.Old,
			New:     c.New,
			Channel: journal.GrantChannelCentral,
			Actor:   orgFirst(issuer, "unknown-issuer"),
			Source:  source,
			Reason:  "central org manifest applied (" + c.Label + ")",
		})
	}
	return rows
}

// orgEmitCentralGrants writes those rows and returns how many landed. j is nil outside
// a session; AppendCapabilityGrant no-ops on a nil receiver, so an unjournaled run needs
// no guard here and stays byte-identical.
func orgEmitCentralGrants(j *journal.Journal, fold policy.OrgFold, issuer string, version uint64) int {
	rows := orgCentralGrantRows(fold, issuer, version)
	for _, r := range rows {
		j.AppendCapabilityGrant(r)
	}
	return len(rows)
}
