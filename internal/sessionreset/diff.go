package sessionreset

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// diff.go — issue #1575: the managed-context BEFORE/AFTER DIFF for every reset, a
// pure read/render layer over data BuildSeed/BuildResetTransaction already compute.
// #1582 (transaction.go) already records what a reset changed — SeedDigest,
// Contributors, OmittedSpans (each with a Reason), WarmPrefixDigest — as payload-free
// audit handles. #1575 does not add a new capture mechanism: it classifies those
// same handles, plus the Seed's own rendered Parts, into the four buckets the issue's
// "In scope" line names by name (survived, expired, summarized, must-be-re-queried),
// and renders both a human-readable Explain() and a Markdown() report — the same
// pairing ctxplan.Preview (#1574) and memview.Timeline (#1599) already ship.
//
// THE FOUR BUCKETS.
//
//	survived        a seed Part that carries forward near-verbatim: verbatim_tail and
//	                any other contributor whose Meta does not mark it a compaction
//	                (task_distill is also near-verbatim — a two-line extract, not a
//	                lossy fold — so it survives too).
//	summarized      a seed Part that is a LOSSY fold over more source material than it
//	                renders — durability_facts (many transcript lines -> a kept-fact
//	                list) is the shipped example; a future contributor can opt in by
//	                naming itself in summarizingContributors.
//	must_requery    cold handles that need an explicit follow-up to page back in: the
//	                warm-prefix descriptor (replay from the vCache prefix-DAG) and any
//	                OmittedSpan whose Reason is a pointer-style reason
//	                (stable_prefix_descriptor) rather than a plain drop.
//	expired         every remaining OmittedSpan — transcript bytes the reset let go
//	                with no recovery handle beyond the digest (tool results, turn-scoped
//	                ephemera, or a summarized_or_clipped line already folded into
//	                "summarized" so it is not double-counted as lost).
//
// Every OmittedSpan lands in exactly one of {must_requery, expired}, and every Seed
// Part lands in exactly one of {survived, summarized} — ResetDiffOf never drops a row,
// mirroring ctxplan.Preview's row-accounting invariant (checked by
// TestResetDiffCoversEveryPartAndSpan).

// Reset-diff bucket names — the product-facing vocabulary the issue's "Done
// condition" asks a reset to show: what survived, what was summarized, what expired,
// and what must be re-queried.
const (
	BucketSurvived    = "survived"
	BucketSummarized  = "summarized"
	BucketExpired     = "expired"
	BucketMustRequery = "must_requery"
)

// summarizingContributors names the seed contributors whose Part is a lossy fold over
// more source material than it renders, rather than a near-verbatim carry-forward.
// durability_facts is the shipped example (many transcript lines -> a kept-fact
// list); a future contributor opts into the "summarized" bucket by adding its Name()
// here rather than by the diff re-guessing from the rendered text.
var summarizingContributors = map[string]bool{
	"durability_facts": true,
}

// requeryOmitReasons names the OmittedSpan.Reason values that mean "cold, but a
// recovery handle exists — an explicit follow-up query pages it back in", matching
// the reasons omittedReason (transaction.go) actually emits for a pointer-style span.
// Every other reason means the bytes are gone with only a digest left (expired).
var requeryOmitReasons = map[string]bool{
	"stable_prefix_descriptor": true,
}

// DiffPart is one seed Part's projection into the diff — which bucket it landed in,
// plus enough of its own accounting to explain why, without re-deriving it from the
// Seed.
type DiffPart struct {
	Name   string `json:"name"`
	Bucket string `json:"bucket"`
	Text   string `json:"text,omitempty"`
	Chars  int    `json:"chars"`
}

// DiffSpan is one omitted transcript span's projection into the diff — the
// payload-free handle plus which bucket it landed in.
type DiffSpan struct {
	Index  int    `json:"index"`
	Role   string `json:"role,omitempty"`
	Bucket string `json:"bucket"`
	Digest string `json:"digest"`
	Reason string `json:"reason,omitempty"`
}

// ResetDiff is the rendered before/after view of one reset: what the drained session
// held (Before, in terms of transcript spans) against what the fresh session opens
// with (After, the seed's carried-over parts), classified into the four buckets a
// human asks about after a reset. It carries the SeedDigest and BudgetRearm from the
// ResetTransaction so a diff is self-describing without a second lookup, and it is
// pure/deterministic: same Input+Seed+ResetTransaction in, same ResetDiff out.
type ResetDiff struct {
	OldTrace    string                   `json:"old_trace,omitempty"`
	NewTrace    string                   `json:"new_trace,omitempty"`
	SeedDigest  string                   `json:"seed_digest,omitempty"`
	BudgetRearm session.ResetBudgetRearm `json:"budget_rearm,omitempty,omitzero"`

	Survived    []DiffPart `json:"survived"`
	Summarized  []DiffPart `json:"summarized"`
	MustRequery []DiffSpan `json:"must_requery"`
	Expired     []DiffSpan `json:"expired"`

	// BeforeSpans/AfterChars give the O(1) size-of-the-delta an operator wants without
	// re-summing every bucket: how many transcript spans existed before the reset
	// (len(Input.Messages)) against how many characters of seed text the fresh session
	// actually opens with (len(Seed.Recap)).
	BeforeSpans int `json:"before_spans"`
	AfterChars  int `json:"after_chars"`
}

// DiffReset renders the before/after diff for one reset from the same three values
// the reset call site already builds (Input, the Seed BuildSeed produced from it, and
// the ResetTransaction BuildResetTransaction produced from both) — no new capture, no
// re-reading of dropped bytes: OmittedSpans and Seed.Parts already are the full
// record of what happened. Never a partial answer: every Part and every OmittedSpan
// is classified into exactly one bucket, so DiffReset never has to be paired with a
// second call to trust its totals (RowCount()).
func DiffReset(in Input, seed Seed, tx session.ResetTransaction) ResetDiff {
	d := ResetDiff{
		OldTrace:    tx.OldTrace,
		NewTrace:    tx.NewTrace,
		SeedDigest:  tx.SeedDigest,
		BudgetRearm: tx.BudgetRearm,
		BeforeSpans: len(in.Messages),
		AfterChars:  len(seed.Recap),
	}
	for _, p := range seed.Parts {
		dp := DiffPart{Name: p.Name, Text: p.Text, Chars: len(p.Text)}
		if summarizingContributors[p.Name] {
			dp.Bucket = BucketSummarized
			d.Summarized = append(d.Summarized, dp)
		} else {
			dp.Bucket = BucketSurvived
			d.Survived = append(d.Survived, dp)
		}
	}
	for _, s := range tx.OmittedSpans {
		ds := DiffSpan{Index: s.Index, Role: s.Role, Digest: s.Digest, Reason: s.Reason}
		if requeryOmitReasons[s.Reason] {
			ds.Bucket = BucketMustRequery
			d.MustRequery = append(d.MustRequery, ds)
		} else {
			ds.Bucket = BucketExpired
			d.Expired = append(d.Expired, ds)
		}
	}
	// Warm-prefix carries its own recovery handle even when no OmittedSpan reason
	// pointed at it (the system preamble it describes is fully rendered VIA the
	// warm_prefix Part, not omitted) — surface it as an explicit must-requery row so
	// "replay the stable prefix from the vCache prefix-DAG" is never invisible in a
	// diff just because the bytes were not dropped outright.
	if tx.WarmPrefixDigest != "" {
		d.MustRequery = append(d.MustRequery, DiffSpan{
			Bucket: BucketMustRequery,
			Digest: tx.WarmPrefixDigest,
			Reason: "warm_prefix_replay",
		})
	}
	sortDiffSpans(d.MustRequery)
	sortDiffSpans(d.Expired)
	return d
}

func sortDiffSpans(spans []DiffSpan) {
	sort.SliceStable(spans, func(i, j int) bool {
		if spans[i].Index != spans[j].Index {
			return spans[i].Index < spans[j].Index
		}
		return spans[i].Digest < spans[j].Digest
	})
}

// RowCount is the total number of classified rows across all four buckets — for a
// diff built from a well-formed reset this always equals
// len(seed.Parts)+len(tx.OmittedSpans)+(1 if a warm prefix digest is present else 0),
// the invariant TestResetDiffCoversEveryPartAndSpan checks.
func (d ResetDiff) RowCount() int {
	return len(d.Survived) + len(d.Summarized) + len(d.MustRequery) + len(d.Expired)
}

// Explain renders the diff as an operator-readable report: one section per bucket,
// then the lineage/digest footer — the "what survived, what expired, what was
// summarized, what needs a follow-up query" account the issue's Done condition asks
// every reset to show.
func (d ResetDiff) Explain() string {
	var b strings.Builder
	fmt.Fprintf(&b, "reset diff: %s -> %s\n", nz(d.OldTrace), nz(d.NewTrace))
	fmt.Fprintf(&b, "  before: %d transcript span(s)\n", d.BeforeSpans)
	fmt.Fprintf(&b, "  after:  %d char(s) of carried-over seed, seed_digest=%s\n", d.AfterChars, shortHash(d.SeedDigest))
	fmt.Fprintf(&b, "  budget rearm: context_tokens=%d/%d turns=%s tokens=%s\n",
		d.BudgetRearm.ContextTokensLeft, d.BudgetRearm.ContextTokensCap,
		unboundedLabel(d.BudgetRearm.TurnsLeft), unboundedLabel(d.BudgetRearm.TokensLeft))
	writeDiffParts(&b, "SURVIVED     (carried forward near-verbatim)", d.Survived)
	writeDiffParts(&b, "SUMMARIZED   (folded/distilled into the seed)", d.Summarized)
	writeDiffSpans(&b, "MUST-REQUERY (cold, recoverable via an explicit follow-up)", d.MustRequery)
	writeDiffSpans(&b, "EXPIRED      (dropped, no recovery handle beyond the digest)", d.Expired)
	return b.String()
}

func writeDiffParts(b *strings.Builder, title string, parts []DiffPart) {
	fmt.Fprintf(b, "  %s: %d part(s)\n", title, len(parts))
	for _, p := range parts {
		fmt.Fprintf(b, "     %-18s %5d char(s)  %s\n", p.Name, p.Chars, truncate(oneLine(p.Text), 60))
	}
}

func writeDiffSpans(b *strings.Builder, title string, spans []DiffSpan) {
	fmt.Fprintf(b, "  %s: %d span(s)\n", title, len(spans))
	for _, s := range spans {
		if s.Index == 0 && s.Role == "" && s.Reason == "warm_prefix_replay" {
			fmt.Fprintf(b, "     [warm prefix] reason=%-24s handle=%s\n", s.Reason, short(s.Digest))
			continue
		}
		fmt.Fprintf(b, "     [span %-4d] %-10s reason=%-24s handle=%s\n", s.Index, s.Role, s.Reason, short(s.Digest))
	}
}

// Markdown renders the same four buckets as a Markdown report — the shareable form
// for a teammate reviewing a reset outside a terminal (mirrors
// ctxplan.Preview.Markdown / memview.Timeline.Render's convention).
func (d ResetDiff) Markdown() string {
	var b strings.Builder
	b.WriteString("# Session reset diff\n\n")
	fmt.Fprintf(&b, "- lineage: `%s` -> `%s`\n", nz(d.OldTrace), nz(d.NewTrace))
	fmt.Fprintf(&b, "- seed digest: `%s`\n", nz(d.SeedDigest))
	fmt.Fprintf(&b, "- before: %d transcript span(s)\n", d.BeforeSpans)
	fmt.Fprintf(&b, "- after: %d char(s) carried over\n", d.AfterChars)
	fmt.Fprintf(&b, "- budget rearm: context_tokens=%d/%d\n\n", d.BudgetRearm.ContextTokensLeft, d.BudgetRearm.ContextTokensCap)

	b.WriteString("## Survived (carried forward near-verbatim)\n\n")
	writeDiffPartsMD(&b, d.Survived)
	b.WriteString("## Summarized (folded/distilled into the seed)\n\n")
	writeDiffPartsMD(&b, d.Summarized)
	b.WriteString("## Must re-query (cold, recoverable via an explicit follow-up)\n\n")
	writeDiffSpansMD(&b, d.MustRequery)
	b.WriteString("## Expired (dropped, no recovery handle beyond the digest)\n\n")
	writeDiffSpansMD(&b, d.Expired)
	return b.String()
}

func writeDiffPartsMD(b *strings.Builder, parts []DiffPart) {
	if len(parts) == 0 {
		b.WriteString("_none_\n\n")
		return
	}
	b.WriteString("| part | chars | preview |\n|---|---|---|\n")
	for _, p := range parts {
		fmt.Fprintf(b, "| %s | %d | %s |\n", mdEscape(p.Name), p.Chars, mdEscape(truncate(oneLine(p.Text), 80)))
	}
	b.WriteByte('\n')
}

func writeDiffSpansMD(b *strings.Builder, spans []DiffSpan) {
	if len(spans) == 0 {
		b.WriteString("_none_\n\n")
		return
	}
	b.WriteString("| index | role | reason | handle |\n|---|---|---|---|\n")
	for _, s := range spans {
		fmt.Fprintf(b, "| %d | %s | %s | %s |\n", s.Index, mdEscape(s.Role), mdEscape(s.Reason), short(s.Digest))
	}
	b.WriteByte('\n')
}

func nz(s string) string {
	if s == "" {
		return "(none)"
	}
	return s
}

func unboundedLabel(n int) string {
	if n < 0 {
		return "unbounded"
	}
	return strconv.Itoa(n)
}

func oneLine(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

func truncate(s string, max int) string {
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	return string(r[:max]) + "…"
}

func short(digest string) string {
	if digest == "" {
		return "(none)"
	}
	if len(digest) <= 12 {
		return digest
	}
	return digest[:12]
}

func shortHash(digest string) string {
	return short(digest)
}

func mdEscape(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, "|", "\\|"), "\n", " ")
}
