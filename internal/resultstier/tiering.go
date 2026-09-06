package resultstier

import (
	"fmt"
	"path"
	"path/filepath"
	"strings"
)

// Tier represents the storage tier for an artifact.
type Tier int

const (
	// TierUnknown is the zero value fail-closed state: must force a decision, never default silently.
	TierUnknown Tier = 0
	// TierClaim stays in Git: diffable, reviewable, schema-validated manifests, metrics, scores, prose summaries, hashes.
	TierClaim Tier = 1
	// TierPayload belongs in blob/external store: machine-written raw dumps, predictions, logs, unaggregated sample streams, caches.
	TierPayload Tier = 2
)

// String returns the string representation of the Tier.
func (t Tier) String() string {
	switch t {
	case TierClaim:
		return "claim"
	case TierPayload:
		return "payload"
	default:
		return "unknown"
	}
}

// Known returns true if the tier is TierClaim or TierPayload.
func (t Tier) Known() bool {
	return t == TierClaim || t == TierPayload
}

type roleRule struct {
	pattern string
	desc    string
}

var claimRoles = []roleRule{
	{pattern: "INDEX.md", desc: "generated index"},
	{pattern: "payload-index.json", desc: "index of externalized payload objects"},
	{pattern: "*manifest*.json", desc: "schema-validated claim / run manifest"},
	{pattern: "perf.json", desc: "latency and hardware performance metrics"},
	{pattern: "row.json", desc: "measurement rows"},
	{pattern: "entry.json", desc: "build/timing entry"},
	{pattern: "*summary*.json", desc: "summary aggregates"},
	{pattern: "*census*.json", desc: "census counts"},
	{pattern: "*score*.json", desc: "scores/verdicts"},
	{pattern: "outcome.json", desc: "verdicts"},
	{pattern: "funnel.json", desc: "stage counts"},
	{pattern: "events.json", desc: "ordered event record"},
	{pattern: "*.md", desc: "prose claims"},
	{pattern: "*.sha256", desc: "provenance hashes"},
	{pattern: "*.sql", desc: "source queries"},
	{pattern: "*.sh", desc: "driver scripts"},
}

var payloadRoles = []roleRule{
	{pattern: "*layerinfo*.json", desc: "per-layer dumps"},
	{pattern: "predictions*.json", desc: "raw model predictions"},
	{pattern: "*diff.json", desc: "machine tactic diffs"},
	{pattern: "per-frame*.csv", desc: "per-frame tabular output"},
	{pattern: "*.log", desc: "raw builder/runner stdout logs"},
	{pattern: "times-*.json", desc: "raw per-repeat sample files"},
	{pattern: "pmon-*.txt", desc: "raw nvidia-smi / GPU monitor samples"},
	{pattern: "*.start", desc: "raw start timestamps"},
	{pattern: "*.end", desc: "raw end timestamps"},
	{pattern: "prefill.json", desc: "raw load-test prompts"},
	{pattern: "*.rle", desc: "mask binary bytes"},
	{pattern: "*.cache", desc: "execution timing caches"},
	{pattern: "*.csv", desc: "tabular measurement output"},
	{pattern: "*.tsv", desc: "tabular output"},
	{pattern: "*.jsonl", desc: "streaming record lines"},
	{pattern: "*.npy", desc: "raw tensor dumps"},
	{pattern: "*.png", desc: "rendered images/crop sheets"},
}

// TierOf evaluates the base name of path p against the role rules.
// It checks Claim roles first, then Payload roles.
// If empty or unmatched, it returns TierUnknown with an explanation.
func TierOf(p string) (Tier, string) {
	cleanPath := filepath.ToSlash(strings.TrimSpace(p))
	if cleanPath == "" {
		return TierUnknown, "empty path"
	}
	base := path.Base(cleanPath)
	if base == "." || base == "/" {
		return TierUnknown, fmt.Sprintf("invalid base name for path %q", p)
	}

	for _, r := range claimRoles {
		matched, _ := path.Match(r.pattern, base)
		if matched {
			return TierClaim, r.desc
		}
	}
	for _, r := range payloadRoles {
		matched, _ := path.Match(r.pattern, base)
		if matched {
			return TierPayload, r.desc
		}
	}

	return TierUnknown, fmt.Sprintf("unmatched result role for %q", base)
}

// Census tallies file counts and sizes across tiers.
type Census struct {
	ClaimFiles   int
	ClaimBytes   int64
	PayloadFiles int
	PayloadBytes int64
	UnknownFiles int
	UnknownBytes int64
	UnknownExts  map[string]int
}

// TotalFiles returns the total number of files across all tiers.
func (c Census) TotalFiles() int {
	return c.ClaimFiles + c.PayloadFiles + c.UnknownFiles
}

// TotalBytes returns the total size in bytes across all tiers.
func (c Census) TotalBytes() int64 {
	return c.ClaimBytes + c.PayloadBytes + c.UnknownBytes
}

// PayloadShare returns the fraction of total bytes in the payload tier, clamped to [0, 1].
func (c Census) PayloadShare() float64 {
	total := c.TotalBytes()
	if total <= 0 {
		return 0.0
	}
	share := float64(c.PayloadBytes) / float64(total)
	if share < 0 {
		return 0.0
	}
	if share > 1.0 {
		return 1.0
	}
	return share
}

// Shrink returns the reduction multiplier when payload is externalized (TotalBytes / RetainedBytes).
// RetainedBytes is ClaimBytes + UnknownBytes.
// If RetainedBytes is 0: returns 1.0 if TotalBytes is 0, or float64(TotalBytes) if TotalBytes > 0.
func (c Census) Shrink() float64 {
	retained := c.ClaimBytes + c.UnknownBytes
	if retained <= 0 {
		if c.TotalBytes() == 0 {
			return 1.0
		}
		return float64(c.TotalBytes())
	}
	return float64(c.TotalBytes()) / float64(retained)
}

// String returns a human-readable summary of the Census.
func (c Census) String() string {
	return fmt.Sprintf("Census(claim: %d files / %d bytes, payload: %d files / %d bytes, unknown: %d files / %d bytes, payload_share: %.2f%%, shrink: %.2fx)",
		c.ClaimFiles, c.ClaimBytes,
		c.PayloadFiles, c.PayloadBytes,
		c.UnknownFiles, c.UnknownBytes,
		c.PayloadShare()*100,
		c.Shrink(),
	)
}

// Split classifies paths into claim, payload, and unknown buckets.
type Split struct {
	Claim   []string
	Payload []string
	Unknown []string
}

// Classify separates paths into their corresponding tiers based on TierOf.
func Classify(paths []string) Split {
	s := Split{
		Claim:   make([]string, 0),
		Payload: make([]string, 0),
		Unknown: make([]string, 0),
	}
	for _, p := range paths {
		tier, _ := TierOf(p)
		switch tier {
		case TierClaim:
			s.Claim = append(s.Claim, p)
		case TierPayload:
			s.Payload = append(s.Payload, p)
		default:
			s.Unknown = append(s.Unknown, p)
		}
	}
	return s
}
