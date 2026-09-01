package storagepressure

import (
	"time"

	"github.com/anthony-chaudhary/fak/internal/treedoctor"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

const Schema = "fak-storage-pressure/1"

const (
	OwnerWorktrees = "fak worktree worker reap --all-cold"
	OwnerGoTmp     = "fak tree-doctor --go-tmp --json"
	OwnerGoCache   = "fak git-daily --dry-run --json"
)

type Filesystem struct {
	Path             string `json:"path"`
	TotalBytes       int64  `json:"total_bytes,omitempty"`
	FreeBytes        int64  `json:"free_bytes,omitempty"`
	Known            bool   `json:"known"`
	WarningFreeBytes int64  `json:"warning_free_bytes"`
	RefuseFreeBytes  int64  `json:"refuse_free_bytes"`
	Warning          bool   `json:"warning"`
	Refuse           bool   `json:"refuse"`
}

type Contributor struct {
	Name                     string   `json:"name"`
	OwnerCommand             string   `json:"owner_command"`
	ObservedBytes            int64    `json:"observed_bytes,omitempty"`
	ObservedBytesComplete    bool     `json:"observed_bytes_complete"`
	ReclaimableBytes         int64    `json:"reclaimable_bytes,omitempty"`
	ReclaimableBytesComplete bool     `json:"reclaimable_bytes_complete"`
	UnknownEntries           int      `json:"unknown_entries,omitempty"`
	Reasons                  []string `json:"reasons,omitempty"`
}

type Report struct {
	Schema                   string        `json:"schema"`
	GeneratedAt              time.Time     `json:"generated_at"`
	Filesystem               Filesystem    `json:"filesystem"`
	Contributors             []Contributor `json:"contributors"`
	ObservedBytes            int64         `json:"observed_bytes,omitempty"`
	ObservedBytesComplete    bool          `json:"observed_bytes_complete"`
	ReclaimableBytes         int64         `json:"reclaimable_bytes,omitempty"`
	ReclaimableBytesComplete bool          `json:"reclaimable_bytes_complete"`
}

func New(now time.Time, filesystem Filesystem, contributors ...Contributor) Report {
	filesystem.Warning = filesystem.Known && filesystem.WarningFreeBytes > 0 && filesystem.FreeBytes < filesystem.WarningFreeBytes
	filesystem.Refuse = filesystem.Known && filesystem.RefuseFreeBytes > 0 && filesystem.FreeBytes < filesystem.RefuseFreeBytes

	r := Report{
		Schema:                   Schema,
		GeneratedAt:              now.UTC(),
		Filesystem:               filesystem,
		Contributors:             contributors,
		ObservedBytesComplete:    true,
		ReclaimableBytesComplete: true,
	}
	for _, c := range contributors {
		r.ObservedBytes += c.ObservedBytes
		r.ReclaimableBytes += c.ReclaimableBytes
		if !c.ObservedBytesComplete {
			r.ObservedBytesComplete = false
		}
		if !c.ReclaimableBytesComplete {
			r.ReclaimableBytesComplete = false
		}
	}
	return r
}

func Worktrees(items []workerworktree.ColdWorktree) Contributor {
	c := Contributor{Name: "managed_worktrees", OwnerCommand: OwnerWorktrees, ObservedBytesComplete: true, ReclaimableBytesComplete: true}
	for _, item := range items {
		if item.ReclaimBytesKnown {
			c.ObservedBytes += item.ReclaimBytes
		} else {
			c.ObservedBytesComplete = false
		}
		if item.Eligible && item.ReclaimBytesKnown {
			c.ReclaimableBytes += item.ReclaimBytes
			continue
		}
		// Retained worktree bytes are observations, not owner-approved eligibility.
		// Active leases, dirty/unlanded work, young trees, and failed probes all
		// keep the reclaimable estimate explicitly incomplete.
		reason := item.Reason
		if item.Eligible && !item.ReclaimBytesKnown {
			reason = "unknown-size eligible worktree"
		}
		markUnknown(&c, reason)
	}
	return c
}

func GoTmp(report treedoctor.GoTmpReport) Contributor {
	c := Contributor{Name: "managed_gotmp", OwnerCommand: OwnerGoTmp, ObservedBytes: report.TotalBytes, ObservedBytesComplete: true, ReclaimableBytesComplete: true}
	if report.Err != "" {
		c.ObservedBytesComplete = false
		markUnknown(&c, report.Err)
	}
	if report.ProcessErr != "" {
		markUnknown(&c, "process census: "+report.ProcessErr)
	}
	ownerComplete := report.Err == "" && report.ProcessErr == ""
	for _, entry := range report.Entries {
		switch {
		case entry.Truncated:
			c.ObservedBytesComplete = false
			markUnknown(&c, "truncated entry: "+entry.Path)
		case entry.ScanErr != "":
			c.ObservedBytesComplete = false
			markUnknown(&c, "scan error: "+entry.Path)
		case entry.Verdict == treedoctor.GoTmpReap && ownerComplete:
			c.ReclaimableBytes += entry.Bytes
		case entry.Verdict == treedoctor.GoTmpReap:
			markUnknown(&c, "owner census incomplete: "+entry.Path)
		case entry.Verdict == treedoctor.GoTmpKeepLive || entry.Verdict == treedoctor.GoTmpKeepIndeterminate:
			markUnknown(&c, entry.Reason)
		}
	}
	return c
}

func GoCache(report treedoctor.GoCacheReport) Contributor {
	complete := report.ScanComplete && report.Err == "" && report.IncompleteReason == "" && report.CandidateBytesUnknown == 0
	c := Contributor{
		Name:                     "ambient_gocache",
		OwnerCommand:             OwnerGoCache,
		ObservedBytes:            report.BytesBefore,
		ObservedBytesComplete:    complete,
		ReclaimableBytesComplete: complete,
	}
	if complete {
		c.ReclaimableBytes = report.CandidateBytes
	}
	if report.Err != "" {
		markUnknown(&c, report.Err)
	}
	if report.IncompleteReason != "" {
		markUnknown(&c, report.IncompleteReason)
	}
	if report.CandidateBytesUnknown > 0 {
		c.UnknownEntries += report.CandidateBytesUnknown
		appendReason(&c, "unknown-size cache candidates")
		c.ReclaimableBytesComplete = false
	}
	if !report.ScanComplete && report.IncompleteReason == "" {
		markUnknown(&c, "incomplete cache census")
	}
	return c
}

func markUnknown(c *Contributor, reason string) {
	c.UnknownEntries++
	c.ReclaimableBytesComplete = false
	appendReason(c, reason)
}

func appendReason(c *Contributor, reason string) {
	if reason == "" {
		reason = "eligibility unknown"
	}
	for _, existing := range c.Reasons {
		if existing == reason {
			return
		}
	}
	c.Reasons = append(c.Reasons, reason)
}
