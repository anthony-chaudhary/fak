package storagepressure

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/treedoctor"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

func TestNewAggregatesOnlyOwnerApprovedReclaimableBytes(t *testing.T) {
	worktrees := Worktrees([]workerworktree.ColdWorktree{
		{Path: "/cold", Eligible: true, ReclaimBytes: 100, ReclaimBytesKnown: true},
		{Path: "/active", LeaseLive: true, Reason: "kept: worker lane lease still live"},
	})
	gotmp := GoTmp(treedoctor.GoTmpReport{
		TotalBytes: 70,
		Entries: []treedoctor.GoTmpEntry{
			{Path: "/tmp/old", Bytes: 40, Verdict: treedoctor.GoTmpReap},
			{Path: "/tmp/live", Bytes: 30, Verdict: treedoctor.GoTmpKeepLive, Reason: "referenced by live process"},
		},
	})
	gocache := GoCache(treedoctor.GoCacheReport{BytesBefore: 300, CandidateBytes: 60, ScanComplete: true})

	report := New(time.Unix(10, 0), Filesystem{
		Path: "/", TotalBytes: 1000, FreeBytes: 80, Known: true,
		WarningFreeBytes: 100, RefuseFreeBytes: 50,
	}, worktrees, gotmp, gocache)

	if got, want := report.ObservedBytes, int64(470); got != want {
		t.Fatalf("observed bytes = %d, want %d", got, want)
	}
	if got, want := report.ReclaimableBytes, int64(200); got != want {
		t.Fatalf("reclaimable bytes = %d, want %d", got, want)
	}
	if report.ReclaimableBytesComplete {
		t.Fatal("active worktree/GOTMP entries must keep the aggregate reclaimable total incomplete")
	}
	if !report.Filesystem.Warning || report.Filesystem.Refuse {
		t.Fatalf("threshold verdict = warning %v refuse %v, want true/false", report.Filesystem.Warning, report.Filesystem.Refuse)
	}
}

func TestIncompleteContributorPropagatesWithoutInventingReclaimableBytes(t *testing.T) {
	cache := GoCache(treedoctor.GoCacheReport{
		BytesBefore: 200, CandidateBytes: 75, ScanComplete: false,
		IncompleteReason: "deadline exceeded",
	})
	gotmp := GoTmp(treedoctor.GoTmpReport{
		TotalBytes: 90,
		Entries:    []treedoctor.GoTmpEntry{{Path: "/tmp/large", Bytes: 90, Truncated: true, Verdict: treedoctor.GoTmpReap}},
	})
	report := New(time.Time{}, Filesystem{Known: false, WarningFreeBytes: 100, RefuseFreeBytes: 50}, cache, gotmp)

	if report.ObservedBytesComplete || report.ReclaimableBytesComplete {
		t.Fatalf("completeness = observed %v reclaimable %v, want both false", report.ObservedBytesComplete, report.ReclaimableBytesComplete)
	}
	if got := report.ReclaimableBytes; got != 0 {
		t.Fatalf("reclaimable bytes = %d, want 0 from incomplete owner census", got)
	}
	if report.Filesystem.Warning || report.Filesystem.Refuse {
		t.Fatal("unknown filesystem capacity must not synthesize warning/refuse")
	}
}

func TestThresholdsUseFilesystemOnly(t *testing.T) {
	report := New(time.Time{}, Filesystem{
		Known: true, FreeBytes: 10, WarningFreeBytes: 20, RefuseFreeBytes: 15,
	}, Contributor{ReclaimableBytes: 1 << 40, ObservedBytesComplete: true, ReclaimableBytesComplete: true})
	if !report.Filesystem.Warning || !report.Filesystem.Refuse {
		t.Fatalf("threshold verdict = warning %v refuse %v, want true/true", report.Filesystem.Warning, report.Filesystem.Refuse)
	}
}

func TestThresholdBoundariesAreInclusiveAndUnknownStaysUnclassified(t *testing.T) {
	atWarning := AssessFilesystem(Filesystem{
		Known: true, FreeBytes: DefaultWarningFreeBytes,
		WarningFreeBytes: DefaultWarningFreeBytes, RefuseFreeBytes: DefaultRefuseFreeBytes,
	})
	if !atWarning.Warning || atWarning.Refuse {
		t.Fatalf("warning boundary = warning %v refuse %v, want true/false", atWarning.Warning, atWarning.Refuse)
	}
	atRefusal := AssessFilesystem(Filesystem{
		Known: true, FreeBytes: DefaultRefuseFreeBytes,
		WarningFreeBytes: DefaultWarningFreeBytes, RefuseFreeBytes: DefaultRefuseFreeBytes,
	})
	if !atRefusal.Warning || !atRefusal.Refuse {
		t.Fatalf("refusal boundary = warning %v refuse %v, want true/true", atRefusal.Warning, atRefusal.Refuse)
	}
	unknown := AssessFilesystem(Filesystem{
		Known: false, FreeBytes: 0,
		WarningFreeBytes: DefaultWarningFreeBytes, RefuseFreeBytes: DefaultRefuseFreeBytes,
	})
	if unknown.Warning || unknown.Refuse {
		t.Fatal("unknown capacity must not synthesize warning or refusal")
	}
}
