// fak-test:runtime fast est=1s lane=default
package workerworktree

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestCheckTrunkUnpublishedOrDiverged(t *testing.T) {
	fixedNow := time.Date(2026, 11, 27, 12, 0, 0, 0, time.UTC)
	prevNow := trunkUnpublishedNow
	trunkUnpublishedNow = func() time.Time { return fixedNow }
	t.Cleanup(func() { trunkUnpublishedNow = prevNow })

	t.Run("non-main branch fails open", func(t *testing.T) {
		fg := newFakeGit()
		if res := checkTrunkUnpublishedOrDiverged("/repo", "refs/heads/feature", "/wt", fg.run); res != nil {
			t.Fatalf("expected nil for non-main branch, got %+v", res)
		}
	})

	t.Run("missing origin/main fails open", func(t *testing.T) {
		fg := newFakeGit()
		fg.reply("rev-list", 128, "fatal: bad revision")
		if res := checkTrunkUnpublishedOrDiverged("/repo", trunkMainRef, "/wt", fg.run); res != nil {
			t.Fatalf("expected nil when origin/main absent, got %+v", res)
		}
	})

	t.Run("in-sync trunk passes without fetch", func(t *testing.T) {
		fg := newFakeGit()
		fg.reply("rev-list", 0, "0\t3\n")
		if res := checkTrunkUnpublishedOrDiverged("/repo", trunkMainRef, "/wt", fg.run); res != nil {
			t.Fatalf("expected nil when ahead=0, got %+v", res)
		}
	})

	t.Run("unpublished within 30m grace passes", func(t *testing.T) {
		fg := newFakeGit()
		fg.reply("rev-list", 0, "2\t0\n")
		fg.reply("log", 0, fmt.Sprintf("%d\n", fixedNow.Add(-10*time.Minute).Unix()))
		if res := checkTrunkUnpublishedOrDiverged("/repo", trunkMainRef, "/wt", fg.run); res != nil {
			t.Fatalf("expected nil within grace, got %+v", res)
		}
	})

	t.Run("unpublished past 30m grace refuses with ERR_FLOW_TRUNK_UNPUBLISHED", func(t *testing.T) {
		fg := newFakeGit()
		fg.reply("rev-list", 0, "3\t0\n")
		fg.reply("log", 0, fmt.Sprintf("%d\n", fixedNow.Add(-45*time.Minute).Unix()))
		fg.reply("fetch", 0, "")
		res := checkTrunkUnpublishedOrDiverged("/repo", trunkMainRef, "/wt", fg.run)
		if res == nil || res.OK || res.Code != LandResultTrunkUnpublished || !res.Preserved {
			t.Fatalf("expected ERR_FLOW_TRUNK_UNPUBLISHED refusal, got %+v", res)
		}
		if !strings.Contains(res.Reason, "unpublished") {
			t.Fatalf("expected unpublished reason, got %q", res.Reason)
		}
	})

	t.Run("diverged trunk refuses immediately with ERR_FLOW_TRUNK_UNPUBLISHED", func(t *testing.T) {
		fg := newFakeGit()
		fg.reply("rev-list", 0, "2\t5\n")
		fg.reply("fetch", 0, "")
		res := checkTrunkUnpublishedOrDiverged("/repo", trunkMainRef, "/wt", fg.run)
		if res == nil || res.OK || res.Code != LandResultTrunkUnpublished || !res.Preserved {
			t.Fatalf("expected ERR_FLOW_TRUNK_UNPUBLISHED refusal on diverged trunk, got %+v", res)
		}
		if !strings.Contains(res.Reason, "diverged") {
			t.Fatalf("expected diverged reason, got %q", res.Reason)
		}
	})

	t.Run("fetch clearing stale origin/main unblocks land", func(t *testing.T) {
		fg := newFakeGit()
		fg.replyOnce("rev-list", 0, "1\t2\n")
		fg.replyOnce("fetch", 0, "")
		fg.replyOnce("rev-list", 0, "0\t0\n")
		if res := checkTrunkUnpublishedOrDiverged("/repo", trunkMainRef, "/wt", fg.run); res != nil {
			t.Fatalf("expected nil after fetch cleared divergence, got %+v", res)
		}
	})
}
