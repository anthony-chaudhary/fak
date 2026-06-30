package marketing

import (
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// highwater.go — the marketing high-water mark: a single git ref, refs/fak/marketing/last,
// pointing at the last commit a marketing tick already announced. Every trigger (the serve
// bgloop, the post-commit hook, cron) reads it to scope "what is genuinely new" and advances
// it on a successful post, so multiple triggers over the same commit do not double-announce.
//
// HONEST SCOPE (load-bearing, mirroring internal/leaseref's own boundary note): a git ref is
// a SAME-HOST compare-and-swap. `git update-ref ref <new> <old>` takes git's ref lock and
// fails if the current value isn't <old>, so two triggers on the same machine racing the same
// commit cleanly resolve to one winner. It is NOT cross-machine arbitration — git's fetch/push
// converges the SET of refs, it does not pick a winner across clones. For marketing that is
// fine: the worst cross-machine outcome is the same digest posted from two hosts, which the
// DedupeKey + a dedupe-aware post then collapses. The ref is the primary guard; the dedupe key
// is the backstop.
//
// SIDE-REF ONLY (the safety contract, same as leaseref): every write targets ONLY
// refs/fak/marketing/last. It NEVER mutates main / HEAD / refs/heads, never force-pushes,
// never touches a commit object — just an ordinary side ref via `git update-ref`.

// highWaterRef is the dedicated ref the marketing mark lives under.
const highWaterRef = "refs/fak/marketing/last"

// ReadHighWater returns the commit the marketing mark currently points at, or "" if the ref
// does not exist yet (the first-run case — the caller bootstraps a starting range). A git
// failure other than "ref absent" also returns "" (best-effort: a missing mark must not wedge
// the loop; the worst case is re-scanning a window the DedupeKey then collapses).
func ReadHighWater(root string) string {
	cmd := exec.Command("git", "rev-parse", "--verify", "--quiet", highWaterRef)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return "" // ref absent (rev-parse --quiet exits 1) or git error — treat as unset
	}
	return strings.TrimSpace(string(out))
}

// AdvanceHighWater moves the mark to newSHA, requiring the current value to be expectedOld
// (the compare-and-swap). expectedOld "" creates the ref only if it does not already exist
// (the first-advance case). It returns true on success; false means the CAS lost — another
// trigger advanced the mark first, so this caller must NOT post (the winner already did).
//
// A git-missing error also returns false (the caller treats a failed advance as "someone else
// owns this window"): never post without having claimed the window.
func AdvanceHighWater(root, newSHA, expectedOld string) bool {
	args := []string{"update-ref", highWaterRef, newSHA}
	if expectedOld == "" {
		// Create-only: the zero oldvalue tells git "fail if the ref already exists",
		// so a racing first-advance has exactly one winner.
		args = append(args, "0000000000000000000000000000000000000000")
	} else {
		args = append(args, expectedOld)
	}
	cmd := exec.Command("git", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = root
	return cmd.Run() == nil
}

// headSHA returns the full HEAD sha at root, or "" on error. The mark stores full shas (git
// update-ref wants the full object name); CollectShips short-forms them only for display.
func headSHA(root string) string {
	cmd := exec.Command("git", "rev-parse", "HEAD")
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}
