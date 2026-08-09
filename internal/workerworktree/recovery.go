package workerworktree

import (
	"fmt"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	// RecoveryRefPrefix roots local candidate commits before trunk CAS.
	RecoveryRefPrefix = "refs/fak/worker-land/"
	// RecoveryMirrorPrefix is read-only local state fetched/read back from remotes.
	RecoveryMirrorPrefix = "refs/fak/remoteworkerland/"
	// RecoveryMirrorStampPrefix is a reflogged receipt updated only after a
	// successful remote fetch/read-back.
	RecoveryMirrorStampPrefix = "refs/fak/remoteworkerland-stamp/"
	recoveryReflogMessage     = "fak worker land candidate"
	mirrorReflogMessage       = "fak worker land remote mirror witnessed"
)

const (
	DurabilityLocalOnly  = "LOCAL_ONLY"
	DurabilityReplicated = "REPLICATED"
	DurabilityRemoteOnly = "REMOTE_ONLY"
	DurabilityLanded     = "LANDED"
)

// RecoveryEntry is one named off-branch land candidate and its current
// reachability/durability classification.
type RecoveryEntry struct {
	Ref          string `json:"ref"`
	SHA          string `json:"sha"`
	Worktree     string `json:"worktree"`
	State        string `json:"state"`
	Durability   string `json:"durability"`
	Remote       string `json:"remote,omitempty"`
	MirrorRef    string `json:"mirror_ref,omitempty"`
	MirrorAgeSec int64  `json:"mirror_age_sec,omitempty"`
	Action       string `json:"action"`
}

// RemoteReadback is the independently read-back result of publishing/fetching a
// candidate. Witnessed is true only when the remote reported exactly SHA.
type RemoteReadback struct {
	Remote    string `json:"remote"`
	Ref       string `json:"ref"`
	SHA       string `json:"sha"`
	MirrorRef string `json:"mirror_ref,omitempty"`
	Witnessed bool   `json:"witnessed"`
	Reason    string `json:"reason,omitempty"`
}

func RecoveryRef(wtPath, candidateSHA string) (string, error) {
	sha := strings.TrimSpace(candidateSHA)
	if len(sha) < 7 || !isRefObjectToken(sha) {
		return "", fmt.Errorf("invalid candidate object id %q", sha)
	}
	name := recoveryName(filepath.Base(filepath.Clean(wtPath)))
	if name == "" || name == "." {
		return "", fmt.Errorf("invalid worktree identity %q", wtPath)
	}
	return RecoveryRefPrefix + name + "/" + sha, nil
}

func recoveryName(s string) string {
	var b strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' || r == '.' {
			b.WriteRune(r)
		} else {
			b.WriteByte('-')
		}
	}
	return strings.Trim(b.String(), ".-")
}

func isRefObjectToken(s string) bool {
	for _, r := range s {
		if !((r >= '0' && r <= '9') || (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z')) {
			return false
		}
	}
	return true
}

func AnchorRecoveryEntry(root, wtPath, candidateSHA string, git GitRunner) (string, error) {
	ref, err := RecoveryRef(wtPath, candidateSHA)
	if err != nil {
		return "", err
	}
	rc, out := run(git, root, []string{"update-ref", "--create-reflog", "-m", recoveryReflogMessage, ref, strings.TrimSpace(candidateSHA)})
	if rc != 0 {
		return ref, fmt.Errorf("could not anchor recovery ref %s (git error): %s", ref, strings.TrimSpace(out))
	}
	return ref, nil
}

func mirrorRef(remote, ref string) (string, error) {
	if !strings.HasPrefix(ref, RecoveryRefPrefix) {
		return "", fmt.Errorf("ref is outside %s", RecoveryRefPrefix)
	}
	r := recoveryName(remote)
	if r == "" {
		return "", fmt.Errorf("remote is empty")
	}
	return RecoveryMirrorPrefix + r + "/" + strings.TrimPrefix(ref, RecoveryRefPrefix), nil
}

func mirrorStampRef(remote string) (string, error) {
	r := recoveryName(remote)
	if r == "" {
		return "", fmt.Errorf("remote is empty")
	}
	return RecoveryMirrorStampPrefix + r, nil
}

// PublishRecoveryRef pushes one candidate and independently reads the remote ref
// back before claiming host-loss durability. Local recovery remains intact on any
// failure. The mirror and stamp update only after exact SHA read-back.
func PublishRecoveryRef(root, remote, ref, sha string, git GitRunner) RemoteReadback {
	receipt := RemoteReadback{Remote: remote, Ref: ref, SHA: strings.TrimSpace(sha)}
	mref, err := mirrorRef(remote, ref)
	if err != nil {
		receipt.Reason = err.Error()
		return receipt
	}
	receipt.MirrorRef = mref
	if rc, out := run(git, root, []string{"push", remote, ref + ":" + ref}); rc != 0 {
		receipt.Reason = "remote push failed; local recovery ref preserved: " + strings.TrimSpace(out)
		return receipt
	}
	rc, out := run(git, root, []string{"ls-remote", "--refs", remote, ref})
	if rc != 0 {
		receipt.Reason = "remote read-back failed; local recovery ref preserved: " + strings.TrimSpace(out)
		return receipt
	}
	fields := strings.Fields(out)
	if len(fields) < 2 || fields[0] != receipt.SHA || fields[1] != ref {
		receipt.Reason = "remote read-back did not match candidate; local recovery ref preserved"
		return receipt
	}
	if rc, out := run(git, root, []string{"update-ref", "--create-reflog", "-m", mirrorReflogMessage, mref, receipt.SHA}); rc != 0 {
		receipt.Reason = "remote matched but local mirror receipt failed: " + strings.TrimSpace(out)
		return receipt
	}
	stamp, _ := mirrorStampRef(remote)
	if rc, out := run(git, root, []string{"update-ref", "--create-reflog", "-m", mirrorReflogMessage, stamp, receipt.SHA}); rc != 0 {
		receipt.Reason = "remote matched but freshness stamp failed: " + strings.TrimSpace(out)
		return receipt
	}
	receipt.Witnessed = true
	return receipt
}

// FetchRecoveryMirror refreshes the disjoint read-only mirror namespace. --prune
// prevents a deleted remote ref from continuing to appear as fresh local evidence.
func FetchRecoveryMirror(root, remote string, git GitRunner) error {
	r := recoveryName(remote)
	if r == "" {
		return fmt.Errorf("remote is empty")
	}
	refspec := "+" + RecoveryRefPrefix + "*:" + RecoveryMirrorPrefix + r + "/*"
	if rc, out := run(git, root, []string{"fetch", "--prune", remote, refspec}); rc != 0 {
		return fmt.Errorf("worker-land mirror fetch failed: %s", strings.TrimSpace(out))
	}
	// A successful empty fetch still needs a freshness receipt. Point the stamp at
	// HEAD: the reflog time is the evidence; it does not claim a candidate exists.
	rc, head := run(git, root, []string{"rev-parse", "HEAD"})
	if rc != 0 || strings.TrimSpace(head) == "" {
		return fmt.Errorf("mirror fetched but HEAD could not seed freshness stamp")
	}
	stamp, _ := mirrorStampRef(remote)
	if rc, out := run(git, root, []string{"update-ref", "--create-reflog", "-m", mirrorReflogMessage, stamp, strings.TrimSpace(head)}); rc != 0 {
		return fmt.Errorf("mirror fetched but freshness stamp failed: %s", strings.TrimSpace(out))
	}
	return nil
}

func RecoveryEntries(root string, git GitRunner) ([]RecoveryEntry, error) {
	rows, err := listRecoveryRows(root, RecoveryRefPrefix, git)
	if err != nil {
		return nil, err
	}
	mirrors, err := listRecoveryRows(root, RecoveryMirrorPrefix, git)
	if err != nil {
		return nil, err
	}
	bySHA := map[string][]rawRecoveryRow{}
	for _, m := range mirrors {
		bySHA[m.SHA] = append(bySHA[m.SHA], m)
	}
	items := make([]RecoveryEntry, 0, len(rows)+len(mirrors))
	localSHA := map[string]bool{}
	for _, row := range rows {
		localSHA[row.SHA] = true
		item := recoveryEntry(root, row.Ref, row.SHA, git)
		item.Durability = DurabilityLocalOnly
		if ms := bySHA[row.SHA]; len(ms) > 0 {
			item.Durability = DurabilityReplicated
			item.MirrorRef = ms[0].Ref
			item.Remote = mirrorRemote(ms[0].Ref)
			item.MirrorAgeSec = mirrorAge(root, item.Remote, git)
		}
		if item.State == "LANDED" {
			item.Durability = DurabilityLanded
		}
		items = append(items, item)
	}
	for _, row := range mirrors {
		if localSHA[row.SHA] {
			continue
		}
		ref := RecoveryRefPrefix + mirrorTail(row.Ref)
		item := recoveryEntry(root, ref, row.SHA, git)
		item.Durability = DurabilityRemoteOnly
		item.MirrorRef = row.Ref
		item.Remote = mirrorRemote(row.Ref)
		item.MirrorAgeSec = mirrorAge(root, item.Remote, git)
		item.Action = "restore local ref with git update-ref " + ref + " " + row.SHA + "; then inspect/land"
		if item.State == "LANDED" {
			item.Durability = DurabilityLanded
		}
		items = append(items, item)
	}
	sort.Slice(items, func(i, j int) bool { return items[i].Ref < items[j].Ref })
	return items, nil
}

type rawRecoveryRow struct{ Ref, SHA string }

func listRecoveryRows(root, prefix string, git GitRunner) ([]rawRecoveryRow, error) {
	rc, out := run(git, root, []string{"for-each-ref", "--format=%(refname)%00%(objectname)", prefix})
	if rc != 0 {
		return nil, fmt.Errorf("could not list recovery refs under %s: %s", prefix, strings.TrimSpace(out))
	}
	var rows []rawRecoveryRow
	for _, line := range strings.Split(strings.TrimRight(out, "\r\n"), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		i := strings.IndexByte(line, 0)
		if i < 0 {
			i = strings.Index(line, "\\x00")
		}
		parts := []string{}
		if i >= 0 {
			sep := 1
			if strings.HasPrefix(line[i:], "\\x00") {
				sep = 4
			}
			parts = []string{line[:i], line[i+sep:]}
		}
		if len(parts) != 2 || !strings.HasPrefix(parts[0], prefix) {
			return nil, fmt.Errorf("malformed recovery ref row %q", line)
		}
		rows = append(rows, rawRecoveryRow{parts[0], strings.TrimSpace(parts[1])})
	}
	return rows, nil
}

func recoveryEntry(root, ref, sha string, git GitRunner) RecoveryEntry {
	rest := strings.TrimPrefix(ref, RecoveryRefPrefix)
	wt := strings.SplitN(rest, "/", 2)[0]
	item := RecoveryEntry{Ref: ref, SHA: sha, Worktree: wt, State: "RECOVERABLE", Action: "inspect with git show " + ref + "; re-run worktree worker land or cherry-pick " + ref}
	if rc, _ := run(git, root, []string{"merge-base", "--is-ancestor", sha, "HEAD"}); rc == 0 {
		item.State = "LANDED"
		item.Action = "safe to clean with fak worktree worker recover --cleanup " + ref
	}
	return item
}

func mirrorRemote(ref string) string {
	rest := strings.TrimPrefix(ref, RecoveryMirrorPrefix)
	return strings.SplitN(rest, "/", 2)[0]
}
func mirrorTail(ref string) string {
	rest := strings.TrimPrefix(ref, RecoveryMirrorPrefix)
	parts := strings.SplitN(rest, "/", 2)
	if len(parts) == 2 {
		return parts[1]
	}
	return ""
}
func mirrorAge(root, remote string, git GitRunner) int64 {
	stamp, err := mirrorStampRef(remote)
	if err != nil {
		return 0
	}
	rc, out := run(git, root, []string{"reflog", "show", "--format=%ct", "-1", stamp})
	if rc != 0 {
		return 0
	}
	sec, err := strconv.ParseInt(strings.TrimSpace(out), 10, 64)
	if err != nil {
		return 0
	}
	age := time.Now().Unix() - sec
	if age < 0 {
		return 0
	}
	return age
}

// RemoteCleanupReport is report-first: Apply=false performs no mutation. A remote
// candidate is eligible only when the remote default branch proves it reachable.
type RemoteCleanupReport struct {
	Remote   string `json:"remote"`
	Ref      string `json:"ref"`
	SHA      string `json:"sha,omitempty"`
	Eligible bool   `json:"eligible"`
	Applied  bool   `json:"applied"`
	Reason   string `json:"reason"`
}

// CleanupRemoteRecoveryRef deletes one remote worker-land ref only after fetching
// the remote default branch and proving candidate ancestry there. Ref ownership
// is encoded by worktree name; peer deletion requires allowPeer.
func CleanupRemoteRecoveryRef(root, remote, ref, localWorktree string, allowPeer, apply bool, git GitRunner) RemoteCleanupReport {
	p := RemoteCleanupReport{Remote: remote, Ref: ref}
	if !strings.HasPrefix(ref, RecoveryRefPrefix) || strings.Contains(ref, "..") {
		p.Reason = "ref outside worker-land recovery namespace"
		return p
	}
	owner := strings.SplitN(strings.TrimPrefix(ref, RecoveryRefPrefix), "/", 2)[0]
	if !allowPeer && localWorktree != "" && owner != recoveryName(localWorktree) {
		p.Reason = "peer candidate deletion requires --allow-peer"
		return p
	}
	rc, out := run(git, root, []string{"ls-remote", "--refs", remote, ref})
	fields := strings.Fields(out)
	if rc != 0 || len(fields) < 2 || fields[1] != ref {
		p.Reason = "remote candidate ref not witnessed"
		return p
	}
	p.SHA = fields[0]
	// FETCH_HEAD is updated only after a successful fetch of the remote's HEAD.
	if rc, out = run(git, root, []string{"fetch", "--no-tags", remote, "HEAD"}); rc != 0 {
		p.Reason = "remote default branch could not be fetched: " + strings.TrimSpace(out)
		return p
	}
	if rc, _ = run(git, root, []string{"merge-base", "--is-ancestor", p.SHA, "FETCH_HEAD"}); rc != 0 {
		p.Reason = "candidate not proven reachable from remote default branch"
		return p
	}
	p.Eligible = true
	p.Reason = "candidate reachable from remote default branch"
	if !apply {
		return p
	}
	if rc, out = run(git, root, []string{"push", remote, ":" + ref}); rc != 0 {
		p.Eligible = false
		p.Reason = "remote deletion failed: " + strings.TrimSpace(out)
		return p
	}
	p.Applied = true
	p.Reason = "remote candidate deleted after ancestry witness"
	return p
}

func DeleteRecoveryRef(root, ref string, force bool, git GitRunner) error {
	if !strings.HasPrefix(ref, RecoveryRefPrefix) || strings.Contains(ref, "..") {
		return fmt.Errorf("ref is outside %s", RecoveryRefPrefix)
	}
	rc, sha := run(git, root, []string{"rev-parse", "--verify", ref})
	if rc != 0 || strings.TrimSpace(sha) == "" {
		return fmt.Errorf("recovery ref not found: %s", ref)
	}
	sha = strings.TrimSpace(sha)
	if !force {
		if ancRC, _ := run(git, root, []string{"merge-base", "--is-ancestor", sha, "HEAD"}); ancRC != 0 {
			return fmt.Errorf("ref %s is not landed in HEAD; refusing cleanup without --force", ref)
		}
	}
	if delRC, out := run(git, root, []string{"update-ref", "-d", ref, sha}); delRC != 0 {
		return fmt.Errorf("could not delete recovery ref %s: %s", ref, strings.TrimSpace(out))
	}
	return nil
}
