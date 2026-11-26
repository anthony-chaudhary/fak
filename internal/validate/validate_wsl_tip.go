package validate

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/committedtree"
	"github.com/anthony-chaudhary/fak/internal/processalive"
)

// wslTipArchiveDirPrefix names the owned temp directory holding a staged tip
// archive; the prefix is what lets the next run reap an abandoned one.
const wslTipArchiveDirPrefix = "fak-validate-tip-"

// validateWSLTipArchive stages the committed tip as a tar that WSL can read. It
// returns the WSL path of the archive and a func that removes the staged file.
// It is a package seam so tests can stand in for a real repository.
var validateWSLTipArchive = stageWSLTipArchive

// stageWSLTipArchive snapshots the full committed tree. WSL runs the tests from
// this snapshot, so export-ignored paths must be present, which git archive
// would drop.
func stageWSLTipArchive(ctx context.Context, repo, tip string) (string, func(), error) {
	local, err := writeWSLTipArchive(ctx, repo, tip)
	if err != nil {
		return "", nil, err
	}
	// The archive lives in its own owned directory, so removing the directory
	// also removes anything else staged there.
	dir := filepath.Dir(local)
	remove := func() { _ = os.RemoveAll(dir) }
	wslPath, ok := wslMountPath(local)
	if !ok {
		remove()
		return "", nil, fmt.Errorf("staged archive %s is not on a Windows drive", local)
	}
	return wslPath, remove, nil
}

// writeWSLTipArchive streams the committed tip into a tar under an owned temp
// directory. A process killed mid-run leaves that directory behind, and the
// next OwnedTempDir with the same prefix reaps it once the owner is dead.
func writeWSLTipArchive(ctx context.Context, repo, tip string) (string, error) {
	dir, err := processalive.OwnedTempDir(wslTipArchiveDirPrefix)
	if err != nil {
		return "", err
	}
	name := filepath.Join(dir, "tip.tar")
	file, err := os.Create(name)
	if err != nil {
		_ = os.RemoveAll(dir)
		return "", err
	}
	out := bufio.NewWriterSize(file, 1<<20)
	err = committedtree.WriteTar(ctx, repo, tip, out)
	if err == nil {
		err = out.Flush()
	}
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		_ = os.RemoveAll(dir)
		return "", err
	}
	return name, nil
}

// extractCommittedTipWSLWithin returns the WSL workspace and the boot id of the
// VM it was built in ("" when the script did not report one).
func extractCommittedTipWSLWithin(ctx context.Context, repo, tip string) (string, string, error) {
	wslArchive, removeArchive, err := validateWSLTipArchive(ctx, repo, tip)
	if err != nil {
		if ctx.Err() != nil {
			return "", "", ctx.Err()
		}
		return "", "", fmt.Errorf("materialize committed tip in WSL: stage archive: %w", err)
	}
	defer removeArchive()
	wslDir := fmt.Sprintf("/tmp/fak-validate-%d-%d", os.Getpid(), validateNow().UnixNano())
	quotedDir := posixQuote(wslDir)
	gitFlags := ""
	if gitDir, ok := readWSLWorktreeGitDir(repo); ok {
		gitFlags = "--git-dir=" + posixQuote(gitDir) + " "
	}
	script := "set -euo pipefail; " +
		"rm -rf -- " + quotedDir + "; mkdir -p -- " + quotedDir + "; " +
		"trap 'rm -rf -- " + quotedDir + "' ERR INT TERM; " +
		"tar -xf " + posixQuote(wslArchive) + " -C " + quotedDir + "; " +
		"git -C " + quotedDir + " init --quiet; " +
		"mkdir -p -- " + quotedDir + "/.git/objects/info; " +
		"git " + gitFlags + "rev-parse --path-format=absolute --git-path objects > " + quotedDir + "/.git/objects/info/alternates; " +
		"git -C " + quotedDir + " read-tree " + posixQuote(tip) + "; " +
		"git -C " + quotedDir + " update-ref --no-deref HEAD " + posixQuote(tip) + "; " +
		"trap - ERR INT TERM; " +
		"printf '" + validateWSLBootIDPrefix + "%s\\n' \"$(cat " + validateWSLBootIDPath + " 2>/dev/null || true)\""
	// WSL git reads .git/packed-refs through drvfs non-atomically while Windows-side
	// landers rewrite it, so a torn read is transient: rerun the idempotent script (it
	// recreates wslDir) with a bounded backoff for that signature only; every other
	// failure stays fail-closed.
	const attempts = 4
	wait := 250 * time.Millisecond
	for attempt := 1; ; attempt++ {
		out, err := validateWSLCommand(ctx, "--cd", repo, "-e", "bash", "-lc", script)
		if err == nil {
			return wslDir, parseValidateWSLBootID(out), nil
		}
		if ctx.Err() != nil {
			cleanupValidateWSLDir(wslDir)
			return "", "", ctx.Err()
		}
		detail := strings.TrimSpace(string(out))
		if detail == "" {
			detail = err.Error()
		}
		if !isTransientPackedRefsRead(detail) {
			cleanupValidateWSLDir(wslDir)
			return "", "", fmt.Errorf("materialize committed tip in WSL: %s", detail)
		}
		if attempt == attempts {
			cleanupValidateWSLDir(wslDir)
			return "", "", fmt.Errorf("materialize committed tip in WSL: %s (transient packed-refs read persisted across %d attempts)", detail, attempts)
		}
		timer := time.NewTimer(wait)
		select {
		case <-ctx.Done():
			timer.Stop()
			cleanupValidateWSLDir(wslDir)
			return "", "", ctx.Err()
		case <-timer.C:
		}
		wait *= 2
	}
}
