package rehome

import (
	"io"
	"os"
	"path/filepath"
)

// RehomeTranscript copies a session's transcript (and its sidecar <sid>/ dir) from the
// throttled owner's config dir into the healthy target account's config dir. It is the
// Go port of fleet_resume_watchdog.rehome_transcript.
//
// `claude --resume <sid>` is CLAUDE_CONFIG_DIR + cwd scoped: it only finds the
// conversation under <config>/projects/<sanitized-cwd>/<sid>.jsonl, so to resume on a
// different account the transcript must physically live there first. Returns false
// (the caller skips the resume and falls back to a plain pin) when the source
// transcript is missing.
//
// destProjects lands the copy under ADDITIONAL project slugs beyond the owner's
// original project — the cross-directory resume fix, so a resume works whether the
// operator launches from the session's birth directory or a different one. The
// owner's own slug is always included and slugs are de-duped.
//
// The self-copy (mirroring within the owner account, where a slug resolves dst == src)
// is skipped and counts as success. A per-slug copy error is swallowed (a live process
// may hold the transcript under a Windows mandatory lock) so one failed slug never
// aborts the others or crashes the resolver — the launcher's fail-open contract.
//
// The optional agent-memory co-travel (memory_cotravel, default FAK_MEMORY_COTRAVEL=
// shadow → copy nothing) is intentionally not ported here: its default path is a no-op,
// so omitting it preserves the observable copy behavior.
func RehomeTranscript(srcCfg, dstCfg, project, sid string, destProjects []string) bool {
	src := filepath.Join(srcCfg, "projects", project, sid+".jsonl")
	if fi, err := os.Stat(src); err != nil || fi.IsDir() {
		return false
	}
	side := filepath.Join(srcCfg, "projects", project, sid)

	slugs := []string{project}
	seen := map[string]bool{project: true}
	for _, p := range destProjects {
		if p != "" && !seen[p] {
			seen[p] = true
			slugs = append(slugs, p)
		}
	}

	copiedAny := false
	for _, slug := range slugs {
		dstDir := filepath.Join(dstCfg, "projects", slug)
		dst := filepath.Join(dstDir, sid+".jsonl")
		if sameFile(dst, src) {
			// Mirroring within the owner account: the owner's own slug resolves
			// dst == src; copying a file onto itself is a no-op (and errors on
			// Windows), so treat it as already-present.
			copiedAny = true
			continue
		}
		if err := os.MkdirAll(dstDir, 0o755); err != nil {
			continue
		}
		if err := copyFile(src, dst); err != nil {
			continue
		}
		copiedAny = true
		if fi, err := os.Stat(side); err == nil && fi.IsDir() {
			// Best-effort sidecar copy (subagents/workflows dir); never fatal.
			_ = copyTree(side, filepath.Join(dstDir, sid))
		}
	}
	return copiedAny
}

// sameFile reports whether a and b resolve to the same absolute path, mirroring the
// Python os.path.abspath(dst) == os.path.abspath(src) self-copy guard.
func sameFile(a, b string) bool {
	aa, err1 := filepath.Abs(a)
	bb, err2 := filepath.Abs(b)
	if err1 != nil || err2 != nil {
		return false
	}
	return aa == bb
}

// copyFile copies src to dst, preserving the source mode and mtime (the shutil.copy2
// contract). The resolver separately stamps the re-homed copy as newest afterward.
func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	fi, err := in.Stat()
	if err != nil {
		return err
	}
	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, fi.Mode().Perm())
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	if err := out.Close(); err != nil {
		return err
	}
	_ = os.Chmod(dst, fi.Mode().Perm())
	_ = os.Chtimes(dst, fi.ModTime(), fi.ModTime())
	return nil
}

// copyTree recursively copies srcDir into dstDir, creating dstDir and merging into an
// existing one (the shutil.copytree(dirs_exist_ok=True) contract).
func copyTree(srcDir, dstDir string) error {
	return filepath.Walk(srcDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(srcDir, path)
		if err != nil {
			return err
		}
		target := filepath.Join(dstDir, rel)
		if info.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		return copyFile(path, target)
	})
}
