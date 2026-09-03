package tb4bench

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// WorkspaceManager manages the preparation, execution, and state snapshotting of a task workspace.
type WorkspaceManager struct {
	engine      ContainerEngine
	containerID string
	taskID      string
	localDir    string // optional local workspace path when running in mock/local mode
}

// NewWorkspaceManager creates a workspace manager for a container instance.
func NewWorkspaceManager(engine ContainerEngine, containerID, taskID, localDir string) *WorkspaceManager {
	return &WorkspaceManager{
		engine:      engine,
		containerID: containerID,
		taskID:      taskID,
		localDir:    localDir,
	}
}

// WorkspaceSnapshot holds cryptographic hashes and diff artifacts for a task execution.
type WorkspaceSnapshot struct {
	TaskID        string    `json:"task_id"`
	InitialDigest string    `json:"initial_digest"`
	FinalDigest   string    `json:"final_digest"`
	UnifiedDiff   string    `json:"unified_diff"`
	ModifiedFiles []string  `json:"modified_files"`
	CapturedAt    time.Time `json:"captured_at"`
}

// SeedWorkspace initializes the workspace directory, writes fixture files, instructions,
// and establishes a baseline git commit.
func (w *WorkspaceManager) SeedWorkspace(ctx context.Context, prompt string, files map[string][]byte) (string, error) {
	// 1. Ensure directory and git baseline
	_, err := w.engine.ExecCommand(ctx, w.containerID, ExecConfig{
		Cmd:        []string{"sh", "-c", "git init -q && git config user.name 'tb4' && git config user.email 'tb4@fak.internal'"},
		WorkingDir: "/workspace",
	})
	if err != nil {
		return "", fmt.Errorf("failed to initialize git in workspace: %w", err)
	}

	// 2. Write INSTRUCTION.md
	if prompt != "" {
		if files == nil {
			files = make(map[string][]byte)
		}
		files["INSTRUCTION.md"] = []byte(prompt)
	}

	// 3. Write files into workspace
	for relPath, content := range files {
		// Escape or write via sh/echo or direct write if local
		if w.localDir != "" {
			fullPath := filepath.Join(w.localDir, relPath)
			_ = os.MkdirAll(filepath.Dir(fullPath), 0755)
			if err := os.WriteFile(fullPath, content, 0644); err != nil {
				return "", err
			}
		} else {
			// Write file via container exec
			dir := filepath.Dir(relPath)
			cmd := fmt.Sprintf("mkdir -p %s && cat << 'EOF' > %s\n%s\nEOF", dir, relPath, string(content))
			_, err := w.engine.ExecCommand(ctx, w.containerID, ExecConfig{
				Cmd:        []string{"sh", "-c", cmd},
				WorkingDir: "/workspace",
			})
			if err != nil {
				return "", fmt.Errorf("failed to seed file %s: %w", relPath, err)
			}
		}
	}

	// 4. Commit baseline in git
	_, err = w.engine.ExecCommand(ctx, w.containerID, ExecConfig{
		Cmd:        []string{"sh", "-c", "git add . && git commit -m 'baseline' -q || true"},
		WorkingDir: "/workspace",
	})
	if err != nil {
		return "", fmt.Errorf("failed to commit baseline: %w", err)
	}

	// 5. Compute initial digest
	initialDigest, err := w.ComputeWorkspaceDigest(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to compute initial digest: %w", err)
	}

	return initialDigest, nil
}

// Exec runs a command inside the container workspace with timeout and stdout/stderr capture.
func (w *WorkspaceManager) Exec(ctx context.Context, cmd []string, timeout time.Duration) (*ExecResult, error) {
	return w.engine.ExecCommand(ctx, w.containerID, ExecConfig{
		Cmd:        cmd,
		WorkingDir: "/workspace",
		Timeout:    timeout,
	})
}

// SnapshotDiff captures git diff HEAD, status porcelain, and post-run workspace tree digest.
func (w *WorkspaceManager) SnapshotDiff(ctx context.Context, initialDigest string) (*WorkspaceSnapshot, error) {
	// Run git diff HEAD
	diffRes, err := w.engine.ExecCommand(ctx, w.containerID, ExecConfig{
		Cmd:        []string{"sh", "-c", "git diff HEAD"},
		WorkingDir: "/workspace",
	})
	if err != nil {
		return nil, fmt.Errorf("failed to capture git diff: %w", err)
	}

	// Run git status --porcelain to list modified/untracked files
	statusRes, err := w.engine.ExecCommand(ctx, w.containerID, ExecConfig{
		Cmd:        []string{"sh", "-c", "git status --porcelain"},
		WorkingDir: "/workspace",
	})
	if err != nil {
		return nil, fmt.Errorf("failed to capture git status: %w", err)
	}

	var modifiedFiles []string
	lines := strings.Split(string(statusRes.Stdout), "\n")
	for _, l := range lines {
		trimmed := strings.TrimSpace(l)
		if len(trimmed) > 3 {
			filePath := strings.TrimSpace(trimmed[2:])
			modifiedFiles = append(modifiedFiles, filePath)
		}
	}
	sort.Strings(modifiedFiles)

	finalDigest, err := w.ComputeWorkspaceDigest(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to compute final workspace digest: %w", err)
	}

	return &WorkspaceSnapshot{
		TaskID:        w.taskID,
		InitialDigest: initialDigest,
		FinalDigest:   finalDigest,
		UnifiedDiff:   string(diffRes.Stdout),
		ModifiedFiles: modifiedFiles,
		CapturedAt:    time.Now().UTC(),
	}, nil
}

// ComputeWorkspaceDigest computes a deterministic SHA-256 tree hash over /workspace.
func (w *WorkspaceManager) ComputeWorkspaceDigest(ctx context.Context) (string, error) {
	if w.localDir != "" {
		return HashDirectoryTree(w.localDir)
	}

	// Run deterministic sha256sum pipeline inside container (ignoring .git directory)
	res, err := w.engine.ExecCommand(ctx, w.containerID, ExecConfig{
		Cmd:        []string{"sh", "-c", `cd /workspace && find . -path './.git' -prune -o -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}'`},
		WorkingDir: "/workspace",
	})
	if err != nil {
		return "", err
	}
	digest := strings.TrimSpace(string(res.Stdout))
	if len(digest) != 64 {
		// Fallback: if find/sha256sum formatting differs
		parts := strings.Fields(digest)
		if len(parts) > 0 && len(parts[0]) == 64 {
			return parts[0], nil
		}
		return "", fmt.Errorf("invalid digest output: %q", digest)
	}
	return digest, nil
}

// HashDirectoryTree computes a deterministic SHA-256 digest of a directory tree on disk,
// ignoring `.git` directories.
func HashDirectoryTree(root string) (string, error) {
	type fileEntry struct {
		relPath string
		hash    [32]byte
	}
	var entries []fileEntry

	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			if info.Name() == ".git" {
				return filepath.SkipDir
			}
			return nil
		}

		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)

		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		h := sha256.Sum256(data)
		entries = append(entries, fileEntry{relPath: rel, hash: h})
		return nil
	})
	if err != nil {
		return "", err
	}

	sort.Slice(entries, func(i, j int) bool {
		return entries[i].relPath < entries[j].relPath
	})

	hasher := sha256.New()
	for _, e := range entries {
		hasher.Write([]byte(e.relPath))
		hasher.Write(e.hash[:])
	}

	return hex.EncodeToString(hasher.Sum(nil)), nil
}

// ExportArchive creates a tar.gz archive of the workspace directory.
func ExportArchive(sourceDir string, outPath string) error {
	outFile, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer outFile.Close()

	gw := gzip.NewWriter(outFile)
	defer gw.Close()
	tw := tar.NewWriter(gw)
	defer tw.Close()

	return filepath.Walk(sourceDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(sourceDir, path)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		if rel == "." {
			return nil
		}

		header, err := tar.FileInfoHeader(info, info.Name())
		if err != nil {
			return err
		}
		header.Name = rel

		if err := tw.WriteHeader(header); err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return nil
		}

		f, err := os.Open(path)
		if err != nil {
			return err
		}
		defer f.Close()
		_, err = io.Copy(tw, f)
		return err
	})
}
