package main

import (
	"archive/tar"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// ciPreflightFailure is shared by repository validation commands that retain
// their own migration lifecycle after ci-preflight moves to fak-dev.
type ciPreflightFailure struct {
	Step   string   `json:"step"`
	Detail string   `json:"detail,omitempty"`
	Files  []string `json:"files,omitempty"`
}

func gitRevParse(r, ref string) (string, error) {
	cmd := windowgate.Command("git", "-C", r, "rev-parse", ref)
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

func extractCommittedTip(r, sha string) (string, error) {
	dir, err := os.MkdirTemp("", "fak-committed-tip-*")
	if err != nil {
		return "", err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	cmd := windowgate.CommandContext(ctx, "git", "-C", r, "archive", "--format=tar", sha)
	windowgate.ConfigureBackgroundCommand(cmd)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		os.RemoveAll(dir)
		return "", err
	}
	var stderr strings.Builder
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		os.RemoveAll(dir)
		return "", fmt.Errorf("git archive start: %w", err)
	}
	untarErr := untarCommittedTip(stdout, dir)
	if untarErr != nil {
		_ = cmd.Process.Kill()
	}
	_, _ = io.Copy(io.Discard, stdout)
	waitErr := cmd.Wait()
	if ctx.Err() != nil {
		os.RemoveAll(dir)
		return "", fmt.Errorf("git archive timed out: %w", ctx.Err())
	}
	if untarErr != nil {
		os.RemoveAll(dir)
		return "", fmt.Errorf("untar archive: %w", untarErr)
	}
	if waitErr != nil {
		os.RemoveAll(dir)
		return "", fmt.Errorf("git archive: %w (%s)", waitErr, strings.TrimSpace(stderr.String()))
	}
	return dir, nil
}

func untarCommittedTip(r io.Reader, dir string) error {
	tr := tar.NewReader(r)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		rel := filepath.Clean(filepath.FromSlash(hdr.Name))
		if filepath.IsAbs(rel) || filepath.VolumeName(rel) != "" || rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
			return fmt.Errorf("archive entry %q escapes extraction root", hdr.Name)
		}
		target := filepath.Join(dir, rel)
		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			f, err := os.OpenFile(target, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, os.FileMode(hdr.Mode)&0o777|0o600)
			if err != nil {
				return err
			}
			if _, err := io.Copy(f, tr); err != nil {
				_ = f.Close()
				return err
			}
			if err := f.Close(); err != nil {
				return err
			}
		}
	}
}
