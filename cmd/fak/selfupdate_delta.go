package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/selfinstall"
)

const (
	selfUpdateDeltaFormat   = "zstd-patch"
	selfUpdateDeltaMaxRatio = 0.80
)

type selfUpdateTransferReceipt struct {
	ChosenPath     string `json:"chosen_path"`
	DeltaBytes     int64  `json:"delta_bytes"`
	FullBytes      int64  `json:"full_bytes"`
	TotalMS        int64  `json:"elapsed_total_ms"`
	Verification   string `json:"verification"`
	FallbackReason string `json:"fallback_reason,omitempty"`
	FallbackBytes  int64  `json:"fallback_bytes"`
	FallbackMS     int64  `json:"fallback_ms"`
}

var selfUpdateTransportNow = time.Now

// acquireSelfUpdateArtifact prefers only a signed, exact-source zstd patch that has a
// worthwhile transfer ratio. Every delta failure falls back to the independently signed full
// artifact; source bytes are used only as zstd's dictionary and are never activation material.
func acquireSelfUpdateArtifact(ctx context.Context, run selfinstall.Runner, installedPath string, target selfUpdateArtifactTarget, dir string) (path string, receipt selfUpdateTransferReceipt, err error) {
	started := selfUpdateTransportNow()
	defer func() { receipt.TotalMS = elapsedSelfUpdateMS(started, selfUpdateTransportNow()) }()

	delta, fallbackReason := selectSelfUpdateDelta(installedPath, target)
	if delta != nil {
		patchPath, n, downloadErr := downloadSelfUpdateBlob(ctx, delta.URL, delta.SHA256, delta.Size, dir, "fak-self-update-delta-*", false)
		receipt.DeltaBytes = n
		if downloadErr != nil {
			fallbackReason = "delta_download_verification_failed"
		} else {
			defer os.Remove(patchPath)
			if _, ok := run(ctx, "", "zstd", "--version"); !ok {
				fallbackReason = "zstd_unavailable"
			} else {
				candidate, candidateErr := reserveSelfUpdateOutput(dir)
				if candidateErr != nil {
					fallbackReason = "delta_output_prepare_failed"
				} else {
					_, ok := run(ctx, "", "zstd", "-d", "--patch-from="+filepath.Clean(installedPath), patchPath, "-o", candidate)
					if !ok {
						_ = os.Remove(candidate)
						fallbackReason = "zstd_patch_failed"
					} else if verifyErr := verifySelfUpdateFile(candidate, target.SHA256, target.Size); verifyErr != nil {
						_ = os.Remove(candidate)
						fallbackReason = "patched_target_verification_failed"
					} else if chmodErr := os.Chmod(candidate, 0o755); chmodErr != nil {
						_ = os.Remove(candidate)
						fallbackReason = "patched_target_mode_failed"
					} else {
						receipt.ChosenPath = "delta"
						receipt.Verification = "signed_target_size_sha256_verified"
						return candidate, receipt, nil
					}
				}
			}
		}
	}

	receipt.ChosenPath = "full"
	receipt.FallbackReason = fallbackReason
	fallbackStarted := selfUpdateTransportNow()
	fullPath, n, fullErr := downloadSelfUpdateBlob(ctx, target.URL, target.SHA256, target.Size, dir, "fak-self-update-artifact-*", true)
	receipt.FullBytes = n
	receipt.FallbackBytes = n
	receipt.FallbackMS = elapsedSelfUpdateMS(fallbackStarted, selfUpdateTransportNow())
	if fullErr != nil {
		receipt.Verification = "failed"
		return "", receipt, fmt.Errorf("verified full artifact fallback: %w", fullErr)
	}
	receipt.Verification = "signed_full_size_sha256_verified"
	return fullPath, receipt, nil
}

func selectSelfUpdateDelta(installedPath string, target selfUpdateArtifactTarget) (*selfUpdateArtifactDelta, string) {
	if len(target.Deltas) == 0 {
		return nil, "no_delta_available"
	}
	digest, _, err := selfUpdateFileIdentity(installedPath)
	if err != nil {
		return nil, "source_digest_unavailable"
	}
	for i := range target.Deltas {
		delta := &target.Deltas[i]
		if !strings.EqualFold(digest, delta.SourceSHA256) {
			continue
		}
		if float64(delta.Size)/float64(target.Size) >= selfUpdateDeltaMaxRatio {
			return nil, "poor_delta_ratio"
		}
		return delta, ""
	}
	return nil, "source_digest_mismatch"
}

func reserveSelfUpdateOutput(dir string) (string, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	f, err := os.CreateTemp(dir, "fak-self-update-patched-*")
	if err != nil {
		return "", err
	}
	path := f.Name()
	if err := f.Close(); err != nil {
		_ = os.Remove(path)
		return "", err
	}
	if err := os.Remove(path); err != nil {
		return "", err
	}
	return path, nil
}

func verifySelfUpdateFile(path, wantDigest string, wantSize int64) error {
	digest, size, err := selfUpdateFileIdentity(path)
	if err != nil {
		return err
	}
	if size != wantSize {
		return fmt.Errorf("size mismatch: got %d want %d", size, wantSize)
	}
	if !strings.EqualFold(digest, wantDigest) {
		return errors.New("SHA-256 mismatch")
	}
	return nil
}

func selfUpdateFileIdentity(path string) (string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return "", n, err
	}
	return hex.EncodeToString(h.Sum(nil)), n, nil
}

func elapsedSelfUpdateMS(start, end time.Time) int64 {
	if end.Before(start) {
		return 0
	}
	return end.Sub(start).Milliseconds()
}
