package harnesscontrolpacket

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const Schema = "fak-harness-control-packet/1"

type File struct {
	Path   string `json:"path"`
	Digest string `json:"digest"`
}

type Manifest struct {
	Schema        string `json:"schema"`
	StudyID       string `json:"study_id"`
	Arm           string `json:"arm"`
	SourceCommit  string `json:"source_commit"`
	BinaryVersion string `json:"binary_version"`
	Files         []File `json:"files"`
}

type CreateOptions struct {
	Arm           string
	MaterialsDir  string
	BinaryPath    string
	ReceiptPath   string
	OutputDir     string
	SourceCommit  string
	BinaryVersion string
}

func Create(opts CreateOptions) (Manifest, error) {
	if opts.Arm != "default-control" && opts.Arm != "scratch" {
		return Manifest{}, fmt.Errorf("arm must be default-control or scratch")
	}
	for name, value := range map[string]string{"materials directory": opts.MaterialsDir, "binary": opts.BinaryPath, "receipt": opts.ReceiptPath, "output directory": opts.OutputDir, "source commit": opts.SourceCommit, "binary version": opts.BinaryVersion} {
		if strings.TrimSpace(value) == "" {
			return Manifest{}, fmt.Errorf("%s is required", name)
		}
	}
	if err := os.MkdirAll(opts.OutputDir, 0o755); err != nil {
		return Manifest{}, err
	}
	entries, err := os.ReadDir(opts.MaterialsDir)
	if err != nil {
		return Manifest{}, fmt.Errorf("materials: %w", err)
	}
	allowed := map[string]bool{"arm-card.md": true, "task-card.md": true}
	if opts.Arm == "default-control" {
		for _, name := range []string{"kernel-component.txt", "product.json", "product.lock.json", "selection.json"} {
			allowed[name] = true
		}
	}
	for _, entry := range entries {
		if entry.IsDir() || !allowed[entry.Name()] {
			return Manifest{}, fmt.Errorf("%s arm contains unexpected material %q", opts.Arm, entry.Name())
		}
	}
	for name := range allowed {
		if err := copyFile(filepath.Join(opts.MaterialsDir, name), filepath.Join(opts.OutputDir, name), 0o644); err != nil {
			return Manifest{}, fmt.Errorf("material %s: %w", name, err)
		}
	}
	if err := copyFile(opts.BinaryPath, filepath.Join(opts.OutputDir, "fak"), 0o755); err != nil {
		return Manifest{}, fmt.Errorf("binary: %w", err)
	}
	if err := copyFile(opts.ReceiptPath, filepath.Join(opts.OutputDir, "receipt.json"), 0o644); err != nil {
		return Manifest{}, fmt.Errorf("receipt: %w", err)
	}

	manifest := Manifest{Schema: Schema, StudyID: "default-control-vs-scratch-2026-08", Arm: opts.Arm, SourceCommit: opts.SourceCommit, BinaryVersion: opts.BinaryVersion}
	packetEntries, err := os.ReadDir(opts.OutputDir)
	if err != nil {
		return Manifest{}, err
	}
	for _, entry := range packetEntries {
		if entry.IsDir() || entry.Name() == "packet.json" {
			continue
		}
		digest, err := fileDigest(filepath.Join(opts.OutputDir, entry.Name()))
		if err != nil {
			return Manifest{}, err
		}
		manifest.Files = append(manifest.Files, File{Path: entry.Name(), Digest: digest})
	}
	sort.Slice(manifest.Files, func(i, j int) bool { return manifest.Files[i].Path < manifest.Files[j].Path })
	raw, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return Manifest{}, err
	}
	if err := os.WriteFile(filepath.Join(opts.OutputDir, "packet.json"), append(raw, '\n'), 0o644); err != nil {
		return Manifest{}, err
	}
	return manifest, nil
}

func Parse(raw []byte) (Manifest, error) {
	var manifest Manifest
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&manifest); err != nil {
		return Manifest{}, err
	}
	if manifest.Schema != Schema || manifest.StudyID == "" || manifest.SourceCommit == "" || manifest.BinaryVersion == "" {
		return Manifest{}, fmt.Errorf("packet provenance is incomplete")
	}
	if manifest.Arm != "default-control" && manifest.Arm != "scratch" {
		return Manifest{}, fmt.Errorf("invalid arm %q", manifest.Arm)
	}
	return manifest, nil
}

func Verify(root string, manifest Manifest) error {
	seen := map[string]bool{}
	for _, file := range manifest.Files {
		if file.Path == "" || filepath.Base(file.Path) != file.Path || seen[file.Path] {
			return fmt.Errorf("unsafe or duplicate packet path %q", file.Path)
		}
		seen[file.Path] = true
		got, err := fileDigest(filepath.Join(root, file.Path))
		if err != nil {
			return fmt.Errorf("%s: %w", file.Path, err)
		}
		if got != file.Digest {
			return fmt.Errorf("%s digest mismatch: got %s want %s", file.Path, got, file.Digest)
		}
	}
	for _, required := range []string{"arm-card.md", "task-card.md", "fak", "receipt.json"} {
		if !seen[required] {
			return fmt.Errorf("packet omits %s", required)
		}
	}
	if manifest.Arm == "scratch" {
		for name := range seen {
			if strings.Contains(name, "product") || strings.Contains(name, "selection") || strings.Contains(name, "kernel") {
				return fmt.Errorf("scratch packet leaks default-control material %s", name)
			}
		}
	}
	return nil
}

func copyFile(from, to string, mode os.FileMode) error {
	in, err := os.Open(from)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(to, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, mode)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(out, in)
	closeErr := out.Close()
	if copyErr != nil {
		return copyErr
	}
	return closeErr
}

func fileDigest(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil)), nil
}
