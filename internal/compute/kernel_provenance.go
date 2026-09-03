package compute

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

const KernelProvenanceSchema = "fak.kernel-provenance/1"

// KernelReuseDecision records how a reviewed upstream kernel relates to fak.
type KernelReuseDecision string

const (
	KernelCopyDirectly KernelReuseDecision = "COPY DIRECTLY"
	KernelAdapt        KernelReuseDecision = "ADAPT"
	KernelDelegate     KernelReuseDecision = "DELEGATE"
	KernelExclude      KernelReuseDecision = "EXCLUDE"
)

// KernelProvenanceManifest binds reviewed upstreams to copied or adapted files.
type KernelProvenanceManifest struct {
	Schema    string                   `json:"schema"`
	Upstreams []KernelProvenanceSource `json:"upstreams"`
	Kernels   []KernelProvenanceEntry  `json:"kernels"`
}

// KernelProvenanceSource is one immutable, license-reviewed upstream revision.
type KernelProvenanceSource struct {
	ID              string `json:"id"`
	Repository      string `json:"repository"`
	SHA             string `json:"sha"`
	License         string `json:"license"`
	LicenseFile     string `json:"license_file"`
	CopyrightNotice string `json:"copyright_notice"`
	NoticeFile      string `json:"notice_file"`
	NoticeRule      string `json:"notice_rule"`
}

// KernelProvenanceEntry is the source-to-destination decision for one kernel.
type KernelProvenanceEntry struct {
	ID            string              `json:"id"`
	Upstream      string              `json:"upstream"`
	SourcePath    string              `json:"source_path"`
	Destination   string              `json:"destination"`
	Decision      KernelReuseDecision `json:"decision"`
	Attribution   string              `json:"attribution"`
	ParityWitness string              `json:"parity_witness"`
}

var kernelProvenanceSHA = regexp.MustCompile(`^[0-9a-f]{40}$`)

// DecodeKernelProvenance decodes and validates one manifest.
func DecodeKernelProvenance(data []byte) (KernelProvenanceManifest, error) {
	var manifest KernelProvenanceManifest
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&manifest); err != nil {
		return KernelProvenanceManifest{}, fmt.Errorf("decode kernel provenance: %w", err)
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		if err == nil {
			err = fmt.Errorf("multiple JSON values")
		}
		return KernelProvenanceManifest{}, fmt.Errorf("decode kernel provenance: %w", err)
	}
	if err := manifest.Validate(); err != nil {
		return KernelProvenanceManifest{}, err
	}
	return manifest, nil
}

// LoadKernelProvenance reads and validates one manifest from disk.
func LoadKernelProvenance(filename string) (KernelProvenanceManifest, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return KernelProvenanceManifest{}, fmt.Errorf("read kernel provenance: %w", err)
	}
	return DecodeKernelProvenance(data)
}

// Validate checks schema completeness without inspecting the working tree.
func (m KernelProvenanceManifest) Validate() error {
	if m.Schema != KernelProvenanceSchema {
		return fmt.Errorf("schema = %q, want %q", m.Schema, KernelProvenanceSchema)
	}
	if len(m.Upstreams) == 0 {
		return fmt.Errorf("upstreams are required")
	}
	if len(m.Kernels) == 0 {
		return fmt.Errorf("kernels are required")
	}

	sources := make(map[string]KernelProvenanceSource, len(m.Upstreams))
	for i, source := range m.Upstreams {
		field := fmt.Sprintf("upstreams[%d]", i)
		if strings.TrimSpace(source.ID) == "" {
			return fmt.Errorf("%s.id is required", field)
		}
		if _, exists := sources[source.ID]; exists {
			return fmt.Errorf("%s.id %q is duplicated", field, source.ID)
		}
		if err := validateKernelRepository(source.Repository); err != nil {
			return fmt.Errorf("%s.repository: %w", field, err)
		}
		if !kernelProvenanceSHA.MatchString(source.SHA) {
			return fmt.Errorf("%s.sha must be a pinned 40-character lowercase hexadecimal commit", field)
		}
		if source.License != "MIT" && source.License != "Apache-2.0" {
			return fmt.Errorf("%s.license must be MIT or Apache-2.0", field)
		}
		if err := validateKernelPath(field+".license_file", source.LicenseFile); err != nil {
			return err
		}
		if strings.TrimSpace(source.CopyrightNotice) == "" {
			return fmt.Errorf("%s.copyright_notice is required", field)
		}
		if source.NoticeFile == "" {
			return fmt.Errorf("%s.notice_file is required (use NONE only when the pinned revision has no NOTICE)", field)
		}
		if source.NoticeFile != "NONE" {
			if err := validateKernelPath(field+".notice_file", source.NoticeFile); err != nil {
				return err
			}
		}
		if strings.TrimSpace(source.NoticeRule) == "" {
			return fmt.Errorf("%s.notice_rule is required", field)
		}
		sources[source.ID] = source
	}

	entries := make(map[string]struct{}, len(m.Kernels))
	destinations := make(map[string]string, len(m.Kernels))
	for i, entry := range m.Kernels {
		field := fmt.Sprintf("kernels[%d]", i)
		if strings.TrimSpace(entry.ID) == "" {
			return fmt.Errorf("%s.id is required", field)
		}
		if _, exists := entries[entry.ID]; exists {
			return fmt.Errorf("%s.id %q is duplicated", field, entry.ID)
		}
		entries[entry.ID] = struct{}{}
		source, exists := sources[entry.Upstream]
		if !exists {
			return fmt.Errorf("%s.upstream %q is not declared", field, entry.Upstream)
		}
		if err := validateKernelPath(field+".source_path", entry.SourcePath); err != nil {
			return err
		}
		if err := validateKernelPath(field+".destination", entry.Destination); err != nil {
			return err
		}
		if previous, exists := destinations[entry.Destination]; exists {
			return fmt.Errorf("%s.destination %q is already owned by kernel %q", field, entry.Destination, previous)
		}
		destinations[entry.Destination] = entry.ID
		switch entry.Decision {
		case KernelCopyDirectly, KernelAdapt, KernelDelegate, KernelExclude:
		default:
			return fmt.Errorf("%s.decision %q is invalid", field, entry.Decision)
		}
		if strings.TrimSpace(entry.Attribution) == "" {
			return fmt.Errorf("%s.attribution is required", field)
		}
		if !strings.Contains(entry.Attribution, source.SHA) {
			return fmt.Errorf("%s.attribution must include upstream SHA %s", field, source.SHA)
		}
		if !strings.Contains(entry.Attribution, source.License) {
			return fmt.Errorf("%s.attribution must include license %s", field, source.License)
		}
		if err := validateKernelPath(field+".parity_witness", entry.ParityWitness); err != nil {
			return err
		}
	}
	return nil
}

// ValidateTree checks declared COPY DIRECTLY and ADAPT destinations against root.
func (m KernelProvenanceManifest) ValidateTree(root string) error {
	if err := m.Validate(); err != nil {
		return err
	}
	for _, entry := range m.Kernels {
		if entry.Decision != KernelCopyDirectly && entry.Decision != KernelAdapt {
			continue
		}
		destination := filepath.Join(root, filepath.FromSlash(entry.Destination))
		data, err := os.ReadFile(destination)
		if err != nil {
			return fmt.Errorf("kernel %q destination: %w", entry.ID, err)
		}
		data = bytes.ReplaceAll(data, []byte("\r\n"), []byte("\n"))
		if !bytes.Contains(data, []byte(entry.Attribution)) {
			return fmt.Errorf("kernel %q destination %q is missing declared attribution", entry.ID, entry.Destination)
		}
		witness := filepath.Join(root, filepath.FromSlash(entry.ParityWitness))
		info, err := os.Stat(witness)
		if err != nil {
			return fmt.Errorf("kernel %q parity witness: %w", entry.ID, err)
		}
		if info.IsDir() {
			return fmt.Errorf("kernel %q parity witness %q is a directory", entry.ID, entry.ParityWitness)
		}
	}
	return nil
}

func validateKernelRepository(repository string) error {
	u, err := url.Parse(repository)
	if err != nil || u.Scheme != "https" || u.Host != "github.com" || strings.Trim(u.Path, "/") == "" {
		return fmt.Errorf("must be an https://github.com/<owner>/<repo> URL")
	}
	if u.RawQuery != "" || u.Fragment != "" {
		return fmt.Errorf("must not include a query or fragment")
	}
	return nil
}

func validateKernelPath(field, value string) error {
	if strings.TrimSpace(value) == "" {
		return fmt.Errorf("%s is required", field)
	}
	clean := filepath.Clean(filepath.FromSlash(value))
	if filepath.IsAbs(clean) || clean == "." || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return fmt.Errorf("%s must be a repository-relative path", field)
	}
	return nil
}
