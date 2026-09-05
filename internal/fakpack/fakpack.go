package fakpack

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

const (
	// ErrBundleDigestMismatch is emitted when layer or bundle digests do not match declarations.
	ErrBundleDigestMismatch = "BUNDLE_DIGEST_MISMATCH"
	// ErrBundleCorrupt is emitted when an archive cannot be unpacked or parsed.
	ErrBundleCorrupt = "BUNDLE_CORRUPT"
	// ErrAirgapViolation is emitted when outbound HTTP/HTTPS URLs are detected.
	ErrAirgapViolation = "AIRGAP_VIOLATION"
	// ErrAirgapUnresolvedRemoteDependency is emitted when outbound HTTP/HTTPS URLs are detected in lock/bundle dependencies.
	ErrAirgapUnresolvedRemoteDependency = "AIRGAP_UNRESOLVED_REMOTE_DEPENDENCY"
	// ErrLockMismatch is emitted when the bundle lock ID does not match the expected lock ID.
	ErrLockMismatch = "LOCK_MISMATCH"
	// ErrComponentMissing is emitted when a declared component binary is absent.
	ErrComponentMissing = "COMPONENT_MISSING"
	// ErrAssetMissing is emitted when a declared asset file is absent.
	ErrAssetMissing = "ASSET_MISSING"
)

const (
	// MediaTypeManifest designates the OCI image manifest media type.
	MediaTypeManifest = "application/vnd.oci.image.manifest.v1+json"
	// MediaTypeConfig identifies metadata describing the execution environment and layer mappings for the collection.
	MediaTypeConfig = "application/vnd.fak.pack.config.v1+json"
	// MediaTypeLockV2 designates the version 2 lock layer media type.
	MediaTypeLockV2 = "application/vnd.fak.pack.lock.v2+json"
	// MediaTypeLockV1 designates the version 1 lock layer media type.
	MediaTypeLockV1 = "application/vnd.fak.pack.lock.v1+json"
	// MediaTypeFloor designates the security floor descriptor layer media type.
	MediaTypeFloor = "application/vnd.fak.pack.policy.v1+json"
	// MediaTypeAssets designates the tar.gz assets layer media type.
	MediaTypeAssets = "application/vnd.fak.pack.assets.v1+tar+gzip"
	// MediaTypeBinaries designates the tar.gz binaries layer media type.
	MediaTypeBinaries = "application/vnd.fak.pack.bin.v1+tar+gzip"
	// MediaTypeModel designates the model octet-stream layer media type.
	MediaTypeModel = "application/vnd.fak.pack.model.v1+octet-stream"
	// ArtifactTypeBundle defines the top-level collection artifact type.
	ArtifactTypeBundle = "application/vnd.fak.pack.collection.v1"
)

// Error describes typed packaging failures.
type Error struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Err     error  `json:"-"`
}

func (e *Error) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("%s: %s: %v", e.Code, e.Message, e.Err)
	}
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

func (e *Error) Unwrap() error {
	return e.Err
}

func wrapError(code, message string, err error) error {
	return &Error{Code: code, Message: message, Err: err}
}

// Code extracts the machine-readable error token if available.
func Code(err error) string {
	var e *Error
	if errors.As(err, &e) {
		return e.Code
	}
	return ""
}

// Descriptor conforms to the OCI content descriptor specification.
type Descriptor struct {
	MediaType    string            `json:"mediaType"`
	Digest       string            `json:"digest"`
	Size         int64             `json:"size"`
	Annotations  map[string]string `json:"annotations,omitempty"`
	ArtifactType string            `json:"artifactType,omitempty"`
}

// Manifest adheres to the OCI image manifest specification.
type Manifest struct {
	SchemaVersion int               `json:"schemaVersion"`
	MediaType     string            `json:"mediaType"`
	ArtifactType  string            `json:"artifactType,omitempty"`
	Config        Descriptor        `json:"config"`
	Layers        []Descriptor      `json:"layers"`
	Annotations   map[string]string `json:"annotations,omitempty"`
}

// Config specifies collection metadata and layer descriptors.
type Config struct {
	Schema     string       `json:"schema"`
	Created    string       `json:"created"`
	LockID     string       `json:"lock_id"`
	LockSchema string       `json:"lock_schema"`
	Platforms  []string     `json:"platforms,omitempty"`
	Layers     []Descriptor `json:"layers"`
}

// CreateOptions configures hermetic OCI bundle creation.
type CreateOptions struct {
	LockPath   string
	AssetsDir  string
	BinDir     string
	ModelPath  string
	PolicyPath string
	OutPath    string
}

// CreateResult summarizes the created bundle artifact.
type CreateResult struct {
	BundlePath     string       `json:"bundle_path"`
	ManifestDigest string       `json:"manifest_digest"`
	LockID         string       `json:"lock_id"`
	Layers         []Descriptor `json:"layers"`
	TotalSize      int64        `json:"total_size"`
}

// VerifyOptions configures offline verification of a .fakpack bundle.
type VerifyOptions struct {
	BundlePath       string
	ExpectedLockPath string
}

// VerifyResult summarizes verification outcomes.
type VerifyResult struct {
	BundlePath     string       `json:"bundle_path"`
	ManifestDigest string       `json:"manifest_digest"`
	LockID         string       `json:"lock_id"`
	LayersVerified int          `json:"layers_verified"`
	Layers         []Descriptor `json:"layers"`
	TotalSize      int64        `json:"total_size"`
	AirGapVerified bool         `json:"airgap_verified"`
	LockMatches    bool         `json:"lock_matches,omitempty"`
}

// LockSummary describes key metadata from a harness product lock.
type LockSummary struct {
	ID             string   `json:"id"`
	Schema         string   `json:"schema"`
	Components     []string `json:"components"`
	Assets         []string `json:"assets"`
	ComponentCount int      `json:"component_count"`
	AssetCount     int      `json:"asset_count"`
}

// InspectResult details inspectable bundle contents and metadata.
type InspectResult struct {
	BundlePath  string       `json:"bundle_path"`
	Manifest    Manifest     `json:"manifest"`
	Layers      []Descriptor `json:"layers"`
	LockSummary LockSummary  `json:"lock_summary"`
	Platforms   []string     `json:"platforms,omitempty"`
	TotalSize   int64        `json:"total_size"`
	CreatedTime string       `json:"created_time,omitempty"`
}

// Digest computes the canonical sha256 hex string with sha256 prefix.
func Digest(b []byte) string {
	h := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(h[:])
}

type layerPayload struct {
	desc Descriptor
	data []byte
}

func checkAirgap(lock harnesskit.ProductLock) error {
	for _, c := range lock.Components {
		for _, field := range []string{c.ID, c.Version, c.Digest, c.Source, c.Reason, c.Provider} {
			if hasHTTP(field) {
				return wrapError(ErrAirgapUnresolvedRemoteDependency, fmt.Sprintf("%s: component %q contains outbound URL: %s", ErrAirgapViolation, c.ID, field), nil)
			}
		}
		for _, p := range c.Provides {
			if hasHTTP(p) {
				return wrapError(ErrAirgapUnresolvedRemoteDependency, fmt.Sprintf("%s: component %q provides outbound URL: %s", ErrAirgapViolation, c.ID, p), nil)
			}
		}
		for _, r := range c.Requires {
			if hasHTTP(r.Capability) || hasHTTP(r.Range) {
				return wrapError(ErrAirgapUnresolvedRemoteDependency, fmt.Sprintf("%s: component %q requirement contains outbound URL", ErrAirgapViolation, c.ID), nil)
			}
		}
	}
	for _, a := range lock.Assets {
		for _, field := range []string{a.ID, a.Kind, a.Value, a.Ref, a.Boundary, a.Source} {
			if hasHTTP(field) {
				return wrapError(ErrAirgapUnresolvedRemoteDependency, fmt.Sprintf("%s: asset %q contains outbound URL: %s", ErrAirgapViolation, a.ID, field), nil)
			}
		}
	}
	return nil
}

func hasHTTP(s string) bool {
	lower := strings.ToLower(s)
	return strings.Contains(lower, "http://") || strings.Contains(lower, "https://")
}

func loadAndNormalizeLock(path string) (harnesskit.ProductLock, []byte, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return harnesskit.ProductLock{}, nil, fmt.Errorf("read lock file %s: %w", path, err)
	}

	lock, err := harnesskit.ParseProductLock(raw)
	if err != nil {
		var pl harnesskit.ProductLock
		if jerr := json.Unmarshal(raw, &pl); jerr == nil && len(pl.Components) > 0 {
			if pl.Schema == "" {
				pl.Schema = harnesskit.ProductLockSchemaV2
			}
			if pl.ID == "" {
				if id, idErr := harnesskit.LockID(pl); idErr == nil {
					pl.ID = id
				}
			}
			reencoded, _ := json.Marshal(pl)
			if parsed, pErr := harnesskit.ParseProductLock(reencoded); pErr == nil {
				return parsed, reencoded, nil
			}
			return pl, raw, nil
		}
		return harnesskit.ProductLock{}, nil, fmt.Errorf("parse product lock: %w", err)
	}
	return lock, raw, nil
}

// Create builds a hermetic OCI bundle archive (.fakpack) from specified components.
func Create(opts CreateOptions) (*CreateResult, error) {
	if opts.LockPath == "" {
		return nil, errors.New("lock path is required")
	}
	if opts.OutPath == "" {
		return nil, errors.New("out path is required")
	}

	lock, lockBytes, err := loadAndNormalizeLock(opts.LockPath)
	if err != nil {
		return nil, err
	}

	// Air-gap enforcement
	if err := checkAirgap(lock); err != nil {
		return nil, err
	}

	// Validate declared components and assets exist or are packaged
	if opts.AssetsDir != "" {
		fi, err := os.Stat(opts.AssetsDir)
		if err != nil || !fi.IsDir() {
			return nil, fmt.Errorf("assets directory %s does not exist or is not a directory", opts.AssetsDir)
		}
	}
	if opts.BinDir != "" {
		fi, err := os.Stat(opts.BinDir)
		if err != nil || !fi.IsDir() {
			return nil, fmt.Errorf("bin directory %s does not exist or is not a directory", opts.BinDir)
		}
	}
	if opts.PolicyPath != "" {
		if _, err := os.Stat(opts.PolicyPath); err != nil {
			return nil, fmt.Errorf("policy path %s does not exist: %w", opts.PolicyPath, err)
		}
	}
	if opts.ModelPath != "" {
		if _, err := os.Stat(opts.ModelPath); err != nil {
			return nil, fmt.Errorf("model path %s does not exist: %w", opts.ModelPath, err)
		}
	}

	// Check assets
	for _, asset := range lock.Assets {
		if asset.Value != "" || asset.Ref != "" {
			continue
		}
		if opts.AssetsDir == "" {
			return nil, wrapError(ErrAssetMissing, fmt.Sprintf("asset %q requires assets dir but none provided", asset.ID), nil)
		}
		found := false
		for _, name := range []string{asset.ID, asset.Source, filepath.Base(asset.Source)} {
			if _, err := os.Stat(filepath.Join(opts.AssetsDir, name)); err == nil {
				found = true
				break
			}
		}
		if !found {
			return nil, wrapError(ErrAssetMissing, fmt.Sprintf("asset %q (source %s) not found in assets dir %s", asset.ID, asset.Source, opts.AssetsDir), nil)
		}
	}

	// Check components
	if opts.BinDir != "" {
		for _, comp := range lock.Components {
			found := false
			for _, name := range []string{comp.ID, comp.ID + ".exe", comp.Source, filepath.Base(comp.Source)} {
				if _, err := os.Stat(filepath.Join(opts.BinDir, name)); err == nil {
					found = true
					break
				}
			}
			if !found {
				return nil, wrapError(ErrComponentMissing, fmt.Sprintf("component %q not found in bin dir %s", comp.ID, opts.BinDir), nil)
			}
		}
	}

	var layers []layerPayload

	// 1. Lock layer
	lockMediaType := MediaTypeLockV1
	if lock.Schema == harnesskit.ProductLockSchemaV2 {
		lockMediaType = MediaTypeLockV2
	}
	lockDesc := Descriptor{
		MediaType: lockMediaType,
		Digest:    Digest(lockBytes),
		Size:      int64(len(lockBytes)),
		Annotations: map[string]string{
			"org.opencontainers.image.title": "harness.lock.json",
			"io.fak.object.kind":             "lock",
		},
	}
	layers = append(layers, layerPayload{desc: lockDesc, data: lockBytes})

	// 2. Policy layer
	var policyBytes []byte
	if opts.PolicyPath != "" {
		pb, err := os.ReadFile(opts.PolicyPath)
		if err != nil {
			return nil, fmt.Errorf("read policy file %s: %w", opts.PolicyPath, err)
		}
		policyBytes = pb
	} else {
		policyBytes = []byte("{}\n")
	}
	floorDesc := Descriptor{
		MediaType: MediaTypeFloor,
		Digest:    Digest(policyBytes),
		Size:      int64(len(policyBytes)),
		Annotations: map[string]string{
			"org.opencontainers.image.title": "policy.json",
			"io.fak.object.kind":             "policy",
		},
	}
	layers = append(layers, layerPayload{desc: floorDesc, data: policyBytes})

	// 3. Assets layer
	var assetsBytes []byte
	if opts.AssetsDir != "" {
		ab, err := buildTarGzFromDir(opts.AssetsDir)
		if err != nil {
			return nil, fmt.Errorf("package assets dir: %w", err)
		}
		assetsBytes = ab
	} else {
		ab, err := buildEmptyTarGz()
		if err != nil {
			return nil, err
		}
		assetsBytes = ab
	}
	assetsDesc := Descriptor{
		MediaType: MediaTypeAssets,
		Digest:    Digest(assetsBytes),
		Size:      int64(len(assetsBytes)),
		Annotations: map[string]string{
			"org.opencontainers.image.title": "assets.tar.gz",
			"io.fak.object.kind":             "assets",
		},
	}
	layers = append(layers, layerPayload{desc: assetsDesc, data: assetsBytes})

	// 4. Binaries layer
	var binBytes []byte
	if opts.BinDir != "" {
		bb, err := buildTarGzFromDir(opts.BinDir)
		if err != nil {
			return nil, fmt.Errorf("package bin dir: %w", err)
		}
		binBytes = bb
	} else {
		bb, err := buildEmptyTarGz()
		if err != nil {
			return nil, err
		}
		binBytes = bb
	}
	binDesc := Descriptor{
		MediaType: MediaTypeBinaries,
		Digest:    Digest(binBytes),
		Size:      int64(len(binBytes)),
		Annotations: map[string]string{
			"org.opencontainers.image.title": "bin.tar.gz",
			"io.fak.object.kind":             "binaries",
		},
	}
	layers = append(layers, layerPayload{desc: binDesc, data: binBytes})

	// 5. Optional model layer
	if opts.ModelPath != "" {
		mb, err := os.ReadFile(opts.ModelPath)
		if err != nil {
			return nil, fmt.Errorf("read model file %s: %w", opts.ModelPath, err)
		}
		modelDesc := Descriptor{
			MediaType: MediaTypeModel,
			Digest:    Digest(mb),
			Size:      int64(len(mb)),
			Annotations: map[string]string{
				"org.opencontainers.image.title": filepath.Base(opts.ModelPath),
				"io.fak.object.kind":             "model",
			},
		}
		layers = append(layers, layerPayload{desc: modelDesc, data: mb})
	}

	// Prepare layer descriptors for config & manifest
	layerDescs := make([]Descriptor, len(layers))
	blobs := make(map[string][]byte)
	for i, l := range layers {
		layerDescs[i] = l.desc
		hexDigest := strings.TrimPrefix(l.desc.Digest, "sha256:")
		blobs["blobs/sha256/"+hexDigest] = l.data
	}

	platforms := make([]string, 0, len(lock.Platforms))
	for _, p := range lock.Platforms {
		platforms = append(platforms, p.String())
	}
	if len(platforms) == 0 && (lock.Environment.OS != "" || lock.Environment.Arch != "") {
		platforms = append(platforms, lock.Environment.OS+"/"+lock.Environment.Arch)
	}

	createdTime := time.Now().UTC().Format(time.RFC3339)

	cfg := Config{
		Schema:     "fak.pack.config/v1",
		Created:    createdTime,
		LockID:     lock.ID,
		LockSchema: lock.Schema,
		Platforms:  platforms,
		Layers:     layerDescs,
	}
	cfgBytes, err := json.Marshal(cfg)
	if err != nil {
		return nil, fmt.Errorf("marshal config: %w", err)
	}
	cfgDesc := Descriptor{
		MediaType: MediaTypeConfig,
		Digest:    Digest(cfgBytes),
		Size:      int64(len(cfgBytes)),
	}
	blobs["blobs/sha256/"+strings.TrimPrefix(cfgDesc.Digest, "sha256:")] = cfgBytes

	manifest := Manifest{
		SchemaVersion: 2,
		MediaType:     MediaTypeManifest,
		ArtifactType:  ArtifactTypeBundle,
		Config:        cfgDesc,
		Layers:        layerDescs,
		Annotations: map[string]string{
			"org.opencontainers.image.created": createdTime,
		},
	}
	manifestBytes, err := json.Marshal(manifest)
	if err != nil {
		return nil, fmt.Errorf("marshal manifest: %w", err)
	}
	manifestDigest := Digest(manifestBytes)
	blobs["blobs/sha256/"+strings.TrimPrefix(manifestDigest, "sha256:")] = manifestBytes
	blobs["manifest.json"] = manifestBytes
	blobs["oci-layout"] = []byte("{\"imageLayoutVersion\":\"1.0.0\"}\n")

	if err := os.MkdirAll(filepath.Dir(opts.OutPath), 0755); err != nil && filepath.Dir(opts.OutPath) != "" {
		return nil, err
	}

	if err := writeArchive(opts.OutPath, blobs); err != nil {
		return nil, fmt.Errorf("write bundle archive: %w", err)
	}

	fi, err := os.Stat(opts.OutPath)
	if err != nil {
		return nil, err
	}

	return &CreateResult{
		BundlePath:     opts.OutPath,
		ManifestDigest: manifestDigest,
		LockID:         lock.ID,
		Layers:         layerDescs,
		TotalSize:      fi.Size(),
	}, nil
}

// Verify validates a .fakpack bundle offline against tampering, airgap rules, and optional expected lock.
func Verify(opts VerifyOptions) (*VerifyResult, error) {
	if opts.BundlePath == "" {
		return nil, errors.New("bundle path is required")
	}

	files, err := readArchive(opts.BundlePath)
	if err != nil {
		return nil, err
	}

	manifestData, ok := files["manifest.json"]
	if !ok {
		return nil, wrapError(ErrBundleCorrupt, "manifest.json missing from bundle", nil)
	}

	var manifest Manifest
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		return nil, wrapError(ErrBundleCorrupt, "invalid manifest.json", err)
	}
	if manifest.SchemaVersion != 2 {
		return nil, wrapError(ErrBundleCorrupt, fmt.Sprintf("unsupported schemaVersion %d", manifest.SchemaVersion), nil)
	}

	// Verify config digest
	if manifest.Config.Digest == "" || !strings.HasPrefix(manifest.Config.Digest, "sha256:") {
		return nil, wrapError(ErrBundleCorrupt, "invalid config descriptor", nil)
	}
	cfgHex := strings.TrimPrefix(manifest.Config.Digest, "sha256:")
	cfgData, ok := files["blobs/sha256/"+cfgHex]
	if !ok {
		return nil, wrapError(ErrBundleCorrupt, fmt.Sprintf("config blob %s missing", manifest.Config.Digest), nil)
	}
	if int64(len(cfgData)) != manifest.Config.Size || Digest(cfgData) != manifest.Config.Digest {
		return nil, wrapError(ErrBundleDigestMismatch, fmt.Sprintf("config digest mismatch: got %s want %s", Digest(cfgData), manifest.Config.Digest), nil)
	}

	// Recompute and check SHA-256 digest of every layer
	for _, layer := range manifest.Layers {
		if layer.MediaType == "" || !strings.HasPrefix(layer.Digest, "sha256:") {
			return nil, wrapError(ErrBundleCorrupt, "invalid layer descriptor", nil)
		}
		hexStr := strings.TrimPrefix(layer.Digest, "sha256:")
		blob, ok := files["blobs/sha256/"+hexStr]
		if !ok {
			return nil, wrapError(ErrBundleCorrupt, fmt.Sprintf("layer blob %s missing", layer.Digest), nil)
		}
		if int64(len(blob)) != layer.Size {
			return nil, wrapError(ErrBundleDigestMismatch, fmt.Sprintf("layer size mismatch for %s: got %d want %d", layer.Digest, len(blob), layer.Size), nil)
		}
		computed := Digest(blob)
		if computed != layer.Digest {
			return nil, wrapError(ErrBundleDigestMismatch, fmt.Sprintf("layer digest mismatch: got %s want %s", computed, layer.Digest), nil)
		}
	}

	// Extract and parse lockfile from lock layer
	var lockData []byte
	for _, layer := range manifest.Layers {
		if layer.MediaType == MediaTypeLockV2 || layer.MediaType == MediaTypeLockV1 {
			hexStr := strings.TrimPrefix(layer.Digest, "sha256:")
			lockData = files["blobs/sha256/"+hexStr]
			break
		}
	}
	if len(lockData) == 0 {
		return nil, wrapError(ErrBundleCorrupt, "lock layer missing from bundle", nil)
	}

	var lock harnesskit.ProductLock
	if err := json.Unmarshal(lockData, &lock); err != nil {
		return nil, wrapError(ErrBundleCorrupt, "failed to parse bundle lock json", err)
	}

	// Enforce airgap
	if err := checkAirgap(lock); err != nil {
		return nil, err
	}

	// Check ExpectedLockPath if provided
	lockMatches := false
	if opts.ExpectedLockPath != "" {
		expectedLock, _, err := loadAndNormalizeLock(opts.ExpectedLockPath)
		if err != nil {
			return nil, fmt.Errorf("read expected lock %s: %w", opts.ExpectedLockPath, err)
		}
		if lock.ID != expectedLock.ID {
			return nil, wrapError(ErrLockMismatch, fmt.Sprintf("lock ID mismatch: bundle lock %s != expected lock %s", lock.ID, expectedLock.ID), nil)
		}
		lockMatches = true
	}

	// Verify all components/assets in lock exist in the bundle
	for _, layer := range manifest.Layers {
		if layer.MediaType == MediaTypeAssets {
			hexStr := strings.TrimPrefix(layer.Digest, "sha256:")
			assetsMap, err := listTarGzFiles(files["blobs/sha256/"+hexStr])
			if err != nil {
				return nil, wrapError(ErrBundleCorrupt, "failed reading assets layer", err)
			}
			for _, asset := range lock.Assets {
				if asset.Value != "" || asset.Ref != "" {
					continue
				}
				found := false
				for _, name := range []string{asset.ID, asset.Source, filepath.Base(asset.Source)} {
					if assetsMap[name] {
						found = true
						break
					}
				}
				if !found {
					return nil, wrapError(ErrAssetMissing, fmt.Sprintf("asset %q missing from bundle assets layer", asset.ID), nil)
				}
			}
		}

		if layer.MediaType == MediaTypeBinaries {
			hexStr := strings.TrimPrefix(layer.Digest, "sha256:")
			binMap, err := listTarGzFiles(files["blobs/sha256/"+hexStr])
			if err != nil {
				return nil, wrapError(ErrBundleCorrupt, "failed reading binaries layer", err)
			}
			if len(binMap) > 0 {
				for _, comp := range lock.Components {
					found := false
					for _, name := range []string{comp.ID, comp.ID + ".exe", comp.Source, filepath.Base(comp.Source)} {
						if binMap[name] {
							found = true
							break
						}
					}
					if !found {
						return nil, wrapError(ErrComponentMissing, fmt.Sprintf("component %q missing from bundle binaries layer", comp.ID), nil)
					}
				}
			}
		}
	}

	fi, err := os.Stat(opts.BundlePath)
	totalSize := int64(0)
	if err == nil {
		totalSize = fi.Size()
	}

	return &VerifyResult{
		BundlePath:     opts.BundlePath,
		ManifestDigest: Digest(manifestData),
		LockID:         lock.ID,
		LayersVerified: len(manifest.Layers),
		Layers:         manifest.Layers,
		TotalSize:      totalSize,
		AirGapVerified: true,
		LockMatches:    lockMatches,
	}, nil
}

// Inspect reads bundle metadata without extracting outside or making network calls.
func Inspect(bundlePath string) (*InspectResult, error) {
	if bundlePath == "" {
		return nil, errors.New("bundle path is required")
	}

	files, err := readArchive(bundlePath)
	if err != nil {
		return nil, err
	}

	manifestData, ok := files["manifest.json"]
	if !ok {
		return nil, wrapError(ErrBundleCorrupt, "manifest.json missing from bundle", nil)
	}

	var manifest Manifest
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		return nil, wrapError(ErrBundleCorrupt, "invalid manifest.json", err)
	}

	var lock harnesskit.ProductLock
	for _, layer := range manifest.Layers {
		if layer.MediaType == MediaTypeLockV2 || layer.MediaType == MediaTypeLockV1 {
			hexStr := strings.TrimPrefix(layer.Digest, "sha256:")
			_ = json.Unmarshal(files["blobs/sha256/"+hexStr], &lock)
			break
		}
	}

	compNames := make([]string, 0, len(lock.Components))
	for _, c := range lock.Components {
		compNames = append(compNames, c.ID)
	}
	assetNames := make([]string, 0, len(lock.Assets))
	for _, a := range lock.Assets {
		assetNames = append(assetNames, a.ID)
	}

	platforms := make([]string, 0, len(lock.Platforms))
	for _, p := range lock.Platforms {
		platforms = append(platforms, p.String())
	}
	if len(platforms) == 0 && (lock.Environment.OS != "" || lock.Environment.Arch != "") {
		platforms = append(platforms, lock.Environment.OS+"/"+lock.Environment.Arch)
	}

	fi, _ := os.Stat(bundlePath)
	totalSize := int64(0)
	if fi != nil {
		totalSize = fi.Size()
	}

	createdTime := ""
	if manifest.Annotations != nil {
		createdTime = manifest.Annotations["org.opencontainers.image.created"]
	}

	return &InspectResult{
		BundlePath: bundlePath,
		Manifest:   manifest,
		Layers:     manifest.Layers,
		LockSummary: LockSummary{
			ID:             lock.ID,
			Schema:         lock.Schema,
			Components:     compNames,
			Assets:         assetNames,
			ComponentCount: len(lock.Components),
			AssetCount:     len(lock.Assets),
		},
		Platforms:   platforms,
		TotalSize:   totalSize,
		CreatedTime: createdTime,
	}, nil
}

// ExtractLock reads and returns the raw lock JSON bytes from a bundle archive.
func ExtractLock(bundlePath string) ([]byte, error) {
	files, err := readArchive(bundlePath)
	if err != nil {
		return nil, err
	}
	manifestData, ok := files["manifest.json"]
	if !ok {
		return nil, wrapError(ErrBundleCorrupt, "manifest.json missing from bundle", nil)
	}
	var manifest Manifest
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		return nil, wrapError(ErrBundleCorrupt, "invalid manifest.json", err)
	}
	for _, layer := range manifest.Layers {
		if layer.MediaType == MediaTypeLockV2 || layer.MediaType == MediaTypeLockV1 {
			hexStr := strings.TrimPrefix(layer.Digest, "sha256:")
			if data, ok := files["blobs/sha256/"+hexStr]; ok {
				return data, nil
			}
		}
	}
	for name, data := range files {
		if name == "harness.lock.json" || strings.HasSuffix(name, "/harness.lock.json") {
			return data, nil
		}
	}
	return nil, wrapError(ErrBundleCorrupt, "lock layer missing from bundle", nil)
}

// Unpack extracts the layers of a .fakpack bundle into the specified destination directory.
func Unpack(bundlePath, destDir string) error {
	files, err := readArchive(bundlePath)
	if err != nil {
		return err
	}
	manifestData, ok := files["manifest.json"]
	if !ok {
		return wrapError(ErrBundleCorrupt, "manifest.json missing from bundle", nil)
	}
	var manifest Manifest
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		return wrapError(ErrBundleCorrupt, "invalid manifest.json", err)
	}

	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return err
	}

	for _, layer := range manifest.Layers {
		hexStr := strings.TrimPrefix(layer.Digest, "sha256:")
		blob, ok := files["blobs/sha256/"+hexStr]
		if !ok {
			return wrapError(ErrBundleCorrupt, fmt.Sprintf("blob %s missing", layer.Digest), nil)
		}

		switch layer.MediaType {
		case MediaTypeLockV1, MediaTypeLockV2:
			if err := os.WriteFile(filepath.Join(destDir, "harness.lock.json"), blob, 0o644); err != nil {
				return err
			}
		case MediaTypeFloor:
			if err := os.WriteFile(filepath.Join(destDir, "policy.json"), blob, 0o644); err != nil {
				return err
			}
		case MediaTypeAssets:
			assetsDir := filepath.Join(destDir, "assets")
			if err := extractTarGzToDir(blob, assetsDir); err != nil {
				return fmt.Errorf("unpack assets: %w", err)
			}
		case MediaTypeBinaries:
			binDir := filepath.Join(destDir, "bin")
			if err := extractTarGzToDir(blob, binDir); err != nil {
				return fmt.Errorf("unpack bin: %w", err)
			}
		case MediaTypeModel:
			modelName := "model.bin"
			if layer.Annotations != nil && layer.Annotations["org.opencontainers.image.title"] != "" {
				modelName = layer.Annotations["org.opencontainers.image.title"]
			}
			if err := os.WriteFile(filepath.Join(destDir, modelName), blob, 0o644); err != nil {
				return err
			}
		}
	}
	return nil
}

func extractTarGzToDir(data []byte, destDir string) error {
	if len(data) == 0 {
		return nil
	}
	gr, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return err
	}
	defer gr.Close()

	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return err
	}

	tr := tar.NewReader(gr)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		target := filepath.Join(destDir, filepath.FromSlash(hdr.Name))
		if hdr.FileInfo().IsDir() {
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}
		mode := hdr.FileInfo().Mode()
		if mode&0o111 != 0 {
			mode = 0o755
		} else {
			mode = 0o644
		}
		f, err := os.OpenFile(target, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, mode)
		if err != nil {
			return err
		}
		if _, err := io.Copy(f, tr); err != nil {
			_ = f.Close()
			return err
		}
		_ = f.Close()
	}
	return nil
}

func buildTarGzFromDir(dir string) ([]byte, error) {
	var paths []string
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if p != dir {
			paths = append(paths, p)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("walk dir %s: %w", dir, err)
	}
	sort.Strings(paths)

	var buf bytes.Buffer
	gw := gzip.NewWriter(&buf)
	gw.Header.ModTime = time.Unix(0, 0).UTC()
	gw.Header.OS = 255
	tw := tar.NewWriter(gw)

	for _, p := range paths {
		info, err := os.Lstat(p)
		if err != nil {
			_ = tw.Close()
			_ = gw.Close()
			return nil, err
		}
		rel, err := filepath.Rel(dir, p)
		if err != nil {
			_ = tw.Close()
			_ = gw.Close()
			return nil, err
		}
		cleanRel := filepath.ToSlash(rel)

		hdr, err := tar.FileInfoHeader(info, "")
		if err != nil {
			_ = tw.Close()
			_ = gw.Close()
			return nil, err
		}
		hdr.Name = cleanRel
		hdr.ModTime = time.Unix(0, 0).UTC()
		hdr.Uid = 0
		hdr.Gid = 0
		hdr.Uname = ""
		hdr.Gname = ""

		if info.IsDir() {
			hdr.Name += "/"
			if err := tw.WriteHeader(hdr); err != nil {
				_ = tw.Close()
				_ = gw.Close()
				return nil, err
			}
			continue
		}

		if err := tw.WriteHeader(hdr); err != nil {
			_ = tw.Close()
			_ = gw.Close()
			return nil, err
		}
		f, err := os.Open(p)
		if err != nil {
			_ = tw.Close()
			_ = gw.Close()
			return nil, err
		}
		if _, err := io.Copy(tw, f); err != nil {
			_ = f.Close()
			_ = tw.Close()
			_ = gw.Close()
			return nil, err
		}
		_ = f.Close()
	}

	if err := tw.Close(); err != nil {
		_ = gw.Close()
		return nil, err
	}
	if err := gw.Close(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func buildEmptyTarGz() ([]byte, error) {
	var buf bytes.Buffer
	gw := gzip.NewWriter(&buf)
	gw.Header.ModTime = time.Unix(0, 0).UTC()
	gw.Header.OS = 255
	tw := tar.NewWriter(gw)
	_ = tw.Close()
	_ = gw.Close()
	return buf.Bytes(), nil
}

func listTarGzFiles(data []byte) (map[string]bool, error) {
	if len(data) == 0 {
		return map[string]bool{}, nil
	}
	gr, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	defer gr.Close()

	tr := tar.NewReader(gr)
	files := make(map[string]bool)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		clean := filepath.ToSlash(hdr.Name)
		clean = strings.TrimPrefix(clean, "./")
		clean = strings.TrimSuffix(clean, "/")
		files[clean] = true
		files[filepath.Base(clean)] = true
	}
	return files, nil
}

func writeArchive(outPath string, files map[string][]byte) error {
	f, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer f.Close()

	tw := tar.NewWriter(f)
	defer tw.Close()

	keys := make([]string, 0, len(files))
	for k := range files {
		if k != "manifest.json" {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	ordered := append([]string{"manifest.json"}, keys...)

	for _, name := range ordered {
		data := files[name]
		hdr := &tar.Header{
			Name:    name,
			Mode:    0o644,
			Size:    int64(len(data)),
			ModTime: time.Unix(0, 0).UTC(),
		}
		if err := tw.WriteHeader(hdr); err != nil {
			return err
		}
		if _, err := tw.Write(data); err != nil {
			return err
		}
	}
	return nil
}

func readArchive(bundlePath string) (map[string][]byte, error) {
	f, err := os.Open(bundlePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	header := make([]byte, 2)
	n, _ := f.ReadAt(header, 0)
	var r io.Reader = f
	if n == 2 && header[0] == 0x1f && header[1] == 0x8b {
		gr, err := gzip.NewReader(f)
		if err != nil {
			return nil, wrapError(ErrBundleCorrupt, "invalid gzip header", err)
		}
		defer gr.Close()
		r = gr
	}

	tr := tar.NewReader(r)
	files := make(map[string][]byte)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, wrapError(ErrBundleCorrupt, "invalid or truncated tar header", err)
		}
		content, err := io.ReadAll(tr)
		if err != nil {
			return nil, wrapError(ErrBundleCorrupt, "failed reading tar entry content", err)
		}
		files[hdr.Name] = content
	}

	if len(files) == 0 || files["manifest.json"] == nil {
		return nil, wrapError(ErrBundleCorrupt, "bundle does not contain valid manifest.json", nil)
	}
	return files, nil
}
