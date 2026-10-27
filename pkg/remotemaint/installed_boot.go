package remotemaint

import (
	"encoding/hex"
	"fmt"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// InstalledBootScope is a source-selected finite boot-artifact operation. It
// confers no activation, module-load, reboot, or follow-on apply authority. The
// host Authority must additionally validate the actual files and maintenance
// fence against its own source definition before every effect.
type InstalledBootScope struct {
	Phase                 string                    `json:"phase"`
	ArtifactSHA256        string                    `json:"artifact_sha256"`
	PrivateCommit         string                    `json:"private_commit"`
	PublicCommit          string                    `json:"public_commit"`
	PublicArchiveSHA256   string                    `json:"public_archive_sha256"`
	ManifestSHA256        string                    `json:"manifest_sha256"`
	StageDigest           string                    `json:"stage_digest"`
	PlanDigest            string                    `json:"plan_digest"`
	IdentitySHA256        string                    `json:"identity_sha256"`
	ServiceIdentitySHA256 string                    `json:"service_identity_sha256"`
	ReadPaths             []string                  `json:"read_paths"`
	WritePaths            []string                  `json:"write_paths"`
	BackupPrefix          string                    `json:"backup_prefix"`
	RequiredNodeIDs       []string                  `json:"required_node_ids"`
	CatalogGeneration     string                    `json:"catalog_generation"`
	ExpectedStoreEpoch    uint64                    `json:"expected_store_epoch"`
	LocalScopes           []InstalledBootLocalScope `json:"local_scopes"`
}

// InstalledBootLocalScope declares source-owned local writes separately from
// POSIX target paths. Roots and paths are absolute native host paths. The host
// authority must check roots against its canonical store and OS folder policy.
type InstalledBootLocalScope struct {
	Root         string       `json:"root"`
	Paths        []string     `json:"paths"`
	PathPrefixes []PathPrefix `json:"path_prefixes"`
}

type InstalledBootFileState struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Mode   string `json:"mode"`
	UID    uint32 `json:"uid"`
	GID    uint32 `json:"gid"`
}

// InstalledBootObservation carries hashes and bounded file metadata, never raw
// plans or secret bytes. Observations remain untrusted content at result floors.
// A preflight may omit stage/plan/identity hashes when no artifact is staged.
type InstalledBootObservation struct {
	Kind                  string                   `json:"kind"`
	ArtifactSHA256        string                   `json:"artifact_sha256"`
	StageDigest           string                   `json:"stage_digest"`
	PlanDigest            string                   `json:"plan_digest"`
	IdentitySHA256        string                   `json:"identity_sha256"`
	ServiceIdentitySHA256 string                   `json:"service_identity_sha256"`
	RootImageSHA256       string                   `json:"root_image_sha256"`
	ESPImageSHA256        string                   `json:"esp_image_sha256"`
	Files                 []InstalledBootFileState `json:"files,omitempty"`
}

func installedBootOperation(op string) bool {
	switch op {
	case OperationInstalledBootStagePlan, OperationInstalledBootApply, OperationInstalledBootReadback, OperationInstalledBootRollback:
		return true
	}
	return false
}

func validateInstalledBootGrant(g Grant, now time.Time) error {
	b := g.InstalledBoot
	if b == nil || g.Reservation != nil || len(g.Units) != 0 || len(g.PathPrefixes) != 0 ||
		!validName(g.ProfileID, 128) || !isDigest(g.OperationDigest) || !isDigest(g.ContextID) ||
		!validName(g.Target, 512) || !validHostKey(g.HostKey) || !validName(g.Holder, 256) ||
		!validName(g.BootID, 256) || !isDigest(g.PriorStateSHA256) || !isDigest(g.PayloadSHA256) {
		return fmt.Errorf("%w: invalid installed boot identity", ErrScope)
	}
	if !isDigest(b.ArtifactSHA256) || b.ArtifactSHA256 != g.PayloadSHA256 ||
		!validBootCommit(b.PrivateCommit) || !validBootCommit(b.PublicCommit) ||
		!isDigest(b.PublicArchiveSHA256) || !isDigest(b.ManifestSHA256) ||
		!validName(b.CatalogGeneration, 256) || b.ExpectedStoreEpoch == 0 {
		return fmt.Errorf("%w: missing installed boot source bindings", ErrScope)
	}
	phaseOperation := map[string]string{"preflight": OperationInstalledBootReadback, "stage-plan": OperationInstalledBootStagePlan,
		"apply": OperationInstalledBootApply, "readback": OperationInstalledBootReadback, "rollback": OperationInstalledBootRollback}
	if expected, ok := phaseOperation[b.Phase]; !ok || g.Operation != expected {
		return fmt.Errorf("%w: installed boot phase mismatch", ErrScope)
	}
	for _, digest := range []string{b.StageDigest, b.PlanDigest, b.IdentitySHA256, b.ServiceIdentitySHA256} {
		if digest != "" && !isDigest(digest) {
			return fmt.Errorf("%w: invalid installed boot binding", ErrScope)
		}
	}
	if b.Phase != "preflight" && (!isDigest(b.StageDigest) || !isDigest(b.IdentitySHA256) || !isDigest(b.ServiceIdentitySHA256) ||
		(b.Phase != "stage-plan" && !isDigest(b.PlanDigest))) {
		return fmt.Errorf("%w: missing installed boot execution bindings", ErrScope)
	}
	if len(b.ReadPaths) == 0 || len(b.ReadPaths) > 64 || len(b.WritePaths) > 64 ||
		!canonicalStrings(b.ReadPaths) || !canonicalStrings(b.WritePaths) || !canonicalStrings(g.Paths) || len(g.Paths) > 64 {
		return fmt.Errorf("%w: invalid installed boot path scope", ErrScope)
	}
	paths := append(append([]string(nil), b.ReadPaths...), b.WritePaths...)
	for _, p := range paths {
		if !validBootPath(p) {
			return fmt.Errorf("%w: invalid installed boot path", ErrScope)
		}
	}
	sort.Strings(paths)
	paths = compactStrings(paths)
	if !equalBootStrings(paths, g.Paths) {
		return fmt.Errorf("%w: installed boot paths do not match grant", ErrScope)
	}
	passive := b.Phase == "preflight" || b.Phase == "readback"
	if b.Phase == "readback" {
		for _, p := range b.WritePaths {
			if p != "/run/fak-installed-boot.lock" && p != "/run/fak-raider-installed-boot.lock" {
				return fmt.Errorf("%w: installed boot readback write is not a lock", ErrScope)
			}
		}
	}
	if b.Phase == "preflight" && len(b.WritePaths) != 0 || passive && b.BackupPrefix != "" || !passive &&
		(len(b.WritePaths) == 0 || !validBootPath(b.BackupPrefix) || !validPathPrefix(PathPrefix{Directory: path.Dir(b.BackupPrefix), FilenamePrefix: path.Base(b.BackupPrefix)})) {
		return fmt.Errorf("%w: invalid installed boot write scope", ErrScope)
	}
	if err := validateInstalledBootLocalScopes(b.LocalScopes); err != nil {
		return err
	}
	if len(b.RequiredNodeIDs) == 0 || len(b.RequiredNodeIDs) > 16 || !canonicalStrings(b.RequiredNodeIDs) || len(g.LeasePins) != len(b.RequiredNodeIDs) {
		return fmt.Errorf("%w: installed boot lease node set", ErrScope)
	}
	if g.ExpiresAt.IsZero() || !g.ExpiresAt.Equal(g.ExpiresAt.UTC()) || !g.ExpiresAt.After(now) {
		return fmt.Errorf("%w: installed boot grant", ErrExpired)
	}
	seen := make(map[string]bool, len(g.LeasePins))
	for i, pin := range g.LeasePins {
		if !validName(pin.NodeID, 256) || pin.NodeID != b.RequiredNodeIDs[i] || !validName(pin.ID, 256) || seen[pin.ID] || pin.Epoch == 0 || pin.Epoch > b.ExpectedStoreEpoch {
			return fmt.Errorf("%w: installed boot lease identity", ErrScope)
		}
		if pin.ExpiresAt.IsZero() || !pin.ExpiresAt.Equal(pin.ExpiresAt.UTC()) || !pin.ExpiresAt.After(now) || pin.ExpiresAt.Before(g.ExpiresAt) {
			return fmt.Errorf("%w: installed boot lease", ErrExpired)
		}
		seen[pin.ID] = true
	}
	return nil
}

func validateInstalledBootLocalScopes(scopes []InstalledBootLocalScope) error {
	if len(scopes) == 0 || len(scopes) > 16 {
		return fmt.Errorf("%w: installed boot local scope bounds", ErrScope)
	}
	last := ""
	for _, s := range scopes {
		if !validBootLocalPath(s.Root) || s.Root <= last || len(s.Paths)+len(s.PathPrefixes) == 0 ||
			len(s.Paths) > 64 || len(s.PathPrefixes) > 16 || !canonicalStrings(s.Paths) {
			return fmt.Errorf("%w: installed boot local root", ErrScope)
		}
		last = s.Root
		within := func(p string) bool {
			if !validBootLocalPath(p) {
				return false
			}
			rel, err := filepath.Rel(s.Root, p)
			return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
		}
		for _, p := range s.Paths {
			if !within(p) {
				return fmt.Errorf("%w: installed boot local path escape", ErrScope)
			}
		}
		previous := PathPrefix{}
		for i, p := range s.PathPrefixes {
			if !within(p.Directory) || !validPathPrefix(PathPrefix{Directory: "/local", FilenamePrefix: p.FilenamePrefix}) ||
				i > 0 && (previous.Directory > p.Directory || previous.Directory == p.Directory && previous.FilenamePrefix >= p.FilenamePrefix) {
				return fmt.Errorf("%w: installed boot local prefix", ErrScope)
			}
			previous = p
		}
	}
	return nil
}

func validBootLocalPath(p string) bool {
	return validName(p, 4096) && filepath.IsAbs(p) && filepath.Clean(p) == p && filepath.Dir(p) != p && !strings.ContainsAny(p, "*?[]{}")
}

func validBootCommit(commit string) bool {
	if len(commit) != 40 && len(commit) != 64 {
		return false
	}
	_, err := hex.DecodeString(commit)
	return err == nil && strings.ToLower(commit) == commit
}

func validBootPath(p string) bool {
	return len(p) <= 4096 && validName(p, 4096) && validAbsolutePOSIX(p) && !strings.ContainsAny(p, "*?[]{}")
}

func equalBootStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
