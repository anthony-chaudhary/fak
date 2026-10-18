package model

import "fmt"

// AuthenticatedArtifactIdentity is a loader receipt for a complete ordered GGUF
// shard set. It records byte membership, not an upstream release assertion or
// numerical qualification. Digests are lowercase SHA-256 hex. ManifestSHA256
// retains the trusted path-dependent manifest; ShardSetSHA256 hashes JSON of
// ordered {size,raw_file_sha256} entries, without paths. ChatTemplateSHA256 binds
// the exact first-shard string, or is empty when that metadata is absent.
// Retained files and model weights must remain immutable for their lifetime.
type AuthenticatedArtifactIdentity struct {
	ManifestSHA256     string
	ShardSetSHA256     string
	ChatTemplateSHA256 string
}

// AuthenticatedArtifactIdentity returns a copy of the load-time receipt. False
// means checkpoint membership is unknown; architecture or tensor shapes cannot
// promote that result to a named release.
func (m *Model) AuthenticatedArtifactIdentity() (AuthenticatedArtifactIdentity, bool) {
	if m == nil || m.authenticatedArtifact == nil {
		return AuthenticatedArtifactIdentity{}, false
	}
	return *m.authenticatedArtifact, true
}

// BindAuthenticatedArtifactIdentity is a trusted-loader-only transfer, before
// Build publishes the model. It does not authenticate caller-supplied strings:
// callers must first verify every retained shard against a trusted manifest.
// The GGUF authenticated opener supplies that evidence; ordinary loaders never
// call this method. There is no post-Build model identity setter.
func (b *QuantBuilder) BindAuthenticatedArtifactIdentity(id AuthenticatedArtifactIdentity) error {
	if b == nil || b.m == nil || b.built || b.m.authenticatedArtifact != nil {
		return fmt.Errorf("model: authenticated artifact identity already bound or builder unavailable")
	}
	valid := func(s string) bool {
		if len(s) != 64 {
			return false
		}
		for _, c := range s {
			if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
				return false
			}
		}
		return true
	}
	if !valid(id.ManifestSHA256) || !valid(id.ShardSetSHA256) || id.ChatTemplateSHA256 != "" && !valid(id.ChatTemplateSHA256) {
		return fmt.Errorf("model: malformed authenticated artifact identity")
	}
	b.m.authenticatedArtifact = &id
	return nil
}
