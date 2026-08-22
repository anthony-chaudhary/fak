package harnessartifact

import (
	"crypto/ed25519"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
)

const Schema = "fak.harness-artifact/v1alpha1"
const ChannelSchema = "fak.harness-channel/v1alpha1"
const RevocationsSchema = "fak.harness-revocations/v1alpha1"
const ReceiptSchema = "fak.harness-artifact-receipt/v1alpha1"

type Metadata struct {
	Name      string `json:"name"`
	Publisher string `json:"publisher"`
	Support   string `json:"support"`
	Lifecycle string `json:"lifecycle"`
	BuiltAt   string `json:"built_at"`
	Source    string `json:"source,omitempty"`
}
type Payload struct {
	Schema   string              `json:"schema"`
	Metadata Metadata            `json:"metadata"`
	Lock     harnessresolve.Lock `json:"lock"`
}
type Artifact struct {
	ID        string  `json:"id"`
	Payload   Payload `json:"payload"`
	PublicKey string  `json:"public_key"`
	Signature string  `json:"signature"`
}
type Channel struct {
	Schema     string   `json:"schema"`
	Name       string   `json:"name"`
	ArtifactID string   `json:"artifact_id"`
	Previous   []string `json:"previous,omitempty"`
	UpdatedAt  string   `json:"updated_at"`
}
type Revocations struct {
	Schema     string   `json:"schema"`
	Publishers []string `json:"publishers,omitempty"`
	Artifacts  []string `json:"artifacts,omitempty"`
}
type Receipt struct {
	Schema               string `json:"schema"`
	ArtifactID           string `json:"artifact_id"`
	LockID               string `json:"lock_id"`
	Publisher            string `json:"publisher"`
	PublisherFingerprint string `json:"publisher_fingerprint"`
	Support              string `json:"support"`
	Lifecycle            string `json:"lifecycle"`
	Source               string `json:"source"`
	VerifiedAt           string `json:"verified_at"`
}
type PublishResult struct {
	Artifact    Artifact `json:"artifact"`
	Path        string   `json:"path"`
	ChannelPath string   `json:"channel_path,omitempty"`
}
type FetchResult struct {
	Artifact     Artifact `json:"artifact"`
	LockPath     string   `json:"lock_path"`
	ArtifactPath string   `json:"artifact_path"`
	ReceiptPath  string   `json:"receipt_path"`
	CacheHit     bool     `json:"cache_hit"`
}
type PublishOptions struct {
	Lock       harnessresolve.Lock
	Metadata   Metadata
	PrivateKey ed25519.PrivateKey
	Registry   string
	Channel    string
	Now        time.Time
}
type FetchOptions struct {
	Registry    string
	Cache       string
	Ref         string
	TrustKey    string
	Revocations string
	Output      string
	Offline     bool
	Now         time.Time
}
