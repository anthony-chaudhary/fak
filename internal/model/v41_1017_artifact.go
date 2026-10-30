package model

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// ErrV41ArtifactProfile is the typed fail-closed refusal for DeepSeek V4.1
// release artifact intake: a presented shard set or profile structure that does
// not exactly match a pinned release profile. It is metadata/provenance
// admission only; it enables no execution path.
var ErrV41ArtifactProfile = errors.New("model: DeepSeek V4.1 release artifact profile mismatch")

const (
	// DeepSeekV41Flash1017ModelID and DeepSeekV41Flash1017Revision pin the
	// official 1017 target release. They are a DISTINCT artifact identity from
	// DeepSeekV41FlashModelID/DeepSeekV41FlashRevision (the pre-1017 release):
	// the old constants stay untouched, and no model architecture enum gains a
	// date-suffix value — the 1017 release is the same deepseek_v41 family with
	// its own artifact/provenance profile.
	DeepSeekV41Flash1017ModelID  = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
	DeepSeekV41Flash1017Revision = "1bae0e1088993b9b82de08f59de5be5a34e42170"
)

// V41FilePin binds one release file by exact name, content identity and byte
// size. LFS-published weight shards carry their lowercase SHA-256 hex; plain
// git objects (config.json, the tokenizer files, the safetensors index) carry
// the lowercase git blob SHA-1 hex the registry reports. A pin never mixes the
// two schemes, and this package deliberately performs no cross-format
// comparison: a GGUF extent hash is not comparable to a safetensors whole-file
// hash, and attempting such a comparison here would fabricate membership.
type V41FilePin struct {
	Name string `json:"name"`
	ID   string `json:"id"`
	Size int64  `json:"size"`
	LFS  bool   `json:"lfs"`
}

// V41DraftSelection records how a release profile sources its bundled DSpark
// draft. V41DraftStatusUnavailable is the header-only fixture path: the
// inspected storage revision cannot supply the draft, and Repository/Revision
// pin that inspected storage while Reason records the witnessed gap.
// V41DraftStatusSelected names a separately qualified draft repository and
// revision instead. Neither status enables speculative decoding:
// RefuseDeepSeekV41DSpark stays the fail-closed serving gate.
type V41DraftSelection struct {
	Status     string `json:"status"`
	Repository string `json:"repository"`
	Revision   string `json:"revision"`
	Reason     string `json:"reason,omitempty"`
}

const (
	V41DraftStatusUnavailable = "unavailable"
	V41DraftStatusSelected    = "selected"
)

// V41DSparkIntake retains the release's bundled DSpark draft declaration read
// from its pinned config: the multi-token-prediction head count and the
// dspark_* markers. It is intake metadata only and admits no draft execution.
type V41DSparkIntake struct {
	NextNPredictLayers int   `json:"num_nextn_predict_layers"`
	BlockSize          int   `json:"dspark_block_size"`
	TargetLayerIDs     []int `json:"dspark_target_layer_ids"`
	MarkovRank         int   `json:"dspark_markov_rank"`
	NumRoutedExperts   int   `json:"dspark_n_routed_experts"`
	NumExpertsPerTok   int   `json:"dspark_num_experts_per_tok"`
}

// DeepSeekV41ReleaseArtifactProfile is a distinct, pinned artifact/provenance
// profile for one DeepSeek V4.1 Flash release. It binds the target revision
// and the complete ordered weight-shard manifest, the tokenizer/encoder and
// config identities, the Engram encoded extents, the bundled DSpark intake,
// and a separately selected draft revision or an explicit unavailable status.
// It prevents silently pairing old weights, Engram or draft with the new
// target: admission below demands the exact pinned content identity of every
// shard, so a same-name/same-size shard from another release fails closed.
//
// The profile is metadata intake only. It is not an artifact receipt, makes no
// checkpoint-membership or numerical-qualification claim, and is not the
// legacy geometry-based graph selector (that remains
// DeepSeekV41NumericalProfileRequest, which stays fail-closed).
type DeepSeekV41ReleaseArtifactProfile struct {
	SchemaVersion int `json:"schema_version"`

	TargetRepository string `json:"target_repository"`
	TargetRevision   string `json:"target_revision"`

	// WeightShards is the complete ordered 48-shard LFS manifest.
	WeightShards []V41FilePin `json:"weight_shards"`

	// Metadata pins the config and tokenizer/encoder files (git blob ids).
	Metadata []V41FilePin `json:"metadata"`

	// EngramShardNames names the WeightShards entries carrying the Engram
	// payload; EngramLayerIDs/EngramNumEmbeddings retain the published encoded
	// row extents per Engram layer.
	EngramShardNames    []string `json:"engram_shard_names"`
	EngramLayerIDs      []int    `json:"engram_layer_ids"`
	EngramNumEmbeddings []int    `json:"engram_num_embeddings"`

	DSpark V41DSparkIntake   `json:"dspark"`
	Draft  V41DraftSelection `json:"draft"`
}

// v41LowerHex reports whether s is exactly wantLen lowercase hex characters.
func v41LowerHex(s string, wantLen int) bool {
	if len(s) != wantLen {
		return false
	}
	for _, c := range s {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}

// v41RequiredV41MetadataNames is the closed set of non-weight files a release
// profile must pin.
var v41RequiredV41MetadataNames = []string{
	"config.json",
	"tokenizer.json",
	"tokenizer_config.json",
	"model.safetensors.index.json",
}

// Validate checks the profile's internal consistency: schema version, target
// identity, the ordered 48-shard manifest naming, well-formed pins, the
// required metadata files, Engram extents inside the manifest, a coherent
// DSpark intake and a coherent draft selection. It validates profile
// structure only; it makes no claim about any loaded artifact.
func (p *DeepSeekV41ReleaseArtifactProfile) Validate() error {
	if p == nil {
		return fmt.Errorf("%w: nil profile", ErrV41ArtifactProfile)
	}
	if p.SchemaVersion != 1 {
		return fmt.Errorf("%w: schema_version=%d, want 1", ErrV41ArtifactProfile, p.SchemaVersion)
	}
	if strings.TrimSpace(p.TargetRepository) == "" || !v41LowerHex(p.TargetRevision, 40) {
		return fmt.Errorf("%w: malformed target identity %q@%q", ErrV41ArtifactProfile, p.TargetRepository, p.TargetRevision)
	}
	if len(p.WeightShards) != 48 {
		return fmt.Errorf("%w: manifest carries %d shards, want 48", ErrV41ArtifactProfile, len(p.WeightShards))
	}
	shardNames := make(map[string]bool, len(p.WeightShards))
	for i, s := range p.WeightShards {
		want := fmt.Sprintf("model-%05d-of-00048.safetensors", i+1)
		if s.Name != want {
			return fmt.Errorf("%w: shard %d named %q, want %q", ErrV41ArtifactProfile, i+1, s.Name, want)
		}
		if !s.LFS || !v41LowerHex(s.ID, 64) || s.Size <= 0 {
			return fmt.Errorf("%w: shard %s has malformed pin (lfs=%t id-len=%d size=%d)", ErrV41ArtifactProfile, s.Name, s.LFS, len(s.ID), s.Size)
		}
		if shardNames[s.Name] {
			return fmt.Errorf("%w: duplicate shard %s", ErrV41ArtifactProfile, s.Name)
		}
		shardNames[s.Name] = true
	}
	metaSeen := make(map[string]bool, len(p.Metadata))
	for _, m := range p.Metadata {
		if m.LFS || !v41LowerHex(m.ID, 40) || m.Size <= 0 {
			return fmt.Errorf("%w: metadata %s has malformed pin (lfs=%t id-len=%d size=%d)", ErrV41ArtifactProfile, m.Name, m.LFS, len(m.ID), m.Size)
		}
		if metaSeen[m.Name] {
			return fmt.Errorf("%w: duplicate metadata %s", ErrV41ArtifactProfile, m.Name)
		}
		metaSeen[m.Name] = true
	}
	for _, name := range v41RequiredV41MetadataNames {
		if !metaSeen[name] {
			return fmt.Errorf("%w: metadata lacks %s", ErrV41ArtifactProfile, name)
		}
	}
	if len(p.EngramShardNames) == 0 {
		return fmt.Errorf("%w: no Engram shard extents declared", ErrV41ArtifactProfile)
	}
	for _, name := range p.EngramShardNames {
		if !shardNames[name] {
			return fmt.Errorf("%w: Engram extent %s is not a manifest shard", ErrV41ArtifactProfile, name)
		}
	}
	if len(p.EngramLayerIDs) == 0 || len(p.EngramLayerIDs) != len(p.EngramNumEmbeddings) {
		return fmt.Errorf("%w: Engram layers %d vs embeddings %d must be equal nonzero counts", ErrV41ArtifactProfile, len(p.EngramLayerIDs), len(p.EngramNumEmbeddings))
	}
	for _, n := range p.EngramNumEmbeddings {
		if n <= 0 {
			return fmt.Errorf("%w: nonpositive Engram embedding extent %d", ErrV41ArtifactProfile, n)
		}
	}
	d := &p.DSpark
	if d.NextNPredictLayers <= 0 || d.BlockSize <= 0 || len(d.TargetLayerIDs) == 0 ||
		d.MarkovRank <= 0 || d.NumRoutedExperts <= 0 || d.NumExpertsPerTok <= 0 || d.NumExpertsPerTok > d.NumRoutedExperts {
		return fmt.Errorf("%w: incoherent DSpark intake %+v", ErrV41ArtifactProfile, *d)
	}
	sel := &p.Draft
	switch sel.Status {
	case V41DraftStatusUnavailable:
		// The unavailable fixture still pins the inspected storage whose gap
		// was witnessed, so a later re-inspection can diff against it.
		if strings.TrimSpace(sel.Reason) == "" || strings.TrimSpace(sel.Repository) == "" || !v41LowerHex(sel.Revision, 40) {
			return fmt.Errorf("%w: unavailable draft selection lacks inspected-storage identity or reason (%+v)", ErrV41ArtifactProfile, *sel)
		}
	case V41DraftStatusSelected:
		if strings.TrimSpace(sel.Repository) == "" || !v41LowerHex(sel.Revision, 40) {
			return fmt.Errorf("%w: selected draft lacks repository/revision (%+v)", ErrV41ArtifactProfile, *sel)
		}
	default:
		return fmt.Errorf("%w: unknown draft status %q", ErrV41ArtifactProfile, sel.Status)
	}
	return nil
}

// AdmitV41ArtifactAgainstProfile validates a presented weight-shard set
// against the profile's pinned manifest: every pinned shard present with the
// exact name, content identity and size, and nothing else. Presentation order
// is irrelevant. A same-name/same-size shard from another release (for example
// the pre-1017 Engram shards under the 1017 profile) fails closed because its
// content identity differs even though its shape matches.
func AdmitV41ArtifactAgainstProfile(p *DeepSeekV41ReleaseArtifactProfile, presented []V41FilePin) error {
	if err := p.Validate(); err != nil {
		return err
	}
	if len(presented) != len(p.WeightShards) {
		return fmt.Errorf("%w: presented %d shards, manifest pins %d", ErrV41ArtifactProfile, len(presented), len(p.WeightShards))
	}
	seen := make(map[string]V41FilePin, len(presented))
	for _, s := range presented {
		if _, dup := seen[s.Name]; dup {
			return fmt.Errorf("%w: duplicate presented shard %s", ErrV41ArtifactProfile, s.Name)
		}
		seen[s.Name] = s
	}
	for _, want := range p.WeightShards {
		got, ok := seen[want.Name]
		if !ok {
			return fmt.Errorf("%w: missing shard %s", ErrV41ArtifactProfile, want.Name)
		}
		if got.ID != want.ID {
			return fmt.Errorf("%w: shard %s content identity mismatch: got %s want %s", ErrV41ArtifactProfile, want.Name, got.ID, want.ID)
		}
		if got.Size != want.Size {
			return fmt.Errorf("%w: shard %s size mismatch: got %d want %d", ErrV41ArtifactProfile, want.Name, got.Size, want.Size)
		}
	}
	return nil
}

// ReleaseProvenance returns the stable cache/trace provenance tag distinguishing
// this release from every other: repository@revision. Two profiles of the same
// family differ on this tag whenever their target revisions differ, so caches
// and traces keyed by it cannot alias across releases.
func (p *DeepSeekV41ReleaseArtifactProfile) ReleaseProvenance() string {
	return p.TargetRepository + "@" + p.TargetRevision
}

// ProvenanceDigest hashes the profile's complete pinned content (manifest,
// metadata, Engram extents, DSpark intake, draft selection) into a lowercase
// SHA-256 hex over the canonical JSON encoding. It distinguishes releases even
// when a repository/revision pair is reused or a single pin drifts, and is the
// value trace/cache consumers key on for release separation.
func (p *DeepSeekV41ReleaseArtifactProfile) ProvenanceDigest() (string, error) {
	if err := p.Validate(); err != nil {
		return "", err
	}
	raw, err := json.Marshal(p)
	if err != nil {
		return "", fmt.Errorf("%w: canonicalize: %w", ErrV41ArtifactProfile, err)
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:]), nil
}

// DeepSeekV41Flash1017Artifact is the pinned 1017 release profile. The weight
// manifest and metadata identities were read from the official registry API at
// revision 1bae0e10... (?blobs=true, observed 2026-10-29); the raw config.json
// bytes hash to SHA-256 8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879,
// matching the retained V41Placement1017ConfigSHA256 constant. All 48 shards
// changed SHA-256 versus the pre-1017 release dba1be0a..., including the
// same-name/same-size Engram shards 47/48, while config.json, the tokenizer
// files and the safetensors index are byte-identical across the two releases
// (same registry blob ids). The DSpark intake values are read from the pinned
// config.json text_config. No full model load or hardware performance is
// claimed; this is header/metadata intake only.
var DeepSeekV41Flash1017Artifact = DeepSeekV41ReleaseArtifactProfile{
	SchemaVersion:    1,
	TargetRepository: DeepSeekV41Flash1017ModelID,
	TargetRevision:   DeepSeekV41Flash1017Revision,
	WeightShards: []V41FilePin{
		{Name: "model-00001-of-00048.safetensors", ID: "13bd121075f0b68cb7e8e8021a1011d42a81a83f557a2602fd9c07d52fddb0ec", Size: 970533624, LFS: true},
		{Name: "model-00002-of-00048.safetensors", ID: "43616eebd88dd4f1f1b642772356fc160478d1d9d04edd240431e7d065d2eda7", Size: 1323858272, LFS: true},
		{Name: "model-00003-of-00048.safetensors", ID: "41611994fc9b27829b757d72257570d65a976fe65b8c6df8641adcd139d65ded", Size: 7389759032, LFS: true},
		{Name: "model-00004-of-00048.safetensors", ID: "e8e901b3cb1ab46f32729c7eaab37bd774a09094f0a02c51a60e883f779f2c39", Size: 7389759032, LFS: true},
		{Name: "model-00005-of-00048.safetensors", ID: "c5df71c02d6a22bcb20a0dbbcda9c6b65aa104f44415102a9a6a03bee92bb9de", Size: 7405953784, LFS: true},
		{Name: "model-00006-of-00048.safetensors", ID: "a5d42afde2aee33619fb7a674b4c737c38d1f1d47a636a26f3edfb3b54164167", Size: 7389759032, LFS: true},
		{Name: "model-00007-of-00048.safetensors", ID: "24856942add8f73f5e1fbaea1e9c9b6ff597fd1b4aa4651d789833dd3a5a2ab9", Size: 7389759032, LFS: true},
		{Name: "model-00008-of-00048.safetensors", ID: "7bf5f7afb5d2bacad49feffb049989fb365d6dbf96986e76265bd7a7a3469564", Size: 7389759032, LFS: true},
		{Name: "model-00009-of-00048.safetensors", ID: "e6ae47e9daca138fda8c04f833c8328eb21671c37d1c8e2c3f7a82aac26649f6", Size: 7389759032, LFS: true},
		{Name: "model-00010-of-00048.safetensors", ID: "835e9a8b0a28cc571cb1cf9e6014b3d5eb84ac41990a8aa4519e1ac435d7e23c", Size: 7389759032, LFS: true},
		{Name: "model-00011-of-00048.safetensors", ID: "5df968edefe59033e451ff8512bef57da2cf97b57b4e7a1380118489dd02915b", Size: 7405953784, LFS: true},
		{Name: "model-00012-of-00048.safetensors", ID: "4f97bf292f58178ecdf593b55252a401d6e9102f18625a4fccc52b8ee57e7be0", Size: 7389759032, LFS: true},
		{Name: "model-00013-of-00048.safetensors", ID: "1b8567854887396840eb5392ca763101308064a142cb36156159096e581659eb", Size: 7389761368, LFS: true},
		{Name: "model-00014-of-00048.safetensors", ID: "e5ba6d7e4380728ad70b8208676075a898f80a64c3618060f1b81e4218a159d1", Size: 7389761368, LFS: true},
		{Name: "model-00015-of-00048.safetensors", ID: "f4b29d936024e68f44fb14d0ffffb15b42b27be04b4a0296acdae695afa94c30", Size: 7389761368, LFS: true},
		{Name: "model-00016-of-00048.safetensors", ID: "3aa67228351c80e708dab4ac7f6e56b19000649cce2cd655def3168a3de4115c", Size: 7389761368, LFS: true},
		{Name: "model-00017-of-00048.safetensors", ID: "b704eec7b7aa1bad6c9d57fa7f61f2db28862e8e5fc2468fd9d872689c778c7d", Size: 7405956128, LFS: true},
		{Name: "model-00018-of-00048.safetensors", ID: "1405cf03376739f46e72e6cdd4d404969f1b94c717bb01782a300e2fbbd17e29", Size: 7389761368, LFS: true},
		{Name: "model-00019-of-00048.safetensors", ID: "a84beaded1913e8d6119ac111f25040c8ec42e59f200dbe04c3a991cac512669", Size: 7389761368, LFS: true},
		{Name: "model-00020-of-00048.safetensors", ID: "038914f79cf5499da90f0bfb8e1b7761cc37d5f4dd132e2431a6d02c9fb9fdfe", Size: 7389761368, LFS: true},
		{Name: "model-00021-of-00048.safetensors", ID: "2ba49f59c8cec144e41370f9f820f268899d0b5cd0501e881908ca65bccd2dc2", Size: 7389761368, LFS: true},
		{Name: "model-00022-of-00048.safetensors", ID: "52f12cceca5e2fdd0191dfbd33b808b6c7ce844cf9351c119413c5ba3bcd28c8", Size: 7389761368, LFS: true},
		{Name: "model-00023-of-00048.safetensors", ID: "d6e47e8b7147c64eb6619732694dda811e1a9f8af1894977919af9522724ca31", Size: 7400713088, LFS: true},
		{Name: "model-00024-of-00048.safetensors", ID: "9d611d65374d37f5e02fdae14e680cb69bf55af8a8478802ac465b97c65be22b", Size: 7389761368, LFS: true},
		{Name: "model-00025-of-00048.safetensors", ID: "8db532a7cbb23cd4b5aa775636ca13af95b758501290fb935074348f660a29cd", Size: 7389761368, LFS: true},
		{Name: "model-00026-of-00048.safetensors", ID: "2d6d7261937d2a98c7398a55ed48dbf129d9b4f188aa9148e46ea9f720255115", Size: 7389761368, LFS: true},
		{Name: "model-00027-of-00048.safetensors", ID: "8e76de6a5cdc7ac0923526d1c1a2e6459c55bbfa692793b29cdfe07b4f9fbc52", Size: 7395337384, LFS: true},
		{Name: "model-00028-of-00048.safetensors", ID: "3c0e5ad6a4e8e811c1a55fd8f5b59c9fa329b241c671997c28fc7a01944d5a77", Size: 7389761368, LFS: true},
		{Name: "model-00029-of-00048.safetensors", ID: "392a950bcb062b89f683abc52d49a17e92c66f80cd1cb6fb92b71091da75319b", Size: 7389761368, LFS: true},
		{Name: "model-00030-of-00048.safetensors", ID: "0b44d1364e578fc26dc0cc86564e49a92871a51063fb7f2f3b51cb020de44ab0", Size: 7389761368, LFS: true},
		{Name: "model-00031-of-00048.safetensors", ID: "d3230ffc32e11c5a34505c5acaeaff642aed5c770b297695e1f7a94071d361ac", Size: 7395337384, LFS: true},
		{Name: "model-00032-of-00048.safetensors", ID: "96d284016ac885415859346d971c0de43bd9e7947337edfb53c34462866ef1d7", Size: 7389761368, LFS: true},
		{Name: "model-00033-of-00048.safetensors", ID: "313953b3b6aaa6d4c60bbcfc177dda5d8d1380c9fe72d82560ba078f55278a8d", Size: 7389761368, LFS: true},
		{Name: "model-00034-of-00048.safetensors", ID: "98eb27655774796bb3feadb35ec64a9849104b88c4c6ae7cabf6fc1ee317a056", Size: 7389761368, LFS: true},
		{Name: "model-00035-of-00048.safetensors", ID: "4a321a28c048389039e4595641fe0b1f130428c61fc8f655545afe0083d37cd9", Size: 7395337384, LFS: true},
		{Name: "model-00036-of-00048.safetensors", ID: "97c7910a8a14002dd941f45f6ddb2bd335eaf6fa05f4a76b2ff9285d6557b65d", Size: 7389761368, LFS: true},
		{Name: "model-00037-of-00048.safetensors", ID: "d37ee013b805344619536e299c73adf92df03744aa0a242f2c5465590015e5be", Size: 7389761368, LFS: true},
		{Name: "model-00038-of-00048.safetensors", ID: "7713bf7a78319eb1c6214b312e0a2da8cac88d77dce851d50f1113d78f8975a4", Size: 7389761368, LFS: true},
		{Name: "model-00039-of-00048.safetensors", ID: "658a77d47d217995b9b78612e69ee0d0e33f6c2240f4ec629fecdca3ecb03f4c", Size: 7395337384, LFS: true},
		{Name: "model-00040-of-00048.safetensors", ID: "6a9dbcc8d5894ba614b5e962764f45f6d0df7f14f1287b17f173d1f2cd762186", Size: 7389761368, LFS: true},
		{Name: "model-00041-of-00048.safetensors", ID: "458308f08ed29b34850bd6b9d10bf769620729d3725c77f7bc0c1d14e01b47cc", Size: 7389761368, LFS: true},
		{Name: "model-00042-of-00048.safetensors", ID: "3d283a4ed6eb2d5013bacc97c3500361926a3261957943434513304df52adfd7", Size: 7389761368, LFS: true},
		{Name: "model-00043-of-00048.safetensors", ID: "4ae12b941c3facf8616ce62dd45bc6868076c2aeb82106f98123ea6c5bb64182", Size: 1323837624, LFS: true},
		{Name: "model-00044-of-00048.safetensors", ID: "3e8ac315c2bcca3e83c5e6e25c5758a75c03732063a6ff0c1b0e24b9ff49f6e6", Size: 2652728736, LFS: true},
		{Name: "model-00045-of-00048.safetensors", ID: "9c73b27c4343236790008979d5e0b565f687aa64440031b9e096fed1ab8039c5", Size: 2573998176, LFS: true},
		{Name: "model-00046-of-00048.safetensors", ID: "1dccc19a3741972c661de734572dc7d4b899e6e99bfbdf3ae4e057c707e5b339", Size: 2706402896, LFS: true},
		{Name: "model-00047-of-00048.safetensors", ID: "76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5", Size: 101535150936, LFS: true},
		{Name: "model-00048-of-00048.safetensors", ID: "9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd", Size: 101537926640, LFS: true},
	},
	Metadata: []V41FilePin{
		{Name: "config.json", ID: "09917a9139b22d5bf8be52132787f435147b1020", Size: 3311, LFS: false},
		{Name: "tokenizer.json", ID: "6a15814dd25c934028034531da744c689f87ff21", Size: 6367257, LFS: false},
		{Name: "tokenizer_config.json", ID: "f3dad388a2bbfd6a8605bd02754acd86d9ca5112", Size: 801, LFS: false},
		{Name: "model.safetensors.index.json", ID: "54c85064dd92c8550471302e9ae59bedbbf96ca4", Size: 7470294, LFS: false},
	},
	EngramShardNames:    []string{"model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors"},
	EngramLayerIDs:      []int{1, 14},
	EngramNumEmbeddings: []int{384006168, 384016682},
	DSpark: V41DSparkIntake{
		NextNPredictLayers: 3,
		BlockSize:          5,
		TargetLayerIDs:     []int{37, 38, 39},
		MarkovRank:         256,
		NumRoutedExperts:   128,
		NumExpertsPerTok:   3,
	},
	Draft: V41DraftSelection{
		Status:     V41DraftStatusUnavailable,
		Repository: "mrfakename/DeepSeek-V4.1-Flash-1017-GGUF",
		Revision:   "1df54e7d1b1e442051ad393ce871870a929f243e",
		Reason:     "inspected Q4_K_M tree carries 1046 tensors with no mtp/nextn or block-40+ tensors, so this storage revision cannot supply the bundled DSpark draft; a separately selected draft must name its own repository and revision",
	},
}
