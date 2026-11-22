package newmodel

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/hfhub"
)

// HFCensusSchema stamps provenance.json written from a header-only census.
const HFCensusSchema = "fak.hf-header-census/1"

// MaxSafetensorsHeaderBytes caps one shard's JSON header; a larger length
// prefix means a corrupt or hostile file, never a real checkpoint.
const MaxSafetensorsHeaderBytes int64 = 100 << 20

// maxHFSmallFileBytes caps config.json and the shard index.
const maxHFSmallFileBytes = 16 << 20

const (
	hfConfigFile      = "config.json"
	hfIndexFile       = "model.safetensors.index.json"
	hfSingleShardFile = "model.safetensors"
)

// HFRangeSource reads pinned repo files. ReadRange returns exactly n bytes at
// off plus the file's total size.
type HFRangeSource interface {
	ReadRange(ctx context.Context, file string, off, n int64) ([]byte, int64, error)
	ReadFile(ctx context.Context, file string) ([]byte, error)
}

// HubRangeSource reads from a Hugging Face Hub (or mirror) by HTTP Range so a
// census never downloads tensor payloads.
type HubRangeSource struct {
	Client *hfhub.Client
	Ref    hfhub.Ref
}

func (s HubRangeSource) base() string {
	if s.Client.BaseURL == "" {
		return hfhub.DefaultBaseURL
	}
	return s.Client.BaseURL
}

func (s HubRangeSource) do(ctx context.Context, url, rangeHeader string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	if rangeHeader != "" {
		req.Header.Set("Range", rangeHeader)
	}
	if s.Client.Token != "" {
		req.Header.Set("Authorization", "Bearer "+s.Client.Token)
	}
	hc := s.Client.HTTP
	if hc == nil {
		hc = http.DefaultClient
	}
	return hc.Do(req)
}

func (s HubRangeSource) get(ctx context.Context, file, rangeHeader string) (*http.Response, error) {
	ref := hfhub.Ref{Repo: s.Ref.Repo, Revision: s.Ref.Revision, File: file}
	resp, err := s.do(ctx, ref.ResolveURL(s.base()), rangeHeader)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode == http.StatusNotFound {
		resp.Body.Close()
		return nil, fmt.Errorf("%s: %w", file, os.ErrNotExist)
	}
	return resp, nil
}

// ReadRange refuses any response other than 206 so a server that ignores
// Range cannot turn a header read into a full shard download.
func (s HubRangeSource) ReadRange(ctx context.Context, file string, off, n int64) ([]byte, int64, error) {
	resp, err := s.get(ctx, file, fmt.Sprintf("bytes=%d-%d", off, off+n-1))
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusPartialContent {
		return nil, 0, fmt.Errorf("%s: range read got HTTP %d, want 206", file, resp.StatusCode)
	}
	total, err := contentRangeTotal(resp.Header.Get("Content-Range"))
	if err != nil {
		return nil, 0, fmt.Errorf("%s: %w", file, err)
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, n+1))
	if err != nil {
		return nil, 0, err
	}
	if int64(len(data)) != n {
		return nil, 0, fmt.Errorf("%s: range read returned %d bytes, want %d", file, len(data), n)
	}
	return data, total, nil
}

func contentRangeTotal(h string) (int64, error) {
	_, total, ok := strings.Cut(h, "/")
	if !ok || !strings.HasPrefix(h, "bytes ") {
		return 0, fmt.Errorf("malformed Content-Range %q", h)
	}
	n, err := strconv.ParseInt(total, 10, 64)
	if err != nil || n < 0 {
		return 0, fmt.Errorf("Content-Range %q has no total size", h)
	}
	return n, nil
}

func (s HubRangeSource) ReadFile(ctx context.Context, file string) ([]byte, error) {
	resp, err := s.get(ctx, file, "")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s: HTTP %d", file, resp.StatusCode)
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxHFSmallFileBytes+1))
	if err != nil {
		return nil, err
	}
	if len(data) > maxHFSmallFileBytes {
		return nil, fmt.Errorf("%s: larger than %d bytes", file, maxHFSmallFileBytes)
	}
	return data, nil
}

// HFFileLister lists a pinned revision's repo-relative file names, so files
// the shard index does not name can still be censused.
type HFFileLister interface {
	ListFiles(ctx context.Context) ([]string, error)
}

// HFListingStatusError is a file listing answered with a non-200 status.
type HFListingStatusError struct{ Status int }

func (e *HFListingStatusError) Error() string {
	return fmt.Sprintf("file listing got HTTP %d", e.Status)
}

// ListFiles reads the Hub revision listing's siblings.
func (s HubRangeSource) ListFiles(ctx context.Context) ([]string, error) {
	resp, err := s.do(ctx, s.base()+"/api/models/"+s.Ref.Repo+"/revision/"+s.Ref.Revision, "")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, &HFListingStatusError{Status: resp.StatusCode}
	}
	var info struct {
		Siblings []struct {
			Rfilename string `json:"rfilename"`
		} `json:"siblings"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, maxHFSmallFileBytes)).Decode(&info); err != nil {
		return nil, fmt.Errorf("file listing: %w", err)
	}
	files := make([]string, 0, len(info.Siblings))
	for _, sib := range info.Siblings {
		files = append(files, sib.Rfilename)
	}
	return files, nil
}

// ParseHFPin parses "<owner>/<repo>@<40-hex commit>". Branches and tags are
// refused so the emitted spine is reproducible from its recorded revision.
func ParseHFPin(arg string) (hfhub.Ref, error) {
	ref, err := hfhub.ParseURI("hf://" + arg)
	if err != nil || ref.File != "" || !strings.Contains(arg, "@") {
		return hfhub.Ref{}, &Refusal{Schema: RefusalSchema, Reason: RefusalPinInvalid, Phase: PhaseHeaderPreflight, Axis: "hf", Detail: fmt.Sprintf("%q must be <owner>/<repo>@<40-hex commit>", arg)}
	}
	if !isHexLen(strings.ToLower(ref.Revision), 40) {
		return hfhub.Ref{}, &Refusal{Schema: RefusalSchema, Reason: RefusalPinInvalid, Phase: PhaseHeaderPreflight, Axis: "hf.revision", Detail: fmt.Sprintf("%q is not a 40-hex commit", ref.Revision)}
	}
	return ref, nil
}

type HFTensor struct {
	Shape []int  `json:"shape"`
	DType string `json:"dtype"`
}

type HFShardHeader struct {
	File         string `json:"file"`
	ShardBytes   int64  `json:"shard_bytes"`
	HeaderBytes  int64  `json:"header_bytes"`
	HeaderSHA256 string `json:"header_sha256"`
}

// HFCensus is a header-only view of a pinned safetensors checkpoint. Config
// holds the upstream config.json bytes verbatim. Shards lists the main
// weights first, then root-level sidecar safetensors the index does not name,
// whose tensors are keyed "<file stem>:<name>". SidecarListing is the HTTP
// status of a failed file listing, so a skipped sidecar census stays visible.
type HFCensus struct {
	Repo, Revision string
	Config         []byte
	HasIndex       bool
	Shards         []HFShardHeader
	Tensors        map[string]HFTensor
	SidecarListing int
}

func headerRefusal(reason RefusalReason, axis, format string, args ...any) error {
	r := HeaderRefusal(reason, axis, fmt.Sprintf(format, args...))
	return &r
}

// CensusHF reads config.json, the shard index (or the single
// model.safetensors) and every shard's safetensors header, and refuses a
// weight_map that disagrees with the headers.
func CensusHF(ctx context.Context, src HFRangeSource, repo, rev string) (*HFCensus, error) {
	cfg, err := src.ReadFile(ctx, hfConfigFile)
	if err != nil {
		return nil, headerRefusal(RefusalHeaderConfigInvalid, hfConfigFile, "%v", err)
	}
	var probe map[string]json.RawMessage
	if err := json.Unmarshal(cfg, &probe); err != nil {
		return nil, headerRefusal(RefusalHeaderConfigInvalid, hfConfigFile, "not a JSON object: %v", err)
	}

	census := &HFCensus{Repo: repo, Revision: rev, Config: cfg, Tensors: map[string]HFTensor{}}
	var weightMap map[string]string
	shardFiles := []string{hfSingleShardFile}
	index, err := src.ReadFile(ctx, hfIndexFile)
	switch {
	case errors.Is(err, os.ErrNotExist):
	case err != nil:
		return nil, headerRefusal(RefusalHeaderConfigInvalid, hfIndexFile, "%v", err)
	default:
		var idx struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if err := json.Unmarshal(index, &idx); err != nil || len(idx.WeightMap) == 0 {
			return nil, headerRefusal(RefusalHeaderConfigInvalid, hfIndexFile, "weight_map is missing or empty (%v)", err)
		}
		weightMap = idx.WeightMap
		census.HasIndex = true
		seen := map[string]bool{}
		for _, shard := range weightMap {
			seen[shard] = true
		}
		shardFiles = sortedKeys(seen)
	}

	owner := map[string]string{}
	absent := "single-file weights absent and the repo has no sharded weight map"
	if census.HasIndex {
		absent = "shard named by the index is absent"
	}
	for _, shard := range shardFiles {
		header, tensors, err := readSafetensorsHeader(ctx, src, shard, absent)
		if err != nil {
			return nil, err
		}
		census.Shards = append(census.Shards, header)
		for name, ts := range tensors {
			if prev, dup := owner[name]; dup {
				return nil, headerRefusal(RefusalHeaderConfigInvalid, name, "tensor appears in both %s and %s", prev, shard)
			}
			owner[name] = shard
			census.Tensors[name] = ts
		}
	}
	if weightMap != nil {
		if err := checkWeightMap(weightMap, owner); err != nil {
			return nil, err
		}
	}
	if err := censusSidecars(ctx, src, census, shardFiles); err != nil {
		return nil, err
	}
	return census, nil
}

// censusSidecars reads the header of every root-level *.safetensors file the
// main weights do not cover, such as an MTP head shipped beside the index.
func censusSidecars(ctx context.Context, src HFRangeSource, census *HFCensus, mainFiles []string) error {
	lister, ok := src.(HFFileLister)
	if !ok {
		return nil
	}
	listed, err := lister.ListFiles(ctx)
	var status *HFListingStatusError
	if errors.As(err, &status) {
		census.SidecarListing = status.Status
		return nil
	}
	if err != nil {
		return err
	}
	covered := map[string]bool{hfSingleShardFile: true}
	for _, f := range mainFiles {
		covered[f] = true
	}
	var sidecars []string
	for _, f := range listed {
		if strings.HasSuffix(f, ".safetensors") && !strings.Contains(f, "/") && !covered[f] {
			covered[f] = true
			sidecars = append(sidecars, f)
		}
	}
	sort.Strings(sidecars)
	for _, file := range sidecars {
		header, tensors, err := readSafetensorsHeader(ctx, src, file, "sidecar named by the file listing is absent")
		if err != nil {
			return err
		}
		census.Shards = append(census.Shards, header)
		stem := strings.TrimSuffix(file, ".safetensors")
		for name, ts := range tensors {
			key := stem + ":" + name
			if _, dup := census.Tensors[key]; dup {
				return headerRefusal(RefusalHeaderConfigInvalid, key, "sidecar tensor name collides with an existing tensor")
			}
			census.Tensors[key] = ts
		}
	}
	return nil
}

func checkWeightMap(weightMap, owner map[string]string) error {
	var missing, orphans []string
	for name, shard := range weightMap {
		if owner[name] != shard {
			missing = append(missing, name+" ("+shard+")")
		}
	}
	for name := range owner {
		if _, ok := weightMap[name]; !ok {
			orphans = append(orphans, name)
		}
	}
	sort.Strings(missing)
	sort.Strings(orphans)
	if len(missing) > 0 {
		return headerRefusal(RefusalTensorMissing, hfIndexFile, "%d weight_map tensors absent from their shard header: %s", len(missing), strings.Join(missing, ", "))
	}
	if len(orphans) > 0 {
		return headerRefusal(RefusalTensorOrphan, hfIndexFile, "%d header tensors absent from weight_map: %s", len(orphans), strings.Join(orphans, ", "))
	}
	return nil
}

func readSafetensorsHeader(ctx context.Context, src HFRangeSource, shard, absent string) (HFShardHeader, map[string]HFTensor, error) {
	prefix, total, err := src.ReadRange(ctx, shard, 0, 8)
	if errors.Is(err, os.ErrNotExist) {
		return HFShardHeader{}, nil, headerRefusal(RefusalTensorMissing, shard, "%s: %v", absent, err)
	}
	if err != nil {
		return HFShardHeader{}, nil, err
	}
	n := binary.LittleEndian.Uint64(prefix)
	if n == 0 || n > uint64(MaxSafetensorsHeaderBytes) || int64(n) > total-8 {
		return HFShardHeader{}, nil, headerRefusal(RefusalHeaderConfigInvalid, shard, "safetensors header length %d is outside (0, min(%d, %d)]", n, MaxSafetensorsHeaderBytes, total-8)
	}
	raw, _, err := src.ReadRange(ctx, shard, 8, int64(n))
	if err != nil {
		return HFShardHeader{}, nil, err
	}
	var entries map[string]json.RawMessage
	if err := json.Unmarshal(raw, &entries); err != nil {
		return HFShardHeader{}, nil, headerRefusal(RefusalHeaderConfigInvalid, shard, "safetensors header is not a JSON object: %v", err)
	}
	tensors := make(map[string]HFTensor, len(entries))
	for name, entry := range entries {
		if name == "__metadata__" {
			continue
		}
		var ts HFTensor
		if err := json.Unmarshal(entry, &ts); err != nil || ts.DType == "" || ts.Shape == nil {
			return HFShardHeader{}, nil, headerRefusal(RefusalHeaderConfigInvalid, shard+":"+name, "tensor entry lacks dtype or shape")
		}
		tensors[name] = ts
	}
	sum := sha256.Sum256(raw)
	return HFShardHeader{File: shard, ShardBytes: total, HeaderBytes: int64(n), HeaderSHA256: hex.EncodeToString(sum[:])}, tensors, nil
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
