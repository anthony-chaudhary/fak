package ggufload

import (
	"errors"
	"os"
)

// errPageCacheResidencyUnsupported reports a platform without a page-cache
// residency query (mincore).
var errPageCacheResidencyUnsupported = errors.New("ggufload: page-cache residency unsupported on this platform")

// Startup cases a PageCacheResidency classifies.
const (
	startupCaseCold = "cold"
	startupCaseWarm = "warm"
)

// pageCacheColdFraction is the cached share of the GGUF bytes below which a start
// reads as cold: a dropped cache measures 0 bytes, an immediate V4.1 restart on
// strix3 kept 3-48 GiB of 246 GiB (>= 1.2%).
const pageCacheColdFraction = 200

// PageCacheResidency is how much of a model's GGUF files the OS page cache held
// when it was measured, before the load read them.
type PageCacheResidency struct {
	ResidentBytes int64 `json:"resident_bytes"`
	FileBytes     int64 `json:"file_bytes"`
	Files         int   `json:"files"`
}

// StartupCase is cold when less than 1/200 of the GGUF bytes were cached, warm
// otherwise, and empty when nothing was measured.
func (r PageCacheResidency) StartupCase() string {
	if r.FileBytes <= 0 {
		return ""
	}
	if r.ResidentBytes*pageCacheColdFraction < r.FileBytes {
		return startupCaseCold
	}
	return startupCaseWarm
}

// ShardFiles lists every file of the GGUF at path: all -N-of-M shards for a split
// model, else path alone.
func ShardFiles(path string) []string {
	count, ok := shardPathCount(path)
	if !ok || count <= 1 {
		return []string{path}
	}
	first, err := firstShardPath(path)
	if err != nil {
		return []string{path}
	}
	paths, err := shardPaths(first, count)
	if err != nil {
		return []string{path}
	}
	return paths
}

// MeasurePageCacheResidency sums the page-cache resident bytes and the sizes of
// paths without reading their contents.
func MeasurePageCacheResidency(paths []string) (PageCacheResidency, error) {
	var out PageCacheResidency
	for _, p := range paths {
		f, err := os.Open(p)
		if err != nil {
			return PageCacheResidency{}, err
		}
		resident, size, err := fileResidentBytes(f)
		_ = f.Close()
		if err != nil {
			return PageCacheResidency{}, err
		}
		out.ResidentBytes += resident
		out.FileBytes += size
		out.Files++
	}
	return out, nil
}
