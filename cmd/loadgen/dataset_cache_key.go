// Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from ai-dynamo/aiperf (src/aiperf/dataset/mmap_cache.py:330@ab610add4e88):
// content-addressed dataset cache key over input bytes, tokenizer identity,
// exhaustive settings payload, and manifest version-bump contract (cache-bust excluded).
package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

// DatasetCacheManifestVersion is bumped whenever the on-disk dataset cache
// schema or serialization changes, invalidating older cached manifests.
const DatasetCacheManifestVersion = 1

// DatasetCacheKeyInput describes all content-addressing inputs for a cached dataset.
// CacheBust is explicitly excluded from the computed key so cache-busting flags do
// not perturb content-addressed identity.
type DatasetCacheKeyInput struct {
	InputBytes      []byte
	TokenizerID     string
	Settings        map[string]any
	ManifestVersion int
	CacheBust       string // intentionally excluded from key derivation
}

// ComputeDatasetCacheKey returns a deterministic SHA-256 hex digest over
// manifest version, input bytes, tokenizer identity, and canonicalized settings.
func ComputeDatasetCacheKey(in DatasetCacheKeyInput) (string, error) {
	ver := in.ManifestVersion
	if ver <= 0 {
		ver = DatasetCacheManifestVersion
	}
	settingsJSON, err := json.Marshal(in.Settings) // Go encoding/json sorts map keys deterministically
	if err != nil {
		return "", fmt.Errorf("loadgen: marshal cache key settings: %w", err)
	}

	h := sha256.New()
	writeLenPrefixed := func(b []byte) {
		var lenBuf [8]byte
		binary.BigEndian.PutUint64(lenBuf[:], uint64(len(b)))
		h.Write(lenBuf[:])
		h.Write(b)
	}

	var verBuf [8]byte
	binary.BigEndian.PutUint64(verBuf[:], uint64(ver))
	h.Write(verBuf[:])
	writeLenPrefixed(in.InputBytes)
	writeLenPrefixed([]byte(in.TokenizerID))
	writeLenPrefixed(settingsJSON)

	return hex.EncodeToString(h.Sum(nil)), nil
}

// DatasetManifestVersionCompatible reports whether a cached manifest's version
// matches the current DatasetCacheManifestVersion.
func DatasetManifestVersionCompatible(storedVersion int) bool {
	return storedVersion == DatasetCacheManifestVersion
}
