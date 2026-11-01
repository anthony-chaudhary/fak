package model

// SPDX-License-Identifier: Apache-2.0
// State-spec planning adapted from Tokenomos.
// Copyright 2026 Shih-Ying Yeh (aka KohakuBlueLeaf, KBlueLeaf)
// Revision eeb47ee09a3c5e66411e90313b017c211d6a3fbb:
// src/tokenomos/models/contract.py and src/tokenomos/core/pools.py.
// Modified for FAK: checked native dimensions and explicit dtype; retain FAK's
// host row ownership and HAL convolution order, not the donor's GPU slot pools
// or transposed convolution view. The Apache-2.0 license is in root LICENSE.

import (
	"fmt"
	"strings"
)

// Qwen35StateLayout describes one layer's convolution history and recurrent
// matrix before allocation. Byte counts cover payload only, not allocator or
// slice metadata. A valid layout proves addressable arithmetic, not memory fit.
// Consumers must retain their own admission, ownership and lifecycle checks.
type Qwen35StateLayout struct {
	StateDtype           string
	ElementBytes         int
	ConvRows             int
	ConvDim              int
	ConvElements         int
	RecurrentHeads       int
	KeyHeadDim           int
	ValueHeadDim         int
	RecurrentRowElements int
	RecurrentElements    int
	ConvBytes            int64
	RecurrentBytes       int64
	BytesPerLayer        int64
}

// ComputeQwen35StateLayout derives the immutable per-layer plan without
// allocating state. Empty dtype means f32; f16/bf16 are accounting contracts,
// not permission for an F32 consumer to change its realized storage.
func ComputeQwen35StateLayout(cfg Config, stateDtype string) (Qwen35StateLayout, error) {
	dtype := strings.ToLower(strings.TrimSpace(stateDtype))
	if dtype == "" {
		dtype = "f32"
	}
	elementBytes := 4
	switch dtype {
	case "f32":
	case "f16", "bf16":
		elementBytes = 2
	default:
		return Qwen35StateLayout{}, fmt.Errorf("model: qwen35 state layout: unsupported dtype %q", stateDtype)
	}
	nK, nV := cfg.LinearNumKeyHeads, cfg.LinearNumValueHeads
	kHd, vHd := cfg.LinearKeyHeadDim, cfg.LinearValueHeadDim
	if nK <= 0 || nV <= 0 || kHd <= 0 || vHd <= 0 || cfg.LinearConvKernelDim < 1 {
		return Qwen35StateLayout{}, fmt.Errorf("model: qwen35 state layout: nonpositive head geometry or convolution width")
	}
	maxInt := int(^uint(0) >> 1)
	product := func(a, b int) (int, bool) {
		if a < 0 || b < 0 || (b != 0 && a > maxInt/b) {
			return 0, false
		}
		return a * b, true
	}
	keyDim, keyOK := product(nK, kHd)
	valueDim, valueOK := product(nV, vHd)
	row, rowOK := product(kHd, vHd)
	if !keyOK || !valueOK || !rowOK || keyDim > (maxInt-valueDim)/2 {
		return Qwen35StateLayout{}, fmt.Errorf("model: qwen35 state layout: projection or recurrent row overflow")
	}
	convDim := 2*keyDim + valueDim
	convRows := cfg.LinearConvKernelDim - 1
	conv, convOK := product(convRows, convDim)
	recurrent, recurrentOK := product(nV, row)
	if !convOK || !recurrentOK || conv > maxInt-recurrent || conv+recurrent > maxInt/elementBytes {
		return Qwen35StateLayout{}, fmt.Errorf("model: qwen35 state layout: state payload overflow")
	}
	return Qwen35StateLayout{
		StateDtype: dtype, ElementBytes: elementBytes,
		ConvRows: convRows, ConvDim: convDim, ConvElements: conv,
		RecurrentHeads: nV, KeyHeadDim: kHd, ValueHeadDim: vHd,
		RecurrentRowElements: row, RecurrentElements: recurrent,
		ConvBytes:      int64(conv) * int64(elementBytes),
		RecurrentBytes: int64(recurrent) * int64(elementBytes),
		BytesPerLayer:  int64(conv+recurrent) * int64(elementBytes),
	}, nil
}
