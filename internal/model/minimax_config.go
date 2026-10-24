// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
)

// ErrMiniMaxConfig identifies malformed or unsupported MiniMax metadata.
var ErrMiniMaxConfig = errors.New("model: unsupported MiniMax config")

type miniMaxSparseConfig struct {
	Enabled      *bool  `json:"use_sparse_attention"`
	HeadDim      int    `json:"sparse_index_dim"`
	Heads        int    `json:"sparse_num_index_heads"`
	TopK         int    `json:"sparse_topk_blocks"`
	BlockSize    int    `json:"sparse_block_size"`
	ScoreType    string `json:"sparse_score_type"`
	InitBlocks   int    `json:"sparse_init_block"`
	LocalBlocks  int    `json:"sparse_local_block"`
	LayerFreq    []int  `json:"sparse_attention_freq"`
	DisableValue []int  `json:"sparse_disable_index_value"`
}

func miniMaxConfigError(field string) error {
	return fmt.Errorf("%w: %s", ErrMiniMaxConfig, field)
}

func (c *Config) decodeMiniMaxConfig(wrapper, text, frequency []byte) error {
	frequency = bytes.TrimSpace(frequency)
	if !c.isMiniMax() {
		if len(frequency) != 0 {
			return json.Unmarshal(frequency, &c.MoELayerFreq)
		}
		return nil
	}
	if len(frequency) != 0 && !bytes.Equal(frequency, []byte("null")) {
		if frequency[0] == '[' {
			var schedule []int
			if err := json.Unmarshal(frequency, &schedule); err != nil || c.NumLayers <= 0 || len(schedule) != c.NumLayers {
				return miniMaxConfigError("moe_layer_freq")
			}
			c.MoELayerFreq = 0
			c.MoELayerSchedule = schedule
		} else {
			if err := json.Unmarshal(frequency, &c.MoELayerFreq); err != nil {
				return miniMaxConfigError("moe_layer_freq")
			}
			c.MoELayerSchedule = nil
		}
	}
	if len(c.MoELayerSchedule) != 0 && len(c.MoELayerSchedule) != c.NumLayers {
		return miniMaxConfigError("moe_layer_schedule")
	}
	var metadata struct {
		Sparse       *miniMaxSparseConfig `json:"sparse_attention_config"`
		UseGemmaNorm *bool                `json:"use_gemma_norm"`
	}
	if err := json.Unmarshal(text, &metadata); err != nil {
		return miniMaxConfigError("text_config")
	}
	if !bytes.Equal(wrapper, text) {
		if err := json.Unmarshal(wrapper, &metadata); err != nil {
			return miniMaxConfigError("config")
		}
	}
	var explicitNorm struct {
		NormGain *bool `json:"norm_gain_1p"`
	}
	_ = json.Unmarshal(text, &explicitNorm)
	_ = json.Unmarshal(wrapper, &explicitNorm)
	if explicitNorm.NormGain != nil {
		c.NormGain1p = *explicitNorm.NormGain
	} else if metadata.UseGemmaNorm != nil {
		c.NormGain1p = *metadata.UseGemmaNorm
	}
	if metadata.Sparse == nil {
		return nil
	}
	return c.applyMiniMaxSparseConfig(*metadata.Sparse)
}

func (c *Config) applyMiniMaxSparseConfig(s miniMaxSparseConfig) error {
	if c.NumLayers <= 0 || len(s.LayerFreq) != c.NumLayers || len(s.DisableValue) != c.NumLayers {
		return miniMaxConfigError("sparse per-layer schedules")
	}
	// Adapted from get_minimax_sparse_layer_ids and
	// get_minimax_sparse_disable_value_layer_ids in SGLang model_config.py:448-459
	// at 7cefa607657b9e63abe6946cb795e77d4d31c9c0 (Apache-2.0).
	layers := make([]string, c.NumLayers)
	hasSparse := false
	for layer, frequency := range s.LayerFreq {
		layers[layer] = "full_attention"
		if frequency != 0 {
			hasSparse = true
			layers[layer] = "minimax_m3_sparse"
			if s.DisableValue[layer] == 0 {
				return miniMaxConfigError("sparse index value projection")
			}
		}
	}
	if hasSparse {
		if s.Enabled != nil && !*s.Enabled {
			return miniMaxConfigError("use_sparse_attention")
		}
		if s.HeadDim <= 0 || s.HeadDim < c.rotaryDim() || s.Heads <= 0 || s.BlockSize <= 0 || s.TopK <= 0 || s.LocalBlocks < 0 || s.LocalBlocks > s.TopK {
			return miniMaxConfigError("sparse geometry")
		}
		if s.ScoreType != "max" || s.InitBlocks != 0 {
			return miniMaxConfigError("sparse score or initial blocks")
		}
	}
	if len(c.LayerTypes) != 0 {
		if len(c.LayerTypes) != len(layers) {
			return miniMaxConfigError("layer_types")
		}
		for layer := range layers {
			if c.LayerTypes[layer] != layers[layer] {
				return miniMaxConfigError("layer_types")
			}
		}
	}
	for _, axis := range []struct{ flat, nested int }{
		{c.IndexHeadDim, s.HeadDim}, {c.IndexNHeads, s.Heads},
		{c.IndexBlockSize, s.BlockSize}, {c.IndexTopKBlocks, s.TopK},
		{c.IndexLocalBlocks, s.LocalBlocks},
	} {
		if axis.flat != 0 && axis.flat != axis.nested {
			return miniMaxConfigError("conflicting sparse geometry")
		}
	}
	c.LayerTypes = layers
	c.IndexHeadDim, c.IndexNHeads = s.HeadDim, s.Heads
	c.IndexBlockSize, c.IndexTopKBlocks, c.IndexLocalBlocks = s.BlockSize, s.TopK, s.LocalBlocks
	return nil
}
