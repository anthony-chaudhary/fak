// SPDX-License-Identifier: Apache-2.0
// Draft KV dtype selection adapted from vLLM (Copyright vLLM contributors),
// vllm/config/speculative.py, revision
// 31f84f6b957b5f5309f66130efc3c1df07d8a995. No Python source is copied.

package model

// ResolveDraftKVDtype selects the declared KV precision for a draft model.
// A nil override inherits the target declaration; a non-nil override wins,
// including an explicit empty declaration. For a drafter without a KV cache
// (such as ngram or suffix lookup), it returns the empty declaration and ignores
// both inputs. usesKV describes the actual drafter, not its proposed token count.
//
// This is a configuration component only: it does not allocate a cache, validate
// backend support, or activate a draft model. The eventual cache constructor must
// admit the selected precision. In particular, naming a precision here does not
// establish FP8 support or a compatible shared target/draft cache layout.
func ResolveDraftKVDtype(target KVPrecision, override *KVPrecision, usesKV bool) KVPrecision {
	if !usesKV {
		return ""
	}
	if override != nil {
		return *override
	}
	return target
}
