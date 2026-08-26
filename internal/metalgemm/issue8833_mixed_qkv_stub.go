//go:build !darwin || !arm64 || !cgo

package metalgemm

func executeMixedQKV(selector MixedQKVSelector, in MixedQKVInput) (MixedQKVResult, error) {
	id := nextMixedQKVCallID()
	return MixedQKVResult{CallID: id}, &MixedQKVError{
		CallID: id,
		Stage:  MixedQKVDeclined,
		Detail: "native mixed QKV owner unavailable",
	}
}
