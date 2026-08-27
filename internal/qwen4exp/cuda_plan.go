package qwen4exp

import "errors"

const CUDAResidencySchema = "fak.qwen4exp.cuda-plan/1"

type CUDAGPU struct {
	ID            string `json:"id"`
	Architecture  string `json:"architecture"`
	PhysicalBytes int64  `json:"physical_bytes"`
	ReservedBytes int64  `json:"reserved_bytes"`
}
type CUDAResidency struct {
	Schema            string          `json:"schema"`
	Artifact          string          `json:"artifact"`
	ArtifactBytes     int64           `json:"artifact_bytes"`
	DType             string          `json:"dtype"`
	GPUs              []CUDAGPU       `json:"gpus"`
	HostOffloadBytes  int64           `json:"host_offload_bytes"`
	HostPhysicalBytes int64           `json:"host_physical_bytes"`
	Ops               map[string]bool `json:"ops"`
	Engine            string          `json:"engine"`
	Fallback          string          `json:"fallback"`
}

func (p CUDAResidency) Validate() error {
	if p.Schema != CUDAResidencySchema {
		return errors.New("qwen4exp cuda: invalid schema")
	}
	if p.Artifact == "" || p.ArtifactBytes <= 0 || p.DType == "" || len(p.GPUs) == 0 {
		return errors.New("qwen4exp cuda: incomplete identity")
	}
	if p.Engine != "fak-native" || p.Fallback != "none" {
		return errors.New("qwen4exp cuda: non-native or fallback plan")
	}
	var available int64
	ids := map[string]bool{}
	for _, g := range p.GPUs {
		if g.ID == "" || ids[g.ID] || g.Architecture == "" || g.PhysicalBytes <= 0 || g.ReservedBytes < 0 || g.ReservedBytes >= g.PhysicalBytes {
			return errors.New("qwen4exp cuda: invalid GPU envelope")
		}
		ids[g.ID] = true
		available += g.PhysicalBytes - g.ReservedBytes
	}
	if p.HostOffloadBytes < 0 || p.HostOffloadBytes > p.HostPhysicalBytes {
		return errors.New("qwen4exp cuda: invalid host offload")
	}
	if available+p.HostOffloadBytes < p.ArtifactBytes {
		return errors.New("qwen4exp cuda: insufficient physical residency")
	}
	for _, op := range []string{"gdn", "qsa_top2048", "sparse_moe", "shared_expert", "ple_ngram"} {
		if !p.Ops[op] {
			return errors.New("qwen4exp cuda: incomplete kernel coverage")
		}
	}
	return nil
}
