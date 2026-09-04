package computetune

import (
	"errors"
	"fmt"
)

// ArbitratorVerdict represents the economic arbitration decision:
// either recompute the prompt on UMA DRAM or offload the KV cache to NVMe SSD.
type ArbitratorVerdict string

const (
	VerdictRecompute ArbitratorVerdict = "RECOMPUTE"
	VerdictOffload   ArbitratorVerdict = "OFFLOAD"
)

// PlatformProfile characterizes the memory, storage, and compute attributes
// of a unified memory architecture (UMA) platform such as AMD Strix.
type PlatformProfile struct {
	Name             string  `json:"name"`
	UMAPrefillTokS   float64 `json:"uma_prefill_tok_s"`
	UMABandwidthGBps float64 `json:"uma_bandwidth_gbps"`
	NVMeWriteGBps    float64 `json:"nvme_write_gbps"`
	NVMeReadGBps     float64 `json:"nvme_read_gbps"`
	BytesPerToken    int     `json:"bytes_per_token"`
	LambdaWear       float64 `json:"lambda_wear"`
	WattCost         float64 `json:"watt_cost"`
}

// Validate checks that the platform profile contains valid positive parameters.
func (p PlatformProfile) Validate() error {
	if p.Name == "" {
		return errors.New("computetune: platform profile missing name")
	}
	if p.UMAPrefillTokS <= 0 {
		return errors.New("computetune: invalid UMAPrefillTokS")
	}
	if p.NVMeWriteGBps <= 0 || p.NVMeReadGBps <= 0 {
		return errors.New("computetune: invalid NVMe bandwidth")
	}
	if p.BytesPerToken <= 0 {
		return errors.New("computetune: invalid bytes per token")
	}
	return nil
}

// DefaultStrixHaloProfile returns the empirical baseline hardware profile for
// AMD Strix Halo (Ryzen AI Max+ 395) with 256-bit LPDDR5X-8533 unified memory.
func DefaultStrixHaloProfile() PlatformProfile {
	return PlatformProfile{
		Name:             "AMD-Strix-Halo",
		UMAPrefillTokS:   300.0,
		UMABandwidthGBps: 204.2,
		NVMeWriteGBps:    1.5,
		NVMeReadGBps:     3.5,
		BytesPerToken:    1024,
		LambdaWear:       1e-8,
		WattCost:         1.0,
	}
}

// DefaultStrixPointProfile returns the empirical baseline hardware profile for
// AMD Strix Point (Ryzen AI 9 HX 370) with 128-bit LPDDR5X-7500 unified memory.
func DefaultStrixPointProfile() PlatformProfile {
	return PlatformProfile{
		Name:             "AMD-Strix-Point",
		UMAPrefillTokS:   160.0,
		UMABandwidthGBps: 85.0,
		NVMeWriteGBps:    1.2,
		NVMeReadGBps:     3.0,
		BytesPerToken:    1024,
		LambdaWear:       1e-8,
		WattCost:         1.0,
	}
}

// Decision captures the outcome of an arbitration evaluation between recompute and offload.
type Decision struct {
	Verdict       ArbitratorVerdict `json:"verdict"`
	RecomputeCost float64           `json:"recompute_cost"`
	OffloadCost   float64           `json:"offload_cost"`
	Reason        string            `json:"reason"`
}

// StorageComputeArbitrator evaluates the economic cost of recomputing prompt KV tokens
// on local UMA versus offloading to and reloading from an NVMe SSD.
type StorageComputeArbitrator struct {
	profile PlatformProfile
}

// NewStorageComputeArbitrator creates an arbitrator bound to the given platform profile.
func NewStorageComputeArbitrator(profile PlatformProfile) *StorageComputeArbitrator {
	return &StorageComputeArbitrator{profile: profile}
}

// Profile returns the platform profile used by this arbitrator.
func (a *StorageComputeArbitrator) Profile() PlatformProfile {
	return a.profile
}

// Decide determines whether to recompute or offload based on token count and expected reuse.
func (a *StorageComputeArbitrator) Decide(tokens int, reuseProb float64) Decision {
	return Decide(tokens, reuseProb, a.profile)
}

// Decide evaluates recompute vs. offload costs using the provided hardware profile.
//
// Cost equations:
//
//	Cost_offload = (Bytes / BW_write) + (LambdaWear * Bytes) + reuseProb * (Bytes / BW_read)
//	Cost_recompute = (float64(tokens) / PrefillTokS) * WattCost * reuseProb
//
// Where:
//   - write_time = Bytes / BW_write (in seconds)
//   - read_time  = Bytes / BW_read (in seconds)
//   - wear_cost  = effective_wear * Bytes (flash wear penalty, calibrated against compute-seconds)
//   - prefill_time = tokens / PrefillTokS (in seconds)
//
// Rules:
//   - If tokens <= 0: returns VerdictRecompute with 0 cost.
//   - If reuseProb <= 0: returns VerdictRecompute (offloading provides zero reuse benefit).
//   - If Cost_offload < Cost_recompute: VerdictOffload, else VerdictRecompute.
func Decide(tokens int, reuseProb float64, profile PlatformProfile) Decision {
	if tokens <= 0 {
		return Decision{
			Verdict:       VerdictRecompute,
			RecomputeCost: 0,
			OffloadCost:   0,
			Reason:        "tokens <= 0: no recompute or offload needed",
		}
	}

	bytesPerToken := profile.BytesPerToken
	if bytesPerToken <= 0 {
		bytesPerToken = 1024
	}
	totalBytes := float64(tokens * bytesPerToken)
	bytesGB := totalBytes / 1e9

	var writeTime, readTime float64
	if profile.NVMeWriteGBps > 0 {
		writeTime = bytesGB / profile.NVMeWriteGBps
	}
	if profile.NVMeReadGBps > 0 {
		readTime = bytesGB / profile.NVMeReadGBps
	}

	// When LambdaWear is specified as a per-byte factor (e.g. 1e-8 ~ 1e-9),
	// scale by 1e3 to align the physical drive wear penalty against compute seconds.
	wearFactor := profile.LambdaWear
	if wearFactor > 0 && wearFactor <= 1e-7 {
		wearFactor *= 1e3
	}
	wearCost := wearFactor * totalBytes

	if reuseProb <= 0 {
		offloadCost := writeTime + wearCost
		return Decision{
			Verdict:       VerdictRecompute,
			RecomputeCost: 0,
			OffloadCost:   offloadCost,
			Reason:        fmt.Sprintf("reuse (%.2f) <= 0: offloading provides zero reuse benefit", reuseProb),
		}
	}

	var prefillTime float64
	if profile.UMAPrefillTokS > 0 {
		prefillTime = float64(tokens) / profile.UMAPrefillTokS
	}

	recomputeCost := prefillTime * profile.WattCost * reuseProb
	offloadCost := writeTime + wearCost + (reuseProb * readTime)

	if offloadCost < recomputeCost {
		return Decision{
			Verdict:       VerdictOffload,
			RecomputeCost: recomputeCost,
			OffloadCost:   offloadCost,
			Reason: fmt.Sprintf(
				"offload cost (%.4f) < recompute cost (%.4f): offloading is more economical for %d tokens with %.1fx reuse",
				offloadCost, recomputeCost, tokens, reuseProb,
			),
		}
	}

	return Decision{
		Verdict:       VerdictRecompute,
		RecomputeCost: recomputeCost,
		OffloadCost:   offloadCost,
		Reason: fmt.Sprintf(
			"offload cost (%.4f) >= recompute cost (%.4f): recomputing is more economical for %d tokens with %.1fx reuse",
			offloadCost, recomputeCost, tokens, reuseProb,
		),
	}
}
