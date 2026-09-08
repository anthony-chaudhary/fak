//go:build darwin && arm64 && cgo

package metalgemm

/*
typedef struct {
    int command_buffers;
    int encoders;
    int icb_dispatches;
    int contiguous_blocks;
    double host_encode_ms;
    double host_wait_ms;
    double gpu_ms;
    double total_ms;
    int icb_used;
    int mode;
} mg_decode_receipt;

void mg_decode_set_mode(int mode);
int  mg_decode_get_mode(void);
int  mg_decode_icb_supported(void);
int  mg_decode_step_receipt(const float* xEmbed, const float* Kctx, const float* Vctx, int L,
                            float* lastPre, float* newKraw, float* newKpost, float* newV, float* logits,
                            int seedFlag, int dispatchMode, mg_decode_receipt* receipt);
*/
import "C"

import (
	"unsafe"
)

// ICBDecodeMode defines the orchestration mechanism for the autoregressive decode pass.
type ICBDecodeMode int

const (
	// DecodeModeDirectMultiCB executes multiple command buffers per step with per-op/block host synchronization.
	DecodeModeDirectMultiCB ICBDecodeMode = 0
	// DecodeModeDirectOneCB executes a single command buffer per step with direct CPU-encoded dispatches.
	DecodeModeDirectOneCB ICBDecodeMode = 1
	// DecodeModeICBReplay executes transformer decode layers via pre-allocated Indirect Command Buffers (ICB).
	DecodeModeICBReplay ICBDecodeMode = 2
)

// ICBDecodeReceipt records execution facts and hardware timing witnessed during a decode step.
type ICBDecodeReceipt struct {
	CommandBuffers   int           `json:"command_buffers"`
	Encoders         int           `json:"encoders"`
	ICBDispatches    int           `json:"icb_dispatches"`
	ContiguousBlocks int           `json:"contiguous_blocks"`
	HostEncodeMs     float64       `json:"host_encode_ms"`
	HostWaitMs       float64       `json:"host_wait_ms"`
	GPUMs            float64       `json:"gpu_ms"`
	TotalMs          float64       `json:"total_ms"`
	ICBUsed          bool          `json:"icb_used"`
	Mode             ICBDecodeMode `json:"mode"`
}

// SetDecodeDispatchMode sets the active decode dispatch mode.
func SetDecodeDispatchMode(mode ICBDecodeMode) {
	C.mg_decode_set_mode(C.int(mode))
}

// GetDecodeDispatchMode returns the currently active decode dispatch mode.
func GetDecodeDispatchMode() ICBDecodeMode {
	return ICBDecodeMode(C.mg_decode_get_mode())
}

// SetICBDecodeEnabled toggles ICB decode replay. Passing false selects DecodeModeDirectOneCB.
func SetICBDecodeEnabled(enabled bool) {
	if enabled {
		SetDecodeDispatchMode(DecodeModeICBReplay)
	} else {
		SetDecodeDispatchMode(DecodeModeDirectOneCB)
	}
}

// ICBDecodeSupported returns true if the host Apple Silicon device supports compute ICBs.
func ICBDecodeSupported() bool {
	return Available() && C.mg_decode_icb_supported() == 1
}

// DecodeStepWithReceipt runs one decode token and returns execution receipts.
func DecodeStepWithReceipt(xEmbed, Kctx, Vctx []float32, L, nLayers, w, H, vocab int, seed bool) (lastPre, newKpost, newV, logits []float32, receipt ICBDecodeReceipt, ok bool) {
	return DecodeStepWithMode(-1, xEmbed, Kctx, Vctx, L, nLayers, w, H, vocab, seed)
}

// DecodeStepWithMode runs one decode token using an explicit dispatch mode.
func DecodeStepWithMode(mode ICBDecodeMode, xEmbed, Kctx, Vctx []float32, L, nLayers, w, H, vocab int, seed bool) (lastPre, newKpost, newV, logits []float32, receipt ICBDecodeReceipt, ok bool) {
	if !Available() || len(xEmbed) < H {
		return nil, nil, nil, nil, ICBDecodeReceipt{}, false
	}
	lastPre = make([]float32, H)
	newKraw := make([]float32, nLayers*w)
	newKpost = make([]float32, nLayers*w)
	newV = make([]float32, nLayers*w)
	var lp *C.float
	if vocab > 0 {
		logits = make([]float32, vocab)
		lp = (*C.float)(unsafe.Pointer(&logits[0]))
	}
	var kp, vp *C.float
	if seed && L > 0 {
		if len(Kctx) < nLayers*L*w || len(Vctx) < nLayers*L*w {
			return nil, nil, nil, nil, ICBDecodeReceipt{}, false
		}
		kp = (*C.float)(unsafe.Pointer(&Kctx[0]))
		vp = (*C.float)(unsafe.Pointer(&Vctx[0]))
	}
	seedF := C.int(0)
	if seed {
		seedF = 1
	}
	var cRec C.mg_decode_receipt
	r := C.mg_decode_step_receipt((*C.float)(unsafe.Pointer(&xEmbed[0])), kp, vp, C.int(L),
		(*C.float)(unsafe.Pointer(&lastPre[0])), (*C.float)(unsafe.Pointer(&newKraw[0])),
		(*C.float)(unsafe.Pointer(&newKpost[0])), (*C.float)(unsafe.Pointer(&newV[0])), lp,
		seedF, C.int(mode), &cRec)
	if r != 1 {
		return nil, nil, nil, nil, ICBDecodeReceipt{}, false
	}
	receipt = ICBDecodeReceipt{
		CommandBuffers:   int(cRec.command_buffers),
		Encoders:         int(cRec.encoders),
		ICBDispatches:    int(cRec.icb_dispatches),
		ContiguousBlocks: int(cRec.contiguous_blocks),
		HostEncodeMs:     float64(cRec.host_encode_ms),
		HostWaitMs:       float64(cRec.host_wait_ms),
		GPUMs:            float64(cRec.gpu_ms),
		TotalMs:          float64(cRec.total_ms),
		ICBUsed:          cRec.icb_used != 0,
		Mode:             ICBDecodeMode(cRec.mode),
	}
	return lastPre, newKpost, newV, logits, receipt, true
}
