/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// DIRECT-PORT of the NVIDIA XID code -> operator-text closed vocabulary from
// NVIDIA/dcgm-exporter (Apache-2.0). The table below is ported VERBATIM from
//
//	https://github.com/NVIDIA/dcgm-exporter/blob/4e663a526c4624ad0e18ad816f09b7be2c5348a7/internal/pkg/collector/xid_errors.go
//	pin 4e663a526c4624ad0e18ad816f09b7be2c5348a7 (table declared at xid_errors.go:20)
//
// which in turn indexes the vendor XID catalog
// (https://docs.nvidia.com/deploy/xid-errors/#topic_4) frozen at that pin.
// Borrow provenance: docs/research/STUDY-PRODUCTION-SCALING-GPU-SRE-2026-11-07.md,
// candidate dcgm-exporter-1 (native ticket oss-port-dcgm-exporter-dcgm-exporter-1).
//
// What fak adds AROUND the ported data: the fail-closed accessor
// XIDOperatorText, the record-level ExtractXID the classifier rung reads, and
// the GPU_XID class in hostfault.go. The table rows themselves are upstream
// data and are never reworded or re-sorted — a drifted row would silently
// change an operator receipt's fault text, so the pinned values are asserted
// verbatim in TestClassifyHostFault.

package hostfault

import (
	"regexp"
	"strconv"
)

// Based on this doc: https://docs.nvidia.com/deploy/xid-errors/#topic_4
var xidErrCodeToText = []string{
	0:   "No Error",
	1:   "Invalid or corrupted push buffer stream",
	2:   "Invalid or corrupted push buffer stream",
	3:   "Invalid or corrupted push buffer stream",
	4:   "Invalid or corrupted push buffer stream",
	5:   "Unused",
	6:   "Invalid or corrupted push buffer stream",
	7:   "Invalid or corrupted push buffer address",
	8:   "GPU stopped processing",
	9:   "Driver error programming GPU",
	10:  "Unused",
	11:  "Invalid or corrupted push buffer stream",
	12:  "Driver error handling GPU exception",
	13:  "Graphics Engine Exception",
	14:  "Unused",
	15:  "Unused",
	16:  "Display engine hung",
	17:  "Unused",
	18:  "Bus mastering disabled in PCI Config Space",
	19:  "Display Engine error",
	20:  "Invalid or corrupted Mpeg push buffer",
	21:  "Invalid or corrupted Motion Estimation push buffer",
	22:  "Invalid or corrupted Video Processor push buffer",
	23:  "Unused",
	24:  "GPU semaphore timeout",
	25:  "Invalid or illegal push buffer stream",
	26:  "Framebuffer timeout",
	27:  "Video processor exception",
	28:  "Video processor exception",
	29:  "Video processor exception",
	30:  "GPU semaphore access error",
	31:  "GPU memory page fault",
	32:  "Invalid or corrupted push buffer stream",
	33:  "Internal micro-controller error",
	34:  "Video processor exception",
	35:  "Video processor exception",
	36:  "Video processor exception",
	37:  "Driver firmware error",
	38:  "Driver firmware error",
	39:  "Unused",
	40:  "Unused",
	41:  "Unused",
	42:  "Video processor exception",
	43:  "GPU stopped processing",
	44:  "Graphics Engine fault during context switch",
	45:  "Preemptive cleanup, due to previous errors -- Most likely to see when running multiple cuda",
	46:  "GPU stopped processing",
	47:  "Video processor exception",
	48:  "Double Bit ECC Error",
	49:  "Unused",
	50:  "Unused",
	51:  "Unused",
	52:  "Unused",
	53:  "Unused",
	54:  "Auxiliary power is not connected to the GPU board",
	55:  "Unused",
	56:  "Display Engine error",
	57:  "Error programming video memory interface",
	58:  "Unstable video memory interface detected",
	59:  "Internal micro-controller error",
	60:  "Video processor exception",
	61:  "Internal micro-controller breakpoint/warning",
	62:  "Internal micro-controller halt",
	63:  "ECC page retirement or row remapping recording event",
	64:  "ECC page retirement or row remapper recording failure",
	65:  "Video processor exception",
	66:  "Illegal access by driver",
	67:  "Illegal access by driver",
	68:  "NVDEC0 Exception",
	69:  "Graphics Engine class error",
	70:  "CE3: Unknown Error",
	71:  "CE4: Unknown Error",
	72:  "CE5: Unknown Error",
	73:  "NVENC2 Error",
	74:  "NVLINK Error",
	75:  "CE6: Unknown Error",
	76:  "CE7: Unknown Error",
	77:  "CE8: Unknown Error",
	78:  "vGPU Start Error",
	79:  "GPU has fallen off the bus",
	80:  "Corrupted data sent to GPU",
	81:  "VGA Subsystem Error",
	82:  "NVJPG0 Error",
	83:  "NVDEC1 Error",
	84:  "NVDEC2 Error",
	85:  "CE9: Unknown Error",
	86:  "OFA Exception",
	87:  "Reserved",
	88:  "NVDEC3 Error",
	89:  "NVDEC4 Error",
	90:  "Reserved",
	91:  "Reserved",
	92:  "High single-bit ECC error rate",
	93:  "Non-fatal violation of provisioned InfoROM wear limit",
	94:  "Contained ECC error",
	95:  "Uncontained ECC error",
	96:  "NVDEC5 Error",
	97:  "NVDEC6 Error",
	98:  "NVDEC7 Error",
	99:  "NVJPG1 Error",
	100: "NVJPG2 Error",
	101: "NVJPG3 Error",
	102: "NVJPG4 Error",
	103: "NVJPG5 Error",
	104: "NVJPG6 Error",
	105: "NVJPG7 Error",
	106: "SMBPBI Test Message",
	107: "SMBPBI Test Message Silent",
	108: "Reserved",
	109: "Context Switch Timeout Error",
	110: "Security Fault Error",
	111: "Display Bundle Error Event",
	112: "Display Supervisor Error",
	113: "DP Link Training Error",
	114: "Display Pipeline Underflow Error",
	115: "Display Core Channel Error",
	116: "Display Window Channel Error",
	117: "Display Cursor Channel Error",
	118: "Display Pixel Pipeline Error",
	119: "GSP RPC Timeout",
	120: "GSP Error",
	121: "C2C Link Error",
	122: "SPI PMU RPC Read Failure",
	123: "SPI PMU RPC Write Failure",
	124: "SPI PMU RPC Erase Failure",
	125: "Inforom FS Failure",
	126: "Robust Channel CE10 Error",
	127: "Robust Channel CE11 Error",
	128: "Robust Channel CE12 Error",
	129: "Robust Channel CE13 Error",
	130: "Robust Channel CE14 Error",
	131: "Robust Channel CE15 Error",
	132: "Robust Channel CE16 Error",
	133: "Robust Channel CE17 Error",
	134: "Robust Channel CE18 Error",
	135: "Robust Channel CE19 Error",
	136: "ALI Training Fail",
	137: "NVLink Privilege Error",
	138: "Robust Channel DLA Error",
	139: "Robust Channel OFA1 Error",
	140: "Unrecovered ECC Error",
	141: "Robust Channel Fast Path Error",
	142: "Robust Channel NVENVC3 Error",
	143: "GPU Initialization Failure",
	144: "NVLINK: Saw Error",
	145: "NVLINK: RLW Error",
	146: "NVLINK: TLW Error",
	147: "NVLINK: TREX Error",
	148: "NVLINK: NVLPW_CTRL Error",
	149: "NVLINK: NETIR Error",
	150: "NVLINK: MSE Error",
	151: "Robust Channel Key rotation Error",
	152: "Robust Channel DLA SMMU Error",
	153: "Robust Channel DLA timeout Error",
	154: "GPU Recovery Action Changed",
	155: "NVLINK: SW Defined Error",
	156: "Resource Retirement Event",
	157: "Resource Retirement Failure",
	158: "GPU Fatal Timeout",
	159: "Robust Channel CHI Non-Data Error",
	160: "Channel Retirement Event",
	161: "Channel Retirement Failure",
	162: "Power Smoothing HW Circuitry Reengaged",
	163: "Power Smoothing HW Circuitry Disengaged",
	164: "Power Smoothing HW Circuitry Low Lifetime",
	165: "Power Smoothing HW Circuitry Lifetime Exhausted",
	166: "NVLINK Secure Crypto Error",
	167: "PCIE Fatal Timeout",
	168: "Reduced GPU Memory Capacity (Errors found in write protection region)",
	169: "Internal micro-controller halt",
	170: "NVLINK: Interrupt seen in CC mode",
	171: "Uncorrectable DRAM Error",
	172: "Uncorrectable SRAM Error",
}

// XIDOperatorText maps one NVIDIA XID code onto its operator text, fail-closed:
// a negative code or one outside the pinned catalog returns ok=false with no
// text. Codes the catalog indexes but leaves UNDEFINED come back as the pinned
// "Unused"/"Reserved" placeholder with ok=true — the catalog is the authority,
// and an indexed placeholder is a documented "no defined fault here", which is
// different from an unknown code. Callers rendering a receipt must branch on ok
// rather than on the string, so a missing row can never masquerade as text.
func XIDOperatorText(code int) (string, bool) {
	if code < 0 || code >= len(xidErrCodeToText) {
		return "", false
	}
	return xidErrCodeToText[code], true
}

// xidCodeRE matches an XID code as a driver or event-log record names it: the
// standalone token Xid/XID (word-bounded, so words that merely contain the
// letters — "oxidation" — do not match), an optional PCI address, then the
// numeric code. Witnessed shapes: "Xid (PCI:0000:01:00.0): 79", "XID 48",
// "Xid: 119". A PCI address with no code does not match, because the code is
// required.
var xidCodeRE = regexp.MustCompile(`(?i)\bxid(?:\s*\([^)]*\))?\s*:?\s*(\d+)`)

// ExtractXID returns the first XID code named in a record's message or faulting
// identity. The token is a closed signature, not a loose substring: only a
// standalone Xid/XID followed by a code matches, so an unrelated word is never
// fabricated into a GPU fault. A code OUTSIDE the pinned catalog is still
// returned — the event is witnessed, because the driver named a code — while
// its operator text stays unknown, because XIDOperatorText fails closed. That
// split keeps catalog drift visible instead of silently dropping the fault.
func ExtractXID(r WinFaultRecord) (int, bool) {
	for _, s := range []string{r.Message, r.App} {
		m := xidCodeRE.FindStringSubmatch(s)
		if len(m) < 2 {
			continue
		}
		code, err := strconv.Atoi(m[1])
		if err != nil {
			continue
		}
		return code, true
	}
	return 0, false
}
