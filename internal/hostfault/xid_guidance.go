/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

// DIRECT-PORT of NVIDIA/nccl src/ras/diagnostics_xid.cc at
// 3e46c089f4ae4732722a419eb6096aead7e47a30 (Apache-2.0).
//
// Ported: the XID/SXID event types, SXID driver classes, the kernel-log line
// grammar (rasDiagnosticsXidParseLine, ParseCode, ParseBdf), the event
// identity rule, and the XID/SXID operator-action guidance sets with their
// documentation URLs.
//
// fak adds: pure-function Go accessors over text only (no GPU, driver or log
// access), String() names for receipts, and a caller-supplied preAmpere bit
// in place of upstream's communicator compute-capability probe.
//
// This grammar is deliberately separate from the XID extractor in xid.go:
// upstream matches "SXid"/"Xid" case-sensitively, while that extractor is
// case-insensitive and cannot see SXid lines, and the two code tables come from different upstream
// pins so they must never be merged.

package hostfault

import "strings"

// XIDEventType distinguishes GPU Xid events from NVSwitch SXid events.
type XIDEventType uint8

const (
	XIDEventXID  XIDEventType = 1
	XIDEventSXID XIDEventType = 2
)

// String returns the receipt name of the event type.
func (t XIDEventType) String() string {
	switch t {
	case XIDEventXID:
		return "xid"
	case XIDEventSXID:
		return "sxid"
	default:
		return "unknown"
	}
}

// SXIDClass is the driver-reported severity of an SXid summary line.
type SXIDClass uint8

const (
	SXIDClassUnknown  SXIDClass = 0
	SXIDClassNonFatal SXIDClass = 1
	SXIDClassFatal    SXIDClass = 2
)

// String returns the receipt name of the SXID class.
func (c SXIDClass) String() string {
	switch c {
	case SXIDClassNonFatal:
		return "non_fatal"
	case SXIDClassFatal:
		return "fatal"
	default:
		return "unknown"
	}
}

// XIDGuidance is the operator action recommended for an Xid/SXid event.
type XIDGuidance uint8

const (
	XIDGuidanceNoUserAction       XIDGuidance = 0
	XIDGuidanceRestartApplication XIDGuidance = 1
	XIDGuidanceResetGPU           XIDGuidance = 2
	// XIDGuidanceCheckDocumentation covers every code with no specific guidance.
	XIDGuidanceCheckDocumentation XIDGuidance = 3
	XIDGuidancePreAmpere          XIDGuidance = 4
)

const (
	xidDocURL        = "https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html"
	xidArchiveDocURL = "https://docs.nvidia.com/deploy/xid-errors/archive/index.html"
	sxidDocURL       = "https://docs.nvidia.com/hgx-platforms/fabric-manager-user-guide/index.html"
)

// String returns the receipt name of the guidance.
func (g XIDGuidance) String() string {
	switch g {
	case XIDGuidanceNoUserAction:
		return "no_user_action"
	case XIDGuidanceRestartApplication:
		return "restart_application"
	case XIDGuidanceResetGPU:
		return "reset_gpu"
	case XIDGuidanceCheckDocumentation:
		return "check_documentation"
	case XIDGuidancePreAmpere:
		return "pre_ampere"
	default:
		return "unknown"
	}
}

// DocURL returns the vendor document an operator should consult for an event
// of type t carrying this guidance. Pre-Ampere GPUs are not in the current Xid
// catalog, so they point at the archive instead.
func (g XIDGuidance) DocURL(t XIDEventType) string {
	if t == XIDEventSXID {
		return sxidDocURL
	}
	if g == XIDGuidancePreAmpere {
		return xidArchiveDocURL
	}
	return xidDocURL
}

// XIDEvent is one parsed Xid or SXid kernel-log line.
type XIDEvent struct {
	Type      XIDEventType
	Code      uint32
	SXIDClass SXIDClass
	// Link is the NVSwitch link of an SXid event and -1 for Xid events.
	Link int16
	// BDF packs domain<<20 | bus<<12 | device<<4 | function; valid only when BDFValid.
	BDF      int64
	BDFValid bool
}

// SameXIDEvent reports whether a and b are the same event for counting, using
// upstream's identity: type, code, class, link and (when present) BDF.
func SameXIDEvent(a, b XIDEvent) bool {
	return a.Type == b.Type && a.Code == b.Code && a.SXIDClass == b.SXIDClass && a.Link == b.Link &&
		a.BDFValid == b.BDFValid && (!a.BDFValid || a.BDF == b.BDF)
}

// ParseXIDLine parses one kernel-log line. SXid is tried first; a line that
// mentions "SXid" but is not the summary form (code, Non-fatal|Fatal, Link N)
// is rejected rather than reparsed as an Xid, because SXid Severity/Data
// detail lines would otherwise be miscounted as GPU Xids.
func ParseXIDLine(line string) (XIDEvent, bool) {
	if i := strings.Index(line, "SXid"); i >= 0 {
		return parseSXIDLine(line[i+len("SXid"):])
	}
	i := strings.Index(line, "Xid")
	if i < 0 {
		return XIDEvent{}, false
	}
	ev := XIDEvent{Type: XIDEventXID, SXIDClass: SXIDClassUnknown, Link: -1}
	rest := skipBlank(line[i+len("Xid"):])
	if strings.HasPrefix(rest, "(") {
		inner := rest[1:]
		closeIdx := strings.IndexByte(inner, ')')
		if closeIdx < 0 {
			return XIDEvent{}, false
		}
		code, _, ok := parseXIDCode(inner[closeIdx+1:])
		if !ok {
			return XIDEvent{}, false
		}
		ev.Code = code
		ev.BDF, ev.BDFValid = parseXIDBDF(inner[:closeIdx])
		return ev, true
	}
	code, _, ok := parseXIDCode(rest)
	if !ok {
		return XIDEvent{}, false
	}
	ev.Code = code
	return ev, true
}

func parseSXIDLine(rest string) (XIDEvent, bool) {
	rest = skipBlank(rest)
	if !strings.HasPrefix(rest, "(") {
		return XIDEvent{}, false
	}
	inner := rest[1:]
	closeIdx := strings.IndexByte(inner, ')')
	if closeIdx < 0 {
		return XIDEvent{}, false
	}
	code, after, ok := parseXIDCode(inner[closeIdx+1:])
	if !ok {
		return XIDEvent{}, false
	}
	comma := strings.IndexByte(after, ',')
	if comma < 0 {
		return XIDEvent{}, false
	}
	cls := skipBlank(after[comma+1:])
	var class SXIDClass
	switch {
	case hasPrefixFold(cls, "Non-fatal"):
		class, cls = SXIDClassNonFatal, cls[len("Non-fatal"):]
	case hasPrefixFold(cls, "Fatal"):
		class, cls = SXIDClassFatal, cls[len("Fatal"):]
	default:
		return XIDEvent{}, false
	}
	cls = skipBlank(cls)
	if !strings.HasPrefix(cls, ",") {
		return XIDEvent{}, false
	}
	link := skipBlank(cls[1:])
	if !hasPrefixFold(link, "Link") || len(link) == len("Link") ||
		(link[len("Link")] != ' ' && link[len("Link")] != '\t') {
		return XIDEvent{}, false
	}
	linkValue, ok := parseLinkNumber(skipBlank(link[len("Link"):]))
	if !ok {
		return XIDEvent{}, false
	}
	ev := XIDEvent{Type: XIDEventSXID, Code: code, SXIDClass: class, Link: linkValue}
	ev.BDF, ev.BDFValid = parseXIDBDF(inner[:closeIdx])
	return ev, true
}

// parseXIDCode mirrors upstream ParseCode: skip blanks and colons, then a
// strtoul-style decimal bounded to uint32.
func parseXIDCode(s string) (uint32, string, bool) {
	s = strings.TrimLeft(s, " \t:")
	s, neg := trimSign(strings.TrimLeft(s, " \t\n\v\f\r"))
	n := 0
	var v uint64
	for n < len(s) && s[n] >= '0' && s[n] <= '9' {
		v = v*10 + uint64(s[n]-'0')
		if v > 0xffffffff {
			return 0, "", false
		}
		n++
	}
	// strtoul negates a signed operand, so only "-0" stays within bounds.
	if n == 0 || (neg && v != 0) {
		return 0, "", false
	}
	return uint32(v), s[n:], true
}

func parseLinkNumber(s string) (int16, bool) {
	s, neg := trimSign(strings.TrimLeft(s, " \t\n\v\f\r"))
	n := 0
	v := 0
	for n < len(s) && s[n] >= '0' && s[n] <= '9' {
		v = v*10 + int(s[n]-'0')
		if v > 32767 {
			return 0, false
		}
		n++
	}
	if n == 0 || (neg && v != 0) {
		return 0, false
	}
	return int16(v), true
}

func trimSign(s string) (string, bool) {
	if s != "" && (s[0] == '+' || s[0] == '-') {
		return s[1:], s[0] == '-'
	}
	return s, false
}

// parseXIDBDF mirrors upstream ParseBdf: an optional "PCI:" prefix, then
// domain:bus:device[.function] in hex, with the whole field consumed.
func parseXIDBDF(s string) (int64, bool) {
	s = strings.Trim(s, " \t")
	if len(s) >= 4 && strings.EqualFold(s[:4], "PCI:") {
		s = skipBlank(s[4:])
	}
	if len(s) == 0 || len(s) >= 64 {
		return 0, false
	}
	var function uint64
	head := s
	if dot := strings.IndexByte(s, '.'); dot >= 0 {
		f, ok := parseHexField(s[dot+1:])
		if !ok {
			return 0, false
		}
		function, head = f, s[:dot]
	}
	parts := strings.Split(head, ":")
	if len(parts) != 3 {
		return 0, false
	}
	var fields [3]uint64
	for i, p := range parts {
		v, ok := parseHexField(p)
		if !ok {
			return 0, false
		}
		fields[i] = v
	}
	domain, bus, device := fields[0], fields[1], fields[2]
	if domain > 0xffff || bus > 0xff || device > 0x1f || function > 7 {
		return 0, false
	}
	return int64(domain<<20 | bus<<12 | device<<4 | function), true
}

// parseHexField accepts what one sscanf %x conversion consumes as a whole
// field: leading whitespace, an optional sign and 0x prefix, then hex digits.
// Values saturate past 32 bits, which every BDF bound then rejects.
func parseHexField(s string) (uint64, bool) {
	s, neg := trimSign(strings.TrimLeft(s, " \t\n\v\f\r"))
	if len(s) > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X') {
		s = s[2:]
	}
	if s == "" {
		return 0, false
	}
	var v uint64
	for i := 0; i < len(s); i++ {
		c := s[i]
		var d uint64
		switch {
		case c >= '0' && c <= '9':
			d = uint64(c - '0')
		case c >= 'a' && c <= 'f':
			d = uint64(c - 'a' + 10)
		case c >= 'A' && c <= 'F':
			d = uint64(c - 'A' + 10)
		default:
			return 0, false
		}
		if v <= 0xffffffff {
			v = v<<4 | d
		}
	}
	if neg && v != 0 {
		v = 0xffffffff
	}
	return v, true
}

func skipBlank(s string) string { return strings.TrimLeft(s, " \t") }

func hasPrefixFold(s, prefix string) bool {
	return len(s) >= len(prefix) && strings.EqualFold(s[:len(prefix)], prefix)
}

var (
	xidNoUserAction       = []uint32{63, 106, 107}
	xidRestartApplication = []uint32{8, 11, 13, 25, 31, 32, 39, 40, 41, 60, 68, 69,
		70, 71, 72, 75, 76, 77, 80, 82, 83, 84, 85, 86,
		88, 89, 94, 96, 97, 98, 99, 100, 101, 102, 103, 104,
		105, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 139}
	xidResetGPU      = []uint32{46, 62, 64, 95, 109, 110, 119, 120, 136, 140, 143, 155, 156, 158}
	sxidNoUserAction = []uint32{11012, 11021, 11022, 11023, 12021, 12023, 15008, 15011, 19049, 19055,
		19057, 19059, 19062, 19065, 19068, 19071, 22013, 24001, 24002, 24003}
)

func codeInSet(code uint32, set []uint32) bool {
	for _, v := range set {
		if v == code {
			return true
		}
	}
	return false
}

// ClassifyXIDGuidance maps a GPU Xid code to its operator action. preAmpere
// reports that the host has a GPU below compute capability 8.0, whose Xids
// the current catalog guidance does not cover.
func ClassifyXIDGuidance(code uint32, preAmpere bool) XIDGuidance {
	switch {
	case preAmpere:
		return XIDGuidancePreAmpere
	case codeInSet(code, xidNoUserAction):
		return XIDGuidanceNoUserAction
	case codeInSet(code, xidRestartApplication):
		return XIDGuidanceRestartApplication
	case codeInSet(code, xidResetGPU):
		return XIDGuidanceResetGPU
	default:
		return XIDGuidanceCheckDocumentation
	}
}

// ClassifySXIDGuidance maps an NVSwitch SXid to its operator action; only
// listed non-fatal codes are safe to ignore.
func ClassifySXIDGuidance(code uint32, class SXIDClass) XIDGuidance {
	if class == SXIDClassNonFatal && codeInSet(code, sxidNoUserAction) {
		return XIDGuidanceNoUserAction
	}
	return XIDGuidanceCheckDocumentation
}

// ClassifyXIDEventGuidance applies the per-type guidance rule to a parsed event.
func ClassifyXIDEventGuidance(ev XIDEvent, preAmpere bool) XIDGuidance {
	if ev.Type == XIDEventSXID {
		return ClassifySXIDGuidance(ev.Code, ev.SXIDClass)
	}
	return ClassifyXIDGuidance(ev.Code, preAmpere)
}
