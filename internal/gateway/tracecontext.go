package gateway

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"strings"
)

const traceparentHeader = "traceparent"

type traceContext struct {
	TraceID, ParentID string
	Flags             byte
}

func parseTraceparent(raw string) (traceContext, error) {
	parts := strings.Split(strings.TrimSpace(raw), "-")
	if len(parts) != 4 || parts[0] != "00" || len(parts[1]) != 32 || len(parts[2]) != 16 || len(parts[3]) != 2 {
		return traceContext{}, fmt.Errorf("invalid traceparent v00 shape")
	}
	if _, err := hex.DecodeString(parts[1]); err != nil {
		return traceContext{}, fmt.Errorf("invalid trace id")
	}
	if _, err := hex.DecodeString(parts[2]); err != nil {
		return traceContext{}, fmt.Errorf("invalid parent id")
	}
	flags, err := hex.DecodeString(parts[3])
	if err != nil {
		return traceContext{}, fmt.Errorf("invalid flags")
	}
	if allZeroHex(parts[1]) || allZeroHex(parts[2]) {
		return traceContext{}, fmt.Errorf("zero traceparent id")
	}
	return traceContext{TraceID: strings.ToLower(parts[1]), ParentID: strings.ToLower(parts[2]), Flags: flags[0]}, nil
}
func (c traceContext) String() string {
	return fmt.Sprintf("00-%s-%s-%02x", c.TraceID, c.ParentID, c.Flags)
}
func allZeroHex(s string) bool {
	for _, r := range s {
		if r != '0' {
			return false
		}
	}
	return true
}
func randomHex(bytesN int) string {
	b := make([]byte, bytesN)
	if _, err := rand.Read(b); err != nil {
		panic(err)
	}
	return hex.EncodeToString(b)
}
func newTraceContext(traceID string, flags byte) traceContext {
	if len(traceID) != 32 || allZeroHex(traceID) {
		traceID = randomHex(16)
	}
	return traceContext{TraceID: strings.ToLower(traceID), ParentID: randomHex(8), Flags: flags}
}
