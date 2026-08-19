package fakclient

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

func DecodeRuntimeNDJSON(line []byte) (RuntimeWireEvent, error) {
	return decodeRuntimeWire(bytes.TrimSpace(line))
}
func DecodeRuntimeSSE(frame []byte) (RuntimeWireEvent, error) {
	for _, line := range strings.Split(string(frame), "\n") {
		if strings.HasPrefix(line, "data: ") {
			return decodeRuntimeWire([]byte(strings.TrimPrefix(line, "data: ")))
		}
	}
	return RuntimeWireEvent{}, fmt.Errorf("runtime SSE data frame missing")
}
func decodeRuntimeWire(body []byte) (RuntimeWireEvent, error) {
	var wire RuntimeWireEvent
	if err := json.Unmarshal(body, &wire); err != nil {
		return wire, err
	}
	if wire.Schema != RuntimeEventWireSchema || wire.Event.Schema != RuntimeEventSchema || wire.Event.TraceID == "" || wire.Event.EventID == "" || wire.Event.Sequence == 0 || !wire.Admission.Screened {
		return wire, fmt.Errorf("invalid runtime wire envelope")
	}
	known := false
	for _, kind := range runtimeEventKinds() {
		if wire.Event.Kind == kind {
			known = true
			break
		}
	}
	if !known {
		return wire, fmt.Errorf("unknown runtime kind %q", wire.Event.Kind)
	}
	return wire, nil
}
func runtimeEventKinds() []RuntimeEventKind {
	return []RuntimeEventKind{RuntimeKindTurnStarted, RuntimeKindToolProposed, RuntimeKindToolVerdict, RuntimeKindToolResultAdmitted, RuntimeKindContextChanged, RuntimeKindCostDebited, RuntimeKindTerminalWitness, RuntimeKindTurnTerminal, RuntimeKindError}
}
