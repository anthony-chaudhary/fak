package trajectory

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

const RuntimeWireSchema = "fak-runtime-event-wire/1"
const RuntimeEventMaxBytes = 1 << 20

type RuntimeScreenStamp struct {
	Screened bool   `json:"screened"`
	Taint    string `json:"taint"`
	Screen   string `json:"screen"`
}

type RuntimeWireEvent struct {
	Schema       string             `json:"schema"`
	RuntimeEvent RuntimeEvent       `json:"event"`
	Admission    RuntimeScreenStamp `json:"admission"`
}

type RuntimeTransport string

const (
	RuntimeNDJSON RuntimeTransport = "ndjson"
	RuntimeSSE    RuntimeTransport = "sse"
)

func EncodeRuntimeEvent(event RuntimeEvent, transport RuntimeTransport) ([]byte, error) {
	if err := ValidateRuntimeEvent(event); err != nil {
		return nil, err
	}
	canonical, err := json.Marshal(event)
	if err != nil {
		return nil, err
	}
	if len(canonical) > RuntimeEventMaxBytes {
		return nil, fmt.Errorf("runtime event exceeds %d-byte cap", RuntimeEventMaxBytes)
	}
	if reason, held := ctxmmu.ScreenBytes(canonical); held {
		return nil, fmt.Errorf("runtime event rejected by ctxmmu: %v", reason)
	}
	wire := RuntimeWireEvent{Schema: RuntimeWireSchema, RuntimeEvent: event, Admission: RuntimeScreenStamp{Screened: true, Taint: "tainted", Screen: "ctxmmu/1"}}
	body, err := json.Marshal(wire)
	if err != nil {
		return nil, err
	}
	switch transport {
	case RuntimeNDJSON:
		return append(body, '\n'), nil
	case RuntimeSSE:
		return []byte(fmt.Sprintf("id: %s\nevent: %s\ndata: %s\n\n", event.EventID, event.Kind, body)), nil
	default:
		return nil, fmt.Errorf("unsupported runtime transport %q", transport)
	}
}

func WriteRuntimeEvent(w io.Writer, event RuntimeEvent, transport RuntimeTransport) error {
	frame, err := EncodeRuntimeEvent(event, transport)
	if err != nil {
		return err
	}
	for len(frame) > 0 {
		n, err := w.Write(frame)
		if err != nil {
			return err
		}
		if n <= 0 || n > len(frame) {
			return io.ErrShortWrite
		}
		frame = frame[n:]
	}
	return nil
}
