package tcp

import (
	"io"

	"github.com/anthony-chaudhary/fak/internal/l3server/transport/protocol"
)

// Codec is an alias for protocol.Codec. Kept for backward compatibility.
type Codec = protocol.Codec

// NewCodec wraps a connection with buffered I/O.
func NewCodec(conn io.ReadWriteCloser) *Codec {
	return protocol.NewCodec(conn)
}
