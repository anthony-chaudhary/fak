package protocol

import (
	"bufio"
	"io"
)

// Codec wraps buffered I/O for a single connection.
type Codec struct {
	Reader *bufio.Reader
	Writer *bufio.Writer
	conn   io.ReadWriteCloser
}

// NewCodec wraps a connection with buffered I/O.
func NewCodec(conn io.ReadWriteCloser) *Codec {
	return &Codec{
		Reader: bufio.NewReaderSize(conn, 65536),
		Writer: bufio.NewWriterSize(conn, 65536),
		conn:   conn,
	}
}

// ReadMessage reads a protocol message.
func (c *Codec) ReadMessage() (Message, error) {
	return ReadMessage(c.Reader)
}

// WriteMessage writes a protocol message and flushes.
func (c *Codec) WriteMessage(msg Message) error {
	if err := WriteMessage(c.Writer, msg); err != nil {
		return err
	}
	return c.Writer.Flush()
}

// Close closes the underlying connection.
func (c *Codec) Close() error {
	return c.conn.Close()
}
