package headroom

import "context"

// NoopName is the identity plugin's registry key and the default selection: until
// another plugin is selected (Select / FAK_COMPRESSOR) the result path compresses
// NOTHING and adds zero overhead, so wiring this leaf into the kernel changes no
// behaviour by default — compression is strictly opt-in.
const NoopName = "noop"

type noopCompressor struct{}

// Name returns the identity compressor's registry key ("noop").
func (noopCompressor) Name() string { return NoopName }

func (noopCompressor) Compress(_ context.Context, in Input) (Output, error) {
	return Output{
		Bytes:      in.Bytes,
		Compressed: false,
		Codec:      "identity",
		OrigLen:    len(in.Bytes),
		NewLen:     len(in.Bytes),
	}, nil
}

func init() { Register(noopCompressor{}) }
