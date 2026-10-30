package compute

// V41EngramRowEncodingBackend optionally narrows row decode admission to an
// exact stored dialect. Implementations must report false for unknown dialects.
// This query is selection only: it does not permit replay after dispatch fails.
type V41EngramRowEncodingBackend interface {
	SupportsV41EngramRowEncoding(V41EngramRowEncoding) bool
}

// V41EngramRowDecodeSupportsEncoding checks an exact dialect before device
// selection. Legacy broad capability covers only the original three dialects;
// a future encoding requires both a known stride and explicit backend support.
// In particular, a new enum value must never leak into a legacy native mode
// namespace (the Vulkan shader reserves mode 3 for poison).
func V41EngramRowDecodeSupportsEncoding(backend V41EngramRowDecodeBackend, encoding V41EngramRowEncoding) bool {
	if backend == nil || V41EngramRowStride(encoding) == 0 || !backend.SupportsV41EngramRowDecode() {
		return false
	}
	if exact, ok := backend.(V41EngramRowEncodingBackend); ok {
		return exact.SupportsV41EngramRowEncoding(encoding)
	}
	switch encoding {
	case V41EngramRowPacked, V41EngramRowQ2K, V41EngramRowF32:
		return true
	default:
		return false
	}
}
