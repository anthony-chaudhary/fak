//go:build vulkan && (windows || linux) && cgo

package compute

// SupportsBorrowedExpertUploads admits immutable Q2_K/Q3_K checkpoint views.
// Vulkan's Upload and UploadClass allocate independent packed weight buffers;
// this backend has no UploadAsync route. fvk_h2d copies
// source bytes into owned staging/step-arena or directly mapped destination
// memory before returning, including when GPU submission is deferred. Q3_K tail
// padding is a separate owned allocation; the source is never modified or retained.
func (*vulkanBackend) SupportsBorrowedExpertUploads() bool { return true }
