//go:build darwin && arm64 && cgo

package metalgemm

/*
#include <stdlib.h>
void mg_gdn_set_chunk_source(const char *source);
*/
import "C"
import (
	"embed"
	"unsafe"
)

//go:embed gdn_chunked.metal
var gdnChunkFiles embed.FS

func init() {
	source, _ := gdnChunkFiles.ReadFile("gdn_chunked.metal")
	cSource := C.CString(string(source))
	C.mg_gdn_set_chunk_source(cSource)
	C.free(unsafe.Pointer(cSource))
}
