//go:build darwin && arm64 && cgo

package metalgemm

/*
#include <stdint.h>
#include <stdlib.h>
typedef struct {
    int committed;
    int completed_wait;
    int encoders;
    int host_readbacks;
    int allocated_buffers;
    uint64_t retained_buffer_bytes;
    double gpu_milliseconds;
    double wait_milliseconds;
    int timing_available;
    int status_code;
    int error_code;
    int device_ok;
    char error_text[256];
} mg_graph_receipt;
void *mg_graph_begin(const float *xf, const signed char *xq, const float *xd, int P, int in);
void *mg_graph_encode_q4k(void *graph, int wid);
void *mg_graph_encode_q6k(void *graph, int wid);
void *mg_graph_encode_q8(void *graph, int wid);
void *mg_graph_encode_q4k_from(void *graph, int wid, void *input, int elems);
void *mg_graph_encode_q6k_from(void *graph, int wid, void *input, int elems);
void *mg_graph_quantize_q8(void *graph, void *input, int elems, void **scales);
void *mg_graph_encode_q8_from(void *graph, int wid, void *q, void *d, int elems);
int mg_graph_finish(void *graph, mg_graph_receipt *receipt, int inject_post_submit_failure);
int mg_graph_await_terminal(void *graph);
int mg_graph_read(void *graph, void *result, float *dst, int n);
int mg_graph_read_pack(void *graph, void **results, const int *sizes, int count, float *dst, int total);
void mg_graph_free(void *graph);
int mg_graph_live_owners(void);
int mg_graph_live_buffers(void);
void *mg_graph_test_hold_terminal(void *graph, int wait_limit_ms);
void mg_graph_test_release_terminal(void *gate);
void *mg_graph_xf_buffer(void *graph);
void *mg_graph_upload(void *graph, const float *src, int n);
int mg_graph_set_gemv_vectorized(void *graph, int mode);
int mg_graph_set_gemv_kernel(void *graph, int mode);
int mg_graph_gemv_executed(void *graph, int q6);
int mg_graph_set_gemv_p1(void *graph, int mode);
int mg_graph_set_mm_mode(void *graph, int mode);
int mg_graph_mm_mode(void *graph);
int mg_graph_set_q6k_mode(void *graph, int mode);
int mg_graph_q6k_mode(void *graph);
int mg_graph_set_buffer_pool(void *graph, int depth);
void mg_graph_recycle_result(void *graph, void *result);
void *mg_qwen35_graph_kv_alloc(int elems);
void mg_qwen35_graph_kv_free(void *kv);
int mg_qwen35_graph_kv_upload(void *kv, const float *src, int elems);
int mg_qwen35_graph_kv_download(void *kv, float *dst, int elems);
int mg_qwen35_graph_kv_upload_at(void *kv, long off, const float *src, long elems);
int mg_qwen35_graph_kv_download_at(void *kv, long off, float *dst, long elems);
int mg_qwen35_graph_attention_dkv(void *graph, void *q, void *k, void *v, void *gate,
    const float *qnorm, const float *knorm, const float *cosv, const float *sinv,
    void *kv_kraw, void *kv_kpost, void *kv_v, int kv_off,
    int base, int nh, int nkv, int hd, int rotary, float scale, float qk_eps, int gain1p, int qknorm,
    int qnorm_elems, int knorm_elems,
    void **out, void **kraw, void **kpost, void **vcurrent);
int mg_qwen35_graph_kvq8_ready(void);
void *mg_qwen35_graph_kvq8_alloc(long bytes);
int mg_qwen35_graph_kvq8_write(void *buf, long off, const void *src, long bytes);
int mg_qwen35_graph_kvq8_read(void *buf, long off, void *dst, long bytes);
int mg_qwen35_graph_attention_q8(void *graph, void *q, void *k, void *v, void *gate,
    const float *qnorm, const float *knorm, const float *cosv, const float *sinv,
    const signed char *prefix_kc, const float *prefix_ks, const signed char *prefix_vc, const float *prefix_vs,
    int base, int nh, int nkv, int hd, int rotary, float scale, float qk_eps, int gain1p, int qknorm,
    int qnorm_elems, int knorm_elems,
    void **out, void **kraw, void **kpost, void **vcurrent, void **kc, void **ks, void **vc, void **vs);
int mg_qwen35_graph_attention_dkv_q8(void *graph, void *q, void *k, void *v, void *gate,
    const float *qnorm, const float *knorm, const float *cosv, const float *sinv,
    void *kv_kc, void *kv_ks, void *kv_vc, void *kv_vs, long elem_off,
    int base, int nh, int nkv, int hd, int rotary, float scale, float qk_eps, int gain1p, int qknorm,
    int qnorm_elems, int knorm_elems,
    void **out, void **kraw, void **kpost, void **vcurrent, void **kc, void **ks, void **vc, void **vs);
void *mg_qwen35_graph_norm(void *graph, void *input, const float *weight, int rows, int width, float eps, int gain1p, int last_only);
int mg_qwen35_graph_add(void *graph, void *x, void *y, int n);
int mg_qwen35_graph_bias(void *graph, void *x, const float *bias, int rows, int width);
int mg_qwen35_graph_swiglu(void *graph, void *gate, void *up, int n);
int mg_qwen35_graph_split(void *graph, void *src, int qwidth, int hd, void **q, void **gate);
int mg_qwen35_graph_attention(void *graph, void *q, void *k, void *v, void *gate,
    const float *qnorm, const float *knorm, const float *cosv, const float *sinv,
	const float *prefix_k, const float *prefix_v,
	int base, int nh, int nkv, int hd, int rotary, float scale, float qk_eps, int gain1p, int qknorm,
	int qnorm_elems, int knorm_elems,
    void **out, void **kraw, void **kpost, void **vcurrent);
void *mg_gdn_graph_encode(void *graph, int owner, void *mixed, void *z, void *b, void *a,
    const float *conv, const float *alog, const float *dtbias, const float *norm,
    int tokens, int nk, int nv, int khd, int vhd, int kernel, float eps);
int mg_gdn_graph_checkpoint(void *graph, int live_owner, int backup_owner);
int mg_gdn_state_swap_buffers(int live_owner, int backup_owner);
*/
import "C"

import (
	"errors"
	"math"
	"runtime"
	"sync/atomic"
	"unsafe"
)

// DeviceKV is a caller-owned device-resident KV triple shared by every
// full-attention layer across a panel walk. Unlike a GraphResult it is NOT
// tracked on any ProjectionGraph, so a panel's graph Free() never reclaims it:
// one DeviceKV lives for the whole sequence and each panel's new KRaw/KPost/V
// rows are appended into it by mg_qwen35_graph_attention_dkv, so panel p+1 reads
// the KPost/V prefix on the device instead of paying a host readback + prefix
// re-upload.
//
// Each of the three sides reserves LayerStride = tokens*NumKVHeads*HeadDim floats
// per layer; layer l's rows begin at l*LayerStride. Close frees all three device
// buffers. Side 0 is KRaw (pre-norm key), side 1 is KPost (attention prefix) and
// side 2 is V (raw value). A pair from NewDeviceKVAttendOnly has no KRaw side: the
// attention reads only KPost and V, so a caller whose host cache keeps KRaw (the
// dense decode graph) does not pay a third, never-read device side.
//
// Every pair counts toward DeviceKVResidentBytes until it is freed, and a pair that
// becomes unreachable without Close is freed by a runtime cleanup, so a dropped
// owner (a Session never Closed) cannot pin device memory for the process lifetime.
type DeviceKV struct {
	kraw, kpost, v unsafe.Pointer
	elems          int // per-side capacity in floats
	layerStride    int // floats per layer per side
	bytes          int64
	cleanup        runtime.Cleanup
}

// deviceKVResident is the process-wide byte count of live DeviceKV sides, so a
// caller can budget a new mirror against the device working set.
var deviceKVResident atomic.Int64

// DeviceKVResidentBytes reports the device bytes held by every live DeviceKV.
func DeviceKVResidentBytes() int64 { return deviceKVResident.Load() }

// deviceKVBuffers is the cleanup's copy of a pair's native handles. It holds only C
// pointers, never the *DeviceKV, so it does not keep the pair reachable.
type deviceKVBuffers struct {
	kraw, kpost, v unsafe.Pointer
	bytes          int64
}

func (b deviceKVBuffers) free() {
	for _, p := range []unsafe.Pointer{b.kraw, b.kpost, b.v} {
		if p != nil {
			C.mg_qwen35_graph_kv_free(p)
		}
	}
	deviceKVResident.Add(-b.bytes)
}

// NewDeviceKV allocates a persistent device KV triple sized for tokens tokens
// across layers full-attention layers, each layer holding kvWidth = nKV*hd
// floats per token. Returns nil when any device allocation or the geometry is
// unavailable, which the caller treats as a decline (fail-open to the host walk).
func NewDeviceKV(layers, tokens, kvWidth int) *DeviceKV {
	return newDeviceKV(layers, tokens, kvWidth, true)
}

// NewDeviceKVAttendOnly is NewDeviceKV without the KRaw side: the attention entry
// still appends and reads KPost/V on the device, while KRaw stays a per-panel graph
// result the caller reads back. Side 0 uploads and downloads are refused.
func NewDeviceKVAttendOnly(layers, tokens, kvWidth int) *DeviceKV {
	return newDeviceKV(layers, tokens, kvWidth, false)
}

func newDeviceKV(layers, tokens, kvWidth int, withRaw bool) *DeviceKV {
	if layers <= 0 || tokens <= 0 || kvWidth <= 0 || tokens > int(^uint(0)>>1)/kvWidth {
		return nil
	}
	layerStride := tokens * kvWidth
	if layers > int(^uint(0)>>1)/layerStride {
		return nil
	}
	elems := layers * layerStride
	// The native allocator, the device-KV offsets and the attention entry carry element
	// counts as C int: a side wider than that would silently truncate, so decline it.
	if elems > math.MaxInt32 {
		return nil
	}
	var bufs deviceKVBuffers
	sides := int64(2)
	if withRaw {
		sides = 3
		if bufs.kraw = C.mg_qwen35_graph_kv_alloc(C.int(elems)); bufs.kraw == nil {
			return nil
		}
	}
	if bufs.kpost = C.mg_qwen35_graph_kv_alloc(C.int(elems)); bufs.kpost == nil {
		bufs.free()
		return nil
	}
	if bufs.v = C.mg_qwen35_graph_kv_alloc(C.int(elems)); bufs.v == nil {
		bufs.free()
		return nil
	}
	bufs.bytes = sides * int64(elems) * 4
	deviceKVResident.Add(bufs.bytes)
	d := &DeviceKV{kraw: bufs.kraw, kpost: bufs.kpost, v: bufs.v, elems: elems, layerStride: layerStride, bytes: bufs.bytes}
	d.cleanup = runtime.AddCleanup(d, deviceKVBuffers.free, bufs)
	return d
}

// ResidentBytes reports the device bytes this pair holds (0 once closed).
func (d *DeviceKV) ResidentBytes() int64 {
	if d == nil {
		return 0
	}
	return d.bytes
}

// LayerStride reports the per-layer float width per side the pair was sized for,
// so a caller can validate its geometry before encoding.
func (d *DeviceKV) LayerStride() int {
	if d == nil {
		return 0
	}
	return d.layerStride
}

func (d *DeviceKV) side(side int) (unsafe.Pointer, error) {
	if d == nil || d.kpost == nil || d.v == nil {
		return nil, errors.New("metalgemm: device KV is not allocated")
	}
	switch side {
	case 0:
		if d.kraw == nil {
			return nil, errors.New("metalgemm: device KV has no KRaw side (attend-only pair)")
		}
		return d.kraw, nil
	case 1:
		return d.kpost, nil
	case 2:
		return d.v, nil
	default:
		return nil, errors.New("metalgemm: device KV side must be 0 (KRaw), 1 (KPost) or 2 (V)")
	}
}

// Upload copies host prefix rows into the device side at offset 0. It is used only
// to seed an already-resident prefix (e.g. a pre-existing cache); a fresh walk
// appends every row on the device and never uploads.
func (d *DeviceKV) Upload(side int, src []float32) error {
	return d.UploadRegion(side, 0, src)
}

// UploadRegion copies `src` into a device side starting at float offset `off`, so
// a layer's existing host prefix can be seeded at l*LayerStride() before a walk.
func (d *DeviceKV) UploadRegion(side, off int, src []float32) error {
	buf, err := d.side(side)
	if err != nil {
		return err
	}
	if len(src) == 0 {
		return nil
	}
	if off < 0 || off+len(src) > d.elems {
		return errors.New("metalgemm: device KV upload exceeds capacity")
	}
	// Offset-addressed copy: only [off, off+len) is written, so seeding layer l at
	// l*LayerStride() (or catching up one row) costs O(len), not O(off+len).
	ok := C.mg_qwen35_graph_kv_upload_at(buf, C.long(off), (*C.float)(unsafe.Pointer(&src[0])), C.long(len(src))) != 0
	// The pair's cleanup frees these buffers once d is unreachable; keep d alive
	// across the native copy so a caller dropping its last reference mid-call
	// cannot free the MTLBuffer under the memcpy.
	runtime.KeepAlive(d)
	if !ok {
		return errors.New("metalgemm: device KV upload failed")
	}
	return nil
}

// Download copies a device side back to a host slice once, at the END of a
// device-resident walk, so the host cache contract the decode path depends on
// still holds. This is the single terminal readback that replaces the per-panel
// one.
func (d *DeviceKV) Download(side int, dst []float32) error {
	return d.DownloadRegion(side, 0, dst)
}

// DownloadRegion copies `dst` floats starting at float offset `off` from a device
// side. It lets one persistent triple serve every full-attention layer: layer l's
// rows live at l*LayerStride(), and the walk downloads each slice into the host
// cache row exactly once at the end.
func (d *DeviceKV) DownloadRegion(side, off int, dst []float32) error {
	buf, err := d.side(side)
	if err != nil {
		return err
	}
	if len(dst) == 0 {
		return nil
	}
	if off < 0 || off+len(dst) > d.elems {
		return errors.New("metalgemm: device KV download exceeds capacity")
	}
	// Offset-addressed copy of exactly [off, off+len): no staging of the rows below.
	ok := C.mg_qwen35_graph_kv_download_at(buf, C.long(off), (*C.float)(unsafe.Pointer(&dst[0])), C.long(len(dst))) != 0
	// The pair's cleanup frees these buffers once d is unreachable; keep d alive
	// across the native copy so a caller dropping its last reference mid-call
	// cannot free the MTLBuffer under the memcpy.
	runtime.KeepAlive(d)
	if !ok {
		return errors.New("metalgemm: device KV download failed")
	}
	return nil
}

// Close frees the device KV triple. Safe to call twice.
func (d *DeviceKV) Close() {
	if d == nil || (d.kraw == nil && d.kpost == nil && d.v == nil) {
		return
	}
	// Stop the unreachability cleanup first: it owns the same handles.
	d.cleanup.Stop()
	deviceKVBuffers{kraw: d.kraw, kpost: d.kpost, v: d.v, bytes: d.bytes}.free()
	d.kraw, d.kpost, d.v = nil, nil, nil
	d.elems, d.layerStride, d.bytes = 0, 0, 0
}

// KVQ8BlockSize is the element count sharing one f32 scale in the packed Q8_0 KV
// layout. It equals internal/model's KVQuantQ8_0BlockSize: the packed rows these
// entries produce and consume are the host cache's realized kvPackedRow bytes.
const KVQ8BlockSize = 32

// KVQ8RowBytes is the packed size of one kvWidth-wide K or V row: one int8 code per
// element plus one f32 scale per KVQ8BlockSize elements (1152 B at the Qwen3.8
// kvWidth=1024, against 4096 B as F32).
func KVQ8RowBytes(kvWidth int) int { return kvWidth + kvWidth/KVQ8BlockSize*4 }

// KVQ8Rows is a run of packed Q8_0 KV rows in the host cache's realized layout
// (internal/model/kvcache_q8.go kvPackedRow): row-major int8 codes, one per element,
// and one f32 scale per KVQ8BlockSize elements, for post-RoPE K and raw V.
type KVQ8Rows struct {
	KCodes  []int8
	KScales []float32
	VCodes  []int8
	VScales []float32
}

// Rows reports how many kvWidth-wide rows r holds, or -1 when its four sides
// disagree with each other or with the block geometry.
func (r KVQ8Rows) Rows(kvWidth int) int {
	if kvWidth <= 0 || kvWidth%KVQ8BlockSize != 0 || len(r.KCodes)%kvWidth != 0 {
		return -1
	}
	n := len(r.KCodes) / kvWidth
	blocks := n * (kvWidth / KVQ8BlockSize)
	if len(r.VCodes) != len(r.KCodes) || len(r.KScales) != blocks || len(r.VScales) != blocks {
		return -1
	}
	return n
}

// KVQ8Available reports whether the packed Q8_0 KV pipelines compiled on this
// device. They are a separate library from the F32 graph, so false declines only
// the Q8 entries and never the F32 route. It initializes the device first: probing
// the pipelines before the device exists would latch both graph libraries declined.
func KVQ8Available() bool { return Available() && C.mg_qwen35_graph_kvq8_ready() != 0 }

// KVQ8Codes reinterprets a terminal readback of a KCodes/VCodes result as the int8
// codes it carries. The returned slice aliases words.
func KVQ8Codes(words []float32) []int8 {
	if len(words) == 0 {
		return nil
	}
	return unsafe.Slice((*int8)(unsafe.Pointer(&words[0])), len(words)*4)
}

// DeviceKVQ8 is DeviceKV's packed twin (#12981): a caller-owned device store holding
// every full-attention layer's post-RoPE K and raw V rows as Q8_0 codes plus block
// scales, KVQ8RowBytes per row per side, never an F32 mirror. KRaw is not held: the
// host keeps the f32 pre-RoPE row (FullAttentionDeviceQ8 returns it per panel), which
// is the planner's mixed KVPrecisionQ8 layout. Layer l's rows start at element
// l*tokens*kvWidth of each side, so one store serves every layer of a walk.
type DeviceKVQ8 struct {
	kc, ks, vc, vs          unsafe.Pointer
	layers, tokens, kvWidth int
}

// NewDeviceKVQ8 allocates a packed store for tokens rows across layers layers of
// kvWidth elements. Returns nil when the geometry, the Q8 pipelines or any device
// allocation is unavailable, which the caller treats as a decline.
func NewDeviceKVQ8(layers, tokens, kvWidth int) *DeviceKVQ8 {
	if layers <= 0 || tokens <= 0 || kvWidth <= 0 || kvWidth%KVQ8BlockSize != 0 || tokens > int(^uint(0)>>1)/kvWidth || layers > int(^uint(0)>>1)/(tokens*kvWidth) || !KVQ8Available() {
		return nil
	}
	codes := layers * tokens * kvWidth
	scales := codes / KVQ8BlockSize * 4
	d := &DeviceKVQ8{layers: layers, tokens: tokens, kvWidth: kvWidth}
	for _, side := range []struct {
		p *unsafe.Pointer
		n int
	}{{&d.kc, codes}, {&d.ks, scales}, {&d.vc, codes}, {&d.vs, scales}} {
		if *side.p = C.mg_qwen35_graph_kvq8_alloc(C.long(side.n)); *side.p == nil {
			d.Close()
			return nil
		}
	}
	return d
}

// Tokens reports the per-layer row capacity the store was sized for.
func (d *DeviceKVQ8) Tokens() int {
	if d == nil {
		return 0
	}
	return d.tokens
}

// ResidentBytes reports the packed bytes the store keeps resident: both sides at
// full capacity. At 16 layers x 20480 tokens x kvWidth 1024 that is 0.70 GiB,
// against 2.50 GiB for DeviceKV's F32 KPost+V sides.
func (d *DeviceKVQ8) ResidentBytes() int64 {
	if d == nil || d.kc == nil {
		return 0
	}
	return 2 * int64(d.layers) * int64(d.tokens) * int64(KVQ8RowBytes(d.kvWidth))
}

// region maps rows [row,row+n) of layer to byte offsets/lengths of the code and
// scale sides.
func (d *DeviceKVQ8) region(layer, row, n int) (codeOff, scaleOff, codeLen, scaleLen int, err error) {
	if d == nil || d.kc == nil || d.ks == nil || d.vc == nil || d.vs == nil {
		return 0, 0, 0, 0, errors.New("metalgemm: packed device KV is not allocated")
	}
	if layer < 0 || layer >= d.layers || row < 0 || n < 0 || row > d.tokens-n {
		return 0, 0, 0, 0, errors.New("metalgemm: packed device KV region out of range")
	}
	codeOff, codeLen = (layer*d.tokens+row)*d.kvWidth, n*d.kvWidth
	return codeOff, codeOff / KVQ8BlockSize * 4, codeLen, codeLen / KVQ8BlockSize * 4, nil
}

// WriteRows seeds rows [row, row+n) of layer with host packed rows, e.g. a prefix
// the host cache already holds. A fresh walk appends on the device and never writes.
func (d *DeviceKVQ8) WriteRows(layer, row int, rows KVQ8Rows) error {
	n := rows.Rows(d.width())
	if n < 0 {
		return errors.New("metalgemm: packed Q8 rows disagree with the store geometry")
	}
	codeOff, scaleOff, codeLen, scaleLen, err := d.region(layer, row, n)
	if err != nil || n == 0 {
		return err
	}
	for _, side := range []struct {
		buf      unsafe.Pointer
		off, len int
		src      unsafe.Pointer
	}{{d.kc, codeOff, codeLen, unsafe.Pointer(&rows.KCodes[0])}, {d.ks, scaleOff, scaleLen, unsafe.Pointer(&rows.KScales[0])},
		{d.vc, codeOff, codeLen, unsafe.Pointer(&rows.VCodes[0])}, {d.vs, scaleOff, scaleLen, unsafe.Pointer(&rows.VScales[0])}} {
		if C.mg_qwen35_graph_kvq8_write(side.buf, C.long(side.off), side.src, C.long(side.len)) == 0 {
			return errors.New("metalgemm: packed device KV write failed")
		}
	}
	return nil
}

// ReadRows copies rows [row, row+n) of layer back into the host packed layout. Call
// it only after every graph appending to those rows has finished.
func (d *DeviceKVQ8) ReadRows(layer, row, n int) (KVQ8Rows, error) {
	codeOff, scaleOff, codeLen, scaleLen, err := d.region(layer, row, n)
	if err != nil || n == 0 {
		return KVQ8Rows{}, err
	}
	out := KVQ8Rows{KCodes: make([]int8, codeLen), KScales: make([]float32, scaleLen/4), VCodes: make([]int8, codeLen), VScales: make([]float32, scaleLen/4)}
	for _, side := range []struct {
		buf      unsafe.Pointer
		off, len int
		dst      unsafe.Pointer
	}{{d.kc, codeOff, codeLen, unsafe.Pointer(&out.KCodes[0])}, {d.ks, scaleOff, scaleLen, unsafe.Pointer(&out.KScales[0])},
		{d.vc, codeOff, codeLen, unsafe.Pointer(&out.VCodes[0])}, {d.vs, scaleOff, scaleLen, unsafe.Pointer(&out.VScales[0])}} {
		if C.mg_qwen35_graph_kvq8_read(side.buf, C.long(side.off), side.dst, C.long(side.len)) == 0 {
			return KVQ8Rows{}, errors.New("metalgemm: packed device KV read failed")
		}
	}
	return out, nil
}

func (d *DeviceKVQ8) width() int {
	if d == nil {
		return 0
	}
	return d.kvWidth
}

// Close frees the packed store. Safe to call twice.
func (d *DeviceKVQ8) Close() {
	if d == nil {
		return
	}
	for _, p := range []*unsafe.Pointer{&d.kc, &d.ks, &d.vc, &d.vs} {
		if *p != nil {
			C.mg_qwen35_graph_kv_free(*p)
			*p = nil
		}
	}
	d.layers, d.tokens = 0, 0
}
