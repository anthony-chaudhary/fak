#ifndef FAK_METALGEMM_Q8_BRIDGE_H
#define FAK_METALGEMM_Q8_BRIDGE_H

// Internal cross-translation-unit contract used when q4k.m owns a command buffer and q8.m
// contributes encoders and readback. Both translation units include this header so Clang checks
// the caller and definitions against the same signatures.
int mg_q8_prepare_gemv_group(const int* wids, int n, const signed char* xq, const float* xd,
                             const int* yoff);
int mg_q8_encode_gemv_group(void* command_buffer, const int* wids, int n, const int* yoff);
void mg_q8_read_gemv_group(float* y, int ytot);

#endif
