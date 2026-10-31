#pragma once
#include <stdint.h>
#include "cuda_flashmla_sparse_decode.h"
#ifdef __cplusplus
extern "C" {
#endif
// Synchronous private qualification entry. Input bytes copied into C ownership;
// neither success nor failure retains a Go pointer. No host fallback.
int fak_flashmla_decode_host(const uint16_t*,const uint8_t*,const int32_t*,const int32_t*,const float*,
 const uint8_t*,const int32_t*,const int32_t*,uint16_t*,float*,
 int,int,int,int,int,int,float,int);
#ifdef __cplusplus
}
#endif
