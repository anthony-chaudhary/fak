#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
enum { FAK_DG_WKV_SHAPE=-1, FAK_DG_WKV_UNAVAILABLE=-2, FAK_DG_WKV_BUFFER=-3,
 FAK_DG_WKV_CAPTURE=-4, FAK_DG_WKV_UNCERTAIN=-5, FAK_DG_WKV_NONFINITE=-6 };
typedef struct {int device,major,minor,sms,compiled_sms,binary_version,runtime_version;char name[256];uint8_t uuid[16];} fak_deepgemm_wkv_facts;
int fcuda_deepgemm_wkv_facts(fak_deepgemm_wkv_facts*);
int fcuda_deepgemm_wkv_state(void);
int fcuda_deepgemm_wkv_upload(void*,const void*,size_t);
int fcuda_deepgemm_wkv_release(void*);
int fcuda_deepgemm_wkv_run(const float*,const uint8_t*,const uint32_t*,uint8_t*,uint32_t*,uint32_t*,void*,int);
#ifdef __cplusplus
}
int fak_deepgemm_wkv_launch(const uint8_t*,const uint8_t*,const uint32_t*,const uint32_t*,void*,int,void*);
#endif
