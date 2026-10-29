#pragma once
#ifdef FAK_CUDA_DEEPGEMM_WKV
// Terminal uncertain operands remain live until device teardown. Shared exact
// retirement routes must consult these pointers before erasing ownership.
static bool g_deepgemm_wkv_uncertain=false;
static void *g_deepgemm_wkv_quarantine[7]={};
#endif
