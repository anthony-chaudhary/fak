//go:build cgo

#include "platform.h"
#if FAK_INKLING_METAL_SDK
#include "ops/RelativeAttention.cpp"
#endif
