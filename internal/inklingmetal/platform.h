#ifndef FAK_INKLING_METAL_PLATFORM_H
#define FAK_INKLING_METAL_PLATFORM_H
#include <Availability.h>
// The embedded Metal shaders use MSL 4.0 and Clang's source embedding.
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260400 && defined(__has_embed)
#define FAK_INKLING_METAL_SDK 1
#else
#define FAK_INKLING_METAL_SDK 0
#endif
#endif
