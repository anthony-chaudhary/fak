#ifndef FAK_V41_EMBED_GATHER_GEOMETRY_H
#define FAK_V41_EMBED_GATHER_GEOMETRY_H
#include <stdint.h>
#include <limits.h>
#include <stddef.h>
// Pure pre-dispatch guard used by the real native ABI and host tests.
// IDs are synchronously borrowed, checked in full before any native operation.
// ABI modes: 0=F32, 1=Q2_K, 2=Q5_K. Unknown modes fail before dispatch.
static inline bool v41_eg_geometry(const intptr_t *ids,int rows,int hidden,int vocab,int mode,uint64_t *rb,uint64_t *ob){
 if(!ids||!rb||!ob||rows<=0||hidden<=0||vocab<=0||mode<0||mode>2||(mode&&hidden%256))return false;
 const uint64_t rowbytes=mode?(uint64_t)(hidden/256)*(mode==1?84:176):(uint64_t)hidden*4;
 const uint64_t outputbytes=(uint64_t)rows*(uint64_t)hidden*4;
 if(outputbytes>INT_MAX||rowbytes*(uint64_t)vocab>SIZE_MAX||(mode==2&&rowbytes*(uint64_t)vocab>INT_MAX))return false;
 for(int r=0;r<rows;++r)if(ids[r]<0||ids[r]>=vocab)return false;
 *rb=rowbytes;*ob=outputbytes;return true;
}
#endif
