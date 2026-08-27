//go:build darwin && arm64 && cgo

#import <Metal/Metal.h>
#include <math.h>

extern id<MTLDevice> gDev;
extern void *mg_graph_command_buffer(void *graph);
extern void *mg_graph_alloc_result(void *graph, int n);
extern void *mg_graph_alloc_buffer(void *graph, int n);
extern void mg_graph_note_encoder(void *graph);

static id<MTLComputePipelineState> qgNorm, qgAdd, qgSwiGLU, qgSplit, qgQK, qgAttn, qgQKDecodeBatch, qgAttnDecodeBatch;
static BOOL qgAttempted;

static NSString *qgSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void qg_norm(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                    device float *y [[buffer(2)]], constant int& width [[buffer(3)]],
                    constant float& eps [[buffer(4)]], constant int& gain1p [[buffer(5)]],
                    constant int& sourceRow [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_threadgroup]]) {
    int r = sourceRow >= 0 ? sourceRow : (int)row;
    threadgroup float sums[256]; float ss=0.0f;
    for(int i=(int)lane;i<width;i+=256){float v=x[(long)r*width+i];ss+=v*v;}
    sums[lane]=ss;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint o=128;o;o>>=1){if(lane<o)sums[lane]+=sums[lane+o];threadgroup_barrier(mem_flags::mem_threadgroup);}
    float inv=rsqrt(sums[0]/(float)width+eps);
    int dr=sourceRow>=0?0:(int)row;
    for(int i=(int)lane;i<width;i+=256){float gain=gain1p?1.0f+w[i]:w[i];y[(long)dr*width+i]=x[(long)r*width+i]*inv*gain;}
}

kernel void qg_add(device float*x [[buffer(0)]],device const float*y [[buffer(1)]],constant int&n [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<(uint)n)x[i]+=y[i];}
kernel void qg_swiglu(device float*g [[buffer(0)]],device const float*u [[buffer(1)]],constant int&n [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<(uint)n){float v=g[i];g[i]=(v/(1.0f+exp(-v)))*u[i];}}
kernel void qg_split(device const float*src [[buffer(0)]],device float*q [[buffer(1)]],device float*gate [[buffer(2)]],constant int&qwidth [[buffer(3)]],constant int&hd [[buffer(4)]],uint i [[thread_position_in_grid]]){if(i<(uint)(32*qwidth)){int t=(int)i/qwidth,j=(int)i-t*qwidth,h=j/hd,d=j-h*hd;long base=(long)t*2*qwidth+(long)h*2*hd;q[i]=src[base+d];gate[i]=src[base+hd+d];}}

kernel void qg_qk(device const float*qIn [[buffer(0)]],device const float*kIn [[buffer(1)]],
                  device const float*qw [[buffer(2)]],device const float*kw [[buffer(3)]],
                  device float*qOut [[buffer(4)]],device float*kRaw [[buffer(5)]],device float*kPost [[buffer(6)]],
                  constant int&nH [[buffer(7)]],constant int&nKV [[buffer(8)]],constant int&hd [[buffer(9)]],constant int&rotary [[buffer(10)]],
                  constant int&base [[buffer(11)]],device const float*cosv [[buffer(12)]],device const float*sinv [[buffer(13)]],constant float&qkEps [[buffer(14)]],constant int&gain1p [[buffer(15)]],constant int&qknorm [[buffer(16)]],constant int&qnw [[buffer(17)]],constant int&knw [[buffer(18)]],
                  uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]){
    int h=(int)group.x,t=(int)group.y;threadgroup float qs[256],ks[256];
    float q=0,k=0;if(h<nH&&lane<(uint)hd)q=qIn[((long)t*nH+h)*hd+lane];if(h<nKV&&lane<(uint)hd)k=kIn[((long)t*nKV+h)*hd+lane];
    float qss=0,kss=0;
    if(qnw==hd)qss=q*q;else for(int i=(int)lane;i<nH*hd;i+=256){float z=qIn[(long)t*nH*hd+i];qss+=z*z;}
    if(knw==hd)kss=k*k;else for(int i=(int)lane;i<nKV*hd;i+=256){float z=kIn[(long)t*nKV*hd+i];kss+=z*z;}
    qs[lane]=qss;ks[lane]=kss;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint o=128;o;o>>=1){if(lane<o){qs[lane]+=qs[lane+o];ks[lane]+=ks[lane+o];}threadgroup_barrier(mem_flags::mem_threadgroup);}
    if(qknorm&&h<nH&&lane<(uint)hd){float w=qw[(qnw==hd?0:h*hd)+(int)lane];q*=rsqrt(qs[0]/(float)qnw+qkEps)*(gain1p?1.0f+w:w);}
    if(qknorm&&h<nKV&&lane<(uint)hd){float w=kw[(knw==hd?0:h*hd)+(int)lane];k*=rsqrt(ks[0]/(float)knw+qkEps)*(gain1p?1.0f+w:w);}
    if(h<nH&&lane<(uint)hd)qOut[((long)t*nH+h)*hd+lane]=q;
    if(h<nKV&&lane<(uint)hd){long ix=((long)t*nKV+h)*hd+lane;kRaw[ix]=k;kPost[ix]=k;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int halfn=rotary/2;if(lane<(uint)halfn){int j=(int)lane;float c=cosv[(long)t*halfn+j],s=sinv[(long)t*halfn+j];
        if(h<nH){long i=((long)t*nH+h)*hd+j;float av=qOut[i],bv=qOut[i+halfn];qOut[i]=av*c-bv*s;qOut[i+halfn]=av*s+bv*c;}
        if(h<nKV){long i=((long)t*nKV+h)*hd+j;float av=kPost[i],bv=kPost[i+halfn];kPost[i]=av*c-bv*s;kPost[i+halfn]=av*s+bv*c;}}
}

kernel void qg_attn(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device const float*gate [[buffer(3)]],device float*out [[buffer(4)]],
                    constant int&total [[buffer(5)]],constant int&base [[buffer(6)]],constant int&nH [[buffer(7)]],constant int&nKV [[buffer(8)]],constant int&hd [[buffer(9)]],constant float&scale [[buffer(10)]],
                    uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]){
    int h=(int)group.x,t=(int)group.y,kh=h/(nH/nKV),upto=base+t+1;threadgroup float score[4096],mx,den;
    if(lane==0){float m=-INFINITY;for(int j=0;j<upto;j++){float z=0;for(int d=0;d<hd;d++)z+=q[((long)t*nH+h)*hd+d]*k[((long)j*nKV+kh)*hd+d];z*=scale;score[j]=z;m=max(m,z);}float sum=0;for(int j=0;j<upto;j++){score[j]=exp(score[j]-m);sum+=score[j];}mx=m;den=sum;}threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane<(uint)hd){float z=0;for(int j=0;j<upto;j++)z+=score[j]*v[((long)j*nKV+kh)*hd+lane];long i=((long)t*nH+h)*hd+lane;out[i]=(z/den)/(1.0f+exp(-gate[i]));}
}
kernel void qg_qk_decode_batch(device const float*qgIn [[buffer(0)]],device const float*kIn [[buffer(1)]],
                  device const float*qw [[buffer(2)]],device const float*kw [[buffer(3)]],
                  device const float*cosv [[buffer(4)]],device const float*sinv [[buffer(5)]],
                  device float*qOut [[buffer(6)]],device float*gate [[buffer(7)]],device float*kRaw [[buffer(8)]],device float*kPost [[buffer(9)]],
                  constant int&nH [[buffer(10)]],constant int&nKV [[buffer(11)]],constant int&hd [[buffer(12)]],constant int&rotary [[buffer(13)]],
                  constant float&qkEps [[buffer(14)]],constant int&gain1p [[buffer(15)]],constant int&qknorm [[buffer(16)]],constant int&qnw [[buffer(17)]],constant int&knw [[buffer(18)]],
                  uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]){
    int h=(int)group.x,row=(int)group.y,qwidth=nH*hd,kvwidth=nKV*hd;threadgroup float qs[256],ks[256];float q=0,k=0;
    if(h<nH&&lane<(uint)hd){long src=(long)row*2*qwidth+(long)h*2*hd+lane;q=qgIn[src];gate[((long)row*nH+h)*hd+lane]=qgIn[src+hd];}
    if(h<nKV&&lane<(uint)hd)k=kIn[((long)row*nKV+h)*hd+lane];float qss=0,kss=0;
    if(h<nH){if(qnw==hd)qss=q*q;else for(int i=(int)lane;i<qwidth;i+=256){int qh=i/hd,d=i-qh*hd;float z=qgIn[(long)row*2*qwidth+(long)qh*2*hd+d];qss+=z*z;}}
    if(h<nKV){if(knw==hd)kss=k*k;else for(int i=(int)lane;i<kvwidth;i+=256){float z=kIn[(long)row*kvwidth+i];kss+=z*z;}}
    qs[lane]=qss;ks[lane]=kss;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint o=128;o;o>>=1){if(lane<o){qs[lane]+=qs[lane+o];ks[lane]+=ks[lane+o];}threadgroup_barrier(mem_flags::mem_threadgroup);}
    if(qknorm&&h<nH&&lane<(uint)hd){float w=qw[(qnw==hd?0:h*hd)+(int)lane];q*=rsqrt(qs[0]/(float)qnw+qkEps)*(gain1p?1.0f+w:w);}
    if(qknorm&&h<nKV&&lane<(uint)hd){float w=kw[(knw==hd?0:h*hd)+(int)lane];k*=rsqrt(ks[0]/(float)knw+qkEps)*(gain1p?1.0f+w:w);}
    if(h<nH&&lane<(uint)hd)qOut[((long)row*nH+h)*hd+lane]=q;
    if(h<nKV&&lane<(uint)hd){long ix=((long)row*nKV+h)*hd+lane;kRaw[ix]=k;kPost[ix]=k;}threadgroup_barrier(mem_flags::mem_threadgroup);
    int halfn=rotary/2;if(lane<(uint)halfn){int j=(int)lane;float c=cosv[(long)row*halfn+j],s=sinv[(long)row*halfn+j];
        if(h<nH){long i=((long)row*nH+h)*hd+j;float a=qOut[i],b=qOut[i+halfn];qOut[i]=a*c-b*s;qOut[i+halfn]=a*s+b*c;}
        if(h<nKV){long i=((long)row*nKV+h)*hd+j;float a=kPost[i],b=kPost[i+halfn];kPost[i]=a*c-b*s;kPost[i+halfn]=a*s+b*c;}}
}

kernel void qg_attn_decode_batch(device const float*q [[buffer(0)]],device const float*kPrefix [[buffer(1)]],device const float*vPrefix [[buffer(2)]],
                    device const int*offsets [[buffer(3)]],device const float*kCurrent [[buffer(4)]],device const float*vCurrent [[buffer(5)]],
                    device const float*gate [[buffer(6)]],device float*out [[buffer(7)]],constant int&nH [[buffer(8)]],constant int&nKV [[buffer(9)]],
                    constant int&hd [[buffer(10)]],constant float&scale [[buffer(11)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]){
    int h=(int)group.x,row=(int)group.y,kh=h/(nH/nKV),start=offsets[row],base=offsets[row+1]-start,total=base+1;threadgroup float score[4096],den;
    if(lane==0){float m=-INFINITY;for(int j=0;j<total;j++){float z=0;for(int d=0;d<hd;d++){float kval=j<base?kPrefix[((long)(start+j)*nKV+kh)*hd+d]:kCurrent[((long)row*nKV+kh)*hd+d];z+=q[((long)row*nH+h)*hd+d]*kval;}z*=scale;score[j]=z;m=max(m,z);}float sum=0;for(int j=0;j<total;j++){score[j]=exp(score[j]-m);sum+=score[j];}den=sum;}threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane<(uint)hd){float z=0;for(int j=0;j<total;j++){float vv=j<base?vPrefix[((long)(start+j)*nKV+kh)*hd+lane]:vCurrent[((long)row*nKV+kh)*hd+lane];z+=score[j]*vv;}long i=((long)row*nH+h)*hd+lane;out[i]=(z/den)/(1.0f+exp(-gate[i]));}
}
)MSL";

static int qg_init(void){@synchronized(gDev){if(qgNorm)return qgQKDecodeBatch&&qgAttnDecodeBatch;if(qgAttempted)return 0;qgAttempted=YES;NSError*e=nil;id<MTLLibrary>l=[gDev newLibraryWithSource:qgSource options:nil error:&e];if(!l){NSLog(@"qwen35 graph compile: %@",e);return 0;}qgNorm=[gDev newComputePipelineStateWithFunction:[l newFunctionWithName:@"qg_norm"] error:&e];qgAdd=[gDev newComputePipelineStateWithFunction:[l newFunctionWithName:@"qg_add"] error:&e];qgSwiGLU=[gDev newComputePipelineStateWithFunction:[l newFunctionWithName:@"qg_swiglu"] error:&e];qgSplit=[gDev newComputePipelineStateWithFunction:[l newFunctionWithName:@"qg_split"] error:&e];qgQK=[gDev newComputePipelineStateWithFunction:[l newFunctionWithName:@"qg_qk"] error:&e];qgAttn=[gDev newComputePipelineStateWithFunction:[l newFunctionWithName:@"qg_attn"] error:&e];qgQKDecodeBatch=[gDev newComputePipelineStateWithFunction:[l newFunctionWithName:@"qg_qk_decode_batch"] error:&e];qgAttnDecodeBatch=[gDev newComputePipelineStateWithFunction:[l newFunctionWithName:@"qg_attn_decode_batch"] error:&e];return qgNorm&&qgAdd&&qgSwiGLU&&qgSplit&&qgQK&&qgAttn&&qgQKDecodeBatch&&qgAttnDecodeBatch;}}
static id<MTLBuffer> qg_host(const float*p,int n){return[gDev newBufferWithBytes:p length:(NSUInteger)n*sizeof(float) options:MTLResourceStorageModeShared];}
static id<MTLBuffer> qg_host_i32(const int*p,int n){return[gDev newBufferWithBytes:p length:(NSUInteger)n*sizeof(int) options:MTLResourceStorageModeShared];}
static void qg_dispatch(id<MTLComputeCommandEncoder>e,id<MTLComputePipelineState>p,int n){int t=(int)p.maxTotalThreadsPerThreadgroup;if(t>n)t=n;if(t<1)t=1;[e dispatchThreads:MTLSizeMake((NSUInteger)n,1,1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)t,1,1)];}

void *mg_qwen35_graph_norm(void*g,void*input,const float*w,int rows,int width,float eps,int gain1p,int lastOnly){if(!qg_init()||!g||!input||!w||rows<=0||width<=0||eps<=0||(lastOnly&&rows!=32))return NULL;id<MTLCommandBuffer>cb=(__bridge id<MTLCommandBuffer>)mg_graph_command_buffer(g);id<MTLBuffer>x=(__bridge id<MTLBuffer>)input,wb=qg_host(w,width),y=(__bridge id<MTLBuffer>)mg_graph_alloc_result(g,lastOnly?width:rows*width);if(!cb||!x||!wb||!y)return NULL;id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];[e setComputePipelineState:qgNorm];[e setBuffer:x offset:0 atIndex:0];[e setBuffer:wb offset:0 atIndex:1];[e setBuffer:y offset:0 atIndex:2];[e setBytes:&width length:4 atIndex:3];[e setBytes:&eps length:4 atIndex:4];[e setBytes:&gain1p length:4 atIndex:5];int row=lastOnly?31:-1;[e setBytes:&row length:4 atIndex:6];[e dispatchThreadgroups:MTLSizeMake(lastOnly?1:rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];return(__bridge void*)y;}
int mg_qwen35_graph_add(void*g,void*xp,void*yp,int n){if(!qg_init()||!g||!xp||!yp||n<=0)return 0;id<MTLCommandBuffer>cb=(__bridge id<MTLCommandBuffer>)mg_graph_command_buffer(g);id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];[e setComputePipelineState:qgAdd];[e setBuffer:(__bridge id<MTLBuffer>)xp offset:0 atIndex:0];[e setBuffer:(__bridge id<MTLBuffer>)yp offset:0 atIndex:1];[e setBytes:&n length:4 atIndex:2];qg_dispatch(e,qgAdd,n);[e endEncoding];mg_graph_note_encoder(g);return 1;}
int mg_qwen35_graph_swiglu(void*g,void*gp,void*up,int n){if(!qg_init()||!g||!gp||!up||n<=0)return 0;id<MTLCommandBuffer>cb=(__bridge id<MTLCommandBuffer>)mg_graph_command_buffer(g);id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];[e setComputePipelineState:qgSwiGLU];[e setBuffer:(__bridge id<MTLBuffer>)gp offset:0 atIndex:0];[e setBuffer:(__bridge id<MTLBuffer>)up offset:0 atIndex:1];[e setBytes:&n length:4 atIndex:2];qg_dispatch(e,qgSwiGLU,n);[e endEncoding];mg_graph_note_encoder(g);return 1;}
int mg_qwen35_graph_split(void*g,void*srcp,int qwidth,int hd,void**qout,void**gateout){if(!qg_init()||!g||!srcp||qwidth<=0||hd<=0||!qout||!gateout)return 0;id<MTLBuffer>q=(__bridge id<MTLBuffer>)mg_graph_alloc_result(g,32*qwidth),gate=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,32*qwidth);if(!q||!gate)return 0;id<MTLCommandBuffer>cb=(__bridge id<MTLCommandBuffer>)mg_graph_command_buffer(g);id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];[e setComputePipelineState:qgSplit];[e setBuffer:(__bridge id<MTLBuffer>)srcp offset:0 atIndex:0];[e setBuffer:q offset:0 atIndex:1];[e setBuffer:gate offset:0 atIndex:2];[e setBytes:&qwidth length:4 atIndex:3];[e setBytes:&hd length:4 atIndex:4];qg_dispatch(e,qgSplit,32*qwidth);[e endEncoding];*qout=(__bridge void*)q;*gateout=(__bridge void*)gate;return 1;}

int mg_qwen35_graph_attention(void*g,void*qp,void*kp,void*vp,void*gatep,const float*qw,const float*kw,const float*cosv,const float*sinv,const float*prefixK,const float*prefixV,int base,int nH,int nKV,int hd,int rotary,float scale,float qkEps,int gain1p,int qknorm,int qnw,int knw,void**outp,void**krawp,void**kpostp,void**vcurp){
    if(!qg_init()||!g||!qp||!kp||!vp||!gatep||!qw||!kw||!cosv||!sinv||base<0||base+32>4096||nH<1||nKV<1||hd<2||hd>256||(qnw!=hd&&qnw!=nH*hd)||(knw!=hd&&knw!=nKV*hd)||rotary<2||rotary>hd||rotary%2||qkEps<=0||!outp||!krawp||!kpostp||!vcurp)return 0;int qn=32*nH*hd,kn=32*nKV*hd,total=base+32;
    id<MTLBuffer>qo=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,qn),kr=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,kn),kpo=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,kn),out=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,qn);id<MTLBuffer>qa=qg_host(qw,qnw),ka=qg_host(kw,knw);if(!qo||!kr||!kpo||!out||!qa||!ka)return 0;
    id<MTLBuffer>cbv=qg_host(cosv,32*(rotary/2)),sbv=qg_host(sinv,32*(rotary/2));if(!cbv||!sbv)return 0;id<MTLCommandBuffer>cb=(__bridge id<MTLCommandBuffer>)mg_graph_command_buffer(g);id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];[e setComputePipelineState:qgQK];[e setBuffer:(__bridge id<MTLBuffer>)qp offset:0 atIndex:0];[e setBuffer:(__bridge id<MTLBuffer>)kp offset:0 atIndex:1];[e setBuffer:qa offset:0 atIndex:2];[e setBuffer:ka offset:0 atIndex:3];[e setBuffer:qo offset:0 atIndex:4];[e setBuffer:kr offset:0 atIndex:5];[e setBuffer:kpo offset:0 atIndex:6];[e setBytes:&nH length:4 atIndex:7];[e setBytes:&nKV length:4 atIndex:8];[e setBytes:&hd length:4 atIndex:9];[e setBytes:&rotary length:4 atIndex:10];[e setBytes:&base length:4 atIndex:11];[e setBuffer:cbv offset:0 atIndex:12];[e setBuffer:sbv offset:0 atIndex:13];[e setBytes:&qkEps length:4 atIndex:14];[e setBytes:&gain1p length:4 atIndex:15];[e setBytes:&qknorm length:4 atIndex:16];[e setBytes:&qnw length:4 atIndex:17];[e setBytes:&knw length:4 atIndex:18];[e dispatchThreadgroups:MTLSizeMake((NSUInteger)MAX(nH,nKV),32,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];mg_graph_note_encoder(g);
    int prefixElems=base*nKV*hd;id<MTLBuffer>kall=[gDev newBufferWithLength:(NSUInteger)total*nKV*hd*sizeof(float) options:MTLResourceStorageModeShared],vall=[gDev newBufferWithLength:(NSUInteger)total*nKV*hd*sizeof(float) options:MTLResourceStorageModeShared];if(!kall||!vall)return 0;if(prefixElems){if(!prefixK||!prefixV)return 0;memcpy(kall.contents,prefixK,(size_t)prefixElems*sizeof(float));memcpy(vall.contents,prefixV,(size_t)prefixElems*sizeof(float));}
    id<MTLBlitCommandEncoder>b=[cb blitCommandEncoder];[b copyFromBuffer:kpo sourceOffset:0 toBuffer:kall destinationOffset:(NSUInteger)prefixElems*sizeof(float) size:(NSUInteger)kn*sizeof(float)];[b copyFromBuffer:(__bridge id<MTLBuffer>)vp sourceOffset:0 toBuffer:vall destinationOffset:(NSUInteger)prefixElems*sizeof(float) size:(NSUInteger)kn*sizeof(float)];[b endEncoding];mg_graph_note_encoder(g);
    e=[cb computeCommandEncoder];[e setComputePipelineState:qgAttn];[e setBuffer:qo offset:0 atIndex:0];[e setBuffer:kall offset:0 atIndex:1];[e setBuffer:vall offset:0 atIndex:2];[e setBuffer:(__bridge id<MTLBuffer>)gatep offset:0 atIndex:3];[e setBuffer:out offset:0 atIndex:4];[e setBytes:&total length:4 atIndex:5];[e setBytes:&base length:4 atIndex:6];[e setBytes:&nH length:4 atIndex:7];[e setBytes:&nKV length:4 atIndex:8];[e setBytes:&hd length:4 atIndex:9];[e setBytes:&scale length:4 atIndex:10];[e dispatchThreadgroups:MTLSizeMake((NSUInteger)nH,32,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];mg_graph_note_encoder(g);
    *outp=(__bridge void*)out;*krawp=(__bridge void*)kr;*kpostp=(__bridge void*)kpo;*vcurp=vp;return 1;
}
int mg_qwen35_graph_attention_decode_batch(void*g,void*qp,void*kp,void*vp,const float*qw,const float*kw,const float*cosv,const float*sinv,const float*prefixK,const float*prefixV,const int*prefixOffsets,int batch,int nH,int nKV,int hd,int rotary,float scale,float qkEps,int gain1p,int qknorm,int qnw,int knw,void**outp,void**krawp,void**kpostp){
    if(!qg_init()||!g||!qp||!kp||!vp||!qw||!kw||!cosv||!sinv||!prefixK||!prefixV||!prefixOffsets||batch<2||batch>8||nH<1||nKV<1||nH%nKV||hd<2||hd>256||rotary<2||rotary>hd||rotary%2||scale<=0||qkEps<=0||(qnw!=hd&&qnw!=nH*hd)||(knw!=hd&&knw!=nKV*hd)||!outp||!krawp||!kpostp)return 0;
    int totalPrefix=prefixOffsets[batch];if(prefixOffsets[0]!=0||totalPrefix<=0)return 0;for(int row=0;row<batch;row++){int n=prefixOffsets[row+1]-prefixOffsets[row];if(n<=0||n+1>4096)return 0;}
    int qwidth=nH*hd,kvwidth=nKV*hd,qn=batch*qwidth,kn=batch*kvwidth,halfn=rotary/2;
    id<MTLBuffer>qo=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,qn),gate=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,qn),kr=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,kn),kpo=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,kn),out=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,qn);
    id<MTLBuffer>qa=qg_host(qw,qnw),ka=qg_host(kw,knw),cbv=qg_host(cosv,batch*halfn),sbv=qg_host(sinv,batch*halfn),pk=qg_host(prefixK,totalPrefix*kvwidth),pv=qg_host(prefixV,totalPrefix*kvwidth),offs=qg_host_i32(prefixOffsets,batch+1);
    if(!qo||!gate||!kr||!kpo||!out||!qa||!ka||!cbv||!sbv||!pk||!pv||!offs)return 0;id<MTLCommandBuffer>cb=(__bridge id<MTLCommandBuffer>)mg_graph_command_buffer(g);
    id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];[e setComputePipelineState:qgQKDecodeBatch];[e setBuffer:(__bridge id<MTLBuffer>)qp offset:0 atIndex:0];[e setBuffer:(__bridge id<MTLBuffer>)kp offset:0 atIndex:1];[e setBuffer:qa offset:0 atIndex:2];[e setBuffer:ka offset:0 atIndex:3];[e setBuffer:cbv offset:0 atIndex:4];[e setBuffer:sbv offset:0 atIndex:5];[e setBuffer:qo offset:0 atIndex:6];[e setBuffer:gate offset:0 atIndex:7];[e setBuffer:kr offset:0 atIndex:8];[e setBuffer:kpo offset:0 atIndex:9];[e setBytes:&nH length:4 atIndex:10];[e setBytes:&nKV length:4 atIndex:11];[e setBytes:&hd length:4 atIndex:12];[e setBytes:&rotary length:4 atIndex:13];[e setBytes:&qkEps length:4 atIndex:14];[e setBytes:&gain1p length:4 atIndex:15];[e setBytes:&qknorm length:4 atIndex:16];[e setBytes:&qnw length:4 atIndex:17];[e setBytes:&knw length:4 atIndex:18];[e dispatchThreadgroups:MTLSizeMake((NSUInteger)MAX(nH,nKV),(NSUInteger)batch,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];mg_graph_note_encoder(g);
    e=[cb computeCommandEncoder];[e setComputePipelineState:qgAttnDecodeBatch];[e setBuffer:qo offset:0 atIndex:0];[e setBuffer:pk offset:0 atIndex:1];[e setBuffer:pv offset:0 atIndex:2];[e setBuffer:offs offset:0 atIndex:3];[e setBuffer:kpo offset:0 atIndex:4];[e setBuffer:(__bridge id<MTLBuffer>)vp offset:0 atIndex:5];[e setBuffer:gate offset:0 atIndex:6];[e setBuffer:out offset:0 atIndex:7];[e setBytes:&nH length:4 atIndex:8];[e setBytes:&nKV length:4 atIndex:9];[e setBytes:&hd length:4 atIndex:10];[e setBytes:&scale length:4 atIndex:11];[e dispatchThreadgroups:MTLSizeMake((NSUInteger)nH,(NSUInteger)batch,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];mg_graph_note_encoder(g);
    *outp=(__bridge void*)out;*krawp=(__bridge void*)kr;*kpostp=(__bridge void*)kpo;return 1;
}
