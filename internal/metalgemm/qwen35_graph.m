//go:build darwin && arm64 && cgo

#import <Metal/Metal.h>
#include <math.h>

extern id<MTLDevice> gDev;
extern void *mg_graph_command_buffer(void *graph);
extern void *mg_graph_alloc_result(void *graph, int n);
extern void *mg_graph_alloc_buffer(void *graph, int n);
extern void mg_graph_note_encoder(void *graph);

static id<MTLComputePipelineState> qgNorm, qgAdd, qgSwiGLU, qgSplit, qgQK, qgAttn, qgLaneSplit, qgLaneQK, qgLaneAttn;
static BOOL qgAttempted, qgReady;

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


kernel void qg_lane_split(device const float*src [[buffer(0)]],device float*q [[buffer(1)]],device float*gate [[buffer(2)]],constant int&batch [[buffer(3)]],constant int&qwidth [[buffer(4)]],constant int&hd [[buffer(5)]],uint i [[thread_position_in_grid]]){if(i<(uint)(batch*qwidth)){int r=(int)i/qwidth,j=(int)i-r*qwidth,h=j/hd,d=j-h*hd;long base=(long)r*2*qwidth+(long)h*2*hd;q[i]=src[base+d];gate[i]=src[base+hd+d];}}

kernel void qg_lane_qk(device const float*qIn [[buffer(0)]],device const float*kIn [[buffer(1)]],device const float*qw [[buffer(2)]],device const float*kw [[buffer(3)]],device float*qOut [[buffer(4)]],device float*kRaw [[buffer(5)]],device float*kPost [[buffer(6)]],constant int&batch [[buffer(7)]],constant int&nH [[buffer(8)]],constant int&nKV [[buffer(9)]],constant int&hd [[buffer(10)]],constant int&rotary [[buffer(11)]],device const int*pos [[buffer(12)]],device const float*cosv [[buffer(13)]],device const float*sinv [[buffer(14)]],constant float&qkEps [[buffer(15)]],constant int&gain1p [[buffer(16)]],constant int&qknorm [[buffer(17)]],constant int&qnw [[buffer(18)]],constant int&knw [[buffer(19)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]){
    int h=(int)group.x,r=(int)group.y;threadgroup float qs[256],ks[256];if(r>=batch)return;
    float q=0,k=0;if(h<nH&&lane<(uint)hd)q=qIn[((long)r*nH+h)*hd+lane];if(h<nKV&&lane<(uint)hd)k=kIn[((long)r*nKV+h)*hd+lane];float qss=0,kss=0;
    if(qnw==hd)qss=q*q;else for(int i=(int)lane;i<nH*hd;i+=256){float z=qIn[(long)r*nH*hd+i];qss+=z*z;}if(knw==hd)kss=k*k;else for(int i=(int)lane;i<nKV*hd;i+=256){float z=kIn[(long)r*nKV*hd+i];kss+=z*z;}
    qs[lane]=qss;ks[lane]=kss;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint o=128;o;o>>=1){if(lane<o){qs[lane]+=qs[lane+o];ks[lane]+=ks[lane+o];}threadgroup_barrier(mem_flags::mem_threadgroup);}
    if(qknorm&&h<nH&&lane<(uint)hd){float w=qw[(qnw==hd?0:h*hd)+(int)lane];q*=rsqrt(qs[0]/(float)qnw+qkEps)*(gain1p?1.0f+w:w);}if(qknorm&&h<nKV&&lane<(uint)hd){float w=kw[(knw==hd?0:h*hd)+(int)lane];k*=rsqrt(ks[0]/(float)knw+qkEps)*(gain1p?1.0f+w:w);}
    if(h<nH&&lane<(uint)hd)qOut[((long)r*nH+h)*hd+lane]=q;if(h<nKV&&lane<(uint)hd){long ix=((long)r*nKV+h)*hd+lane;kRaw[ix]=k;kPost[ix]=k;}threadgroup_barrier(mem_flags::mem_threadgroup);
    int halfn=rotary/2;if(lane<(uint)halfn){int j=(int)lane,p=pos[r];float c=cosv[(long)p*halfn+j],sn=sinv[(long)p*halfn+j];if(h<nH){long i=((long)r*nH+h)*hd+j;float x=qOut[i],y=qOut[i+halfn];qOut[i]=x*c-y*sn;qOut[i+halfn]=x*sn+y*c;}if(h<nKV){long i=((long)r*nKV+h)*hd+j;float x=kPost[i],y=kPost[i+halfn];kPost[i]=x*c-y*sn;kPost[i+halfn]=x*sn+y*c;}}
}

kernel void qg_lane_attn(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device const float*gate [[buffer(3)]],device float*out [[buffer(4)]],device const int*offsets [[buffer(5)]],device const int*lengths [[buffer(6)]],constant int&batch [[buffer(7)]],constant int&nH [[buffer(8)]],constant int&nKV [[buffer(9)]],constant int&hd [[buffer(10)]],constant float&scale [[buffer(11)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]){
    int h=(int)group.x,r=(int)group.y,kh=h/(nH/nKV);if(r>=batch)return;int off=offsets[r],n=lengths[r];threadgroup float score[4096],den;if(lane==0){float m=-INFINITY;for(int j=0;j<n;j++){float z=0;for(int d=0;d<hd;d++)z+=q[((long)r*nH+h)*hd+d]*k[((long)(off+j)*nKV+kh)*hd+d];z*=scale;score[j]=z;m=max(m,z);}float sum=0;for(int j=0;j<n;j++){score[j]=exp(score[j]-m);sum+=score[j];}den=sum;}threadgroup_barrier(mem_flags::mem_threadgroup);if(lane<(uint)hd){float z=0;for(int j=0;j<n;j++)z+=score[j]*v[((long)(off+j)*nKV+kh)*hd+lane];long i=((long)r*nH+h)*hd+lane;out[i]=(z/den)/(1.0f+exp(-gate[i]));}
}
)MSL";

static id<MTLComputePipelineState> qg_pipeline(id<MTLLibrary> library, NSString *name, NSError **error) {
    id<MTLFunction> function=[library newFunctionWithName:name];
    if(!function)return nil;
    return[gDev newComputePipelineStateWithFunction:function error:error];
}

static int qg_init(void) {
    @synchronized(gDev) {
        if(qgReady)return 1;
        if(qgAttempted)return 0;
        qgAttempted=YES;
        NSError *error=nil;
        id<MTLLibrary> library=[gDev newLibraryWithSource:qgSource options:nil error:&error];
        if(!library){NSLog(@"qwen35 graph compile: %@",error);return 0;}

        // Keep initialization transactional. Publishing even one pipeline before
        // all lane and P32 functions exist lets a partial set masquerade as a
        // ready graph and turns a clean admission decline into a nil-PSO signal.
        id<MTLComputePipelineState> norm=qg_pipeline(library,@"qg_norm",&error);
        id<MTLComputePipelineState> add=qg_pipeline(library,@"qg_add",&error);
        id<MTLComputePipelineState> swiglu=qg_pipeline(library,@"qg_swiglu",&error);
        id<MTLComputePipelineState> split=qg_pipeline(library,@"qg_split",&error);
        id<MTLComputePipelineState> qk=qg_pipeline(library,@"qg_qk",&error);
        id<MTLComputePipelineState> attn=qg_pipeline(library,@"qg_attn",&error);
        id<MTLComputePipelineState> laneSplit=qg_pipeline(library,@"qg_lane_split",&error);
        id<MTLComputePipelineState> laneQK=qg_pipeline(library,@"qg_lane_qk",&error);
        id<MTLComputePipelineState> laneAttn=qg_pipeline(library,@"qg_lane_attn",&error);
        if(!norm||!add||!swiglu||!split||!qk||!attn||!laneSplit||!laneQK||!laneAttn){
            NSLog(@"qwen35 graph pipeline initialization failed: %@",error);
            return 0;
        }
        qgNorm=norm;qgAdd=add;qgSwiGLU=swiglu;qgSplit=split;qgQK=qk;qgAttn=attn;
        qgLaneSplit=laneSplit;qgLaneQK=laneQK;qgLaneAttn=laneAttn;
        qgReady=YES;
        return 1;
    }
}

int mg_qwen35_graph_ready(void){return qg_init();}
static id<MTLBuffer> qg_host(const float*p,int n){return[gDev newBufferWithBytes:p length:(NSUInteger)n*sizeof(float) options:MTLResourceStorageModeShared];}
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



int mg_qwen35_graph_attention_batch(void*g,void*qgatep,void*kp,void*vp,const float*qw,const float*kw,const float*cosv,const float*sinv,const int*positions,const int*offsets,const int*lengths,const float*prefixK,const float*prefixV,int totalKV,int batch,int nH,int nKV,int hd,int rotary,float scale,float qkEps,int gain1p,int qknorm,int qnw,int knw,void**outp,void**krawp,void**kpostp,void**vcurp){
    if(!qg_init()||!g||!qgatep||!kp||!vp||!qw||!kw||!cosv||!sinv||!positions||!offsets||!lengths||!prefixK||!prefixV||batch<2||batch>8||totalKV<batch)return 0;
    id<MTLCommandBuffer>cb=(__bridge id<MTLCommandBuffer>)mg_graph_command_buffer(g);int qwth=nH*hd,kvw=nKV*hd,maxPos=0;for(int r=0;r<batch;r++)maxPos=MAX(maxPos,positions[r]);
    id<MTLBuffer>q=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,batch*qwth),gate=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,batch*qwth),qout=(__bridge id<MTLBuffer>)mg_graph_alloc_buffer(g,batch*qwth),kr=(__bridge id<MTLBuffer>)mg_graph_alloc_result(g,batch*kvw),kpst=(__bridge id<MTLBuffer>)mg_graph_alloc_result(g,batch*kvw),vc=(__bridge id<MTLBuffer>)mg_graph_alloc_result(g,batch*kvw),out=(__bridge id<MTLBuffer>)mg_graph_alloc_result(g,batch*qwth);
    id<MTLBuffer>qwb=qg_host(qw,qnw==hd?hd:qwth),kwb=qg_host(kw,knw==hd?hd:kvw),cbf=qg_host(cosv,(maxPos+1)*(rotary/2)),sbf=qg_host(sinv,(maxPos+1)*(rotary/2)),posb=[gDev newBufferWithBytes:positions length:(NSUInteger)batch*sizeof(int) options:MTLResourceStorageModeShared],offb=[gDev newBufferWithBytes:offsets length:(NSUInteger)batch*sizeof(int) options:MTLResourceStorageModeShared],lenb=[gDev newBufferWithBytes:lengths length:(NSUInteger)batch*sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer>allK=qg_host(prefixK,totalKV*kvw),allV=qg_host(prefixV,totalKV*kvw);if(!cb||!q||!gate||!qout||!kr||!kpst||!vc||!out||!qwb||!kwb||!cbf||!sbf||!posb||!offb||!lenb||!allK||!allV)return 0;
    id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];[e setComputePipelineState:qgLaneSplit];[e setBuffer:(__bridge id<MTLBuffer>)qgatep offset:0 atIndex:0];[e setBuffer:q offset:0 atIndex:1];[e setBuffer:gate offset:0 atIndex:2];[e setBytes:&batch length:4 atIndex:3];[e setBytes:&qwth length:4 atIndex:4];[e setBytes:&hd length:4 atIndex:5];qg_dispatch(e,qgLaneSplit,batch*qwth);[e endEncoding];
    e=[cb computeCommandEncoder];[e setComputePipelineState:qgLaneQK];[e setBuffer:q offset:0 atIndex:0];[e setBuffer:(__bridge id<MTLBuffer>)kp offset:0 atIndex:1];[e setBuffer:qwb offset:0 atIndex:2];[e setBuffer:kwb offset:0 atIndex:3];[e setBuffer:qout offset:0 atIndex:4];[e setBuffer:kr offset:0 atIndex:5];[e setBuffer:kpst offset:0 atIndex:6];[e setBytes:&batch length:4 atIndex:7];[e setBytes:&nH length:4 atIndex:8];[e setBytes:&nKV length:4 atIndex:9];[e setBytes:&hd length:4 atIndex:10];[e setBytes:&rotary length:4 atIndex:11];[e setBuffer:posb offset:0 atIndex:12];[e setBuffer:cbf offset:0 atIndex:13];[e setBuffer:sbf offset:0 atIndex:14];[e setBytes:&qkEps length:4 atIndex:15];[e setBytes:&gain1p length:4 atIndex:16];[e setBytes:&qknorm length:4 atIndex:17];[e setBytes:&qnw length:4 atIndex:18];[e setBytes:&knw length:4 atIndex:19];[e dispatchThreadgroups:MTLSizeMake(MAX(nH,nKV),batch,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];
    id<MTLBlitCommandEncoder>bl=[cb blitCommandEncoder];for(int r=0;r<batch;r++){NSUInteger dst=((NSUInteger)offsets[r]+lengths[r]-1)*kvw*sizeof(float),src=(NSUInteger)r*kvw*sizeof(float);[bl copyFromBuffer:kpst sourceOffset:src toBuffer:allK destinationOffset:dst size:(NSUInteger)kvw*sizeof(float)];[bl copyFromBuffer:(__bridge id<MTLBuffer>)vp sourceOffset:src toBuffer:allV destinationOffset:dst size:(NSUInteger)kvw*sizeof(float)];[bl copyFromBuffer:(__bridge id<MTLBuffer>)vp sourceOffset:src toBuffer:vc destinationOffset:src size:(NSUInteger)kvw*sizeof(float)];}[bl endEncoding];
    e=[cb computeCommandEncoder];[e setComputePipelineState:qgLaneAttn];[e setBuffer:qout offset:0 atIndex:0];[e setBuffer:allK offset:0 atIndex:1];[e setBuffer:allV offset:0 atIndex:2];[e setBuffer:gate offset:0 atIndex:3];[e setBuffer:out offset:0 atIndex:4];[e setBuffer:offb offset:0 atIndex:5];[e setBuffer:lenb offset:0 atIndex:6];[e setBytes:&batch length:4 atIndex:7];[e setBytes:&nH length:4 atIndex:8];[e setBytes:&nKV length:4 atIndex:9];[e setBytes:&hd length:4 atIndex:10];[e setBytes:&scale length:4 atIndex:11];[e dispatchThreadgroups:MTLSizeMake(nH,batch,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];mg_graph_note_encoder(g);mg_graph_note_encoder(g);mg_graph_note_encoder(g);mg_graph_note_encoder(g);*outp=(__bridge void*)out;*krawp=(__bridge void*)kr;*kpostp=(__bridge void*)kpst;*vcurp=(__bridge void*)vc;return 1;
}
