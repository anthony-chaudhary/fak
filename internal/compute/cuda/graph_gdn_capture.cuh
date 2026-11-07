// SPDX-License-Identifier: Apache-2.0
// Capture-only GDN producer. All arithmetic kernels are the existing pinned
// qwen35_gdn.cuh implementations. Q8 fused projections KEEP F32 activations;
// only Q4_K QKV/Z use the exact frozen graph-owned MMVQ quant operations.
// No ordinary packed-capture refusal, public capability or Go route is changed.
#ifndef FAK_GRAPH_GDN_CAPTURE_CUH
#define FAK_GRAPH_GDN_CAPTURE_CUH

enum { FAK_GRAPH_GDN_BASELINE_V1 = 1 };
struct fak_graph_gdn_record {
  const void *weights[5]; // QKV, Z, B, A, output projection
  const float *scales[5];
  int kind[5]; // 0 F32, 1 Q8_0; QKV/Z alone also allow 2 Q4_K
  int quant_operation[2]; // -1 for fused F32/Q8; exact plan index for Q4_K
  const float *input, *conv_weight, *a_log, *dt_bias, *norm;
  float *conv_state, *recurrent_state, *output;
  float *mixed, *z, *b, *a, *conv_out, *q_norm, *k_norm, *core;
  int hidden, key_heads, value_heads, key_dim, value_dim, conv_kernel;
  int producer, register_state;
  float rms_eps;
};
struct graph_gdn_geometry {
  int key, value, conv, fused, q_threads, v_threads;
  size_t recurrent, history, conv_weight;
};
static bool graph_gdn_floats(const float *p, size_t elements, graph_quant_span *span) {
  // Fused Q8 projections use float4 loads of input/core. Exact device allocation
  // bases are normally much more aligned; reject weaker alignment explicitly.
  return elements && elements <= SIZE_MAX/sizeof(float) &&
    (uintptr_t)p % 16 == 0 && graph_quant_live(p,elements*sizeof(float),span);
}
static bool graph_gdn_weight(const void *p, const float *scale, int kind,
    int rows, int cols, graph_quant_span *weight, graph_quant_span *scales) {
  *scales = {0,0};
  if (rows <= 0 || cols <= 0 || (long long)rows*cols > INT_MAX) return false;
  size_t elements=(size_t)rows*cols, bytes;
  if (kind==0) {
    if (scale || elements > SIZE_MAX/sizeof(float)) return false;
    bytes=elements*sizeof(float);
  } else if (kind==1) {
    if (cols%32 || !graph_gdn_floats(scale,elements/32,scales)) return false;
    bytes=elements;
  } else if (kind==2) {
    if (scale || cols%256 || elements/256 > SIZE_MAX/144) return false;
    bytes=(elements/256)*144;
  } else return false;
  return graph_quant_live(p,bytes,weight);
}
static bool graph_gdn_preflight(fak_graph_once *s,const fak_graph_gdn_record *r,
                                 graph_gdn_geometry *g,int quant_cursor=-1) {
  if (!s || !r || !g || !s->position || r->producer!=FAK_GRAPH_GDN_BASELINE_V1 ||
      (r->register_state!=0 && r->register_state!=1) || r->hidden<=0 ||
      r->hidden>INT_MAX-255 || r->hidden%32 || r->key_heads<=0 || r->value_heads<=0 ||
      r->value_heads%r->key_heads || r->key_dim<=0 || r->key_dim>1024 ||
      r->value_dim<=0 || r->value_dim>1024 || r->conv_kernel<2 || r->conv_kernel>64 ||
      !(r->rms_eps>0.0f) || !isfinite(r->rms_eps)) return false;
  long long key=(long long)r->key_heads*r->key_dim;
  long long value=(long long)r->value_heads*r->value_dim;
  long long conv=2*key+value, fused=conv+value+2LL*r->value_heads;
  long long recurrent=(long long)r->value_heads*r->key_dim*r->value_dim;
  long long history=(r->conv_kernel-1)*conv, conv_weight=r->conv_kernel*conv;
  // Bound padded conv threads and signed stride loops in unchanged kernels.
  if (key>INT_MAX-255 || value>INT_MAX-255 || conv>INT_MAX-255 || fused>INT_MAX ||
      recurrent>INT_MAX || history>INT_MAX || conv_weight>INT_MAX) return false;
  *g={(int)key,(int)value,(int)conv,(int)fused,qwen35_gdn_threads(r->key_dim),
      qwen35_gdn_threads(r->value_dim),(size_t)recurrent,(size_t)history,(size_t)conv_weight};
  if (r->register_state && (r->key_dim>kQwen35GDNDecodeRegisterMaxKeyHeadDim ||
      g->v_threads>kQwen35GDNDecodeRegisterMaxThreads)) return false;
  graph_quant_span read[15],write[11];
  int rows[5]={g->conv,g->value,r->value_heads,r->value_heads,r->hidden};
  int cols[5]={r->hidden,r->hidden,r->hidden,r->hidden,g->value};
  for (int i=0;i<5;++i) {
    if (r->kind[i]<0 || r->kind[i]>(i<2?2:1) ||
        !graph_gdn_weight(r->weights[i],r->scales[i],r->kind[i],rows[i],cols[i],&read[i],&read[5+i])) return false;
  }
  const float *rp[5]={r->input,r->conv_weight,r->a_log,r->dt_bias,r->norm};
  size_t rn[5]={(size_t)r->hidden,g->conv_weight,(size_t)r->value_heads,(size_t)r->value_heads,(size_t)r->value_dim};
  for (int i=0;i<5;++i) if (!graph_gdn_floats(rp[i],rn[i],&read[10+i])) return false;
  float *wp[11]={r->conv_state,r->recurrent_state,r->output,r->mixed,r->z,r->b,r->a,r->conv_out,r->q_norm,r->k_norm,r->core};
  size_t wn[11]={g->history,g->recurrent,(size_t)r->hidden,(size_t)g->conv,(size_t)g->value,(size_t)r->value_heads,(size_t)r->value_heads,(size_t)g->conv,(size_t)g->key,(size_t)g->key,(size_t)g->value};
  uintptr_t position=(uintptr_t)s->position;
  if (position>UINTPTR_MAX-2*sizeof(int)) return false;
  graph_quant_span position_span{position,position+2*sizeof(int)};
  for (int i=0;i<11;++i) {
    if (!graph_gdn_floats(wp[i],wn[i],&write[i]) || graph_quant_overlap(write[i],position_span)) return false;
    for (int j=0;j<15;++j) if (graph_quant_overlap(write[i],read[j])) return false;
    for (int j=0;j<i;++j) if (graph_quant_overlap(write[i],write[j])) return false;
  }
  // Validate BOTH packed projections before recording the first one. Fused Q8
  // is not the activation-quantized Q8 MatMul arm and must not be substituted.
  int next=quant_cursor<0?s->quant_cursor:quant_cursor;
  for (int i=0;i<2;++i) {
    if (r->kind[i]!=2) { if (r->quant_operation[i]!=-1) return false; continue; }
    if (!s->quant_prepared || !s->quant_ops || next<0 || next>=s->quant_count ||
        r->quant_operation[i]!=next) return false;
    const auto &op=((const fak_graph_quant_op *)s->quant_ops)[next];
    if (op.kind!=FAK_GRAPH_QUANT_Q4_K || op.out!=rows[i] || op.in!=r->hidden ||
        op.weights!=r->weights[i] || op.scales || op.input!=r->input ||
        op.output!=(i==0?r->mixed:r->z)) return false;
    ++next;
  }
  return true;
}

// A manifest freezes native descriptor values and allocation extents, not Go
// object generations or contents. The caller still owns/pins every borrowed
// weight/state/transient under the live outer execution scope and cudaMu.
// Exact-address ABA, COW/privacy and arbitrary in-place weight mutation require
// the future Go dependency bridge. No native allocation map proves those facts.
struct graph_gdn_dependency {
  fak_graph_gdn_record record;
  const void *base[26];
  size_t extent[26]; // actual complete allocation extent captured at preparation
};
static void graph_gdn_bases(const fak_graph_gdn_record &r,const void **base) {
  for (int i=0;i<5;++i) { base[i]=r.weights[i];base[5+i]=r.scales[i]; }
  const void *tail[16]={r.input,r.conv_weight,r.a_log,r.dt_bias,r.norm,
    r.conv_state,r.recurrent_state,r.output,r.mixed,r.z,r.b,r.a,r.conv_out,r.q_norm,r.k_norm,r.core};
  for (int i=0;i<16;++i) base[10+i]=tail[i];
}
static int graph_gdn_quant_start(const fak_graph_gdn_record &r) {
  for (int i=0;i<2;++i) if (r.kind[i]==2) return r.quant_operation[i];
  return 0;
}
static bool graph_gdn_entry_live(fak_graph_once *s,const graph_gdn_dependency &d) {
  graph_gdn_geometry geometry;
  if (!graph_gdn_preflight(s,&d.record,&geometry,graph_gdn_quant_start(d.record))) return false;
  const void *base[26];graph_gdn_bases(d.record,base);
  for (int i=0;i<26;++i) {
    if (base[i]!=d.base[i]) return false;
    if (!base[i]) { if (d.extent[i]) return false;continue; }
    auto found=g_live.find((void *)base[i]);
    if (found==g_live.end() || found->second!=d.extent[i] ||
        d.extent[i]>UINTPTR_MAX-(uintptr_t)base[i]) return false;
  }
  return true;
}
static graph_quant_span graph_gdn_extent(const graph_gdn_dependency &d,int i) {
  uintptr_t lo=(uintptr_t)d.base[i];return {lo,lo+d.extent[i]};
}
static bool graph_gdn_manifest_valid(fak_graph_once *s,const graph_gdn_dependency *plan,int count,bool cross) {
  if (!s || !plan || count<=0 || count>256) return false;
  int previous_quant=-1;
  for (int i=0;i<count;++i) {
    if (!graph_gdn_entry_live(s,plan[i])) return false;
    for (int q=0;q<2;++q) if (plan[i].record.kind[q]==2) {
      int operation=plan[i].record.quant_operation[q];
      if (operation<=previous_quant) return false;
      previous_quant=operation; // interleaved non-GDN quant operations are legal
    }
    if (!cross) continue;
    // State pairs must be private to their block. Reusing transient storage
    // across sequential blocks is legal; aliasing any state pair is not.
    for (int j=0;j<count;++j) {
      for (int w=15;w<26;++w) {
        graph_quant_span writable=graph_gdn_extent(plan[i],w);
        for (int k=0;k<15;++k) {
          if (k==10) continue; // a previous output may feed a later input
          if (graph_quant_overlap(writable,graph_gdn_extent(plan[j],k))) return false;
        }
      }
      if (i==j) continue;
      for (int state=15;state<=16;++state)
        for (int k=0;k<26;++k)
          if (graph_quant_overlap(graph_gdn_extent(plan[i],state),graph_gdn_extent(plan[j],k))) return false;
    }
    // Include frozen quant operations outside GDN, e.g. FFN/output heads.
    // None may overwrite private GDN state or immutable GDN weight/constant
    // storage; no GDN writer may overwrite any frozen quant weight or scale.
    for (int q=0;q<s->quant_count;++q) {
      graph_quant_span quant[4];
      if (!s->quant_ops || !graph_quant_op_spans(((const fak_graph_quant_op *)s->quant_ops)[q],quant)) return false;
      for (int k=0;k<17;++k) {
        if (k==10) continue;
        if (graph_quant_overlap(quant[1],graph_gdn_extent(plan[i],k))) return false;
      }
      for (int w=15;w<26;++w)
        if (graph_quant_overlap(graph_gdn_extent(plan[i],w),quant[2]) ||
            graph_quant_overlap(graph_gdn_extent(plan[i],w),quant[3])) return false;
    }
  }
  return true;
}
static bool graph_gdn_manifest_check(fak_graph_once *s,int phase) {
  if (!s || s->terminal || !s->gdn_plan || !s->gdn_retire ||
      phase<fak_graph_once::GDN_BEGIN || phase>fak_graph_once::GDN_COPY ||
      v41_candidates_stream_uncertain()) return false;
  if (s->quant_ops!=s->gdn_quant_plan || s->quant_count!=s->gdn_quant_count) return false;
  if (phase!=fak_graph_once::GDN_BEGIN && s->gdn_cursor!=s->gdn_count) return false;
  cudaStreamCaptureStatus capture;
  cudaError_t e=cudaStreamIsCapturing(g_stream,&capture);
  if (e!=cudaSuccess) { graph_quant_error(s,e,3);return false; }
  const bool recording=phase==fak_graph_once::GDN_END;
  if (capture!=(recording?cudaStreamCaptureStatusActive:cudaStreamCaptureStatusNone)) return false;
  e=cudaPeekAtLastError();
  if (e!=cudaSuccess) { graph_quant_error(s,e,3);return false; }
  if (!recording && !graph_once_quant_idle(s)) return false;
  if (!graph_gdn_manifest_valid(s,(const graph_gdn_dependency *)s->gdn_plan,s->gdn_count,false)) return false;
  if (phase==fak_graph_once::GDN_BEGIN) s->gdn_cursor=0;
  return true;
}
// Called only after checked drain, graph/exec destruction and quant retirement.
// Manifest retirement NEVER frees borrowed state, weights or transient buffers.
static void graph_gdn_manifest_retire(fak_graph_once *s) {
  free(s->gdn_plan);s->gdn_plan=nullptr;s->gdn_count=s->gdn_cursor=0;
  s->gdn_quant_plan=nullptr;s->gdn_quant_count=0;
  s->gdn_check=nullptr;s->gdn_retire=nullptr;
}
// host_budget bounds only the immutable host manifest, not device capacity.
// Prepare once before capture. Caller source descriptors may then go away.
extern "C" int fak_graph_once_gdn_prepare(void *owner,const fak_graph_gdn_record *records,
                                         int count,size_t host_budget) {
  auto *s=(fak_graph_once *)owner;
  if (!s || s->terminal || !s->position || s->graph || s->exec || s->gdn_plan ||
      s->gdn_check || s->gdn_retire || g_capture_open || g_graph_once_capture ||
      v41_candidates_stream_uncertain() || !records || count<=0 || count>256) return 1;
  if ((size_t)count>host_budget/sizeof(graph_gdn_dependency)) return 2;
  cudaStreamCaptureStatus capture;
  cudaError_t e=cudaStreamIsCapturing(g_stream,&capture);
  if (e!=cudaSuccess) return graph_quant_error(s,e,3);
  if (capture!=cudaStreamCaptureStatusNone) return 1;
  e=cudaPeekAtLastError();if (e!=cudaSuccess) return graph_quant_error(s,e,3);
  if (!graph_once_quant_idle(s)) return 1;
  auto *plan=(graph_gdn_dependency *)calloc((size_t)count,sizeof(graph_gdn_dependency));
  if (!plan) return 2; // no submitted work or caller state mutation
  for (int i=0;i<count;++i) {
    plan[i].record=records[i];graph_gdn_bases(records[i],plan[i].base);
    for (int b=0;b<26;++b) {
      if (!plan[i].base[b]) continue;
      auto found=g_live.find((void *)plan[i].base[b]);
      if (found==g_live.end()) { free(plan);return 1; }
      plan[i].extent[b]=found->second;
    }
  }
  if (!graph_gdn_manifest_valid(s,plan,count,true)) { free(plan);return 1; }
  s->gdn_plan=plan;s->gdn_count=count;s->gdn_cursor=0;
  s->gdn_quant_plan=s->quant_ops;s->gdn_quant_count=s->quant_count;
  s->gdn_check=graph_gdn_manifest_check;s->gdn_retire=graph_gdn_manifest_retire;
  return 0;
}

static int graph_gdn_record_status(fak_graph_once *s) {
  cudaError_t e=cudaGetLastError();
  return e==cudaSuccess?0:graph_quant_error(s,e,3);
}
// Caller holds cudaMu and the already-live model execution scope, pins every
// state/weight/transient through checked graph retirement, and commits host
// state generations/frontiers only after successful initial/replay completion.
// This function records one COMPLETE selected GDN block, never a completed token.
extern "C" int fak_graph_once_gdn_record(void *owner,const fak_graph_gdn_record *r) {
  fak_graph_once *s=(fak_graph_once *)owner;
  if (!graph_once_capture_admit(s)) return 1;
  if (s->gdn_plan) {
    const auto *plan=(const graph_gdn_dependency *)s->gdn_plan;
    if (s->gdn_cursor<0 || s->gdn_cursor>=s->gdn_count ||
        r!=&plan[s->gdn_cursor].record || !graph_gdn_entry_live(s,plan[s->gdn_cursor])) {
      s->terminal=true;return 1;
    }
  }
  graph_gdn_geometry g;
  if (v41_candidates_stream_uncertain() || !graph_gdn_preflight(s,r,&g)) { s->terminal=true;return 1; }
  cudaStreamCaptureStatus capture;
  cudaError_t e=cudaStreamIsCapturing(g_stream,&capture);
  if (e!=cudaSuccess) return graph_quant_error(s,e,3);
  if (capture!=cudaStreamCaptureStatusActive) { s->terminal=true;return 1; }
  e=cudaPeekAtLastError();
  if (e!=cudaSuccess) return graph_quant_error(s,e,3);
  for (int i=0;i<2;++i) if (r->kind[i]==2) {
    int rc=fak_graph_once_quant_matmul(s,r->quant_operation[i]);
    if (rc) return rc;
  }
  if (!qwen35_gdn_fused_in_proj_warp_launch(
      r->input,r->weights[0],r->scales[0],r->kind[0],r->weights[1],r->scales[1],r->kind[1],
      r->weights[2],r->scales[2],r->kind[2],r->weights[3],r->scales[3],r->kind[3],
      r->mixed,r->z,r->b,r->a,r->hidden,g.conv,g.value,r->value_heads,1)) {
    FCUDA_TS("k_qwen35_gdn_fused_in_proj", g.fused, k_qwen35_gdn_fused_in_proj<<<g.fused,256,0,g_stream>>>(
        r->input,r->weights[0],r->scales[0],r->kind[0],r->weights[1],r->scales[1],r->kind[1],
        r->weights[2],r->scales[2],r->kind[2],r->weights[3],r->scales[3],r->kind[3],
        r->mixed,r->z,r->b,r->a,r->hidden,g.conv,g.value,r->value_heads));
  }
  if (int rc=graph_gdn_record_status(s)) return rc;
  unsigned blocks=(unsigned)g.conv/256+(g.conv%256!=0);
  FCUDA_TS("k_qwen35_gdn_conv_state", blocks, k_qwen35_gdn_conv_state<<<blocks,256,0,g_stream>>>(r->mixed,r->conv_weight,r->conv_state,r->conv_out,g.conv,r->conv_kernel));
  if (int rc=graph_gdn_record_status(s)) return rc;
  FCUDA_TS("k_qwen35_gdn_qk_norm", r->key_heads, k_qwen35_gdn_qk_norm<<<r->key_heads,g.q_threads,(size_t)2*g.q_threads*sizeof(float),g_stream>>>(r->conv_out,r->q_norm,r->k_norm,r->key_heads,r->key_dim));
  if (int rc=graph_gdn_record_status(s)) return rc;
  if (r->register_state) {
    FCUDA_TS("k_qwen35_gdn_recurrent_gated_norm_regstate", r->value_heads, k_qwen35_gdn_recurrent_gated_norm_regstate<<<r->value_heads,g.v_threads,(size_t)g.v_threads*sizeof(float),g_stream>>>(
        r->conv_out,r->q_norm,r->k_norm,r->z,r->b,r->a,r->a_log,r->dt_bias,r->norm,r->recurrent_state,r->core,r->key_heads,r->value_heads,r->key_dim,r->value_dim,r->rms_eps));
  } else {
    FCUDA_TS("k_qwen35_gdn_recurrent_gated_norm", r->value_heads, k_qwen35_gdn_recurrent_gated_norm<<<r->value_heads,g.v_threads,(size_t)g.v_threads*sizeof(float),g_stream>>>(
        r->conv_out,r->q_norm,r->k_norm,r->z,r->b,r->a,r->a_log,r->dt_bias,r->norm,r->recurrent_state,r->core,r->key_heads,r->value_heads,r->key_dim,r->value_dim,r->rms_eps));
  }
  if (int rc=graph_gdn_record_status(s)) return rc;
  if (!qwen35_gdn_out_proj_warp_launch(r->weights[4],r->scales[4],r->kind[4],r->core,r->output,r->hidden,g.value,1)) {
    FCUDA_TS("k_qwen35_gdn_out_proj", r->hidden, k_qwen35_gdn_out_proj<<<r->hidden,256,0,g_stream>>>(r->weights[4],r->scales[4],r->kind[4],r->core,r->output,r->hidden,g.value));
  }
  if (int rc=graph_gdn_record_status(s)) return rc;
  if (s->gdn_plan) ++s->gdn_cursor; // accepted recording only, not completed execution
  return 0; // No fence, operation counter, completion generation or host-state publication.
}
// Ordinal entry cannot accept a caller-rebound descriptor or skip a GDN block.
extern "C" int fak_graph_once_gdn_record_at(void *owner,int operation) {
  auto *s=(fak_graph_once *)owner;
  if (!s || !graph_once_capture_admit(s)) return 1;
  if (!s->gdn_plan || operation!=s->gdn_cursor || operation<0 || operation>=s->gdn_count) {
    s->terminal=true;return 1;
  }
  const auto *plan=(const graph_gdn_dependency *)s->gdn_plan;
  return fak_graph_once_gdn_record(s,&plan[operation].record);
}
#endif
