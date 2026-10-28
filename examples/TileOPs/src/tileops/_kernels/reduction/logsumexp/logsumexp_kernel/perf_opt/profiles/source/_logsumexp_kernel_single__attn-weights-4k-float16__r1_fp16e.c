#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(1024) inline void main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem, int32_t x_ub_version_counter_1) {
  half x_frag[32];
  half m_frag[8];
  half s_frag[8];
  half out_frag[8];
  #pragma unroll
  for (int32_t i = 0; i < 8; ++i) {
    *(uint2*)(x_frag + (i * 4)) = *(__ubuf__ uint2*)(((__ubuf__ half*)buf_dyn_shmem) + ((((x_ub_version_counter_1 & 1) * 32768) + (i * 4096)) + (((int32_t)threadIdx.x) * 4)));
  }
  #pragma unroll
  for (int32_t i_1 = 0; i_1 < 8; ++i_1) {
    m_frag[i_1] = tl::limits::kFloatNInf;
    #pragma unroll
    for (int32_t rv = 0; rv < 4; ++rv) {
      m_frag[i_1] = max(m_frag[i_1], x_frag[((i_1 * 4) + rv)]);
    }
    asc_syncthreads();
    m_frag[i_1] = tl::AscendAllReduce<tl::MaxOp, 1024, 1, 0>::run(m_frag[i_1], (&(((__ubuf__ half*)buf_dyn_shmem)[65536])));
  }
  #pragma unroll
  for (int32_t i_2 = 0; i_2 < 8; ++i_2) {
    uint2 _;
      uint2 __1;
        uint2 v_ = *(uint2*)(x_frag + (i_2 * 4));
        uint2 __2;
          half v__1 = m_frag[i_2];
          ((half2*)(&(__2)))[0].x = v__1;
          ((half2*)(&(__2)))[0].y = v__1;
          ((half2*)(&(__2)))[1].x = v__1;
          ((half2*)(&(__2)))[1].y = v__1;
        ((half2*)(&(__1)))[0].x = (((half2*)(&(v_)))[0].x-((half2*)(&(__2)))[0].x);
        ((half2*)(&(__1)))[0].y = (((half2*)(&(v_)))[0].y-((half2*)(&(__2)))[0].y);
        ((half2*)(&(__1)))[1].x = (((half2*)(&(v_)))[1].x-((half2*)(&(__2)))[1].x);
        ((half2*)(&(__1)))[1].y = (((half2*)(&(v_)))[1].y-((half2*)(&(__2)))[1].y);
      ((half2*)(&(_)))[0].x = hexp(((half2*)(&(__1)))[0].x);
      ((half2*)(&(_)))[0].y = hexp(((half2*)(&(__1)))[0].y);
      ((half2*)(&(_)))[1].x = hexp(((half2*)(&(__1)))[1].x);
      ((half2*)(&(_)))[1].y = hexp(((half2*)(&(__1)))[1].y);
    *(uint2*)(x_frag + (i_2 * 4)) = _;
  }
  #pragma unroll
  for (int32_t i_3 = 0; i_3 < 8; ++i_3) {
    s_frag[i_3] = half(0x0p+0f/*0.000000e+00*/);
    #pragma unroll
    for (int32_t rv_1 = 0; rv_1 < 4; ++rv_1) {
      s_frag[i_3] = (s_frag[i_3] + x_frag[((i_3 * 4) + rv_1)]);
    }
    asc_syncthreads();
    s_frag[i_3] = tl::AscendAllReduce<tl::SumOp, 1024, 1, 0>::run(s_frag[i_3], (&(((__ubuf__ half*)buf_dyn_shmem)[66560])));
  }
  #pragma unroll
  for (int32_t i_4 = 0; i_4 < 4; ++i_4) {
    uint1 __3;
      float2 __4;
        uint1 v__2 = *(uint1*)(m_frag + (i_4 * 2));
        ((float2*)(&__4))[0] = __half22float2(((half2*)(&v__2))[0]);
      float2 __5;
        float2 __6;
          uint1 v__3 = *(uint1*)(s_frag + (i_4 * 2));
          ((float2*)(&__6))[0] = __half22float2(((half2*)(&v__3))[0]);
        __5.x = logf(__6.x);
        __5.y = logf(__6.y);
      float2 v__4 = __4 + __5;
      ((half2*)(&__3))[0] = __float22half2_rn(((float2*)(&v__4))[0]);
    *(uint1*)(out_frag + (i_4 * 2)) = __3;
  }
  if (((int32_t)threadIdx.x) == 0) {
    #pragma unroll
    for (int32_t i_5 = 0; i_5 < 2; ++i_5) {
      *(__ubuf__ uint2*)(((__ubuf__ half*)buf_dyn_shmem) + ((i_5 * 4) + 67584)) = *(uint2*)(out_frag + (i_5 * 4));
    }
  }
}

extern "C" __global__ __vector__ void main_kernel(__gm__ half* x, __gm__ half* y) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  int32_t x_ub_version_counter_1 = 0;
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
  asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
  x_ub_version_counter_1 = 0;
  for (int32_t w = 0; w < 2; ++w) {
    if (((w * 7) + (((int32_t)block_idx) >> 3)) < 8) {
      for (int32_t s = 0; s < 2; ++s) {
        asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>((x_ub_version_counter_1 & 1)));
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ half*)buf_dyn_shmem)[((x_ub_version_counter_1 & 1) * 32768)]))), (__gm__ uint8_t*)((&(x[(((w * 3670016) + (((int32_t)block_idx) * 65536)) + (s * 32768))]))), 1, 65536, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 65536, 65536);
        asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>((x_ub_version_counter_1 & 1)));
        asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
        asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>((x_ub_version_counter_1 & 1)));
        asc_vf_call<main_kernel_simt_vf_0>(cce::dim3(1024), buf_dyn_shmem, x_ub_version_counter_1);
        asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
        asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>((x_ub_version_counter_1 & 1)));
        asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
        asc_copy_ub2gm_align((__gm__ uint8_t*)((&(y[(((w * 896) + (((int32_t)block_idx) * 16)) + (s * 8))]))), (__ubuf__ uint8_t*)((&(((__ubuf__ half*)buf_dyn_shmem)[67584]))), 1, 16, static_cast<asc_store_l2_cache_mode>(4), 16, 16);
        asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
        x_ub_version_counter_1 = (x_ub_version_counter_1 + 1);
      }
    }
  }
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
  asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
}

