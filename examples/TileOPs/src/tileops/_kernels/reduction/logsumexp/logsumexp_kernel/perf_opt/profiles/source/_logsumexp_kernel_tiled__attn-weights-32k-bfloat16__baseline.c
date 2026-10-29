#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(1024) inline void main_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  if (((int32_t)threadIdx.x) < 1) {
    ((__ubuf__ float*)buf_dyn_shmem)[18432] = tl::limits::kFloatNInf;
    ((__ubuf__ float*)buf_dyn_shmem)[18433] = float(0x0p+0f/*0.000000e+00*/);
  }
}

__simt_vf__ __launch_bounds__(1024) inline void main_kernel_simt_vf_1(__ubuf__ uint8_t* buf_dyn_shmem, int32_t x_ub_version_counter_1) {
  float x_frag[16];
  float m_c[1];
  float s_c[1];
  float prev_frag[1];
  #pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    bfloat16_t x_ub_local_cast[4];
    *(uint2*)(x_ub_local_cast + 0) = *(__ubuf__ uint2*)(((__ubuf__ bfloat16_t*)buf_dyn_shmem) + ((((x_ub_version_counter_1 & 1) * 16384) + (i * 4096)) + (((int32_t)threadIdx.x) * 4)));
    for (int32_t vec = 0; vec < 2; ++vec) {
      float2 _;
        uint1 v_ = *(uint1*)(x_ub_local_cast + (vec * 2));
        ((float2*)(&_))[0] = __bfloat1622float2((reinterpret_cast<bfloat16x2_t*>(&v_))[0]);
      *(float2*)(x_frag + ((i * 4) + (vec * 2))) = _;
    }
  }
  m_c[0] = tl::limits::kFloatNInf;
  #pragma unroll
  for (int32_t rv = 0; rv < 16; ++rv) {
    m_c[0] = max(m_c[0], x_frag[(((rv & 3) * 4) + (rv >> 2))]);
  }
  asc_syncthreads();
  m_c[0] = tl::AscendAllReduce<tl::MaxOp, 1024, 1, 0>::run(m_c[0], (&(((__ubuf__ float*)buf_dyn_shmem)[16384])));
  prev_frag[0] = ((__ubuf__ float*)buf_dyn_shmem)[18432];
  asc_syncthreads();
  ((__ubuf__ float*)buf_dyn_shmem)[18432] = max(((__ubuf__ float*)buf_dyn_shmem)[18432], m_c[0]);
  asc_syncthreads();
  #pragma unroll
  for (int32_t i_1 = 0; i_1 < 8; ++i_1) {
    float2 __1;
      float2 v__1 = *(float2*)(x_frag + (i_1 * 2)) - make_float2(((__ubuf__ float*)buf_dyn_shmem)[18432], ((__ubuf__ float*)buf_dyn_shmem)[18432]);
      __1.x = expf(v__1.x);
      __1.y = expf(v__1.y);
    *(float2*)(x_frag + (i_1 * 2)) = __1;
  }
  s_c[0] = float(0x0p+0f/*0.000000e+00*/);
  #pragma unroll
  for (int32_t rv_1 = 0; rv_1 < 16; ++rv_1) {
    s_c[0] = (s_c[0] + x_frag[(((rv_1 & 3) * 4) + (rv_1 >> 2))]);
  }
  s_c[0] = tl::AscendAllReduce<tl::SumOp, 1024, 1, 0>::run(s_c[0], (&(((__ubuf__ float*)buf_dyn_shmem)[17408])));
  if (((int32_t)threadIdx.x) == 0) {
    ((__ubuf__ float*)buf_dyn_shmem)[18433] = ((((__ubuf__ float*)buf_dyn_shmem)[18433] * expf((prev_frag[0] - ((__ubuf__ float*)buf_dyn_shmem)[18432]))) + s_c[0]);
  }
}

__simt_vf__ __launch_bounds__(1024) inline void main_kernel_simt_vf_2(__ubuf__ uint8_t* buf_dyn_shmem) {
  bfloat16_t out_frag[1];
  out_frag[0] = ((bfloat16_t)(((__ubuf__ float*)buf_dyn_shmem)[18432] + logf(((__ubuf__ float*)buf_dyn_shmem)[18433])));
  if (((int32_t)threadIdx.x) == 0) {
    ((__ubuf__ bfloat16_t*)buf_dyn_shmem)[36880] = out_frag[0];
  }
}

extern "C" __global__ __vector__ void main_kernel(__gm__ bfloat16_t* x, __gm__ bfloat16_t* y) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  int32_t x_ub_version_counter_1 = 0;
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
  asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
  x_ub_version_counter_1 = 0;
  for (int32_t w = 0; w < 19; ++w) {
    if (((w * 7) + (((int32_t)block_idx) >> 3)) < 128) {
      asc_vf_call<main_kernel_simt_vf_0>(cce::dim3(1024), buf_dyn_shmem);
      for (int32_t c = 0; c < 2; ++c) {
        asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>((x_ub_version_counter_1 & 1)));
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[((x_ub_version_counter_1 & 1) * 16384)]))), (__gm__ uint8_t*)((&(x[(((w * 1835008) + (((int32_t)block_idx) * 32768)) + (c * 16384))]))), 1, 32768, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 32768, 32768);
        asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>((x_ub_version_counter_1 & 1)));
        asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>((x_ub_version_counter_1 & 1)));
        asc_vf_call<main_kernel_simt_vf_1>(cce::dim3(1024), buf_dyn_shmem, x_ub_version_counter_1);
        asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>((x_ub_version_counter_1 & 1)));
        x_ub_version_counter_1 = (x_ub_version_counter_1 + 1);
      }
      asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
      asc_vf_call<main_kernel_simt_vf_2>(cce::dim3(1024), buf_dyn_shmem);
      asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
      asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
      asc_copy_ub2gm_align((__gm__ uint8_t*)((&(y[((w * 56) + ((int32_t)block_idx))]))), (__ubuf__ uint8_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[36880]))), 1, 2, static_cast<asc_store_l2_cache_mode>(4), 2, 2);
      asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
    }
  }
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
  asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
}

