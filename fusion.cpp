//
// Created by marko on 11/24/19.
//

#include "fusion.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

void fusion_cpu(const Views& views, const TsdfFusionFunctor functor, float vx_size, int n_threads, Volume& vol) {
    int vx_res3 = vol.depth_ * vol.height_ * vol.width_;

#if defined(_OPENMP)
    omp_set_num_threads(n_threads);
#endif
#pragma omp parallel for
    for(int idx = 0; idx < vx_res3; ++idx) {
        int d,h,w;
        fusion_idx2dhw(idx, vol.width_,vol.height_, d,h,w);
        float x,y,z;
        fusion_dhw2xyz(d,h,w, vx_size, x,y,z);

        functor.before_sample(&vol, d,h,w);
        bool run = true;
        int n_valid_views = 0;
        for(int vidx = 0; vidx < views.n_views_ && run; ++vidx) {
            float ur, vr, vx_d;
            fusion_project(&views, vidx, x,y,z, ur,vr,vx_d);

            int u = int(ur + 0.5f);
            int v = int(vr + 0.5f);
            // printf("  vx %d,%d,%d has center %f,%f,%f and projects to uvd=%f,%f,%f\n", w,h,d, x,y,z, ur,vr,vx_d);

            if(u >= 0 && v >= 0 && u < views.cols_ && v < views.rows_) {
                int dm_idx = (vidx * views.rows_ + v) * views.cols_ + u;
                float dm_d = views.depthmaps_[dm_idx];
                // printf("    is on depthmap[%d,%d] with depth=%f, diff=%f\n", views.cols_,views.rows_, dm_d, dm_d - vx_d);
                run = functor.new_sample(&vol, vx_d, dm_d, d,h,w, &n_valid_views);
            }
        } // for vidx
        functor.after_sample(&vol, d,h,w, n_valid_views);
    }
}


void fusion_tsdf_cpu(const Views &views, float vx_size, float truncation, bool unknown_is_free, int n_threads, Volume &vol) {
    TsdfFusionFunctor functor(truncation, unknown_is_free);
    fusion_cpu(views, functor, vx_size, n_threads, vol);
}

