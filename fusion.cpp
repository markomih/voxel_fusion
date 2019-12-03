//
// Created by marko on 11/24/19.
//

#include "fusion.h"
#include <cmath>

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


void apply_kernel(const Volume* vol, float* weight_matrix, float* color_matrix, int d, int h, int w, int r, int g, int b){
    int w2 = int(vol->kernel_size_/2);
    for (int d_= 0; d_ < vol->kernel_size_; d_++){
        for (int h_= 0; h_ < vol->kernel_size_; h_++) {
            for (int w_= 0; w_ < vol->kernel_size_; w_++){
                int new_d = d + d_ - w2;
                if (new_d < 0 || new_d >= vol->depth_) continue;

                int new_h = h + h_ - w2;
                if (new_h < 0 || new_d >= vol->height_) continue;

                int new_w = w + w_ - w2;
                if (new_w < 0 || new_w >= vol->width_) continue;

                int kidx = (d_*vol->kernel_size_ + h_)*vol->kernel_size_ + w_;
                float weight = vol->color_kernel_[kidx];
                int idx = volume_idx(vol, new_d, new_h, new_w);
                weight_matrix[idx] += 1;

                int r_ind, g_ind, b_ind;
                rgb_volume_idx(vol, new_d, new_h, new_w, r_ind, g_ind, b_ind);
                color_matrix[r_ind] += weight*float(r);
                color_matrix[g_ind] += weight*float(g);
                color_matrix[b_ind] += weight*float(b);
            }
        }
    }
}

void rgb_splatting_cpu(const Views& views, const TsdfFusionFunctor functor, float vx_size, int n_threads, Volume& vol) {
    int vx_res3 = vol.depth_ * vol.height_ * vol.width_;
    float* weight_matrix = new float[vx_res3]{0};
    float* color_matrix = new float[vx_res3*Volume::COLOR_CHANNELS]{0};

    for(int vidx = 0; vidx < views.n_views_; ++vidx) {
        printf("%i/%i\n", vidx, views.n_views_);
        for (int v = 0; v < views.rows_; v++){
            for (int u = 0; u < views.cols_; u++){
                float depth = views.depthmaps_[(vidx * views.rows_ + v) * views.cols_ + u];
                if (depth == -1) continue;
                float x,y,z;
                uvd2xyz(&views, vidx, float(u), float(v),depth,x,y,z);
                if (fabsf(x) > 0.5 || fabsf(y) > 0.5 || fabsf(z) > 0.5) continue; // TODO fix it to work with BOX_SIZE
                int d,h,w;
                xyz2dhw(x,y,z,vx_size, d, h, w);

                // no need to store colors that are not on the border
                float sdf = volume_get(&vol, d,h,w);
                if (sdf == functor.truncation_) continue;

                int rgb_idx = ((vidx * views.rows_ + v) * views.cols_ + u)*Volume::COLOR_CHANNELS;
                int r = views.rgb_images_[rgb_idx+0];
                int g = views.rgb_images_[rgb_idx+1];
                int b = views.rgb_images_[rgb_idx+2];
                apply_kernel(&vol, weight_matrix, color_matrix,d, h,  w,  r,  g,  b);
            }
        }
    }
    printf("copying");
    for(int idx = 0; idx < vx_res3; ++idx) {  // TODO this can be parallelize
        float weight = weight_matrix[idx];
        if (weight == 0) continue;

        int color_idx = idx * 3;
        vol.rgb_data_[color_idx] = int(color_matrix[color_idx]/weight);
        vol.rgb_data_[color_idx+1] = int(color_matrix[color_idx+1]/weight);
        vol.rgb_data_[color_idx+2] = int(color_matrix[color_idx+2]/weight);
    }
    printf("copied");
    delete[] weight_matrix;
    delete[] color_matrix;
}

void rgb_fusion_cpu(const Views& views, const TsdfFusionFunctor functor, float vx_size, int n_threads, Volume& vol) {
    int vx_res3 = vol.depth_ * vol.height_ * vol.width_;

#if defined(_OPENMP)
    omp_set_num_threads(n_threads);
#endif
#pragma omp parallel for
    for(int idx = 0; idx < vx_res3; ++idx) {
        int d,h,w;
        fusion_idx2dhw(idx, vol.width_,vol.height_, d,h,w);
        float sdf = volume_get(&vol, d, h, w);
        if (functor.truncation_ == sdf) continue;

        float x,y,z;
        fusion_dhw2xyz(d,h,w, vx_size, x,y,z);

//        functor.before_sample(&vol, d,h,w);
        int n_valid_views = 0;
        for(int vidx = 0; vidx < views.n_views_; ++vidx) {
            float ur, vr, vx_d;
            fusion_project(&views, vidx, x,y,z, ur,vr,vx_d);

            int u = int(ur + 0.5f);
            int v = int(vr + 0.5f);
            // printf("  vx %d,%d,%d has center %f,%f,%f and projects to uvd=%f,%f,%f\n", w,h,d, x,y,z, ur,vr,vx_d);

            if (u >= 0 && v >= 0 && u < views.cols_ && v < views.rows_) {
                int rgb_idx = ((vidx * views.rows_ + v) * views.cols_ + u)*Volume::COLOR_CHANNELS;
                int r = views.rgb_images_[rgb_idx+0];
                int g = views.rgb_images_[rgb_idx+1];
                int b = views.rgb_images_[rgb_idx+2];
                // printf("    is on depthmap[%d,%d] with depth=%f, diff=%f\n", views.cols_,views.rows_, dm_d, dm_d - vx_d);
                functor.new_color_sample(&vol,r,g,b, d,h,w);
                n_valid_views++;
            }
        } // for vidx
        functor.div_color(&vol, d,h,w, n_valid_views);
    }
}



void fusion_tsdf_cpu(const Views &views, float vx_size, float truncation, bool unknown_is_free, int n_threads, Volume &vol) {
    TsdfFusionFunctor functor(truncation, unknown_is_free);
    fusion_cpu(views, functor, vx_size, n_threads, vol);
//    rgb_fusion_cpu(views, functor, vx_size, n_threads, vol);
    rgb_splatting_cpu(views, functor, vx_size, n_threads, vol);
}

