//
// Created by marko on 11/24/19.
//

#ifndef VOXEL_FUSION_FUSION_H
#define VOXEL_FUSION_FUSION_H

#include <cstdio>
#include <cmath>

class Views {
public:
    int n_views_;
    float* depthmaps_;
    int* rgb_images_;
    int rows_;
    int cols_;
    float* Ks_;
    float* Rs_;
    float* Ts_;

    float* Ks_inv_;
    float* cam2world_;

    Views() : n_views_(0), depthmaps_(0), rgb_images_(0), rows_(0), cols_(0), Ks_(0), Rs_(0), Ts_(0), Ks_inv_(0), cam2world_(0) {}
};

class Volume {
public:
    int depth_;
    int height_;
    int width_;
    float box_size_;
    float* data_;

    int* rgb_data_;
    float* color_kernel_;
    int kernel_size_;
    static const int COLOR_CHANNELS=3;

    Volume() : depth_(0), height_(0), width_(0), box_size_(0), data_(0), rgb_data_(0), color_kernel_(0), kernel_size_(0) {}
};

// index conversion
inline int volume_idx(const Volume* vol, int d, int h, int w) {
    return (d * vol->height_ + h) * vol->width_ + w;
}
inline void rgb_volume_idx(const Volume* vol, int d, int h, int w, int& r, int& g, int& b) {
    int offset = ((d * vol->height_ + h) * vol->width_ + w)*Volume::COLOR_CHANNELS;
    r = offset + 0;
    g = offset + 1;
    b = offset + 2;
}
//FUSION_FUNCTION
inline void fusion_idx2dhw(int idx, int width, int height, int& d, int& h, int &w) {
    w = idx % (width);
    d = idx / (width * height);
    h = ((idx - w) / width) % height;
}

//FUSION_FUNCTION
inline void fusion_dhw2xyz(int d, int h, int w, float vx_size, float& x, float& y, float& z, float box_size) {
    // +0.5: move vx_center from (0,0,0) to (0.5,0.5,0.5), therefore vx range in [0, 1)
    // *vx_size: scale from [0,vx_resolution) to [0,1)
    // -0.5: move box to center, resolution [-.5,0.5)
    x = (((float)w + 0.5f) * vx_size) - 0.5f;
    y = (((float)h + 0.5f) * vx_size) - 0.5f;
    z = (((float)d + 0.5f) * vx_size) - 0.5f;

    x = x*box_size*2.0f;
    y = y*box_size*2.0f;
    z = z*box_size*2.0f;
}
inline void xyz2dhw(float x, float y, float z, float vx_size, int& d, int&h, int& w, float box_size) {
    float scaling = 1.0f/(box_size*2.0f);

    w = int((scaling*x + 0.5f) / vx_size - 0.5f);
    h = int((scaling*y + 0.5f) / vx_size - 0.5f);
    d = int((scaling*z + 0.5f) / vx_size - 0.5f);
}

//FUSION_FUNCTION
inline void fusion_project(const Views* views, int vidx, float x, float y, float z, float& u, float& v, float& d) {
    float* K = views->Ks_ + vidx * 9;
    float* R = views->Rs_ + vidx * 9;
    float* T = views->Ts_ + vidx * 3;

    float xt = R[0] * x + R[1] * y + R[2] * z + T[0];
    float yt = R[3] * x + R[4] * y + R[5] * z + T[1];
    float zt = R[6] * x + R[7] * y + R[8] * z + T[2];
    // printf("  vx has center %f,%f,%f and projects to %f,%f,%f\n", x,y,z, xt,yt,zt);

    u = K[0] * xt + K[1] * yt + K[2] * zt;
    v = K[3] * xt + K[4] * yt + K[5] * zt;
    d = K[6] * xt + K[7] * yt + K[8] * zt;
    u /= d;
    v /= d;
}

inline void uvd2xyz(const Views* views, int vidx, float u, float v, float d, float& x, float& y, float& z){
    float* K_inv = views->Ks_inv_ + vidx*9;
    float* cam2world = views->cam2world_ + vidx*12;

    float x_ = K_inv[0]*u + K_inv[1]*v + K_inv[2]*d;
    float y_ = K_inv[3]*u + K_inv[4]*v + K_inv[5]*d;
    float z_ = K_inv[6]*u + K_inv[7]*v + K_inv[8]*d;

    x = cam2world[0]*x_ + cam2world[1]*y_ + cam2world[2]*z_ + cam2world[3];
    y = cam2world[4]*x_ + cam2world[5]*y_ + cam2world[6]*z_ + cam2world[7];
    z = cam2world[8]*x_ + cam2world[9]*y_ + cam2world[10]*z_ + cam2world[11];
}


// VOLUME ACCESS FUNCTIONS
inline float volume_get(const Volume* vol, int d, int h, int w) {
    return vol->data_[volume_idx(vol, d,h,w)];
}

inline void volume_set(const Volume* vol, int d, int h, int w, float val) {
    vol->data_[volume_idx(vol,d,h,w)] = val;
}

inline void volume_add(const Volume* vol, int d, int h, int w, float val) {
    vol->data_[volume_idx(vol,d,h,w)] += val;
}

inline void volume_div(const Volume* vol, int d, int h, int w, float val) {
    vol->data_[volume_idx(vol,d,h,w)] /= val;
}


// TSDF FOO
struct TsdfFusionFunctor {
    float truncation_;
    bool unknown_is_free_;
    TsdfFusionFunctor(float truncation, bool unknown_is_free) :
            truncation_(truncation), unknown_is_free_(unknown_is_free) {}

//    FUSION_FUNCTION
    void before_sample(Volume* vol, int d, int h, int w) const {
        volume_set(vol,d,h,w, 0);
    }

//    FUSION_FUNCTION
    bool new_sample(Volume* vol, float vx_depth, float dm_depth, int d, int h, int w, int* n_valid_views) const {
        if(unknown_is_free_ && dm_depth < 0) {
            dm_depth = 1e9;
        }
        float dist = dm_depth - vx_depth;
        float truncated_dist = fminf(truncation_, fmaxf(-truncation_, dist));
        if(dm_depth > 0 && dist >= -truncation_) {
            (*n_valid_views)++;
            volume_add(vol,d,h,w, truncated_dist);
        }
        return true;
    }
    void new_color_sample(Volume* vol, int r, int g, int b, int d, int h, int w) const {
        int r_ind, g_ind, b_ind;
        rgb_volume_idx(vol, d,h,w,r_ind, g_ind, b_ind);
        vol->rgb_data_[r_ind] += r;
        vol->rgb_data_[g_ind] += g;
        vol->rgb_data_[b_ind] += b;
    }
    void div_color(Volume* vol, int d, int h, int w, const int n_valid_views) const {
        int r_ind, g_ind, b_ind;
        rgb_volume_idx(vol, d,h,w,r_ind, g_ind, b_ind);
        vol->rgb_data_[r_ind] /= n_valid_views;
        vol->rgb_data_[g_ind] /= n_valid_views;
        vol->rgb_data_[b_ind] /= n_valid_views;
    }

//    FUSION_FUNCTION
    void after_sample(Volume* vol, int d, int h, int w, int n_valid_views) const {
        if(n_valid_views > 0) {
            volume_div(vol,d,h,w, (float)n_valid_views);
        }
        else {
            volume_set(vol,d,h,w, -truncation_);
        }
    }
};


void fusion_tsdf_cpu(const Views& views, float vx_size, float truncation, bool unknown_is_free, int n_threads, Volume& vol);

#endif //VOXEL_FUSION_FUSION_H
