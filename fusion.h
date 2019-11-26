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

    Views() : n_views_(0), depthmaps_(0), rgb_images_(0), rows_(0), cols_(0), Ks_(0), Rs_(0), Ts_(0) {}
};

class Volume {
public:
    int depth_;
    int height_;
    int width_;
    float* data_;

    int* rgb_data_;
    static const int COLOR_CHANNELS=3;

    Volume() : depth_(0), height_(0), width_(0), data_(0), rgb_data_(0) {}
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
inline void fusion_dhw2xyz(int d, int h, int w, float vx_size, float& x, float& y, float& z) {
    // +0.5: move vx_center from (0,0,0) to (0.5,0.5,0.5), therefore vx range in [0, 1)
    // *vx_size: scale from [0,vx_resolution) to [0,1)
    // -0.5: move box to center, resolution [-.5,0.5)
    x = ((w + 0.5) * vx_size) - 0.5;
    y = ((h + 0.5) * vx_size) - 0.5;
    z = ((d + 0.5) * vx_size) - 0.5;
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
