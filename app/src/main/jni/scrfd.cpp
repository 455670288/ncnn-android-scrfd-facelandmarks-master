// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "scrfd.h"

#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "cpu.h"

static inline float intersection_area(const FaceObject& a, const FaceObject& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<FaceObject>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

//     #pragma omp parallel sections
    {
//         #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
//         #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<FaceObject>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<FaceObject>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const FaceObject& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const FaceObject& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            //             float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

// insightface/detection/scrfd/mmdet/core/anchor/anchor_generator.py gen_single_level_base_anchors()
static ncnn::Mat generate_anchors(int base_size, const ncnn::Mat& ratios, const ncnn::Mat& scales)
{
    int num_ratio = ratios.w;
    int num_scale = scales.w;

    ncnn::Mat anchors;
    anchors.create(4, num_ratio * num_scale);

    const float cx = 0;
    const float cy = 0;

    for (int i = 0; i < num_ratio; i++)
    {
        float ar = ratios[i];

        int r_w = round(base_size / sqrt(ar));
        int r_h = round(r_w * ar); //round(base_size * sqrt(ar));

        for (int j = 0; j < num_scale; j++)
        {
            float scale = scales[j];

            float rs_w = r_w * scale;
            float rs_h = r_h * scale;

            float* anchor = anchors.row(i * num_scale + j);

            anchor[0] = cx - rs_w * 0.5f;
            anchor[1] = cy - rs_h * 0.5f;
            anchor[2] = cx + rs_w * 0.5f;
            anchor[3] = cy + rs_h * 0.5f;
        }
    }

    return anchors;
}

static void generate_proposals(const ncnn::Mat& anchors, int feat_stride, const ncnn::Mat& score_blob, const ncnn::Mat& bbox_blob, const ncnn::Mat& kps_blob, float prob_threshold, std::vector<FaceObject>& faceobjects)
{
    int w = score_blob.w;
    int h = score_blob.h;

    // generate face proposal from bbox deltas and shifted anchors
    const int num_anchors = anchors.h;

    for (int q = 0; q < num_anchors; q++)
    {
        const float* anchor = anchors.row(q);

        const ncnn::Mat score = score_blob.channel(q);
        const ncnn::Mat bbox = bbox_blob.channel_range(q * 4, 4);

        // shifted anchor
        float anchor_y = anchor[1];

        float anchor_w = anchor[2] - anchor[0];
        float anchor_h = anchor[3] - anchor[1];

        for (int i = 0; i < h; i++)
        {
            float anchor_x = anchor[0];

            for (int j = 0; j < w; j++)
            {
                int index = i * w + j;

                float prob = score[index];

                if (prob >= prob_threshold)
                {
                    // insightface/detection/scrfd/mmdet/models/dense_heads/scrfd_head.py _get_bboxes_single()
                    float dx = bbox.channel(0)[index] * feat_stride;
                    float dy = bbox.channel(1)[index] * feat_stride;
                    float dw = bbox.channel(2)[index] * feat_stride;
                    float dh = bbox.channel(3)[index] * feat_stride;

                    // insightface/detection/scrfd/mmdet/core/bbox/transforms.py distance2bbox()
                    float cx = anchor_x + anchor_w * 0.5f;
                    float cy = anchor_y + anchor_h * 0.5f;

                    float x0 = cx - dx;
                    float y0 = cy - dy;
                    float x1 = cx + dw;
                    float y1 = cy + dh;

                    FaceObject obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0 + 1;
                    obj.rect.height = y1 - y0 + 1;
                    obj.prob = prob;

                    if (!kps_blob.empty())
                    {
                        const ncnn::Mat kps = kps_blob.channel_range(q * 10, 10);

                        obj.landmark[0].x = cx + kps.channel(0)[index] * feat_stride;
                        obj.landmark[0].y = cy + kps.channel(1)[index] * feat_stride;
                        obj.landmark[1].x = cx + kps.channel(2)[index] * feat_stride;
                        obj.landmark[1].y = cy + kps.channel(3)[index] * feat_stride;
                        obj.landmark[2].x = cx + kps.channel(4)[index] * feat_stride;
                        obj.landmark[2].y = cy + kps.channel(5)[index] * feat_stride;
                        obj.landmark[3].x = cx + kps.channel(6)[index] * feat_stride;
                        obj.landmark[3].y = cy + kps.channel(7)[index] * feat_stride;
                        obj.landmark[4].x = cx + kps.channel(8)[index] * feat_stride;
                        obj.landmark[4].y = cy + kps.channel(9)[index] * feat_stride;
                    }

                    faceobjects.push_back(obj);
                }

                anchor_x += feat_stride;
            }

            anchor_y += feat_stride;
        }
    }
}

int SCRFD::load(const char* modeltype, bool use_gpu) //不运行
{
    scrfd.clear();
    landmarks.clear(); //清除关键点模型

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    scrfd.opt = ncnn::Option();

#if NCNN_VULKAN
    scrfd.opt.use_vulkan_compute = use_gpu;
#endif

    scrfd.opt.num_threads = ncnn::get_big_cpu_count();

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "scrfd_%s-opt2.param", modeltype);
    sprintf(modelpath, "scrfd_%s-opt2.bin", modeltype);

    scrfd.load_param(parampath);
    scrfd.load_model(modelpath);

    has_kps = strstr(modeltype, "_kps") != NULL;

    landmarks.load_param("assets/2d106det_change.param"); //加载关键点模型
    landmarks.load_model("assets/2d106det_change.bin");

    return 0;
}

int SCRFD::load(AAssetManager* mgr, const char* modeltype, bool use_gpu)
{
    scrfd.clear();
    landmarks.clear(); //清除关键点模型

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    scrfd.opt = ncnn::Option();

#if NCNN_VULKAN
    scrfd.opt.use_vulkan_compute = use_gpu;
#endif

    scrfd.opt.num_threads = ncnn::get_big_cpu_count();

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "scrfd_%s-opt2.param", modeltype);
    sprintf(modelpath, "scrfd_%s-opt2.bin", modeltype);

    scrfd.load_param(mgr, parampath);
    scrfd.load_model(mgr, modelpath);

    has_kps = strstr(modeltype, "_kps") != NULL;

    landmarks.load_param(mgr,"2d106det_change.param"); //加载关键点模型权重
    landmarks.load_model(mgr,"2d106det_change.bin");

    return 0;
}

/* 人脸关键点前处理*/
return_d_m pre_process(const cv::Mat& src, int input_size, FaceObject& det){
    cv::Mat dst;
    return_d_m dst_m3;
    int x1 = det.rect.x;
    int y1 = det.rect.y;
    int faceImgWidth = det.rect.width;
    int faceImgHeight = det.rect.height;
//    NCNN_LOGE("x1 = %d",x1);
//    NCNN_LOGE("y1 = %d",y1);
//    NCNN_LOGE("w = %d",faceImgWidth);
//    NCNN_LOGE("h = %d",faceImgHeight);
    float center_w = x1+faceImgWidth/2;
    float center_h = y1+faceImgHeight/2;


    double _scale = input_size / (MAX(faceImgWidth, faceImgHeight) * 1.5);

    //构建仿射变换矩阵矩阵
    cv::Mat m1 = cv::Mat::eye(cv::Size(3,2),CV_64F);
    m1 *= _scale; //缩放矩阵
    cv::Mat m2 = cv::Mat::zeros(cv::Size(3,2),CV_64F);
    m2.at<double >(0,2) = -(center_w * _scale) + input_size/2;
    m2.at<double >(1,2) = -(center_h * _scale) + input_size/2; //平移矩阵
    dst_m3.matri = m1 + m2;
    cv::warpAffine(src, dst_m3.dst,  dst_m3.matri, cv::Size(192,192));

    return dst_m3;
}

/*人脸关键点后处理*/
cv::Mat post_progress (cv::Mat& post_mat, cv::Mat& m){
    cv::Mat im;
    cv::Mat pts(106, 2, CV_32F);
    cv::Mat new_pts;
    cv::Mat im_t;
    cv::Mat im_t_f;
    cv::Mat coord;
    cv::Mat col_cat = cv::Mat::ones(106,1,CV_32F);

    int index = 0;
    for (int i = 0; i < 106; ++i) {
        for (int j = 0; j < 2; ++j){
            pts.at<float>(i,j) = post_mat.at<float>(index,0);
            index ++;
        }
    }

    invertAffineTransform(m, im);
    pts += 1;
    pts *= 192/2;

    cv::hconcat(pts, col_cat, new_pts);

    transpose(im, im_t);
    im_t.convertTo(im_t_f,CV_32F);
    coord = new_pts * im_t_f;

    return coord;
}

int SCRFD::detect(const cv::Mat& rgb, std::vector<FaceObject>& faceobjects, std::vector<cv::Mat>& facelandmarks,float prob_threshold, float nms_threshold)
{
    int width = rgb.cols;
    int height = rgb.rows;

    // insightface/detection/scrfd/configs/scrfd/scrfd_500m.py
    const int target_size = 120;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, width, height, w, h);

    // pad to target_size rectangle
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1/128.f, 1/128.f, 1/128.f};
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = scrfd.create_extractor();

    ex.input("input.1", in_pad);

    std::vector<FaceObject> faceproposals;

    // stride 8
    {
        ncnn::Mat score_blob, bbox_blob, kps_blob;
        ex.extract("score_8", score_blob);
        ex.extract("bbox_8", bbox_blob);
        if (has_kps)
            ex.extract("kps_8", kps_blob);

        const int base_size = 16;
        const int feat_stride = 8;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 1.f;
        scales[1] = 2.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<FaceObject> faceobjects8;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, kps_blob, prob_threshold, faceobjects8);

        faceproposals.insert(faceproposals.end(), faceobjects8.begin(), faceobjects8.end());
    }

    // stride 16
    {
        ncnn::Mat score_blob, bbox_blob, kps_blob;
        ex.extract("score_16", score_blob);
        ex.extract("bbox_16", bbox_blob);
        if (has_kps)
            ex.extract("kps_16", kps_blob);

        const int base_size = 64;
        const int feat_stride = 16;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 1.f;
        scales[1] = 2.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<FaceObject> faceobjects16;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, kps_blob, prob_threshold, faceobjects16);

        faceproposals.insert(faceproposals.end(), faceobjects16.begin(), faceobjects16.end());
    }

    // stride 32
    {
        ncnn::Mat score_blob, bbox_blob, kps_blob;
        ex.extract("score_32", score_blob);
        ex.extract("bbox_32", bbox_blob);
        if (has_kps)
            ex.extract("kps_32", kps_blob);

        const int base_size = 256;
        const int feat_stride = 32;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 1.f;
        scales[1] = 2.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<FaceObject> faceobjects32;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, kps_blob, prob_threshold, faceobjects32);

        faceproposals.insert(faceproposals.end(), faceobjects32.begin(), faceobjects32.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(faceproposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(faceproposals, picked, nms_threshold);

    int face_count = picked.size();

    faceobjects.resize(face_count);
    for (int i = 0; i < face_count; i++)
    {
        faceobjects[i] = faceproposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (faceobjects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (faceobjects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (faceobjects[i].rect.x + faceobjects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (faceobjects[i].rect.y + faceobjects[i].rect.height - (hpad / 2)) / scale;

        x0 = std::max(std::min(x0, (float)width - 1), 0.f);
        y0 = std::max(std::min(y0, (float)height - 1), 0.f);
        x1 = std::max(std::min(x1, (float)width - 1), 0.f);
        y1 = std::max(std::min(y1, (float)height - 1), 0.f);

        faceobjects[i].rect.x = x0;
        faceobjects[i].rect.y = y0;
        faceobjects[i].rect.width = x1 - x0;
        faceobjects[i].rect.height = y1 - y0;

        if (has_kps)
        {
            float x0 = (faceobjects[i].landmark[0].x - (wpad / 2)) / scale;
            float y0 = (faceobjects[i].landmark[0].y - (hpad / 2)) / scale;
            float x1 = (faceobjects[i].landmark[1].x - (wpad / 2)) / scale;
            float y1 = (faceobjects[i].landmark[1].y - (hpad / 2)) / scale;
            float x2 = (faceobjects[i].landmark[2].x - (wpad / 2)) / scale;
            float y2 = (faceobjects[i].landmark[2].y - (hpad / 2)) / scale;
            float x3 = (faceobjects[i].landmark[3].x - (wpad / 2)) / scale;
            float y3 = (faceobjects[i].landmark[3].y - (hpad / 2)) / scale;
            float x4 = (faceobjects[i].landmark[4].x - (wpad / 2)) / scale;
            float y4 = (faceobjects[i].landmark[4].y - (hpad / 2)) / scale;

            faceobjects[i].landmark[0].x = std::max(std::min(x0, (float)width - 1), 0.f);
            faceobjects[i].landmark[0].y = std::max(std::min(y0, (float)height - 1), 0.f);
            faceobjects[i].landmark[1].x = std::max(std::min(x1, (float)width - 1), 0.f);
            faceobjects[i].landmark[1].y = std::max(std::min(y1, (float)height - 1), 0.f);
            faceobjects[i].landmark[2].x = std::max(std::min(x2, (float)width - 1), 0.f);
            faceobjects[i].landmark[2].y = std::max(std::min(y2, (float)height - 1), 0.f);
            faceobjects[i].landmark[3].x = std::max(std::min(x3, (float)width - 1), 0.f);
            faceobjects[i].landmark[3].y = std::max(std::min(y3, (float)height - 1), 0.f);
            faceobjects[i].landmark[4].x = std::max(std::min(x4, (float)width - 1), 0.f);
            faceobjects[i].landmark[4].y = std::max(std::min(y4, (float)height - 1), 0.f);
        }
    }


    /*关键点
     **/
    if (faceobjects.size() > 0){
        for(int i = 0; i < faceobjects.size(); i++){
            return_d_m image_str = pre_process(rgb,192,faceobjects[i]);
            cv::Mat clip_rgb = image_str.dst;

            ncnn::Mat face_input = ncnn::Mat::from_pixels_resize(clip_rgb.data, ncnn::Mat::PIXEL_RGB, clip_rgb.cols, clip_rgb.rows, 192, 192);
            const float face_mean_vals[3] = {0.f, 0.f,0.f};
            const float face_norm_vals[3] = {1.f, 1.f,1.f};
            face_input.substract_mean_normalize(face_mean_vals, face_norm_vals);
            ncnn::Mat face_output;
            ncnn::Extractor ex_face = landmarks.create_extractor();
            ex_face.input("data", face_input); // 推理
            ex_face.extract("fc1", face_output); //face_output.w = 212 face_output.h = 1. 将模型推理输出存储在ncnn::Mat对象


            cv::Mat cvMat(212, 1, CV_32FC1);
            for(int i =0; i < face_output.w; i++){
                cvMat.at<float>(i,0) = face_output[i];
            }

            cv::Mat coord = post_progress(cvMat, image_str.matri);
            NCNN_LOGE("result.rows= %d", coord.rows);
            NCNN_LOGE("result.cols= %d", coord.cols);

            facelandmarks.push_back(coord);

        }
    }
//    ncnn::Mat face_input = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, width, height, 192, 192);
//    const float face_mean_vals[3] = {0.f, 0.f,0.f};
//    const float face_norm_vals[3] = {1.f, 1.f,1.f};
//    face_input.substract_mean_normalize(face_mean_vals, face_norm_vals);
//    ncnn::Mat face_output;
//    ncnn::Extractor ex_face = landmarks.create_extractor();
//    ex_face.input("data", face_input);
//    ex_face.extract("fc1", face_output);

    return 0;
}

int SCRFD::draw(cv::Mat& rgb, const std::vector<FaceObject>& faceobjects, const std::vector<cv::Mat>& facelandmarks)
{
    for (size_t i = 0; i < faceobjects.size(); i++)
    {
        const FaceObject& obj = faceobjects[i];

//         fprintf(stderr, "%.5f at %.2f %.2f %.2f x %.2f\n", obj.prob,
//                 obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(rgb, obj.rect, cv::Scalar(0, 255, 0));

        for(int j =0; j < facelandmarks[i].rows; j++){ //绘制关键点
            float x = facelandmarks[i].at<float>(j, 0);
            float y = facelandmarks[i].at<float>(j, 1);
            cv::circle(rgb,cv::Point2f(x, y),2, cv::Scalar(255, 255, 0), -1);
        }

        if (has_kps)
        {
            cv::circle(rgb, obj.landmark[0], 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(rgb, obj.landmark[1], 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(rgb, obj.landmark[2], 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(rgb, obj.landmark[3], 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(rgb, obj.landmark[4], 2, cv::Scalar(255, 255, 0), -1);
        }

        char text[256];
        sprintf(text, "%.1f%%", obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), -1);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    return 0;
}