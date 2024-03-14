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

#ifndef SCRFD_H
#define SCRFD_H

#include <opencv2/core/core.hpp>

#include <net.h>

struct FaceObject
{
    cv::Rect_<float> rect; //预测框空间参数
    cv::Point2f landmark[5]; //关键点
    float prob; //置信度
};

class SCRFD
{
public:
    int load(const char* modeltype, bool use_gpu = false);

    int load(AAssetManager* mgr, const char* modeltype, bool use_gpu = false); //加载模型

    int detect(const cv::Mat& rgb, std::vector<FaceObject>& faceobjects, std::vector<cv::Mat>& facelandmarks,float prob_threshold = 0.5f, float nms_threshold = 0.45f); //模型推理

    int draw(cv::Mat& rgb, const std::vector<FaceObject>& faceobjects,const std::vector<cv::Mat>& facelandmarks); //根据模型输出绘图

private:
    ncnn::Net scrfd; //声明检测模型
    bool has_kps;
    ncnn::Net landmarks; //声明关键点模型
};

struct  return_d_m //关键点前处理
{
    cv::Mat dst;
    cv::Mat matri;
};

#endif // SCRFD_H
