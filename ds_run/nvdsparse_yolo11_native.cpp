#include <vector>
#include <algorithm>
#include <cmath>
#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseYoloNative(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (outputLayersInfo.empty()) return false;

    const float* output = nullptr;
    int layerIdx = -1;
    int numAnchors = 0;
    int numChannels = 0;
    int numClasses = 0;
    bool channelFirst = false;

    int numClassesConfigured = detectionParams.numClassesConfigured;

    for (unsigned int i = 0; i < outputLayersInfo.size(); ++i) {
        int ndims = outputLayersInfo[i].inferDims.numDims;
        int d0 = outputLayersInfo[i].inferDims.d[0];
        int d1 = (ndims > 1) ? outputLayersInfo[i].inferDims.d[1] : 0;

        int channels = std::min(d0, d1);
        int anchors = std::max(d0, d1);

        int expectedChannels = 4 + numClassesConfigured;

        if (channels == expectedChannels && anchors >= 1000) {
            layerIdx = i;
            numChannels = channels;
            numAnchors = anchors;
            numClasses = numClassesConfigured;
            channelFirst = (d0 == channels);
            break;
        }
    }

    if (layerIdx < 0) {
        for (unsigned int i = 0; i < outputLayersInfo.size(); ++i) {
            int ndims = outputLayersInfo[i].inferDims.numDims;
            int d0 = outputLayersInfo[i].inferDims.d[0];
            int d1 = (ndims > 1) ? outputLayersInfo[i].inferDims.d[1] : 0;

            int channels = std::min(d0, d1);
            int anchors = std::max(d0, d1);

            if (channels == 13 && anchors >= 1000) {
                layerIdx = i;
                numChannels = channels;
                numAnchors = anchors;
                numClasses = 9;
                channelFirst = (d0 == channels);
                break;
            }
        }
    }

    if (layerIdx < 0) {
        return false;
    }

    output = (const float*)outputLayersInfo[layerIdx].buffer;
    int netWidth = networkInfo.width;
    int netHeight = networkInfo.height;
    float threshold = detectionParams.perClassPreclusterThreshold[0];

    auto getVal = [&](int channel, int anchor) -> float {
        if (channelFirst) {
            return output[channel * numAnchors + anchor];
        } else {
            return output[anchor * numChannels + channel];
        }
    };

    for (int i = 0; i < numAnchors; ++i) {
        float cx = getVal(0, i);
        float cy = getVal(1, i);
        float w = getVal(2, i);
        float h = getVal(3, i);

        float maxScore = 0.0f;
        int classId = -1;

        for (int c = 0; c < numClasses; ++c) {
            float cls = getVal(4 + c, i);
            if (cls > maxScore) {
                maxScore = cls;
                classId = c;
            }
        }

        if (maxScore >= threshold && classId >= 0 && classId < numClasses) {
            float left = cx - w / 2.0f;
            float top = cy - h / 2.0f;

            left = std::max(0.0f, std::min(left, (float)netWidth - 1));
            top = std::max(0.0f, std::min(top, (float)netHeight - 1));
            w = std::min(w, (float)netWidth - left);
            h = std::min(h, (float)netHeight - top);

            if (w > 2 && h > 2) {
                NvDsInferParseObjectInfo objInfo;
                objInfo.classId = classId;
                objInfo.detectionConfidence = maxScore;
                objInfo.left = left;
                objInfo.top = top;
                objInfo.width = w;
                objInfo.height = h;
                objectList.push_back(objInfo);
            }
        }
    }

    return true;
}
