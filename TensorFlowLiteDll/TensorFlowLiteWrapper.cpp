#include "pch.h"

using namespace cv;
using namespace tflite;

VideoCapture cap;
std::unique_ptr<tflite::Interpreter> interpreter;

extern "C" __declspec(dllexport) int Initialize() {
    cap.open(0);
    if (!cap.isOpened()) {
        return -1;
    }

    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("selfie_segmentation_landscape.tflite");
    if (!model) {
        return -2;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        return -3;
    }

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        return -4;
    }

    return 0;
}

extern "C" __declspec(dllexport) int CaptureFrameAndSegment(unsigned char* buffer, int width, int height) {
    Mat frame;
    cap >> frame;
    if (frame.empty()) {
        return -5;
    }

    // input size (256x144 for selfie segmentation)
    Mat resized_frame;
    resize(frame, resized_frame, Size(256, 144));
    resized_frame.convertTo(resized_frame, CV_32FC3, 1.0 / 255.0);

    // Input dimensions (float32[1,144,256,3])
    float* input_tensor = interpreter->typed_input_tensor<float>(0);
    memcpy(input_tensor, resized_frame.data, resized_frame.total() * resized_frame.elemSize());

    if (interpreter->Invoke() != kTfLiteOk) {
        return -6;
    }

    // Get output (float32[1,144,256,1])
    const float* output_tensor = interpreter->typed_output_tensor<float>(0);
    Mat segmentation_mask(144, 256, CV_32FC1, (void*)output_tensor);

    Mat segmentation_mask_resized;
    resize(segmentation_mask, segmentation_mask_resized, Size(width, height));

    memcpy(buffer, segmentation_mask_resized.data, width * height * sizeof(unsigned char));

    return 0;
}

extern "C" __declspec(dllexport) void Release() {
    cap.release();
    destroyAllWindows();
}
