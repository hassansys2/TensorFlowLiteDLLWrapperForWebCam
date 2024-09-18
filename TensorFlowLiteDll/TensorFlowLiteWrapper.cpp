#include "pch.h"

using namespace cv;
using namespace tflite;

VideoCapture cap;
std::unique_ptr<tflite::Interpreter> interpreter;
std::unique_ptr<tflite::FlatBufferModel> model;
tflite::ops::builtin::BuiltinOpResolver resolver;

Mat background;
Mat frame;
Mat resized_frame;

Mat smooth_mask_3ch;
Mat segmentation_mask_resized;
Mat binary_mask;

Mat frame_float;
Mat background_float;

Mat foreground;
Mat background_overlay;

int mask_n = 5;
double threshold_value = 0.6;
std::deque<Mat> mask_buffer;

extern "C" __declspec(dllexport) int Initialize() {
    cap.open(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the webcam." << std::endl;
        return -1;
    }

    model = tflite::FlatBufferModel::BuildFromFile("selfie_segmentation_landscape.tflite");
    if (!model) {
        std::cerr << "Failed to load model." << std::endl;
        return -2;
    }

    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to create interpreter." << std::endl;
        return -3;
    }

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
        return -4;
    }

    background = imread("background_image.jpg");

    if (background.empty()) {
        return -7;
    }

    cap >> frame;
    if (frame.empty()) {
        return -6;
    }

    resize(background, background, frame.size());

    return 0;
}

extern "C" __declspec(dllexport) int CaptureFrameAndSegment(unsigned char* buffer, int width, int height, char key) {
    Mat output;
    Mat avg_mask;
    
    cap >> frame;
    if (frame.empty()) {
        return -6;
    }

    resize(frame, resized_frame, Size(256, 144));
    resized_frame.convertTo(resized_frame, CV_32FC3, 1.0 / 255.0);

    memcpy(interpreter->typed_input_tensor<float>(0), resized_frame.data,    resized_frame.total() * resized_frame.elemSize());

    if (interpreter->Invoke() != kTfLiteOk) {
        return -7;
    }

    resize(Mat(144, 256, CV_32FC1, (void*)interpreter->typed_output_tensor<float>(0)), segmentation_mask_resized, frame.size());
    
    mask_buffer.push_back(segmentation_mask_resized);

    if (mask_buffer.size() > mask_n) {
        mask_buffer.pop_front();
    }

    avg_mask = Mat::zeros(segmentation_mask_resized.size(), CV_32FC1);
    for (const Mat& m : mask_buffer) {
        avg_mask += m;
    }
    avg_mask /= static_cast<float>(mask_buffer.size());
    
    threshold(avg_mask, binary_mask, threshold_value, 1, THRESH_BINARY);
    GaussianBlur(binary_mask, binary_mask, Size(15, 15), 0);

    Mat mask_channels[] = { binary_mask, binary_mask, binary_mask };
    merge(mask_channels, 3, smooth_mask_3ch);
    smooth_mask_3ch.convertTo(smooth_mask_3ch, CV_32FC3);

    frame.convertTo(frame_float, CV_32FC3, 1.0 / 255.0);
    background.convertTo(background_float, CV_32FC3, 1.0 / 255.0);

    multiply(frame_float, smooth_mask_3ch, foreground);
    multiply(background_float, Scalar(1, 1, 1) - smooth_mask_3ch, background_overlay);

    add(foreground, background_overlay, output);

    output.convertTo(output, CV_8UC3, 255.0);
    
    memcpy(buffer, output.data, width * height * sizeof(unsigned char));

    switch (key)
    {
        case '+':
                std::cout << "masks: " << (mask_n = std::max(mask_n + 1, 1)) << std::endl;
            break;
        case '-':
                std::cout << "masks: " << (mask_n = std::max(mask_n - 1, 1)) << std::endl;
            break;
        case 'u':
                std::cout << "threshold value: " << (threshold_value = std::min((float)threshold_value + 0.05f, 1.0f)) << std::endl;
            break;
        case 'd':
                std::cout << "threshold value: " << (threshold_value = std::max((float)threshold_value - 0.05f, 0.0f)) << std::endl;
            break;
        /*case 'b':
            current_background_index = (current_background_index + 1) % (backgrounds.size() + 1);
            if (current_background_index == backgrounds.size()) current_background_index = -1;
            std::cout << "Switched to next background: " << (current_background_index == -1 ? "Blurred Background" : std::to_string(current_background_index)) << std::endl;
            break;*/
    default:
        break;
    }

    return 0;
}

extern "C" __declspec(dllexport) void Release() {
    cap.release();
    destroyAllWindows();
}
