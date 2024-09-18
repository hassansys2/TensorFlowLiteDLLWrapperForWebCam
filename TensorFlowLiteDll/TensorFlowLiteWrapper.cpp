#include "pch.h"

using namespace cv;
using namespace tflite;
namespace fs = std::filesystem;

VideoCapture cap;
std::unique_ptr<tflite::Interpreter> interpreter;
std::unique_ptr<tflite::FlatBufferModel> model;
tflite::ops::builtin::BuiltinOpResolver resolver;

Mat background;
Mat frame;

Mat resized_frame;
Mat segmentation_mask_resized;
Mat avg_mask;
Mat binary_mask;
Mat smooth_mask;
Mat smooth_mask_3ch;
Mat frame_float, background_float;
Mat foreground, background_overlay;

int current_background_index = 0;
int mask_n = 5;
double threshold_value = 0.6;

std::vector<Mat> backgrounds;
std::deque<cv::Mat>* mask_buffer = nullptr;

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

    std::string background_folder = "backgrounds";
    for (const auto& entry : fs::directory_iterator(background_folder)) {
        Mat bg = imread(entry.path().string());
        if (!bg.empty()) {
            backgrounds.push_back(bg);
        }
    }

    if (backgrounds.empty()) {
        std::cerr << "Error: No background images found in folder: " << background_folder << std::endl;
        return -1;
    }

    mask_buffer = new std::deque<cv::Mat>();

    if (!mask_buffer) {
        std::cerr << "Masks_buffer Allocation Failed" << std::endl;
        return -1;
    }

    return 0;
}

extern "C" __declspec(dllexport) int CaptureFrameAndSegment(unsigned char* buffer, int width, int height, char key) {
    Mat output;
    
    cap >> frame;
    if (frame.empty()) {
        std::cerr << "Failed to capture frame from webcam." << std::endl;
        return -1;
    }

    if (current_background_index == -1) {
        GaussianBlur(frame, background, Size(45, 45), 0);
    }
    else {
        resize(backgrounds[current_background_index], background, frame.size());
    }
    
    resize(frame, resized_frame, Size(256, 144));
    resized_frame.convertTo(resized_frame, CV_32FC3, 1.0 / 255.0);

    float* input_tensor = interpreter->typed_input_tensor<float>(0);
    memcpy(input_tensor, resized_frame.data, resized_frame.total() * resized_frame.elemSize());

    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke tflite interpreter." << std::endl;
        return -2;
    }

    const float* output_tensor = interpreter->typed_output_tensor<float>(0);
    Mat segmentation_mask(144, 256, CV_32FC1, (void*)output_tensor);

    resize(segmentation_mask, segmentation_mask_resized, frame.size());

    mask_buffer->push_back(segmentation_mask_resized);

    if (mask_buffer->size() > mask_n) {
        mask_buffer->pop_front();
    }

    
    avg_mask = Mat::zeros(segmentation_mask_resized.size(), CV_32FC1);
    for (const Mat& m : *mask_buffer) {
        avg_mask += m;
    }
    
    avg_mask /= static_cast<float>(mask_buffer->size());

    threshold(avg_mask, binary_mask, threshold_value, 1, THRESH_BINARY);

    GaussianBlur(binary_mask, smooth_mask, Size(15, 15), 0);

    Mat mask_channels[] = { smooth_mask, smooth_mask, smooth_mask };
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
        case 'b':
            current_background_index = (current_background_index + 1) % (backgrounds.size() + 1);
            if (current_background_index == backgrounds.size()) current_background_index = -1;
            std::cout << "Switched to next background: " << (current_background_index == -1 ? "Blurred Background" : std::to_string(current_background_index)) << std::endl;
            break;
    default:
        break;
    }

    return 0;
}

extern "C" __declspec(dllexport) void Release() {
    cap.release();
    destroyAllWindows();
}
