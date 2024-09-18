#include "pch.h"

using namespace cv;
using namespace tflite;
namespace fs = std::filesystem;

VideoCapture cap;
std::unique_ptr<tflite::Interpreter> interpreter;
std::unique_ptr<tflite::FlatBufferModel> model;
tflite::ops::builtin::BuiltinOpResolver resolver;

Size frame_size;

Mat background;
Mat frame;

Mat resized_frame;
Mat segmentation_mask_resized;

Mat binary_mask;
Mat smooth_mask;
Mat smooth_mask_3ch;
Mat frame_float, background_float;
Mat foreground, background_overlay;

int current_background_index = 0;
int mask_n = 5;
double threshold_value = 0.6;

std::vector<cv::Mat>* backgrounds = nullptr;
static std::deque<cv::Mat> mask_buffer;

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
    backgrounds = new std::vector<cv::Mat>();
    
    /*mask_buffer = new std::deque<cv::Mat>();
    if (!mask_buffer) {
        std::cerr << "Masks_buffer Allocation Failed" << std::endl;
        return -1;
    }*/

    if (!backgrounds) {
        std::cerr << "backgrounds Allocation Failed" << std::endl;
        return -1;
    }

    for (const auto& entry : fs::directory_iterator(background_folder)) {
        Mat bg = imread(entry.path().string());
        if (!bg.empty()) {
            backgrounds->push_back(bg);
        }
    }

    if (backgrounds->empty()) {
        std::cerr << "Error: No background images found in folder: " << background_folder << std::endl;
        return -1;
    }

    cap >> frame;
    if (frame.empty()) {
        std::cerr << "Failed to capture frame from webcam." << std::endl;
        return -1;
    }

    frame_size = frame.size();
    resize((*backgrounds)[current_background_index], background, frame_size);

    return 0;
}

extern "C" __declspec(dllexport) int CaptureFrameAndSegment(unsigned char* buffer, int width, int height, char key) {
    Mat output_frame;
    Mat avg_mask;

    cap >> frame;
    if (frame.empty()) {
        std::cerr << "Failed to capture frame from webcam." << std::endl;
        return -1;
    }

    resize(frame, resized_frame, Size(256, 144));
    resized_frame.convertTo(resized_frame, CV_32FC3, 1.0 / 255.0);

    memcpy(interpreter->typed_input_tensor<float>(0), resized_frame.data, resized_frame.total() * resized_frame.elemSize());

    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke tflite interpreter." << std::endl;
        return -2;
    }

    Mat segmentation_mask(144, 256, CV_32FC1, (void*)interpreter->typed_output_tensor<float>(0));
    resize(segmentation_mask, segmentation_mask_resized, frame_size);

    avg_mask = Mat::zeros(segmentation_mask_resized.size(), CV_32FC1);
    mask_buffer.push_back(segmentation_mask_resized.clone());

    if (mask_buffer.size() > mask_n) {
        mask_buffer.pop_front();
    }
    
    for (const Mat& m : mask_buffer) {
        avg_mask += m;
    }
    
    /*
    for (int i = 0; i < mask_buffer.size(); i++) {
        avg_mask += mask_buffer[i];
        Mat m(mask_buffer[i].clone());
        m.convertTo(m, CV_8UC1, 255.0);
        cv::imshow(("Masks_abs_" + std::to_string(i)), m);
    }
    */
    
    avg_mask /= static_cast<float>(mask_buffer.size());

    Mat dis(avg_mask);
    dis.convertTo(dis, CV_8UC1, 255.0);
    cv::imshow("Average", dis);
    //cv::waitKey(0);

    threshold(avg_mask, binary_mask, threshold_value, 1, THRESH_BINARY);

    GaussianBlur(binary_mask, smooth_mask, Size(15, 15), 0);

    Mat mask_channels[] = { smooth_mask, smooth_mask, smooth_mask };
    merge(mask_channels, 3, smooth_mask_3ch);
    smooth_mask_3ch.convertTo(smooth_mask_3ch, CV_32FC3);

    frame.convertTo(frame_float, CV_32FC3, 1.0 / 255.0);
    background.convertTo(background_float, CV_32FC3, 1.0 / 255.0);

    multiply(frame_float, smooth_mask_3ch, foreground);
    multiply(background_float, Scalar(1, 1, 1) - smooth_mask_3ch, background_overlay);

    add(foreground, background_overlay, output_frame);

    output_frame.convertTo(output_frame, CV_8UC3, 255.0);
    
    memcpy(buffer, output_frame.data, width * height * sizeof(unsigned char));

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
            current_background_index = (current_background_index + 1) % (backgrounds->size() + 1);
            if (current_background_index == backgrounds->size()) current_background_index = 0;
            std::cout << "Switched to next background: " << std::to_string(current_background_index) << std::endl;
            resize((*backgrounds)[current_background_index], background, frame_size);
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
