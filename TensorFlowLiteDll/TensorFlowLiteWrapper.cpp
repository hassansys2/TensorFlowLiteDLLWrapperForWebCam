#include "pch.h"

using namespace cv;
using namespace tflite;
namespace fs = std::filesystem;

Mat frame;
Size frame_size;
VideoCapture cap;
std::unique_ptr<tflite::Interpreter> interpreter;

Mat resized_frame;
Mat avg_mask;
Mat binary_mask;
Mat smooth_mask;
Mat smooth_mask_3ch;

Mat frame_float, background_float;
Mat segmentation_mask_resized;

//Mat mask_channels[3];

std::deque<Mat> mask_buffer;

Mat foreground, background_overlay, output;

int n = 5;
float threshold_value = 0.6;

Mat background;
std::vector<Mat> backgrounds;
int current_background_index = 0;

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

    std::string background_folder = "backgrounds";
    for (const auto& entry : fs::directory_iterator(background_folder)) {
        Mat bg = imread(entry.path().string());
        if (!bg.empty()) {
            backgrounds.push_back(bg);
        }
    }

    if (backgrounds.empty()) {
        return -7;
    }

    cap >> frame;
    if (frame.empty()) {
        return -6;
    }

    frame_size = frame.size();

    resize(backgrounds[current_background_index], background, frame_size);

    return 0;
}

extern "C" __declspec(dllexport) int CaptureFrameAndSegment(unsigned char* buffer, int width, int height, char key) {
    cap >> frame;
    if (frame.empty()) {
        return -6;
    }

    
    resize(frame, resized_frame, Size(256, 144));
    resized_frame.convertTo(resized_frame, CV_32FC3, 1.0 / 255.0);

    float* input_tensor = interpreter->typed_input_tensor<float>(0);
    memcpy(input_tensor, resized_frame.data, resized_frame.total() * resized_frame.elemSize());

    if (interpreter->Invoke() != kTfLiteOk) {
        return -7;
    }

    // Get output (float32[1,144,256,1])
    const float* output_tensor = interpreter->typed_output_tensor<float>(0);
    resize(Mat(144, 256, CV_32FC1, (void*)output_tensor), segmentation_mask_resized, frame_size);
    
    
    /*
    mask_buffer.push_back(segmentation_mask_resized);

    if (mask_buffer.size() > n) {
        mask_buffer.pop_front();
    }

    avg_mask = Mat::zeros(segmentation_mask_resized.size(), CV_32FC1);
    for (const Mat& m : mask_buffer) {
        avg_mask += m;
    }
    avg_mask /= static_cast<float>(mask_buffer.size());
    */
    
    threshold(segmentation_mask_resized, binary_mask, threshold_value, 1, THRESH_BINARY);
    
    //GaussianBlur(binary_mask, smooth_mask, Size(15, 15), 0);
    Mat mask_channels[] = { binary_mask, binary_mask, binary_mask };
    //mask_channels[0] = mask_channels[1] = mask_channels[2] = binary_mask;
    merge(mask_channels, 3, smooth_mask_3ch);
    smooth_mask_3ch.convertTo(smooth_mask_3ch, CV_32FC3);
    
    frame.convertTo(frame_float, CV_32FC3, 1.0 / 255.0);
    background.convertTo(background_float, CV_32FC3, 1.0 / 255.0);

    multiply(frame_float, smooth_mask_3ch, foreground);
    multiply(background_float, Scalar(1, 1, 1) - smooth_mask_3ch, background_overlay);

    add(foreground, background_overlay, output);
    
    frame.convertTo(output, CV_8UC3, 255.0);
    
    
    memcpy(buffer, background_overlay.data, frame_size.area());

    
    switch (key)
    {
        case '+':
            n = std::min(n + 1, 20);
            std::cout << "masks: " << n << std::endl;
            break;
        case '-':
            n = std::max(n - 1, 1);
            std::cout << "masks: " << n << std::endl;
            break;
        case 'u':
            threshold_value = std::min(threshold_value + 0.05f, 1.0f);
            std::cout << "threshold value: " << threshold_value << std::endl;
            break;
        case 'd':
            threshold_value = std::max(threshold_value - 0.05f, 0.0f);
            std::cout << "threshold value: " << threshold_value << std::endl;
            break;
        case 'b':
            current_background_index = (current_background_index + 1) % (backgrounds.size() + 1);
            if (current_background_index == backgrounds.size()) current_background_index = -1;
            std::cout << "Switched to next background: " << (current_background_index == -1 ? "Blurred Background" : std::to_string(current_background_index)) << std::endl;

            if (current_background_index == -1) {
                GaussianBlur(frame, background, Size(45, 45), 0);
            }
            else {
                resize(backgrounds[current_background_index], background, frame_size);
            }

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
