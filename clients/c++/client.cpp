#include <gflags/gflags.h>
#include <iostream>
#include <http_client.h>
#include <grpc_client.h>
#include "Triton.hpp"

DEFINE_string(input, "", "Input file to load");
DEFINE_int32(protocols, 0, "0 for grpc,1 for http");
DEFINE_string(filename, "", "coin names to load");
DEFINE_string(model, "yolov5", "Inference model name, default yolov5");
DEFINE_int32(width, 640, "Inference model input width, default 640");
DEFINE_int32(height, 640, "Inference model input width, default 640");
DEFINE_string(out, "", "Write output into file instead of displaying it");
DEFINE_double(confidence, 0.6, "Confidence threshold for detected objects, default 0.6");
DEFINE_double(nms, 0.4, "Non-maximum suppression threshold for filtering raw boxes, default 0.4");


int main(int argc, char **argv)
{
    google::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_input.empty())
    {
        std::cerr << "WARNING: no input file" << std::endl;
        exit(1);
    }
    std::string modelName = FLAGS_model;
    std::string modelVersion = "";
    std::string input_img = FLAGS_input;
    int protocol_type = FLAGS_protocols;
    std::string serverAddress;
    Triton::ProtocolType protocol;
    if (protocol_type == 0)
    {
        serverAddress = "localhost:8001";
        protocol = Triton::ProtocolType::GRPC;
    }
    else
    {
        serverAddress = "localhost:8000";
        protocol = Triton::ProtocolType::HTTP;
    }

    std::string url(serverAddress);
    const std::string fileName = FLAGS_filename;

    std::cout << "Server address: " << serverAddress << std::endl;
    std::cout << "Video name: " << input_img << std::endl;
    std::cout << "Protocol:  " << protocol;
    std::cout << "Path to labels name:  " << fileName << std::endl;

    Triton::TritonClient tritonClient;
    tc::Error err;
    tc::Headers http_headers;

    if (protocol == Triton::ProtocolType::GRPC)
    {
        err = tc::InferenceServerGrpcClient::Create(
            &tritonClient.grpcClient, url, false);
    }
    else
    {
        err = tc::InferenceServerHttpClient::Create(
            &tritonClient.httpClient, url, false);
    }
    if (!err.IsOk())
    {
        std::cerr << "error: unable to create client for inference: " << err
                  << std::endl;
        exit(1);
    }
    Triton::TritonModelInfo yoloModelInfo;
    Triton::setModel(yoloModelInfo, 1);
    tc::InferInput *input;
    err = tc::InferInput::Create(
        &input, yoloModelInfo.input_name_, yoloModelInfo.shape_, yoloModelInfo.input_datatype_);
    if (!err.IsOk())
    {
        std::cerr << "unable to get input: " << err << std::endl;
        exit(1);
    }
    std::shared_ptr<tc::InferInput> input_ptr(input);
    std::vector<tc::InferInput *> inputs = {input_ptr.get()};
    std::vector<const tc::InferRequestedOutput *> outputs;
    for (auto output_name : yoloModelInfo.output_names_)
    {
        tc::InferRequestedOutput *output;
        err =
            tc::InferRequestedOutput::Create(&output, output_name);
        if (!err.IsOk())
        {
            std::cerr << "unable to get output: " << err << std::endl;
            exit(1);
        }
        else
            std::cout << "Created output " << output_name << std::endl;
        outputs.push_back(std::move(output));
    }
    tc::InferOptions options(modelName);
    options.model_version_ = modelVersion;

    cv::Mat frame = cv::imread(input_img);
    if (frame.empty())
    {
        std::cerr << "WARNING: input file is empty" << std::endl;
        exit(1);
    }
    std::vector<uint8_t> input_data;
    std::vector<cv::Mat> frameBatch;
    std::vector<std::vector<uint8_t>> input_data_raw;

    Yolo::coin_names = Yolo::readLabelNames(fileName);

    if (Yolo::coin_names.size() != Yolo::CLASS_NUM)
    {
        std::cerr << Yolo::coin_names.size() << std::endl;
        std::cerr << Yolo::CLASS_NUM << std::endl;
        std::cerr << "Wrong labels filename or wrong path to file: " << fileName << std::endl;
        exit(1);
    }
    int it = 0;
    while (it < 1)
    {
        frameBatch.push_back(frame.clone());
        if (frameBatch.size() < 1)
        {
            continue;
        }

        // Reset the input for new request.
        err = input_ptr->Reset();
        if (!err.IsOk())
        {
            std::cerr << "failed resetting input: " << err << std::endl;
            exit(1);
        }

        for (size_t batchId = 0; batchId < 1; batchId++)
        {
            input_data_raw.push_back(Triton::Preprocess(
                frameBatch[batchId], yoloModelInfo.input_format_, yoloModelInfo.type1_, yoloModelInfo.type3_,
                yoloModelInfo.input_c_, cv::Size(yoloModelInfo.input_w_, yoloModelInfo.input_h_)));
            err = input_ptr->AppendRaw(input_data_raw[batchId]);
            if (!err.IsOk())
            {
                std::cerr << "failed setting input: " << err << std::endl;
                exit(1);
            }
        }
        tc::InferResult *result;
        std::unique_ptr<tc::InferResult> result_ptr;
        if (protocol == Triton::ProtocolType::HTTP)
        {
            err = tritonClient.httpClient->Infer(
                &result, options, inputs, outputs);
        }
        else
        {
            err = tritonClient.grpcClient->Infer(
                &result, options, inputs, outputs);
        }
        if (!err.IsOk())
        {
            std::cerr << "failed sending synchronous infer request: " << err
                      << std::endl;
            exit(1);
        }

        const int DETECTION_SIZE = sizeof(Yolo::Detection) / sizeof(float);
        const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * DETECTION_SIZE + 1;
        auto [detections, shape] = Triton::PostprocessYoloV5(result, 1, yoloModelInfo.output_names_, yoloModelInfo.max_batch_size_ != 0);
        std::vector<std::vector<Yolo::Detection>> batch_res(1);
        const float *prob = detections.data();
        for (size_t batchId = 0; batchId < 1; batchId++)
        {
            auto &res = batch_res[batchId];
            Yolo::nms(res, &prob[batchId * OUTPUT_SIZE], 0.6, 0.4);
        }
        for (size_t batchId = 0; batchId < 1; batchId++)
        {
            auto &res = batch_res[batchId];
            cv::Mat img = frameBatch.at(batchId);
            for (size_t j = 0; j < res.size(); j++)
            {
                cv::Rect r = Yolo::get_rect(img, res[j].bbox);
                cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                std::cout<<Yolo::coin_names[(int)res[j].class_id]<<std::endl;
                cv::putText(img, Yolo::coin_names[(int)res[j].class_id], cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
                //cv::imwrite("save.jpg",img);
            }
        }
        frameBatch.clear();
        input_data_raw.clear();
        it++;
    }

    google::ShutDownCommandLineFlags();
    return 0;
}
