#include "clientclass.h"

Client::Client(cv::Mat input_img, int protocols, std::string model, float conf)
    : _img(input_img), _protocol_type(protocols), _model(model), _conf(conf)
{
    if (_img.empty())
    {
        std::cerr << "WARNING: input file is empty" << std::endl;
        exit(1);
    }

    if (_protocol_type == 0)
    {
        serverAddress = "localhost:8001";
        protocol = Triton::ProtocolType::GRPC;
    }
    else
    {
        serverAddress = "localhost:8000";
        protocol = Triton::ProtocolType::HTTP;
    }
    std::cout << "model :  " << _model << std::endl;
    std::cout << "Server address: " << serverAddress << std::endl;
    std::cout << "Protocol:  " << protocol << std::endl;

    if (protocol == Triton::ProtocolType::GRPC)
    {
        err = tc::InferenceServerGrpcClient::Create(
            &tritonClient.grpcClient, serverAddress, false);
    }
    else
    {
        err = tc::InferenceServerHttpClient::Create(
            &tritonClient.httpClient, serverAddress, false);
    }
    if (!err.IsOk())
    {
        std::cerr << "error: unable to create client for inference: " << err
                  << std::endl;
        exit(1);
    }

    Triton::setModel(yoloModelInfo, 1);

    err = tc::InferInput::Create(
        &input, yoloModelInfo.input_name_, yoloModelInfo.shape_, yoloModelInfo.input_datatype_);
    if (!err.IsOk())
    {
        std::cerr << "unable to get input: " << err << std::endl;
        exit(1);
    }
}
std::vector<std::vector<int>> Client::getResult()
{
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
    tc::InferOptions options(_model);
    options.model_version_ = modelVersion;
    err = input_ptr->Reset();
    if (!err.IsOk())
    {
        std::cerr << "failed resetting input: " << err << std::endl;
        exit(1);
    }
    input_data = Triton::Preprocess(
        _img, yoloModelInfo.input_format_, yoloModelInfo.type1_, yoloModelInfo.type3_,
        yoloModelInfo.input_c_, cv::Size(yoloModelInfo.input_w_, yoloModelInfo.input_h_));
    err = input_ptr->AppendRaw(input_data);
    if (!err.IsOk())
    {
        std::cerr << "failed setting input: " << err << std::endl;
        exit(1);
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
        Yolo::nms(res, &prob[batchId * OUTPUT_SIZE], _conf, 0.4);
    }
    for (size_t batchId = 0; batchId < 1; batchId++)
    {
        auto &res = batch_res[batchId];
        for (size_t j = 0; j < res.size(); j++)
        {
            std::vector<int> box;
            cv::Rect r = Yolo::get_rect(_img, res[j].bbox);
            box.push_back(res[j].class_id);
            box.push_back(r.x);
            box.push_back(r.y);
            box.push_back(r.width);
            box.push_back(r.height);
            result_vec.push_back(box);
        }
    }
    return result_vec;
}
