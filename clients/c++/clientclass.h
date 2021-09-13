#pragma once
#ifndef CLASS_H
#define CLASS_H

#include <gflags/gflags.h>
#include <iostream>
#include <http_client.h>
#include <grpc_client.h>
#include "Triton.hpp"

class Client
{
public:
    Client(cv::Mat input_img, int protocols, std::string model, float conf);
    std::vector<std::vector<int>> getResult();

private:
    cv::Mat _img;
    int _protocol_type;
    std::string _model;
    float _conf;

    std::vector<uint8_t> input_data;
    std::string modelVersion = "";
    std::string serverAddress;
    Triton::ProtocolType protocol;
    Triton::TritonClient tritonClient;
    tc::Error err;
    tc::Headers http_headers;

    tc::InferInput *input;
    Triton::TritonModelInfo yoloModelInfo;

    std::vector<std::vector<int>> result_vec;
};

#endif