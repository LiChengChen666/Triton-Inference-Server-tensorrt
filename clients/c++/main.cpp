#include "clientclass.h"
#include <iostream>
#include <fstream>

DEFINE_string(input, "", "Input file to load");
DEFINE_int32(protocols, 0, "0 for grpc,1 for http");
DEFINE_string(filename, "", "coin names to load");
DEFINE_string(model, "yolov5", "Inference model name, default yolov5");
DEFINE_int32(width, 640, "Inference model input width, default 640");
DEFINE_int32(height, 640, "Inference model input width, default 640");
DEFINE_string(out, "", "Write output into file instead of displaying it");
DEFINE_double(confidence, 0.6, "Confidence threshold for detected objects, default 0.6");
DEFINE_double(nms, 0.4, "Non-maximum suppression threshold for filtering raw boxes, default 0.4");


typedef struct Rect  
{  
    int label;
    int x;  
    int y;  
    int width;  
    int height;  
} RECT;

extern "C" Client *Client_ctor(char *buf, int size, int protocols, char *model, float conf);
extern "C" int Client_getresult(Client *self, RECT rect[]);


using namespace std;

int main(int argc, char **argv)
{
    google::ParseCommandLineFlags(&argc, &argv, true);

    std::string strFile = FLAGS_input; // 本地图片位置
    std::ifstream in(strFile, ios::in | ios::binary | ios::ate);
    unsigned int size1 = in.tellg();
    char *buffer = new char[size1 + 1]; // 开辟内存
    if (size1 > 0)
    {
        in.seekg(0, ios::beg);
        in.read(buffer, size1); // 将图片二进制数据放到buffer中
        buffer[size1] = 0;
    }
    in.close();
    char *name = (char *)"yolov5";

    Client *client = Client_ctor(buffer, size1, 0, name, 0.6);
    
    RECT *rect = new Rect[5];

    int res = Client_getresult(client, rect);
    printf("%d\n", res);
    for (int i = 0; i < res; i++)
    {
        std::cout<< rect[i].label << " " << rect[i].x << " "<< rect[i].y << " "<< rect[i].width << " "<<rect[i].height<<std::endl;

    }
    delete [] rect;

    google::ShutDownCommandLineFlags();
    return 0;
}