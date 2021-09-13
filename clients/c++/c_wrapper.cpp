#include "clientclass.h"

typedef struct Rect  
{  
    int label;
    int x;  
    int y;  
    int width;  
    int height;  
} RECT;

extern "C" Client *Client_ctor(char *buf, int size, int protocols, char *model, float conf);
Client *Client_ctor(char *buf, int size, int protocols, char *model, float conf)
{
    std::vector<char> vec_data(buf, buf + size);
    cv::Mat input_img = cv::imdecode(vec_data, 1);
    std::string model_name = model;
    return new Client(input_img, protocols, model_name, conf);
}

extern "C" int Client_getresult(Client *self, RECT rect[])
{
    std::vector<std::vector<int>> res;

    res = self->getResult();
    int size = res.size();

    for (int i = 0; i < res.size(); i++)
    {
        rect[i].label = res[i][0];
        rect[i].x = res[i][1];
        rect[i].y = res[i][2];
        rect[i].width = res[i][3];
        rect[i].height = res[i][4];
    }

    return size;
}
