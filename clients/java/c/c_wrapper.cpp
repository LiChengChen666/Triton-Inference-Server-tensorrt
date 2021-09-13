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
    std::cout << "protocols : " << protocols << std::endl;
    std::vector<char> vec_data(buf, buf + size);
    cv::Mat input_img = cv::imdecode(vec_data, 1);
    cv::imwrite("3.jpg", input_img);
    //cv::Mat input_img = cv::Mat(height, width, CV_8UC3, (void *)buf);
    std::string model_name = model;
    return new Client(input_img, protocols, model_name, conf);
}

extern "C" int Client_getresult(Client *self, RECT rect[])
{
    std::vector<std::vector<int>> res;

    res = self->getResult();

    std::cout << "num of boxes: " << res.size() << std::endl;
    for (int i = 0; i < res.size(); i++)
    {
        for (int j = 0; j < res[i].size(); j++)
        {
            std::cout << res[i][j] << " ";
        }
        std::cout << std::endl;
    }

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

extern "C" void arrary_free(int *p)
{
    if (p)
    {
        free(p);
        p = NULL;
    }
}