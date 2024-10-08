/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : app.cpp
 * Authors    : lqwang@inspur
 * Create Time: 2024-05-28:13:53:14
 * Description:
 * 
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <thread>
#include <unistd.h>

#include <opencv2/opencv.hpp>

#include <mariana_llm/mariana_llm.h>

class Timer {
public:
    Timer() {
        start = (double)cv::getTickCount();
    }
    ~Timer() {
        std::cout<<"timing usage::"<<((double)cv::getTickCount() - start) / cv::getTickFrequency()<<"s"<<std::endl;
    }
    double start = 0.d;
};


static void generate(mariana::GptModel* gpt_model, const std::string& img_path, const std::string& save_path) {
    mariana::ExeContext& context = *gpt_model->context;
    context.prompt = "person. shirtless. chef uniform. hat. not wearing hat. wear face mask. not wear face mask. smoking. fire. opened dustbin. closeed dustbin. people holding cell phone. mouse with tail. mouse. working clothes.";
    cv::Mat img = cv::imread(img_path);
    mariana::AIImage image;
    image.device = mariana::DataOn::CPU;
    image.data = img.data;
    image.batch = 1;
    image.height = img.rows;
    image.width = img.cols;
    image.channel = img.channels();
    context.image = image;
    context.post_info.box_threshold = 0.3f;
    context.post_info.text_threshold = 0.3f;
    mariana::AIResult result;
    {
        Timer t1;
        result = mariana_compute_lmodel(gpt_model);
    }
    
    // {
    //     Timer t2;
    //     result = mariana_compute_lmodel(gpt_model);
    // }
    for (auto it : result.results) {
        std::string prompt;
        for (auto s : it.prompts) {
            prompt += s + " ";
        }
        std::cout<<"boxes:["<<it.bbox.tl.x<<" "<<it.bbox.tl.y<<" "
                 <<it.bbox.br.x<<" "<<it.bbox.br.y<<" ] score:"<<it.score<<" label:"<<prompt<<std::endl;
        cv::Rect roi(it.bbox.tl.x, it.bbox.tl.y, it.bbox.width(), it.bbox.height());
        cv::rectangle(img, roi, cv::Scalar(0, 0, 255), 4);
        cv::putText(img, prompt, cv::Point(it.bbox.tl.x, it.bbox.tl.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,0,0), 2, 8, false);
    }
    cv::imwrite(save_path, img);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout<<"Usage : ./app 1\n";
        return 0;
    }
    mariana::mariana_llm_init();
    mariana::GptParams gpt_params;
    gpt_params.n_threads = std::stoi(argv[1]);
    gpt_params.config_dir = "/home/lqwang/.cache/huggingface/hub/models--IDEA-Research--grounding-dino-tiny/snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71";
    gpt_params.lmodel = mariana::LModelCategory::GroundingDINO;
    gpt_params.backend = mariana::DataOn::GPU;
    
    mariana::GptModel* gpt_model = mariana::mariana_create_lmodel(gpt_params);
    generate(gpt_model, "000000000285.jpg", "res1.jpg");
    // generate(gpt_model, "000000000139.jpg", "res2.jpg");
    // generate(gpt_model, "000000563267.jpg", "res3.jpg");
    mariana::mariana_destroy_lmodel(gpt_model);
    mariana::mariana_llm_finit();
    return 0;
}
