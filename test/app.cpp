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

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout<<"Usage : ./app 1\n";
        return 0;
    }
    mariana::mariana_llm_init();
    
    mariana::ExeContext context;
    context.config_dir = "/home/lqwang/project/chenan/mariana/mariana_llm/build/grounding-dino";
    std::cout<<"D:ssssssssssssssssssssssssssss";
    void* handle = mariana::mariana_create_lmodel_handle(mariana::LModelCategory::GroundingDINO, context);
    
    
    context.n_threads = std::stoi(argv[1]);
    context.prompt = "person. shirtless. chef uniform. hat. not wearing hat.wear face mask. not wear face mask. smoking. fire. opened dustbin. closeed dustbin. people holding cell phone. mouse with tail. mouse. working clothes.";
    cv::Mat img = cv::imread("ele_2b48e6d1f438314430eece215090c5ad.jpg");
    mariana::AIImage image;
    image.device = mariana::DataOn::CPU;
    image.data = img.data;
    image.batch = 1;
    image.height = img.rows;
    image.width = img.cols;
    image.channel = img.channels();
    context.image = image;
    context.runtime_info.img_hws = {image.height, image.width};
    std::cout<<"D:"<<(void*)image.data<<std::endl;;
    mariana::AIResult result = mariana_compute_lmodel_handle(handle, context);

    for (auto it : result.results) {
        std::cout<<"boxes:["<<it.bbox.tl.x<<" "<<it.bbox.tl.y<<" "
                 <<it.bbox.br.x<<" "<<it.bbox.br.y<<" ] score:"<<it.score<<" label:"<<it.prompts[0]<<"\n";
    }
    mariana::mariana_destroy_lmodel_handle(handle, context);
    mariana::mariana_llm_finit();
    return 0;
}
