/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : single_op_run.cpp
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-02:14:22:19
 * Description:
 * 
 */

#include <iostream>
#include <vector>
#include <string>
#include <core/tensor.h>
#include <ops/math.h>
#include <core/device_type.h>
#include <core/impl/thread_pool.h>
#include <models/init_lmodels_module.h>
#include <mariana_llm/mariana_llm.h>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout<<"Usage : ./app 1\n";
        return 0;
    }
    mariana::init_lmodels_module();
    mariana::ExeContext context;
    mariana::AddFunc* add = new mariana::AddFunc{};
    mariana::Tensor a({1,3,224,224}, mariana::DataOn::GPU);
    float* a_ptr = a.mutable_ptr<float>();
    mariana::Tensor b({1,3,224,224}, mariana::DataOn::GPU);
    float* b_ptr = b.mutable_ptr<float>();
    mariana::Tensor c({1,3,224,224}, mariana::DataOn::GPU);
    float* c_ptr = c.mutable_ptr<float>();
    mariana::ThreadPool* tp = new mariana::ThreadPool{std::stoi(argv[1])};
    add->set_thread_pool(tp);
    std::vector<mariana::Tensor> inputs = {a, b};
    std::vector<mariana::Tensor> outputs = {c};
    add->plan_forward(inputs, outputs, context);
    add->on_forward(inputs, outputs, context);
    return 0;
}
