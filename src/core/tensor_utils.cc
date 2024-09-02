/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/tensor_utils.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-23:08:13:44
 * Description:
 * 
 */

#include <fstream>
#include <cstdint>

#include <core/tensor.h>
#include <core/tensor_utils.h>

#include <utils/mariana_define.h>

namespace mariana {

void to_bin(const Tensor& tensor, const std::string& name) {
    std::ofstream out(name, std::ios::binary | std::ios::trunc | std::ios::out);
    if (!out.is_open()) {
        MLOG(ERROR)<<"Open file:"<<name<<" failed.";
        return;
    }
    if (tensor.dtype().match<float>()) {
        out.write(reinterpret_cast<const char*>(tensor.ptr<float>(0)), tensor.total_size()*tensor.dtype().itemsize());
    } else if (tensor.dtype().match<int32_t>()) {
        out.write(reinterpret_cast<const char*>(tensor.ptr<int32_t>(0)), tensor.total_size()*tensor.dtype().itemsize());
    } else if (tensor.dtype().match<uint32_t>()) {
        out.write(reinterpret_cast<const char*>(tensor.ptr<uint32_t>(0)), tensor.total_size()*tensor.dtype().itemsize());
    } else if (tensor.dtype().match<int16_t>()) {
        out.write(reinterpret_cast<const char*>(tensor.ptr<int16_t>(0)), tensor.total_size()*tensor.dtype().itemsize());
    } else if (tensor.dtype().match<double>()) {
        out.write(reinterpret_cast<const char*>(tensor.ptr<double>(0)), tensor.total_size()*tensor.dtype().itemsize());
    } else if (tensor.dtype().match<int8_t>()) {
        out.write(reinterpret_cast<const char*>(tensor.ptr<int8_t>(0)), tensor.total_size()*tensor.dtype().itemsize());
    } else if (tensor.dtype().match<uint8_t>()) {
        out.write(reinterpret_cast<const char*>(tensor.ptr<uint8_t>(0)), tensor.total_size()*tensor.dtype().itemsize());
    } else if (tensor.dtype().match<int64_t>()) {
        out.write(reinterpret_cast<const char*>(tensor.ptr<int64_t>(0)), tensor.total_size()*tensor.dtype().itemsize());
    } else {
        MLOG(ERROR)<<"Unsupport dtype:"<<tensor.dtype().name();
    }
    out.close();
}

void to_txt(const Tensor& tensor, const std::string& name) {
    std::ofstream out(name, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        MLOG(ERROR)<<"Open file:"<<name<<" failed.";
        return;
    }
    out<<"shape: [";
    for (int32_t i = 0; i < tensor.dim_size(); ++i) {

        out<<tensor.dim_at(i);
        if (i != tensor.dim_size()-1) {
            out<<", ";
        }
    }
    out<<"]"<<std::endl;
    out<<"dtype:"<<tensor.dtype().name()<<std::endl;
    for (uint32_t i = 0; i < tensor.total_size(); ++i) {
        if (tensor.dtype().match<float>()) {
            out<<tensor.data<float>(i)<<std::endl;
        } else if (tensor.dtype().match<int32_t>()) {
            out<<tensor.data<int32_t>(i)<<std::endl;
        } else if (tensor.dtype().match<uint32_t>()) {
            out<<tensor.data<uint32_t>(i)<<std::endl;
        } else if (tensor.dtype().match<int16_t>()) {
            out<<tensor.data<int16_t>(i)<<std::endl;
        } else if (tensor.dtype().match<double>()) {
            out<<tensor.data<double>(i)<<std::endl;
        } else if (tensor.dtype().match<int8_t>()) {
            out<<(int)tensor.data<int8_t>(i)<<std::endl;
        } else if (tensor.dtype().match<uint8_t>()) {
            out<<(int)tensor.data<uint8_t>(i)<<std::endl;
        } else if (tensor.dtype().match<int64_t>()) {
            out<<(int)tensor.data<int64_t>(i)<<std::endl;
        } else {
            MLOG(ERROR)<<"Unsupport dtype:"<<tensor.dtype().name();
        }
    }
    
    out.close();
}

} // namespace mariana
