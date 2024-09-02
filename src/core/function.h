/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/function.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-18:07:05:05
 * Description:
 *
 */

#ifndef __CORE_FUNCTION_H__
#define __CORE_FUNCTION_H__

#include <cstdint>
#include <functional>
#include <unordered_map>

#include <core/tensor.h>
#include <core/impl/thread_pool.h>

#include <utils/sys.h>
#include <utils/mariana_define.h>

#include <ops/sched_param.h>

namespace mariana {

struct ExeContext;
using tensor_list = std::vector<Tensor>;
struct ModelParam;
enum class OpCategory : int16_t;
class Node; 

struct Function {
    Function() {}
    virtual ~Function() {}
    virtual void set_thread_pool(ThreadPool* tp) {
        m_tp = tp;
    }
    bool on_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
        return _forward(inputs, outputs, context);
    }
    void set_node(Node* node) {
        m_owner = node;
    }
    virtual bool plan_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)=0;
    virtual bool init(const ModelParam& param, const std::string& node_name) {return true;}
protected:
    virtual bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)=0;
public:
    template<typename F, typename...Args>
    static void _parallel_async(ThreadPool* tp, uint32_t size, F&& f, Args&&... args) {
        std::function<void(SchedParam)> func = std::bind(std::forward<F>(f), std::placeholders::_1, std::forward<Args>(args)...);
        const size_t  thread_size = tp->size();
        const int64_t blk_size    = size/thread_size;
        uint64_t remainder = size%thread_size;
        SchedParam sched_param;
        if (blk_size > 0) {
            sched_param.n_chunk = blk_size;
            sched_param.n_thread = thread_size+remainder;
            for(size_t i = 0; i < thread_size; ++i) {
                sched_param.i_thread = i;
                tp->submit(func, sched_param);
            }
            MVLOG(1)<<"Thread info-->data size:"<<size<<" thread size:"
                    <<sched_param.n_thread<<" data chunk:"<<sched_param.n_chunk;
            if (remainder != 0) {
                sched_param.n_chunk  = 1;
                sched_param.n_offset = blk_size*thread_size;
                for (uint64_t ir = 0; ir < remainder; ++ir) {
                    sched_param.i_thread = ir;
                    tp->submit(func, sched_param);
                }
                MVLOG(1)<<"    -->remainder data size:"<<size<<" thread size:"
                        <<sched_param.n_thread<<" data chunk:"<<sched_param.n_chunk;
            }
        } else {
            sched_param.n_thread = remainder;
            sched_param.n_chunk = 1;
            for(size_t i = 0; i < remainder; ++i) {
                sched_param.i_thread = i;
                tp->submit(func, sched_param);
            }
            MVLOG(1)<<"Thread info-->data size:"<<size<<" thread size:"
                    <<sched_param.n_thread<<" data chunk:"<<sched_param.n_chunk;
        }
    }
    template<typename F, typename...Args>
     static void _parallel_sync(ThreadPool* tp, uint32_t size, F&& f, Args&&... args) {
        _parallel_async(tp, size, f, args...);
        tp->wait_work_complete();
    }
protected:
    Node* m_owner = nullptr;
    ThreadPool* m_tp = nullptr;
};

using FuncMake = std::function<Function*()>;
DECLARE_SOMETHING_HOLDER(Function, OpCategory, FuncMake); // FunctionHolder

} // namespace mariana

#endif /* __CORE_FUNCTION_H__ */

