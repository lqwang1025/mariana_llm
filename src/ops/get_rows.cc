/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/embedding.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-18:11:55:34
 * Description:
 * 
 */

#include <ops/get_rows.h>
#include <ops/backend/cpu/get_rows.h>

#include <utils/mariana_define.h>
#include <models/model_param.h>

namespace mariana {

bool GetRowsFunc::init(const ModelParam& param, const std::string& node_name) {
    TRACE();
    ModelParam::SafeTensorInfo sti;
    TRY_STL(sti = param.sti_map.at(node_name+".weight"), return false);
    Tensor weight(sti.shape, DataOn::CPU, sti.data, sti.dtype, true/*move_data*/);
    m_weight = weight;
    return true;
}

bool GetRowsFunc::plan_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    if (inputs[0].dim_size() != 2) {
        MLOG(ERROR)<<"GetRows input's dimision must be 2";
        return false;
    }
    int32_t nb = inputs[0].dim_at(0);
    int32_t nr = inputs[0].dim_at(1);
    int32_t ne = m_weight.dim_at(1);
    
    if (outputs.empty()) {
        outputs.push_back(Tensor(inputs[0].device()));
    }
    outputs[0].try_realloc({nb, nr, ne}, m_weight.dtype());
    return true;
}

/**
 * Example:
 * thread_size = 128, batch_size = 8, rows = 52
 * Take the example of processing 1 piece of data per thread: total numbers of threads: 8(nb)x52(nr)=416(nbr),
 * In order to more fully utilize cpu resources, it is now handled as follows:
 * Spread(Flatten) the two-dimensional data:,
 * 416/128=3(blk_size) (chunk, Each thread handles 3 pieces of data),
 * 416%128=32(remainder) (For the remaining 32 data, 32 threads are opened and each thread processes one piece of data),
 * A total of 128+32=160 threads are needed to run this function.
 * |----------------------->8x52=416<-----------------------------|
 *  ..............................................................
 * | chunk=3 | chunk=3 | chunk=3 |.....| chunk=3 |  remainder=32  |
 * |  ith=0  |  ith=1  |  ith=2  |.....| ith=127 |ith=[128,128+32]|
 **/

bool GetRowsFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    TRACE();
    _parallel_sync(m_tp, inputs[0].total_size(), get_rows, std::ref(inputs[0]),
                   std::ref(m_weight), std::ref(outputs[0]));
    return true;
}

} // namespace mariana
