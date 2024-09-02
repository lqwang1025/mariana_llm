/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-18:09:05:08
 * Description:
 * 
 */

#include <ops/ops.h>
#include <utils/mariana_define.h>

namespace mariana {

std::string op_to_string(const OpCategory& op_cate) {
#define CASE_ITEM(val)                          \
    case OpCategory::val:                       \
        return std::string(STR(val))
    switch (op_cate) {
        CASE_ITEM(None);
        CASE_ITEM(Conv2D);
        CASE_ITEM(GetRows);
        CASE_ITEM(LayerNorm);
        CASE_ITEM(Add);
        CASE_ITEM(SelfAtt);
        CASE_ITEM(Permute);
        CASE_ITEM(AttMask);
        CASE_ITEM(MatMul);
        CASE_ITEM(GELU);
        CASE_ITEM(SwinLayer);
        CASE_ITEM(Pad);
        CASE_ITEM(Slice);
        CASE_ITEM(Roll);
        CASE_ITEM(SwinPatchMerging);
        CASE_ITEM(SwinStageOutput);
        CASE_ITEM(GroupNorm);
        CASE_ITEM(GroundingDinoSinePositionEmbedding);
        CASE_ITEM(GroundingDinoEncoderLayer);
        CASE_ITEM(GroundingDinoDecoderLayer);
        CASE_ITEM(GroundingDinoEncoderBefore);
        CASE_ITEM(GroundingDinoDecoderBefore);
        CASE_ITEM(GroundingDinoForDetection);
        CASE_ITEM(Pass);
        CASE_ITEM(Mul);
        CASE_ITEM(RELU);
    default:
        return "uninit";
    }
#undef CASE_ITEM
}

} // namespace mariana
