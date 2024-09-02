/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/ops.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-18:07:48:13
 * Description:
 *
 */

#ifndef __OPS_OPS_H__
#define __OPS_OPS_H__

#include <cstdint>
#include <string>

namespace mariana {

enum class OpCategory : int16_t {
    None                               = 0,
    Conv2D                             = 1,
    GetRows                            = 2,
    LayerNorm                          = 3,
    Add                                = 4,
    SelfAtt                            = 5,
    Permute                            = 6,
    AttMask                            = 7,
    MatMul                             = 8,
    GELU                               = 9, //9-50 is activate func
    SwinLayer                          = 51,
    Pad                                = 52,
    Slice                              = 53,
    Roll                               = 54,
    SwinPatchMerging                   = 55,
    SwinStageOutput                    = 56,
    GroupNorm                          = 57,
    GroundingDinoSinePositionEmbedding = 58,
    GroundingDinoEncoderLayer          = 59,
    GroundingDinoEncoderBefore         = 60,
    Pass                               = 61,
    Mul                                = 62,
    RELU                               = 63,
    GroundingDinoDecoderLayer          = 64,
    GroundingDinoDecoderBefore         = 65,
    GroundingDinoForDetection          = 66,
};

std::string op_to_string(const OpCategory& op_cate);

} // namespace mariana

#endif /* __OPS_OPS_H__ */

