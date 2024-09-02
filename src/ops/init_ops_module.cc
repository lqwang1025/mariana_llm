/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : init_ops_module.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-18:07:54:06
 * Description:
 * 
 */

#include <ops/ops.h>
#include <ops/all.h>
#include <ops/init_ops_module.h>

namespace mariana {

#define ADD_FUNC(identity, type)                                        \
    static auto __##identity##_make = []()->Function* { return new type{}; }; \
    FunctionHolder::add_Function(OpCategory::identity, __##identity##_make)

static void _register_ops() {
    ADD_FUNC(Conv2D, Conv2dFunc);
    ADD_FUNC(GetRows, GetRowsFunc);
    ADD_FUNC(LayerNorm, LayerNormFunc);
    ADD_FUNC(GroupNorm, GroupNormFunc);
    ADD_FUNC(Add, AddFunc);
    ADD_FUNC(SelfAtt, SelfAttentionFunc);
    ADD_FUNC(Permute, PermuteFunc);
    ADD_FUNC(AttMask, AttMaskFunc);
    ADD_FUNC(MatMul, MatMulFunc);
    ADD_FUNC(GELU, GELUFunc);
    ADD_FUNC(SwinLayer, SwinLayerFunc);
    ADD_FUNC(Slice, SliceFunc);
    ADD_FUNC(SwinPatchMerging, SwinPatchMergingFunc);
    ADD_FUNC(SwinStageOutput, SwinStageOutputFunc);
    ADD_FUNC(GroundingDinoSinePositionEmbedding, GroundingDinoSinePositionEmbeddingFunc);
    ADD_FUNC(GroundingDinoEncoderLayer, GroundingDinoEncoderLayerFunc);
    ADD_FUNC(GroundingDinoDecoderLayer, GroundingDinoDecoderLayerFunc);
    ADD_FUNC(GroundingDinoDecoderBefore, GroundingDinoDecoderBeforeFunc);
    ADD_FUNC(GroundingDinoEncoderBefore, GroundingDinoEncoderBeforeFunc);
    ADD_FUNC(GroundingDinoForDetection, GroundingDinoForDetectionFunc);
    ADD_FUNC(Pass, PassFunc);
    ADD_FUNC(Mul, MulFunc);
}

void init_ops_module() {
    _register_ops();
}

void uninit_ops_module() {
    
}
    
} // namespace mariana
