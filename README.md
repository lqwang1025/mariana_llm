# HuggingFace.cpp

***

The purpose of this project is to provide a running c++ version of huggingface, through which developers can easily and quickly work seamlessly with huggingface python exported models.  
Mariana, the deepest trench in the world, is a name that for me personally aims at exploring the unknown and enriching my knowledge accordingly!

***

## Get Start

***
1. python models/grounding_dion/export_model_configs.py
2. mkdir build
3. make install -j8
4. cd test & mkdir build
5. make
6. enjoy it.
   
Currently this project only supports GroundingDINO, a large multimodal model.
The supported operators are:
| Operator                                           | CPU                      | CUDA                         |
| :---                                               |    :----:                |          ---:                |
| Conv2D                                             | :white_check_mark:       |:x:                           |
| GetRows                                            | :white_check_mark:       |:white_check_mark:            |
| LayerNorm                                          | :white_check_mark:       | :x:                          |
| Add                                                | :white_check_mark:       | :x:                          |
| SelfAttention                                      | :white_check_mark:       | :x:                          |
| Permute                                            | :white_check_mark:       | :x:                          |
| AttMask                                            | :white_check_mark:       |:white_check_mark:            |
| Matmul                                             | :white_check_mark:       | :x:                          |
| GELU                                               | :white_check_mark:       | :x:                          |
| SwinLayer                                          | :white_check_mark:       | :x:                          |
| Pad                                                | :white_check_mark:       | :x:                          |
| Slice                                              | :white_check_mark:       | :x:                          |
| Roll                                               | :white_check_mark:       | :x:                          |
| SwinPatchMerging                                   | :white_check_mark:       | :x:                          |
| SwinStageOutput                                    | :white_check_mark:       | :x:                          |
| GroupNorm                                          | :white_check_mark:       | :x:                          |
| GroundingDinoSinePositionEmbedding                 | :white_check_mark:       | :x:                          |
| GroundingDinoEncoderLayer                          | :white_check_mark:       | :x:                          |
| GroundingDinoEncoderBefore                         | :white_check_mark:       | :x:                          |
| Pass                                               | :white_check_mark:       | :white_check_mark:           |
| Mul                                                | :white_check_mark:       | :x:                          |
| RELU                                               | :white_check_mark:       | :x:                          |
| GroundingDinoDecoderLayer                          | :white_check_mark:       | :x:                          |
| GroundingDinoDecoderBefore                         | :white_check_mark:       | :x:                          |
| GroundingDinoForDetection                          | :white_check_mark:       | :x:                          |


## Run Log Description

***

This project uses google open source glog log library, you can set the following environment variables to print different levels of runtime information.
export GLOG_v=1 --> All thread information will be printed to the log.  
export GLOG_v=2 --> All memory allocation information will be printed to the log.  
export GLOG_v=3 --> All function code trace information will be printed to the log.  
export GLOG_v=4 --> All model loading parsing information will be printed to the logs.  

## Acknowledgement
***
Mariana_lmm refers to the following projects:
+ [pytorch](https://github.com/pytorch/pytorch)
+ [caffe](https://github.com/BVLC/caffe)
+ [transformers](https://github.com/huggingface/transformers)
+ [glog](https://github.com/google/glog)
+ [abseil-cpp](https://github.com/abseil/abseil-cpp)
+ [protobuf](https://github.com/protocolbuffers/protobuf)
+ [llama.cpp](https://github.com/ggerganov/llama.cpp)


***
https://github.com/user-attachments/assets/aad91973-979a-4590-ab9e-265ebcb60a58
