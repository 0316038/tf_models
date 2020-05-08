#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script is used to compile the code for correlation_cost and create correlation_cost.so.
#
#  Usage:
#    sh ./compile.sh cuda_dir
#  Where cuda_dir points to a directory containing the cuda folder (not the cuda folder itself).
#
#

if [ "$#" -ne 1 ]; then
  echo "Illegal number of parameters, usage: ./compile.sh cuda_dir"
  exit 1
fi
CUDA_DIR=$1

if [ ! -d "${CUDA_DIR}/cuda" ]; then
  echo "cuda_dir must point to a directory containing the cuda folder, not to the cuda folder itself"
  exit 1
fi

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
CUB_DIR=cub
THRUST_DIR=thrust

# Depending on the versions of your nvcc and gcc, the flag --expt-relaxed-constexpr might be required or should be removed.
# If nvcc complains about a too new gcc version, you can point it to another gcc
# version by using something like nvcc -ccbin /path/to/your/gcc6
nvcc -std=c++11 --expt-relaxed-constexpr -I ./ -I ${CUB_DIR}/../ -I ${THRUST_DIR} -I ${CUDA_DIR}/ -c -o correlation_cost_op_gpu.o kernels/correlation_cost_op_gpu.cu.cc -I $TF_INC/ -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -D_GLIBCXX_USE_CXX11_ABI=0

g++ -std=c++11 -I ./ -L ${CUDA_DIR}/cuda/lib64 -shared -o correlation_cost.so ops/correlation_cost_op.cc kernels/correlation_cost_op.cc correlation_cost_op_gpu.o -I $TF_INC/ -fPIC -lcudart -L$TF_LIB -l:libtensorflow_framework.so.2 -D GOOGLE_CUDA=1 -D_GLIBCXX_USE_CXX11_ABI=0
