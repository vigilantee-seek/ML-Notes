## Issues and Solutions

#### Issue 1: fatal error: caffe/proto/caffe.pb.h: No such file or directory

This usually happens when you add new layers to the caffe frastructure. You can solve it by running:

```shell
protoc src/caffe/proto/caffe.proto --cpp_out=.
mkdir -p include/caffe/proto
mv src/caffe/proto/caffe.pb.h include/caffe/proto
```

These commands are encapsulated in `pbhSolution.sh`. You should run it under the `$CAFFE_ROOT_DIR` .

#### Issue 2: Unknown layer type: Python

Are you sure you have modified the configuration file `Makefile.config` under the `$CAFFE_ROOT_DIR` ? Maybe you just disabled the python layer in your caffe (And that's the default configuration in the example file). You can do as follows to restart it:

```shell
# Uncomment to support layers written in Python (will link against Python libs)
WITH_PYTHON_LAYER := 1
```

#### Issue 3: NVCC warning

This is because you enabled out-of-date CUDA architecture, just remove the first two lines of the following command:

```shell
-gencode arch=compute_20,code=20	\
-gencode arch=compute_20,code=21
```

The current command should be like:

```shell
# CUDA architecture setting: going with all of them.
# For CUDA < 6.0, comment the *_50 lines for compatibility.
CUDA_ARCH := -gencode arch=compute_30,code=sm_30 \
                 -gencode arch=compute_35,code=sm_35 \
                 -gencode arch=compute_50,code=sm_50 \
                 -gencode arch=compute_50,code=compute_50
```

