## Caffe Tip Tutorial

#### How to add layers?

For common usage, you will need just four or probably only three steps to add new layers:

Suppose your layer is `newLayer` and the corresponding parameter class is `newParameter`

1. add `new_layer.hpp` to `include/caffe/layers/`

2. add `new_layer.cpp` to `src/caffe/layers/` and never forget to add the following command to your cpp file to register it in the layer factory:

   ```c
   INSTANTIATE_CLASS(newLayer);
   REGISTER_LAYER_CLASS(new);
   ```

3. [optional]  add `new_layer.cu` to `src/caffe/layers` 

4. modify the `src/caffe/proto/caffe.proto`: define new message type `newParameter` and add its instance `new_param` in the message `LayerParameter`. Here, please pay attention to the comment of `LayerParameter` to get the available layer-specific id and update the comment when you add a new layer parameter instance. You need to provide the next available layer-specific id (you can get it by increasing the old id by one). 

Well, people are always seeking for higher performance and better understanding, so probably you are not satisfied with such a rough method mentioned above. If you want a more specific hands-on introduction and sharper insight, please visit https://github.com/BVLC/caffe/wiki/Development for an official tutorial.

#### Image Format

Caffe uses a **BGR color channel** scheme for reading image files. This is due to the underlying OpenCV implementation of `imread`. The assumption of RGB is a common mistake.

#### Makefile.config

Always remember to modify the `Makefile.config` to meet your project requirements. 

