## Tips

### tf.rank()

*I encountered this interesting problem when I was practicing a hands-on tensorflow project.*

**The rank of a tensor is not the same as the rank of a matrix.** The rank of a tensor is the number of indices required to uniquely select each element of the tensor. Rank is also known as "order", "degree", or "ndims." 

If you type:

```python
import tensorflow as tf
X = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
# Explicitly, the rank of the matrix X should be 3 but that's not the case for the tensor X.
with tf.Session() as sess:
    Xrank = sess.run(tf.rank(X))
    print("The rank of tensor X is {rank}.".format(rank=Xrank))
```

and we will get:

```shell
The rank of tensor X is 2.
```

It's especially important for the API `tf.argmax()` which you can refer to at [here](https://tensorflow.google.cn/api_docs/python/tf/argmax). For the parameter `axis`:

> **axis**: A `Tensor`. Must be one of the following types: `int32`, `int64`. int32 or int64, must be in the range `[-tf.rank(input), tf.rank(input))`. Describes which axis of the input Tensor to reduce across. For vectors, use axis = 0.

0 - reduce by column, 1 - reduce by row, 2 - reduce by the third dimension, et cetera.

