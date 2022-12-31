Building core Deep Learning algorithms in Rust.

It's kinda like the big brother version of the [karpathy/micrograd](https://github.com/karpathy/micrograd).
Some key differences is that we support higher order gradients, tensors (through the use of the ndarray crate), and convolutional layers.

-------------------------------------------------------

### Contributing

Any type of contribution is welcome as long as it adds value! i.e
* Bug fixes followed with tests to ensure the bug never resurfaces
* Increasing code readability, or run-time/memory efficiency
* Completing a To-Do task

-------------------------------------------------------

### To-Do

* Weight Initializers
    * [~~Glorot~~](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotNormal)
    * [He](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal)
    * [Lecun](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/LecunNormal)
* [Create optimizer module for optimization algorithms](https://pytorch.org/docs/stable/optim.html)
* Convolutional Neural Networks
    * [~~Convolutional Module~~](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
    * Pooling Module
        * [~~Max Pooling~~](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
        * [~~Average Pooling~~](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html#torch.nn.AvgPool2d)
    * [Flatten Module](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html#torch.nn.Flatten)
    * Padding Support
    * Dilation Support

