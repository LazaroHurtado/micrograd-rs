Building core Deep Learning algorithms in Rust.

It's kinda like the middle child of [karpathy/micrograd](https://github.com/karpathy/micrograd) and [geohot/tinygrad](https://github.com/geohot/tinygrad).

---

### Contributing

Any type of contribution is welcome as long as it adds value! i.e

- Bug fixes followed with tests to ensure the bug never resurfaces
- Increasing code readability, or run-time/memory efficiency
- Completing a To-Do task

---

### To-Do

- Weight Initializers
  - [~~Glorot~~](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotNormal)
  - [He](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal)
  - [Lecun](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/LecunNormal)
- [~~Create optimizer module for optimization algorithms~~](https://pytorch.org/docs/stable/optim.html)
  - [~~SGD~~](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD)
  - [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam)
  - [~~RMSProp~~](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop)
- Convolutional Neural Networks
  - [~~Convolutional Module~~](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
  - Pooling Module
    - [~~Max Pooling~~](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
    - [~~Average Pooling~~](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html#torch.nn.AvgPool2d)
  - ~~Padding Support~~
  - Dilation Support
