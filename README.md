# Self-Supervised Domain Adaptation with Consistency Training
[![arXiv](https://img.shields.io/badge/arXiv-1409.7495-b31b1b.svg)](https://arxiv.org/abs/1409.7495)

Pytorch implementation for the paper ["Unsupervised Domain Adaptation by Backpropagation"](https://arxiv.org/abs/1409.7495).

```
@misc{ganin2015unsupervised,
      title={Unsupervised Domain Adaptation by Backpropagation}, 
      author={Yaroslav Ganin and Victor Lempitsky},
      year={2015},
      eprint={1409.7495},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```

# Usage

As module
```
from torch import nn
from gradient_reversal import GradientReversal

# define a network with Gradient Reversal Layer
net = nn.Sequential(
      nn.Linear(10, 10),
      nn.GradientReversal(alpha=1.)
)
```

As function
```
import torch
from gradient_reversal import revgrad

alpha = torch.tensor([1.])

x = torch.tensor([4.], requires_grad=True)
x_rev = torch.tensor([4.], requires_grad=True)

y = x*5
y = y+6

y_rev = x_rev*5
y_rev = revgrad(y_rev, alpha)
y_rev = y_rev+6

y.backward()
y_rev.backward()

print(f'x gradient: {x.grad}')
print(f'reversed x gradient: {x_rev.grad}')

assert x.grad==-x_rev.grad
```