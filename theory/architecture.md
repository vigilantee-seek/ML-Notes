## Network Architecture

**This chapter collects some tricky design in various types of neural networks.**

#### Skip Connection

It is first raised in ResNet as a . 

#### Connections between ResNet, DenseNet and Higher Order RNN

> Residual networks essentially belong to the family of densely connected networks except that their connections are shared across steps.

***Prove***: (The following prove is provided by *Dual Path Networks* by *Pengcheng Yun*)

We use $$h^t$$ to denote the hidden state of the recurrent neural network at the $$t-th$$ step and use $$k$$ as the index of the current step. Let $$x^t$$ denotes the input at $$t-th$$ step, $$h^0 = x^0$$. For each step, $$f^k_t(·)$$ refers to the feature extracting function which takes the hidden state as input and outputs the extracted information. The $$g^k(·)$$ denotes a transformation function that transforms the gathered information to current hidden state: 
$$
h^k = g^k[\sum_{t=0}^{k-1}f_t^k(h^t)]
$$
For HORNNs, weights are shared across steps, $$i.e. \forall t,k,f^k_t(·) ≡ f_t(·)$$ and $$\forall k, g^k(·) ≡ g(·)$$. For the densely connected networks, each step (micro-block) has its own parameter, which means $$f^k_t (·)$$ and $$g^k(·)$$ are not shared.
$$
r^k \triangleq \sum_{t=1}^{k-1}f_t(h^t) = r^{k-1} + f^{k-1}(h^{k-1})	\\
h^k = g^k(r^k)	\\
\Longrightarrow r^k = r^{k-1} + f_{k-1}(h^{k-1}) = r^{k-1}+f^{k-1}(g^{k-1}(r^{k-1})) = r^{k-1} + \phi^{k-1}(r^{k-1})
$$
Speciﬁcally, when $$\forall k, \phi^k(·) = \phi(·)$$, the above equation degenerates to an RNN; when none of $$\phi ^k(·)$$ is shared and $$x^k = 0,k > 1$$, it produces a residual network. 

