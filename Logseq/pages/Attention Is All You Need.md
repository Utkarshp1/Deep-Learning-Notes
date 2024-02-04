- ![Attention Is All You Need](../assets/1706.03762_1684517468103_0.pdf)
- **Basic Idea**
  * Can we parallelize the training of encoder-decoder models for sequence transduction tasks, because RNN based models are sequential in processing the tokens of the sequences?
  * Can we reduce the number of operations required to relate any two arbitrary tokens in the input or output sequence? In RNNs, it is $O$(distance between the two tokens). The memory vector in the LSTM or GRU is fixed dimensional. So, if the sequence length increases or dependencies length increases, then the same problem that is faced in encoder-decoder model of fixed length content history vector not able to capture everything, will become evident and hence, we need self-attention.
-
- **Model Architecture**
  * **Notations**
      * $(x_{1}, ..., x_{n})$ : input sequence token
      * $(e_{1}, ..., e_{n})$ : embedding of the input sequence (includes normal embedding + positional embedding)
      * $(z_{1}, ..., z_{n})$ : output of encoder
      * $(h_{1}, ..., h_{n})$ : hidden state of the decoder
      * $(y_{1}, ..., y_{m})$ : output sequence
      * $(g_{1}, ..., g_{m})$ : embedding of the output sequence (includes normal embedding + positional embedding)
  * **Architecture**
  #+BEGIN_CENTER
  ((6467b62f-c46c-4f61-8430-a4fb1a965385))
  #+END_CENTER 
      **Encoder**
      ((6467b6ba-3db5-4d2b-8fa1-cf8178e7dce8))
       
       **Decoder**
       ((6467b6e3-9f3d-4bb9-a10e-70ed467963f0))
  
       The masking is important in decoder because while generating sequence we cannot look into the future. ((6467c1c0-ca65-463b-9676-ad9d65f9b18b)) Also, because the model is an autoregressive model i.e., the last predicted token is fed into the model again, the authors shift the output sequence by 1.
  
  * **Scaled-Dot-Product Attention**
     Let's consider the case when the attention is between the encoder output and decoder. In this case:
      * **Query** $Q \in \mathbb{R}^{d_{k}}$: This is some function of $h$. For example, in ConvS2S, it is $Wh_{i} + b + g_{i}$
      * **Key** $K \in \mathbb{R}^{d_{k}}$: This is some function of $z$. For example, in ConvS2S, it is $z_{j}$.
      * **Value** $V \in \mathbb{R}^{d_{v}}$: This is some function of $z$. For example, in ConvS2S, it is $z_{j} + e_{j}$.
      * **Output** $O$: The output vector which is the weighted sum of the values.
     The general equation of attention from ConvS2S can be written as:
  
  #+BEGIN_EXPORT latex
  softmax(align(z_{j}, (Wh_{i} + b + g_{i}))(z_{j} + e_{j})
  #+END_EXPORT
           where align is some function used to align the keys to values. This can be:
              * **Additive** : When a MLP is used to align. (Bahdanau et al. Attention)
              * **Dot-Product** : When dot product is used to align.
  ((6467bf02-3d3b-40b4-be7b-c924cb369591))
           For more variations to this align function, visit [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025).
  
         This is exactly the same attention formula that is used in Transformers (except for the scale factor $\sqrt{d_{k}}$):
  
  #+BEGIN_EXPORT latex
  softmax \left ( \frac{QK^T}{\sqrt{d_{k}}} \right ) V
  #+END_EXPORT
  
  ((6467bf17-fa6b-446b-957a-4395f2842f08))
  ((6467bfc3-e6f2-4ebf-ab37-70928e20a074))
  
  #+BEGIN_CENTER
  ((6467bf80-ee10-4bdb-be12-8b388bea8f6f))
  #+END_CENTER
  
  * **Multi-Head Attention**
  Basically, the same computation as above but the input is transformed (projected) by different projection matrices. Intuition: ((6467c183-bfb6-4abe-8a75-019afae4913a))
  
  
  #+BEGIN_CENTER
  ((6467c209-d89e-41e7-8962-51ab1b25d198))
  #+END_CENTER
  
  * **Positional Embedding**
  ((6467c29c-ed86-498e-830a-ecdf51b9e97b))
	- **Position-wise Feed-Forward Networks**
	- ((65127b76-e983-4618-87ee-17c138d30b66))
	-
- **Self-Attention**
  ((6467c2dd-4c64-4ba1-ac29-ebdfac8b3383))
  The computations of the convolutional versions can be improved by using atrous (dilated) convolutions or separable convolutions. The computations for the convolution model is based on a stride of $k$. The computations of the self-attention, can be done by using Eq. 1 of the paper assuming $Q, K, V \in \mathbb{R}^{n \times d}$.
  ((6467c410-d2b5-41a3-84df-b60972a9e6d1))
  ((6467c3f6-d653-4069-85a7-4c1ab9253bf7))