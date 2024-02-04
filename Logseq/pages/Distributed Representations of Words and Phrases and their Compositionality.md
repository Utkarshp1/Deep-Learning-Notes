- ![1310.4546.pdf](../assets/1310.4546_1696695088705_0.pdf)
- **Basic Idea**:
	- In the earlier Word2Vec paper, the computational complexity of a NNLM was reduced by getting rid of the hidden layer. Can we further reduce this? (Hint: Replace hierarchical softmax with Noise Contrastive Estimation)
	- How do we improve learning by balancing frequently occurring words? Or learn better representations for rarely occurring words?
	- How do we also get representations for phrases like "Air Canada" (an airline company) which has completely different meaning from words "Air" and "Canada"? How do you statistically identify relevant phrases which represent all together different meaning from the constituent words?
	- Additional basic vector operations on word vectors such as `vec("Russia") + vec("River")` should be close to `vec("Volga River")`.
	-
- **Skip-gram Model**:
	- ((651ef207-9db4-4ce8-b284-ce26c1ccb25d))
	- There are two kinds of word representations that are being learnt corresponding to the projection matrix $P1$ and $P2$ (a.k.a. as "input" ($v_{w}$) and "output" ($v'_{w}$) representation). The final word embedding can be either of two, or even the sum/average of the two embeddings.
	- Given a sequence of training words $w_{1}, w_{2}, ..., w_{T}$, the training objective to maximize is given by:
	  id:: 652192b0-140e-451d-95e7-00e6c9b8b4f5
	  \begin{equation}
	  \frac{1}{T}\sum_{t=1}^{T}\sum_{-c \leq j \leq c, j \neq 0} log p(w_{t+j} | w_{t})
	  \end{equation}
	  Note that $c$ is the context window length and higher value leads to higher accuracy due to more training examples but at the cost of increased training time.
	- The probability $p(w_{t+j} | w_{t})$ is modelled using softmax function as follows:
	  \begin{equation}
	  p(w_{O} | w_{I}) = \frac{exp(v'^{T}_{w_{O}} v_{w_{I}})}{\sum_{w=1}^{W} exp(v'^{T} _{w} v_{w_{I}})}
	  \end{equation}
	  where $W$ is the number of words in the vocabulary.
	  Note that the softmax is defined using the cosine similarity between the input and output words because of the [Distributional hypothesis](logseq://graph/Logseq?block-id=65171585-d268-4984-8c18-5197bf423a28) and also if you do the maths of projection matrices, neural network and stuff it works out the same formula.
	- Since, the denominator sums over all the words in the vocabulary (typically $W=10^5-10^7$), it is computationally expensive to calculate $p(w_{t+j} | w_{t})$. Therefore, in order to overcome this, following solutions are used:
		- **Hierarchical Softmax**:
			- Introduced in [Morin and Bengio, 2005](https://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf)
			- ((65219ab9-dbf5-463b-ad7d-7d7e1e8ded33))
			  ((65219ac8-b56c-4329-b9d7-6137dba6a351))
			  ((65219b3e-08de-4908-bd7b-85fa73bbe9fa))
			  Note:
			  * $p(w | w_{I})$ represents the probability of a random walker reaching a leaf node corresponding to word $w$ starting from the root node.
			  * The above equation is based on the fact that:
			  \begin{equation}
			  1 - sigmoid (x) = sigmoid(-x)
			  \end{equation}
			  ((65219c30-866a-4c8a-b82a-3b8f33b2b050))
		-
		- **Negative Sampling or Noise Contrastive Estimation (NCE)**
			- Idea is similar to [FaceNet](logseq://graph/Logseq?page=FaceNet%3A%20A%20Unified%20Embedding%20for%20Face%20Recognition%20and%20Clustering), Contrastive Learning etc.
			- Therefore, the [original objective](logseq://graph/Logseq?block-id=652192b0-140e-451d-95e7-00e6c9b8b4f5) of the Skip-gram model is replaced by:
			  \begin{equation}
			  \max \left [\underbrace{log \sigma (v'^{T}_{w_{O}}v_{w_{I}})}_\text{Positive example probability} + \underbrace{ \sum_{i=1}^{k} \mathbb{E}_{w_{i} \sim P_{n}(w)} \left [ log \sigma (-v'^{T}_{w_{i}} v_{w_{I}}) \right ]}_\text{Negative example probability} \right ]
			  \end{equation}
			  Note for the calculation of negative example probability again the fact $1 - \sigma (x) = \sigma(-x)$ is used.
		- The intuition is to contrast every positive target word against $k$ negative words. The negative words are chosen according to the distribution $P_{n}(w)$ defined as follows:
		  \begin{equation}
		  U(w)^{3/4}/Z
		  \end{equation}
		  where $U(w)$ is the unigram distribution or the frequency of all the unigrams that appear in your training data.
		- ((6521a28a-23ba-4a8f-9668-5321295aaa76))
-
- **Subsampling of Frequent Words**
	- ((6521b11f-0f35-48ed-b797-c414dcc33c2d))
	  ((6521b133-8c36-4681-8014-fa1438dc33c1))
	- **Intuition for the above formula**:
		- $\sqrt{\frac{t}{f(w_{i})}}$ is the probability that the word $w_{i}$ is presented to the network while training. Intuitively, this probability should be inversely proportional to the frequency $f(w_{i})$. Moreover, we take the square root because the typical vocabulary size used is $10^5-10^7$, therefore, the if we just take simple inverse without square the root, the probability of selection of rare word would be quite high and for a frequent word would be very low. So, the square root kind of shrinks the domain of possible probability values. However, since the frequent word will still be available a lot number of time, hence, there is a threshold $t$. Think of this threshold as $\frac{1}{\frac{f(w_{i})}{t}}$, i.e., kind of normalizing the frequency.
-
- **Learning Phrases**
	- How do you statistically identify relevant phrases which represent all together different meaning from the constituent words?
	  ((6521b5f0-8f26-4f11-bd50-014ca84c43d1))
	  "New York Times" or "Toronto Maple Leafs" will appear frequently together but infrequently in other usage of "New York" or "times". However, throughout the training set, there is a high possibility that "this is" will always occur together. Thereby, "this is" is not treated as a separate token.
	- The authors use a simple data-driven approach for forming phrases using unigram and bigram counts using
	  \begin{equation}
	  \text{score}(w_{i}, w_{j}) = \frac{count(w_{i}w_{j}) - \delta}{count(w_{i}) \times count(w_{j})}
	  \end{equation}
	- **Intuition for the above formula**:
		- The above formula is based on the naive definition of probability i.e.,
		  \begin{equation}
		  \mathbb{P}(\text{event}) = \frac{\text{Number of favourable outcomes}}{\text{Total number of possible outcomes}}
		  \end{equation}
		- $count(w_{i}w_{j})$ represents the number of times the words $w_{i}$ and $w_{j}$ appear together in the training set (favourable event).
		- $count(w_{i}) \times count(w_{j})$ represent the number of times the words $w_{i}$ and $w_{j}$ could have appeared together if they were to be distributed independently assuming each occurrence of the word as separate/distinct (total number of possible outcomes).
		- Therefore, $\text{score}(w_{i}, w_{j})$ just represents the probability of the words $w_{i}$, $w_{j}$ occurring together.
		- $\delta$ is a discounting coefficient and prevents too many phrases consisting of very infrequent words to be formed.
	- ((6521ba4e-e5c2-418e-a8de-8ec8cb3b3a23))
-
- **Explanation for Additive Structure of Word Vectors**
	- ((6521bac1-fe2c-4831-b368-1370085c41bc))
-
- **Important hyperparameters for Skip-gram model**
  * Choice of model architecture
  * Size (dimension) of the word vector
  * Subsampling rate
  * Size of the training window ($c$)
-