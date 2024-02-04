- ![WenECCV16.pdf](../assets/WenECCV16_1690827634972_0.pdf)
- **Basic Idea**:
	- The Triplet Loss is based on sample-to-sample comparisons and hence, quickly becomes prohibitive with the increase in the size of the dataset. Moreover, the triplets needs to be carefully chosen (neither too hard nor too easy).
	- Use softmax based classifer to learn good features which can even distinguish unseen faces.
	- The learned features should be such that inter-class separation is maximised whereas intra-class variability is minimised.
-
- **Separable vs Discriminative Features**:
- #+BEGIN_CENTER
  ((64c7fea2-3483-4264-a14d-c6e36007eb86))
  #+END_CENTER
- ((64c7ff26-6924-426a-a59f-b12e2b2d3708))
- ((64c7ff42-93a2-4a6b-b3ba-c335cb6c5b1f))
- ((64c7ff8e-a396-4142-ac77-76013ce02bfb))
-
- **Proposed Approach**:
	- Learn a class center for each class (a vector of the same dimension as the embedding/feature dimension)
	- While training, simultaneously update the centers and minimise the distance between the deep features and their corresponding class centers.
	- Train with both softmax loss and center loss (balanced using a hyper-parameter): Softmax loss ensures features of the different classes stay apart.
	- Note that similar objective can also be achieved using a margin (which is used in later papers). Also, margin is used in Triplet Loss to ensure that discriminative power of the learned features. (**TRY**: Is the Triplet Loss or Contrastive Loss successful without a margin?)
	-
	- **Center Loss**:
		- Motivation: Minimise the intra-class variations while keeping the features of different classes separable.
		- \begin{equation}
		  \mathcal{L}_{C} = \frac{1}{2} \sum_{i=1}^{m} || x_{i} - c_{y_{i}}||^{2}_{2}
		  \end{equation}
		  where $c_{y_{i}} \in \mathbb{R}^d$ denotes the $y_{i}$th class center of deep features.
		- Notice the similarity of training procedure/loss with *K-means clustering*. (**THINK**: Something on the line of *Expectation Minimization Algorithm*).
		- ((64c80859-649c-4011-8cf6-7deeca099d8e)) $c_{y_{i}}$ ((64c8087b-8bcc-4da3-a83c-b991ce0085e6))
		- To address the above problem, they introduce two modifications:
			- ((64c8092c-1c62-4d2a-9e50-cb3c557244df))
			- **Robustness to Noise Labels**: ((64c80939-d8a1-4878-a245-7f57feeb7d1a))
		- The gradients are computed as follows:
		  \begin{equation}
		  \frac{\partial{\mathcal{L}_{C}}}{\partial{\textbf{x}_{i}}} = \textbf{x}_{i} - c_{y_{i}}
		  \end{equation}
		  **NOTE:** Since in the present iteration, $c_{y_{i}}$ is a constant, i.e., it's value will not be influenced by $\textbf{x}_{i}$, hence, $\frac{\partial{\mathcal{L}_{C}}}{\partial{\textbf{c}_{y_{i}}}} = 0$.
		  
		  \begin{equation}
		  \Delta c_{j} = \frac{ \sum_{i=1}^{m} \delta(y_{i} = j) \cdot (c_{j} - x_{i})} {1 + \sum_{i=1}^{m} \delta(y_{i} = j)} 
		  \end{equation}
		  **NOTE:** 
		  * where $\delta(condition)=1$ if the $condition$ is satisfied, and $\delta(condition)=0$ if not. $\alpha$ is restricted in [0, 1].
		  * 1 is added in the denominator to ensure that if no example belongs to class $j$, then the update $\Delta c_{j} = 0$.
		  * The update rule is similar to an incremental update rule for calculating average of a quantity:
		  \begin{equation}
		  \bar{x}_{n+1} = \bar{x}_{n} - \alpha (\bar{x}_{n} - x_{n}) \\
		  NewEstimate \leftarrow OldEstimate+ StepSize \left [ Target - OldEstimate \right ]
		  \end{equation}
		  where $\alpha$ is the learning rate for calculating the average. For exact, average calculation use $\alpha = \frac{1}{n}$. For proof, refer to *Sutton and Barto, Reinforcement Learning, 2nd Edition Pg 31*.
		-
		- Final loss function is given by:
		  \begin{aligned}
		  \mathcal{L} &= \mathcal{L}_{S} + \lambda \mathcal{L}_{C} \\
		   &= - \sum_{i=1}^{m} log \frac {e^{W^{T}_{y_{i}}x_{i} + b_{y_{i}}}}{\sum_{j=1}^{n} e^{W^{T}_{j}x_{i} + b_{j}}} + \frac{\lambda}{2} \sum_{i=1}^{m} || x_{i} - c_{y_{i}}||^{2}_{2}
		  \end{aligned}
		- ((64c811db-8a51-4230-a97f-6260423ea872))
		- **DISCUSSION**:
			- **The necessity of joint supervision.** ((64d4638c-941f-4cbb-8584-d20ee409c776))
	- **Tricks of the Trade**:
		- Use of PReLU activation function.
		- For score computation during testing, extract features from each image and its horizontally flipped one, and concatenate them as the representation. The score is computed by the Cosine Distance of two features after PCA.