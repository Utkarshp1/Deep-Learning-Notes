- ![1708.02002.pdf](../assets/1708.02002_1703598002439_0.pdf)
- **Basic Idea**:
	- Since single-stage detectors do not use region proposals (they do a dense prediction i.e. at each spatial location predict boxes of various shapes, sizes and aspect ratio etc.), there is a high class imbalance between positive (with object) and negative class. Can we alleviate this class imbalance problem by dynamically weighing hard samples (both positive and negative)?
	- Due to this class imbalance the training initially is unstable. Can we use better initialisation technique to overcome this class imbalance?
-
- **NOTE:** There are two concepts overlapped here. Dominant (Negative) vs Rare (Positive) class. Hard vs Easy examples. The assumption is that since negative class is dominant, hence can be easily classified. Therefore, as a proxy for solving class imbalance problem, the authors discount the easily classified examples. Therefore, it can happen that even hard-negatives are discounted less. However, in practice this rarely happens as validated by the following experiment.
  ((658b0403-518e-41d4-91c2-c29be2417c45))
-
- **Why class imbalance is a problem?**
	- ((658adb37-f26d-45dc-a244-5a1f4a70eb5e))
	- ((658adb65-ec1f-48d2-9d53-d3e6862a52c6)) Degenerate model as in which always predicts negative class.
-
- **Earlier attempts to combat class-imbalance**
	- In two-stage detectors, region proposal significantly filters regions with background. Thereafter, while selecting batches sampling heuristics are applied such as ((658ad9c3-76e0-4f37-8ab7-e54afada92ac))
	- ((658ad9e6-2855-4ec8-9ffc-9055ed0898e5))
	- *Balanced cross-entropy loss*: [Used in YOLO](logseq://graph/Logseq?block-id=65872a91-e6e2-45c1-a53c-883fcb26743a).
-
- **Focal Loss**
	- #+BEGIN_CENTER
	  ((658b009b-90e6-4c6b-8dae-a709490d9ee3))
	  #+END_CENTER
	- **Cross-Entropy(CE)** loss for binary classification:
	  \begin{equation}
	  CE(p, y) = 
	  \begin{cases}
	  -log(p) & \text{if } y = 1 \\
	  -log(1-p) & \text{otherwise}
	  \end{cases}
	  \end{equation}
	  where $y \in \{\pm1\}$ is the ground-truth label and $p$ is the predicted probability for class with label $y=1$. Define $p_{t}$ as 
	  \begin{equation}
	  p_{t} =
	  \begin{cases}
	  p & \text{if } y = 1 \\
	  1 - p & \text{otherwise}
	  \end{cases}
	  \end{equation}
	  Therefore, p_{t} represents the predicted probability for the ground-truth label class i.e. the correct class.
	  ((658b0421-b9cc-48b6-a13b-5673922c3d0f))
	- **Balanced Cross-Entropy** [(Used in YOLO)](logseq://graph/Logseq?block-id=65872a91-e6e2-45c1-a53c-883fcb26743a):
	  \begin{equation}
	  CE(p_{t}) = -\alpha_{t}log(p_{t})
	  \end{equation}
	  where $\alpha_{t}$ is a weighing term defined similar to $p_{t}$. The objective of this is to weigh rare class examples higher than dominant class (has the similar effect of over-sampling rare class or maintaining foreground to background ratio in the samples).
	  ((658b065e-5308-47e1-ba96-b5d334035457))
	- **Focal Loss**
	  \begin{equation}
	  FL(p_{t}) = -(1-p_{t})^{\gamma} log(p_{t})
	  \end{equation}
	  where $\gamma \ge 0$ is a tunable focussing parameter.
	- Properties of Focal loss:
		- ((658b071e-3b54-4b1a-b367-5e7b3b27666d))
		  An example can only be misclassified when $p_{t}$ is small.
		- ((658b0768-2f2c-4f43-a1b8-2e3f939653d6))
		- ((658b0864-32e1-44c6-b6de-a6e312996e35))
		  The last one is true because $\gamma \le \left ( 1- \frac{1}{2}\right)^{2} = \frac{1}{4}$.
		- Note that for both hard as well as easy examples, the loss is getting down weighed. It's just that for hard examples it is less and for easy it is more.
		- As $\gamma$ is increased the effect of down-weighing is more pronounced especially for the misclassified samples.
		- ((658b13a6-6ee2-4bd5-a3df-d52c20160e96))
		- ((658b13b4-b94e-4493-9052-303e69dab50b))
		-
	- $\alpha$-balanced version of focal loss works better in practice:
	  \begin{equation}
	  FL(p_{t}) = - \alpha_{t} (1-p_{t})^{\gamma} log(p_{t})
	  \end{equation}
	- ((658b09d2-d587-4d44-8cf1-dfda205f00fc))
	- ((658b13d7-f7cb-4634-b6df-2f9fa84856b8)) ((658b1415-7903-4d53-989e-8f5da45680d7))
	  $\alpha$ gives more emphasis to the foreground (dominant/hard class). So, when $\gamma$ is increased we are already giving more emphasis to hard examples (dominated by foreground class), hence if we keep high values of $\alpha$ then training may only focus on hard examples which is also not something we would want.
	-
- **Class Imbalance and Model Initialization**
	- ((658b0f15-df09-45cd-be24-618cb9f37ade))
	- ((658b0f2b-4d11-41ce-ad72-00b41875c983))
	- This helps the model to initially predict the frequent class.
	- ((658b0f78-e435-4f6b-8e19-284b3bf7b73a)) ((658b0f89-5317-431f-9602-7e0b95c37385))
	  **Intuition**: 
	  \begin{aligned}
	  \frac{1}{1+e^{-z}} &= \pi  \implies z &= - log \left ( \frac{1-\pi}{\pi} \right )
	  \end{aligned}
	  Now, since $\pi$ represents the probability of the rare class, which we want to be low. Therefore, we want $z = w^Tx + b$ to be low. Assume, $x \sim \mathcal{N}(0, \sigma)$ and $\mathbb{E}[w^Tx] = 0$. Therefore, b can be set as $log \left ( \frac{1-\pi}{\pi} \right )$.
	- ((658b1216-65dc-49b3-8e5b-86cbde8d384a))
-
- **Tricks of the trade**:
	- ((658b12c3-fdac-42bc-85a8-10ced5fe469b))
	  This is similar to YOLO and YOLOv2. Class-specific bounding box regressor would be for each class you have separate bounding box regressor. Note that this is different from YOLOv2, which had separate classification head for each anchor box (there is no class-specific bounding box regressor).
	- ((658b1551-ae16-4c1b-a94f-f979a0c28a4b))
	  \begin{equation}
	  smooth_{L1}(x) =
	  \begin{cases}
	  0.5x^2 & \text{if } |x| < 1 \\
	  |x| - 0.5 & \text{otherwise}
	  \end{cases}
	  \end{equation}
	  The above smooth L1 loss is less sensitive to outliers. This is same as the Huber Loss.
-
- **Results**
	- ((658b16a6-dccd-4c9b-9dc3-55af8e5f9741)) The conclusion of this experiment is that for the positive examples the loss is mostly dominated by the top 20% of the hard examples, whereas for the negative examples, the loss is mostly 0 for almost all of the examples except for extremely hard negative examples (indicated by the sharp peak in the graph at 1). Moreover, CDF for negative examples show that for $\gamma = 0$ negative examples get assigned high loss values and therefore indicates the efficacy of focal loss.
	- ((658b1827-03cc-4a42-aeb9-f2f8eeb5ae4f))
-
- **Relation to other losses**
	- *Robust Estimation*: Losses such as Huber loss ((658b1c6f-3e07-4d6c-876a-e89c05f898f0))
	- *Hinge Loss*: ((658b1ca1-efe3-4bdb-b274-1bf42017a084))
	  Look at [this](logseq://graph/Logseq?block-id=64adb328-e769-41a4-ad1d-24858e6722a8) to understand how hinge loss helps in focussing training on hard examples. The reason why I think focal loss works better than hinge loss is that hinge completely ignores easy examples just like OHEM whereas focal loss still takes them into account.
	-