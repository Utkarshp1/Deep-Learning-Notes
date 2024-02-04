- ![taigman_cvpr14.pdf](../assets/taigman_cvpr14_1689015333556_0.pdf)
- **Basic Idea**
	- ((64ac5496-e2e6-486b-9267-0f31a186025e))
	- This paper tries to better align the images of faces using 3D face modelling in order to apply piece-wise affine transformation.
	- Also, this paper tries to learn better representations for the aligned face images.
- **Face Alignment**
	- Revisit this portion after reading the camera and stuff.
	-
- **Representation**
	- Train a multi-class face recognition task (classify each face image).
	- **Tricks of the trade**:
		- *The classic problem of retaining the dense spatial information throughout the network which often gets lost in the classification networks due to max-pooling operation.*
		  id:: 64ac5696-92e1-4a24-b17b-c787d1dc3a0c
		  **Solution**: ((64ac56ad-b6fe-4e80-ab55-a1ed26003bf2))
		-
		- *Locally connected Convolutional Features to learn different discriminative features for different spatial location*.
		  **Solution**: ((64ac57bf-68b0-4c54-bb63-fd6fcd7c3732))
		  **NOTE:** Increasing the number of filters will do the same job because each filter will kind of learn different discriminative feature and will fire up at different locations. The only downside of this approach is that each filter will be ran over the entire image (and hence compute intensive) whereas locally connected convolutional filter will be ran over a local patch in the image (less compute intensive).
		-
		- *Making the representations learnt invariant to illumination changes*
		  **Solution**: ((64ac5967-3016-4568-ba1f-c1dab626aca5)) $f(I) := \bar{G}(I)/||\bar{G}(I)||_{2}$ where $\bar{G}(I)_{i} = G(I)_{i}/max(G_{i}, \epsilon), \epsilon=0.05$ in order to avoid division by a small number.
		  ((64ac5c59-11ec-499a-9a46-49b6bf9eae21)) ((64ac5c66-6151-4d68-b5b7-1a9866ed810c))
		  This is because it can happen that if we scale the intensities, it may go the negative region of ReLU (which was originally in the positive region) and vice-versa. Ex. Image intensity at a location is 127,  weight and bias are 1 and -126 respectively. $WI+b = 127*1 - 126 = 1 > 0$. Now, if we scale image intensity to 64, $WI+b = 64*1-126 = -62 < 0$. The value of ReLU will be different for both of them. 
		  Now, if b=0, then $\alpha Wx$ and $Wx$ will always be of the same sign.
		  **NOTE:** Modern networks alleviate this problem by normalizing the input by diving by the mean and standard deviation across the whole dataset (can also do sample-wise normalization to make it illumination invariant).
		-
	- **Properties of the representations learnt**:
	  id:: 64ac5c8f-013c-4ca5-aa9e-86cf5a61d291
		- It contains non-negative values (Non-negative as in 0 because they apply ReLU).
		- It is very sparse. ((64ac5e63-70ba-4bf7-abb2-d38a4bada45d))
		- Its value are between [0, 1]
		-
- **Metric Learning**
	- **Inner Product**: Taking the inner product between the two normalized feature representations (two query images).
	- **Weighted** $\chi^{2}$ **distance**: Because of the properties mentioned in ((64ac5c8f-013c-4ca5-aa9e-86cf5a61d291)) they use:
	  \begin{equation}
	  \chi^{2}(f_{1}, f_{2}) = \sum_{i} w_{i}\frac{(f_{1}[i] - f_{2}[i])^{2}}{f_{1}[i] + f_{2}[i]}
	  \end{equation}
	  where $f_{1}$ and $f_{2}$ are DeepFace representations. The weight parameters are learned using a liner SVM, applied to vectors of the $(f_{1}[i] - f_{2}[i])^{2}/{f_{1}[i] + f_{2}[i]}$.
	-
	- **Siamese Network**: End-to-end metric learning system to identify whether the features belong to the same person. ((64ac6064-4c82-46e9-afd0-e94a2304f5a5)) The Siamese network's induced distance is:
	  \begin{equation}
	  d(f_{1}, f_{2}) = \sum_{i} \alpha_{i} |f_{1}[i] - f_{2}[i]|
	  \end{equation}
	  where $\alpha_{i}$ are the trainable parameters trained using standard cross-entropy loss and back-propagation of the error. Note that this induced distance is because they are taking the difference between the feature representations.
	-
- **Network with Best Results**
	- Ensemble of 3 networks trained with different inputs:
		- 3D aligned RGB images
		- The gray-level images plus image gradient magnitude and orientation
		- 2D aligned RGB images
	- ((64ac62bb-5430-4026-bbaa-af2ba3e0f52a))
	-