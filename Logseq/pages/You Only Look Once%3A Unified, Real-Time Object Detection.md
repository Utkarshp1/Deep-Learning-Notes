- ![1506.02640.pdf](../assets/1506.02640_1703354830544_0.pdf)
- **Basic Idea**:
	- Can we have a single-stage detector which is faster than two-stage detectors (region-proposal + detection/classification) such as RCNN family?
	- Allow the network to look at the entire image to predict the bounding boxes rather than just local RoI (as in two stage detector). This is indicated by high false positives by RCNN than YOLO.
	-
- **Unified Detection**
	- #+BEGIN_CENTER
	  ((658727a7-130f-49e3-9cb2-2daeb61669cc))
	  #+END_CENTER
	- ((6587231b-fff0-4998-b7a6-fb04cbbb793f))
	- ((6587244b-b8f1-434e-a6d8-312fc84bea67))
		- ((6587245f-b405-4abb-a5a0-9a6e9d2a400c))
		- Confidence ($C$):
		  \begin{equation}
		  C = \text{Pr(Object)} * \text{IOU}^{\text{truth}}_{\text{pred}}
		  \end{equation}
			- $C=0$ if no object belongs to the cell.
			- Else, $C= \text{Intersection over Union (IoU)}$ between predicted box and ground truth.
		- ((65872560-74c5-4678-aafa-a76462055c75))
			- ((65872581-7ef5-4c38-b244-43c453a2581d))
			  The center coordinate is wrt the grid cell because this would enable $[x, y] \in [0, 1] \times [0, 1]$. Moreover, since convolutions have local information, knowing the center wrt to the whole image would be difficult.
			- ((6587258a-66bf-4e07-8565-4db403dff6a2))
		- ((65872ccf-4528-4567-8ab6-cc0c129f61b4))
	-
	- ((658725a4-fa25-4718-94b7-3797b5fb9980))
		- Only predict one set of class probabilities per grid cell.
			- Implies that an object occurs only once in a grid cell (Doubtful).
			- This kind of formulation for sharing conditional probability across bounding boxes helps save number of parameters.
	- **Test time**: Multiply conditional probability and individual box confidence predictions which gives class-specific scores for each box:
	  \begin{equation}
	  \text{Pr(Class}_{i} | \text{Object)} * \text{Pr(Object)} * \text{IOU}^{\text{truth}}_{\text{pred}} = \text{Pr(Class}_{i} \text{)} * \text{IOU}^{\text{truth}}_{\text{pred}}
	  \end{equation}
	  ((6587278a-9932-43a3-8ff2-3818e6e34b14))
	-
	- **Questions**
		- How do they handle cases where there are more than 2 objects being assigned to the same grid cell?
		- ((65872803-6223-4e95-ab93-f86f6abf8528))
		  id:: 658727ec-5c14-4d71-8419-1735090f6f1f
		  How is this true? The conditional probability is estimated using a linear activation function, hence there should be no competition among classes.
		-
- **General Problems with Object Detection/Tricks of the Trade**
	- *Learning fine-grained features.*
	  id:: 658728b3-e83d-4e62-a0bc-ac6d71dedf89
	  Image classification networks trained at $224 \times 224$ whereas the detection network trained at $448 \times 448$.
	  ((6587304b-648c-474d-bc9b-0529c0302c48))
		- ((64ac5696-92e1-4a24-b17b-c787d1dc3a0c))
	-
	- *Detection of smaller objects*
	  id:: 65872fec-b1e9-4827-a491-b0c330294377
	  ((65873004-a7a8-4863-bba7-66996168b29c))
	  Also don't have any mechanism for multi-scale detection.
	-
	- *Loss function not directly aligned with the goal of maximizing average precision.*
	  ((65872aa2-6c81-4702-a14e-d79f6b230674))
	-
	- *Class imbalance: A lot of bounding boxes don't have any object*
	  id:: 65872a91-e6e2-45c1-a53c-883fcb26743a
	  ((65872aea-ef6d-476d-a3b8-d0eb7539d5f7))
	  ((65872b02-99a5-46db-8235-494f16eec8e0))
	  Parameters $\lambda_{coord} = 5$ and $\lambda_{noobj} = 0.5$ accomplish this.
	-
	- *Small error/deviation in large boxes matter less than in small boxes*
	  ((65872c21-dd9a-44a3-9866-7deef0d7b4e8))
	  ((65872c38-6d90-4119-af50-d91e253c778a))
-
	- *Early divergence in training with high learning rate*
	  ((65872ef4-35d4-4181-9bc0-741e219e13b5))
-
	- *Generalization to new or unusual bounding boxes*
	  ((65873037-a806-4c99-b319-3688f3788d5b))
	-
- **Loss Function**
	- #+BEGIN_CENTER
	  ((65872d4f-ba93-412e-b685-048ea7fe9b78))
	  #+END_CENTER
	- $C_{i}$ is the confidence score of object for which bounding box $j$ is "reponsible" in grid cell $i$
	- $p_{i}(c)$ conditional probability of class $c$ being in grid $i$ given there is an object in grid $i$.
	- The last sum happens only if object exists in the grid cell $i$ indicated by $\mathbb{1}^{obj}_{i}$. This is required to ensure that the predicted probability represents the conditional probability $\text{Pr(Class}_{i} | \text{Object)}$.
	- $\lambda_{noobj}$ can be determined using the fraction of bounding boxes that have no object in them. ((658b0580-ef66-453a-a0f1-cb056ed53bde))
-
- **Discussion**
	- *Spatial diversity in bounding box*: ((65873f2f-fd29-4579-96f9-d2642052f824))
	-
	- *Spatial constraints on grid cell proposals*: ((65873f9a-3b31-474c-b033-252fcf3100b1)) 
	  One can think that by spatial constraints they are able to get rid of the region proposals.
	-
	- *Global context reasoning*: ((6587406a-0e62-4821-ab73-0d960c49b3ed))