- ![1612.08242.pdf](../assets/1612.08242_1703532203084_0.pdf)
- **Basic Idea**:
	- Can we improve the performance of one-stage detector (YOLO)?
		- **Better**: In comparison to Fast R-CNN makes more localization error. Also has a lower recall (TPR) i.e., unable to identify boxes with classes. Can we improve recall as well as reduce localization error?
		- **Stronger**: Being able to detect 9000 object categories.
		- **Faster**
-
- **Stronger**
	- **Batch Normalization**:
		- ((6589d7db-03bb-4470-b031-603e5900d784))
-
	- **Fine-grained learning (High-Resolution Classifier)**:
		- {{embed ((658728b3-e83d-4e62-a0bc-ac6d71dedf89))}}
		- ((6589d93c-214f-4863-b5ea-8b6170d178aa))
		- **Solution:** ((6589d953-575b-46d1-b1ab-f19467f23a33))
	-
	- **Fine-grained learning (Detecting Smaller objects)**:
		- {{embed ((65872fec-b1e9-4827-a491-b0c330294377))}}
		- Feature-map output size for YOLO: $13 \times 13$ (sufficient for large objects)
		- ((6589da36-8a16-4b11-837b-fc0f481e396f))
		- **Solution:** Take feature map from previous layer ($26 \times 26 \times 512$) and resize it into $13 \times 13 \times 2048$ and concatenate with the original feature map of the last layer channel-wise. ((6589daf4-ea7f-4535-b9bc-9b5bf861c096))
-
	- **Fine-grained learning (Multi-scale training)**:
		- **Objective**: ((6589db7d-5f56-4394-9c94-88889326e08a))
		- **Solution**: ((6589db90-12f9-4bbe-add5-e840cb8b1e44))
-
	- **Anchor Boxes**:
		- Use hand-picked anchor boxes (priors) like Faster R-CNN for predicting bounding boxes (offsets wrt to these anchor boxes instead of predicting the coordinates directly). ((658ad08e-d82f-4e6a-b633-370ee1779da0)) This is because convolutions have local information, therefore knowing the coordinates wrt to whole image would be difficult.
		- ((658ad0d7-580d-46bd-ac28-79be153be2dd))
		- *YOLO limitation*: ((658727ec-5c14-4d71-8419-1735090f6f1f))
		  ((658ad0ec-803a-49ca-824f-4ed31314e1c0))
		- ((658ad16a-e60e-4eff-82cb-6f07092ad067))
-
	- **Data-driven anchor box (Auto-anchor):**
		- ((658ad1d6-c3a1-43f9-9e65-2c6917f03687))((658ad1e0-46d7-4434-bd39-5574279589b2))
		  \begin{equation}
		  \text{d(box, centroid)} = 1 - \text{IOU(box, centroid)}
		  \end{equation}
		- ((658ad237-7b71-4e41-b712-11a72c0b1727))
		  In order to achieve 100% recall on the training dataset, we can set $k$ to be equal to the number of all possible bounding boxes in the training dataset.
		- *Efficacy of auto-anchor*: Measured by calculating average IoU between the training set bounding boxes and the anchor boxes. ((658ad358-97ec-40f8-b47e-45214655dae2)) Therefore, with 5 (instead of 9) anchor boxes we can save number of parameters.
		- *Downside of auto-anchor*: Lack of generalization. The downside of learning anchor boxes is that it may not transfer well to other datasets.
		-
	- **Direct Location Prediction:**
		- The network training with anchor boxes is highly unstable during the early iterations due to unconstrained prediction of centre of the bounding box.
		  ((658ad5d3-fb29-46f6-a5c3-180875f59570))
		- ((658ad5f5-9a93-434e-873c-33e8f30f7173)) This is because the centre of the object has to lie within the grid cell.
		- #+BEGIN_CENTER
		  ((658ad612-a912-4f1f-b26d-bcd5652c68da))
		  #+END_CENTER
- **Results**
	- #+BEGIN_CENTER
	  ((658ad6ee-47b5-47eb-ba12-0c78ae3f22b6))
	  #+END_CENTER
	-