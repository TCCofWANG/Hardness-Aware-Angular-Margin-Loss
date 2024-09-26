# An Open-Set Image Classification Algorithm Based on Hardness-Aware Angular Margin Loss and Label Noise Filtering
Learning discriminative features is crucial for deep learning-based image classification, particularly in open-set scenarios. Current methods often aim to achieve this by increasing inter-class decision boundaries and compressing intra-class feature distances through the addition of a fixed margin to all samples.  However, assigning the same margin to all samples disregards the varying learning difficulties among different samples. Moreover, label noise is almost unavoidable in large-scale classification datasets.To address these issues, we devise a novel loss function that adaptively adjusts the angular margin based on the difficulty of each sample. Specifically, this paper proposes a quantification function that helps the model identify sample difficulty during training and adaptively adjusts the angular margin for different samples based on these quantification results. Additionally, we propose an automatic detection method for anomalous labels during training, leveraging the Laplace kernel function, which significantly enhances model performance in the presence of noisy label data.To verify the effectiveness of the proposed method, we conduct several experiments using face recognition tasks and typical open-set image recognition as case studies, demonstrating that our method outperforms the state-of-the-art approaches.

# Training
To train face recognition models on HAAML under clean train set, run this command:
```python
python trainOnHAAML.py 
```
To train face recognition models on HAAML under noisy train set, run this command:
```python
python Noise_CASIA_WF_HAAMLLaplace.py 
```
# Datasets
CASIA-WebFace and MS1MV3:https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_
