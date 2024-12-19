
# Unsupervised Medical Imaging: 3D GAN and TransUNET for Advanced COVID-19 CT Segmentation

This repository contains the source code and datasets for our research project on self-supervised segmentation of COVID-19 CT scans using a combination of 3D GAN and TransUNet architectures.
## Authors

Aymane EL FAHSI, Aymane DHIMEN, Mostapha EL ANSARI, Selma KOUDIA, Adam M. KHALI  
Ecole Centrale of Casablanca  
Emails:  
- aymane.elfahsi@centrale-casablanca.ma
- aymane.dhimen@centrale-casablanca.ma
- mostapha.elansari@centrale-casablanca.ma
- selma.koudia@centrale-casablanca.ma
- adam.khali@centrale-casablanca.ma

Dr. BANOUAR Oumayma  
Faculty of Sciences and Technics, Cady Ayyad University  
Email: o.banouar@uca.ac.ma

---

## Project Structure
- **Original Data**: Original dataset from Mosmed [download here](https://drive.google.com/drive/folders/1DAMKsbyvNFgIkUIRvVi6ls5nFo6hPmF2?usp=sharing).
- **Preprocessed Data**: Preprocessed CT scans prepared for the model [download here](https://drive.google.com/drive/folders/1uauGRuoG4i9wB87VK_aAFNwzfC8XwbZs?usp=sharing).
- **Notebooks**: Jupyter notebooks detailing each step from preprocessing to predictions.
- **Results**: Outputs and visualizations from our experiments.

## Highlights
- **Innovative Approach**: Utilizes a self-supervised learning framework to tackle the challenge of limited annotated medical datasets.
- **Advanced Models**: Implementation of a 3D GAN for generating pseudo-masks and a TransUNet for precise segmentation of COVID-19 CT scans.
- **Results**: Our models demonstrate high accuracy and robustness, surpassing traditional methods in various metrics.

## Background
- **Original Pipeline**: The foundational research and pipeline for our work was adapted from [Varut Vardhanabhuti et al., 2020](https://doi.org/10.1016/j.compbiomed.2022.106033).
- **TransUNET Architecture**: Adapted from [mkara44's TransUNET implementation](https://github.com/mkara44/transunet_pytorch).
![image](https://github.com/user-attachments/assets/cecd2051-663b-4fec-8738-7581c1a46a8a)

![image](https://github.com/user-attachments/assets/818cf315-fa43-4009-bb37-382cb788db5a)
---

## Implementation and Training

### Implementation
#### Encoder in TransUNet

1. **Initial Convolution**:
   - The input image is first processed by a convolution layer, reducing its spatial resolution and extracting basic features.
   - Output: A feature map with increased channels and reduced resolution.

2. **Hierarchical Feature Extraction with CNN**:
   - The encoder uses three `EncoderBottleneck` blocks, progressively reducing the spatial resolution while increasing the feature depth.
   - These blocks create a hierarchical representation of the input, capturing local features at multiple scales.

3. **Vision Transformer (ViT) for Global Context**:
   - The deepest feature map (output of the final bottleneck) is reshaped into a sequence of patches and fed into the **ViT**.
   - **Purpose**:
     - The ViT captures **global contextual relationships** across all spatial regions of the image, which traditional CNNs struggle to achieve.
     - It leverages **self-attention** to learn interactions between patches, providing a more holistic understanding of the input.
   - **Steps**:
     - The feature map is divided into patches (size = 1, since it’s already reduced to patches).
     - Each patch is linearly embedded into a token.
     - Positional embeddings are added to the tokens to encode spatial relationships.
     - The tokens pass through multiple transformer blocks, performing self-attention and feedforward transformations.
   - Output: A globally enriched feature map.

4. **Post-ViT Processing**:
   - The output tokens from the ViT are reshaped back into a 2D feature map.
   - This feature map is passed through a final convolution layer to prepare it for the decoder.

#### Decoder in TransUNet

1. **Upsampling with Skip Connections**:
   - The decoder progressively upsamples the feature maps to restore the spatial resolution of the input image.
   - **Skip connections** from the encoder are concatenated with the upsampled features at each stage.
   - **Purpose**:
     - Skip connections help retain fine-grained spatial details from the encoder.
     - The decoder combines global context (from the bottleneck) and local details (from the skip connections).

2. **Decoder Bottlenecks**:
   - Each upsampling step is followed by a `DecoderBottleneck`, which refines the features.
   - **Details**:
     - Each `DecoderBottleneck` contains:
       - A bilinear interpolation layer for upsampling.
       - Convolutions to process and refine the upsampled features.
       - Batch normalization and ReLU activation for better training stability.

3. **Final Convolution**:
   - After the last upsampling step, a **final convolution layer** reduces the number of channels to the number of output classes.
   - This produces the segmentation map.
   - **Example**:
     - If the input is \( [B, 1, 128, 128] \) and the task is binary segmentation, the final output will be \( [B, 1, 128, 128] \), with pixel values representing probabilities of each class.

---

### **Key Workflow of the Decoder**

1. **Input**:
   - The input to the decoder is the bottleneck feature map output by the encoder’s Vision Transformer.

2. **Steps**:
   - **Step 1**: Bottleneck feature map is upsampled.
   - **Step 2**: Concatenate with the skip connection from the corresponding encoder layer.
   - **Step 3**: Refine the combined features using convolutions in the `DecoderBottleneck`.
   - Repeat for each resolution level until the original input size is restored.

3. **Output**:
   - A segmentation map of the same spatial resolution as the input image.



#### **Skip Connection Workflow**
At each decoder stage:
- **Skip Connection Source**: Feature maps \( x_1, x_2, x_3 \) from the encoder.
- **Skip Connection Integration**:
  - Concatenate the skip connection with the upsampled feature map.
  - This ensures that both fine-grained details and global features are used in reconstruction.
---
## Contributing
Interested in contributing? We welcome contributions from the community, whether it's improving the codebase, adding new features, or extending the documentation.

## Contact
For any queries, reach out to us at [aymane.elfahsi@student-cs.fr] or [mostapha.el_ansari@centrale-med.fr].

## Declaration

Conflict of Interest: The authors declare that there are no conflicts of interest regarding this project.

Original Notebooks: The notebooks in this repository are based on the original work by the authors of the pipeline we aimed to enhance.

Supervision: This project was conducted under the supervision and guidance of Dr. Oumayma Banouar, Faculty of Sciences and Technics, Cady Ayyad University, [o.banouar@uca.ac.ma].
