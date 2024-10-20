
# UnsupervisedML_GAN_TransUNET_SegmentationCovid

This repository contains the source code and datasets for our research project on self-supervised segmentation of COVID-19 CT scans using a combination of 3D GAN and TransUNet architectures.

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


## Usage
To replicate our results or use the models for your data:
1. Clone the repository: `git clone https://github.com/aymaneelfahsi1/UnsupervisedML_GAN_TransUNET_SegmentationCovid.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebooks: Jupyter notebook `path_to_notebook.ipynb`

## Contributing
Interested in contributing? We welcome contributions from the community, whether it's improving the codebase, adding new features, or extending the documentation.

## Contact
For any queries, reach out to us at [aymane.elfahsi@student-cs.fr].

## Declaration

Conflict of Interest: The authors declare that there are no conflicts of interest regarding this project.

Original Notebooks: The notebooks in this repository are based on the original work by the authors of the pipeline we aimed to enhance.

Supervision: This project was conducted under the supervision and guidance of Dr. Oumayma Banouar, Faculty of Sciences and Technics, Cady Ayyad University, [o.banouar@uca.ac.ma].
