Code for 
# SLAM-AGS: Slide-Label Aware Multi-Task Pretraining Using Adaptive Gradient Surgery in Computational Cytology
_Marco Acerbis, Swarnadip Chatterjee, Christophe Avenel, Joakim Lindblad_

[ArXiv preprint] 

## Dataset
The Bone Marrow Cytomorphology dataset can be downloaded from [here](https://www.cancerimagingarchive.net/collection/bone-marrow-cytomorphology_mll_helmholtz_fraunhofer/).
To run the full code you need to prepare the following set up:
* After downloading the dataset, use bag_gen.py to create the train/test bags,
* Only for the training set; create two folders: positive and negative. In the first folder copy ALL the patches from positive bags; while in the second copy all the patches from the negative bags.
You will need to pass these two folders to SLAM-AGS.py, while for PAMIL.py train/test you will have to use the folder containing the bags. 

## Run the code
> pip install prerequisites.txt

Usage:
- Use SLAM-AGS.py to pretrain a ResNet18 encoder with SLAM-AGS,
> python SLAM-AGS.py --positive_dir /path/to/positive/dir --negative_dir /path/to/negative/dir --wr (witness rate to name the saving file)

- Once you have a trained encoder, you can use embeddings.py to generate train and test embeddings for PAMIL,
> python embeddings.py --split train/test --data_dir path/to/bags/dir --dim encoder_output_dimension --model path/to/pretrained/model.pth  --pre [weakly, self, wcs]
- At last, run PAMIL.py to train the MIL model and run the test.
> python PAMIL.py --emb_train path/to/train/embeddigs/dir --emb_test path/to/test/embeddigs/dir --nproto number_of_prototypes_to_be_used --dim encoder_output_dimension


Corresponding author: Marco Acerbis (marco.acerbis@it.uu.se)







