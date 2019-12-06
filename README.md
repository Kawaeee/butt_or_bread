# Corgi butt or loaf of bread? 
- This repository will tackle the meme of "Corgi butt or loaf of bread?" using deep learning model to classify it.

<img src="https://img-9gag-fun.9cache.com/photo/aYeP537_700bwp_v2.webp" width="500" height="500">

#### Reference: https://9gag.com/gag/aYeP537/corgi-butt-or-loaf-of-bread

## Datasets
- Bread dataset was acquired from the combination of Google Open Images V5 + Google Images search
- Corgi dataset was acquired from the combination of ImageNet + Stanford Dogs + Google Images search

  **After that, we manually remove incorrect images and apply Perceptual Hashing to get rid of duplication images.**

  * Total images: 6385 images (randomly split using 80:10:10 ratios) with [dataset-split](https://github.com/muriloxyz/dataset-split)
  
    * **Bread**: 3710 images
    
      * Train: 2968 images
      * Valid: 371 images
      * Test: 371 images
    * **Corgi**: 2675 images
    
      * Train: 2140 images
      * Valid: 268 images
      * Test: 267 images
      
## Model
- ResNet-152

## Results
|Set|Loss|Accuracy|
|:--|:--|:--|
|**Train**|0.0077|0.9977|
|**Valid**|0.0132|0.9969|
|**Test**|-|0.9968|

#### You can download our model weight here: [Google Drive](https://drive.google.com/file/d/1DVsTz-gx8NvdMaFseF26CYzE6uVLPAL4/view?usp=sharing)

## Hyperparameters and configurations

| Configuration | Value |
|:--|:--|
|Epoch | 3 |
|Batch Size | 32 |
|Optimizer | ADAM |

## Reproduction
 * In order to reproduce the model, it requires our datasets. You can send me an e-mail at kawaekc@gmail.com if you are interested.
 
 - Install dependencies
    ```Bash
    pip install -r requirements.txt
    ```
    
 - Run the train.py python script
 
    ```Bash
    python train.py 
    ```
    
 - Open and run the notebook for prediction : `predictor.ipynb`
