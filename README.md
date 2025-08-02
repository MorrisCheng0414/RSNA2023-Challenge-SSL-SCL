# RSNA2023-Challenge-SSL-SCL
Our experiment results demonstrate the fact that SSL and SCL pretraining could alleviate the overfitting problem occurs commonly in supervised learning. Moreover, it also yields modest improvement across several metrics. Further analysis, through comparing the experiments results with different feature extractor components, we reveal the image feature extractor is the major contributor to these gains. Lastly, we discover the potential of SCL to reinforce the robustness of a model by switch the backbone model of the feature extractor, providing insights into breaking the model robustness bottleneck.

## Requirements
The experiments are conducted with Cuda version 12.1 and Python version 3.10. Please install the following packages to configure the environment.
```
pip install numpy==1.26.3
pip install pandas==2.2.2
pip install pillow==10.2.0
pip install pytorch-cuda==12.1
pip install scikit-learn==1.5.1
pip install timm==1.0.9
pip install torch==2.6.0
pip install torchvision==0.21.0
pip install tqdm==4.65.0
pip install transformers==4.44.2
```

## Dataset Usage
The dataset we used is provided by [RSNA 2023 Abdominal Trauma Detection](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/).
Below is a list of data we used in the thesis:
1. ```segmentations/```: Model generated pixel-level annotations of the relevant organs and some major bones for a subset of the scans in the training set. This data is provided in the nifti file format. The filenames are series IDs. You can find a description of [the source model (total segmentator) here](https://pubs.rsna.org/doi/10.1148/ryai.230024) and [the data used to train that model here](https://zenodo.org/record/6802614).
2. ```train_images```: The CT scan data, in DICOM format. Scans from dozens of different CT machines have been reprocessed to use the run length encoded lossless compression format but retain other differences such as the number of bits per pixel, pixel range, and pixel representation. 
3. ```train.csv```: Target labels for the train set. Note that patients labeled healthy may still have other medical issues, such as cancer or broken bones, that don't happen to be covered by the competition labels.
4. ```train_series_meta.csv```: Each patient may have been scanned once or twice. Each scan contains a series of images. 
5. ```submission.csv```: A valid sample submission. Only the first few rows are available for download.

## Segmented Dataset
The segmented dataset used in the experiments is stored on Kaggle, please refer to [the dataset webpage](https://www.kaggle.com/datasets/morrisistaken/rsna23-256x256-roi-png). 

After downloading the dataset to your local machine, please change the dataset path in ```dataset_dir.yaml```.

## Folder Structure
```
Root/
│
├── data/                    # shared CSV files
├── dataset.py               # shared PyTorch Dataset class
├── utils.py                 # shared utilities
├── download_weights.py      # script for download pretrain weights
│
├── 2d_pretrain/
│   ├── ssl_methods/             # 2d SSL methods classes
│   ├── pretrain_weights/        # 2d model pretrain weights
│   ├── weights_finetune/        # 2d model train weights (freeze backbone)
│   ├── weights/                 # 2d model train weights
│   ├── img_aug.py           # 2d image augmentation
│   ├── main_organs.py       # 2d pretraining
│   ├── main_lincls.py       # 2d downstream training 
│   ├── model.py             # 2d pretraining/training model definition
│   ├── train.py             # 2d pretrain/train/validation function
│   └── hparam.yaml          # hyper-parameters settings for 2d SSL methods
│
├── 3d_pretrain/
│   ├── ssl_methods/             # 3d SSL, SCL methods classes
│   ├── pretrain_weights/        # 3d model pretrain weights for different folds
│   ├── weights/                 # 3d model train weights
│   ├── img_aug.py           # 3d image augmentation
│   ├── main_organs.py       # 3d pretraining
│   ├── main_lincls.py       # 3d downstream training 
│   ├── model.py             # 3d pretraining/training model definition
│   ├── train.py             # 3d pretrain/train/validation function
│   └── hparam.yaml          # hyper-parameters settings for 3d SSL methods
│
├── README.md                # project documentation
└── requirements.txt         # require enviroment
```

## Execution
### Download pretrain weights (CLI)
Please run the below code for downloading the pretrain weights.
```bash
python -m RSNA2023-Challenge-SSL-SCL.download_weights [Options]
```
#### Options
| Name           | Default | Description |
|----------------|-------------------|----------------------------------------|
| --repo_dest    | Current directory | The destination folder of the cloned repository |
| --repo_name    | RSNA2023-Challenge-Pretrain-Weights | The name of the new directory where the repo will be cloned |

### Download pretrain weights (manual)
1. Download the ```zip``` file in [this repository](https://github.com/MorrisCheng0414/RSNA2023-Challenge-Pretrain-Weights).
2. Manually move the ```2d_pretrain_weights/``` folder under ```RSNA2023-Challenge-SSL-SCL/2d_pretrain/```.
3. Rename it to ```pretrain_weights```.
4. The same applies to ```3d_pretrain_weights/```

### Pretraining
Please run the below code for 2D and 3D pretraining.
#### 2D Pretraining
```bash
python -m RSNA2023-Challenge-SSL-SCL.2d_pretrain.main_organs [Options]
```
#### 3D Pretraining
```bash
python -m RSNA2023-Challenge-SSL-SCL.3d_pretrain.main_organs [Options]
```
#### Options
| Name           | Default | Description |
|----------------|-------------|----------------------------------------|
| --epochs       | 10          | Number of training epochs |
| --batch_size   | 4           | Number of videos in a batch |
| --accumulate   | 16          | Number of steps to gradient accumulation |
| --num_workers  | 0           | Number of CPU workers |
| --lr           | 2e-4        | Initial learning rate |
| --wd           | 1e-4        | Weight decay |
| --warmup_ratio | 2           | Ratio of warmup step to total training step. During the warmup step, the learning rate linearly increase from 0.0 to initial learning rate. |
| --kfold        | 5           | Number of folds for stratified k-fold cross-validation |
| --methods      | All methods | List of models pretrained by different methods to run |

### Training & Validation
Please run the below code for 2D and 3D training and validation.
#### 2D Training & Validation
```bash
python -m RSNA2023-Challenge-SSL-SCL.2d_pretrain.main_lincls [Options]
```
#### 3D Training & Validation
```bash
python -m RSNA2023-Challenge-SSL-SCL.3d_pretrain.main_lincls [Options]
```
#### Options
| Name           | Default | Description |
|----------------|-------------|------------------------------------|
| --epochs       | 15          | Number of training epochs |
| --save_last_k  | 2           | Number of last epochs to save weights |
| --batch_size   | 4           | Number of videos in a batch |
| --accumulate   | 4           | Number of steps to gradient accumulation |
| --num_workers  | 0           | Number of CPU workers |
| --lr           | 2e-4        | Initial learning rate |
| --wd           | 0.01        | Weight decay |
| --warmup_ratio | 0.1         | Ratio of warmup step to total training step. During the warmup step, the learning rate linearly increase from 0.0 to initial learning rate. |
| --kfold        | 5           | Number of folds for stratified k-fold cross-validation |
| --methods      | All methods | List of models pretrained by different methods to run |
| --train_cnn    | True        | Train 2D CNN or not |

