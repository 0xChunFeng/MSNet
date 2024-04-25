# MSNet
Official code repository for: FCN-Transformer Feature Fusion for Polyp Segmentation

2. Usage
2.1 Preparation
Create and activate virtual environment:
python3 -m venv ~/FCBFormer-env
source ~/FCBFormer-env/bin/activate
Clone the repository and navigate to new directory:
git clone https://github.com/ESandML/FCBFormer
cd ./FCBFormer
Install the requirements:
pip install -r requirements.txt
Download and extract the Kvasir-SEG and the CVC-ClinicDB datasets.

Download the PVTv2-B3 weights to ./

2.2 Training
Train FCBFormer on the train split of a dataset:

python train.py --dataset=[train data] --data-root=[path]
Replace [train data] with training dataset name (options: Kvasir; CVC).

Replace [path] with path to parent directory of /images and /masks directories (training on Kvasir-SEG); or parent directory of /Original and /Ground Truth directories (training on CVC-ClinicDB).

To train on multiple GPUs, include --multi-gpu=true.

2.3 Prediction
Generate predictions from a trained model for a test split. Note, the test split can be from a different dataset to the train split:

python predict.py --train-dataset=[train data] --test-dataset=[test data] --data-root=[path]
Replace [train data] with training dataset name (options: Kvasir; CVC).

Replace [test data] with testing dataset name (options: Kvasir; CVC).

Replace [path] with path to parent directory of /images and /masks directories (testing on Kvasir-SEG); or parent directory of /Original and /Ground Truth directories (testing on CVC-ClinicDB).

2.4 Evaluation
Evaluate pre-computed predictions from a trained model for a test split. Note, the test split can be from a different dataset to the train split:

python eval.py --train-dataset=[train data] --test-dataset=[test data] --data-root=[path]
Replace [train data] with training dataset name (options: Kvasir; CVC).

Replace [test data] with testing dataset name (options: Kvasir; CVC).

Replace [path] with path to parent directory of /images and /masks directories (testing on Kvasir-SEG); or parent directory of /Original and /Ground Truth directories (testing on CVC-ClinicDB).
