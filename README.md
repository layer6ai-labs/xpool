<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p> 

<div align="center">
<h1>
<b>
X-Pool: Cross-Modal Language-Video Attention for Text-Video Retrieval
</b>
</h1>
<h4>
<b>
<a href="https://www.cs.toronto.edu/~satyag/">Satya Krishna Gorti*</a>, <a href="https://www.cs.toronto.edu/~nvouitsis/">Noël Vouitsis*</a>, <a href="https://www.linkedin.com/in/jeremy-ma/">Junwei Ma*</a>, <a href="https://www.linkedin.com/in/keyvangolestan/">Keyvan Golestan</a>, <a href="https://www.cs.toronto.edu/~mvolkovs/">Maksims Volkovs</a>, <a href="https://animesh.garg.tech/">Animesh Garg</a>, <a href="http://www.cs.toronto.edu/~guangweiyu/">Guangwei Yu</a>    
</b>
</h4>
  
[Paper](https://arxiv.org/abs/2203.15086) | [Project Page & Demo](https://layer6ai-labs.github.io/xpool/)
</div>

<a name="intro"/>

## Introduction
This repository contains the official implementation of our **CVPR 2022** paper. It includes both training and evaluation code.

<a name="depend"/>

## Dependencies
Our model was developed and evaluated using the following package dependencies:
- PyTorch 1.8.1
- Transformers 4.6.1
- OpenCV 4.5.3

<a name="datasets"/>

## Datasets
We trained models on the MSR-VTT, MSVD and LSMDC datasets. To download the datasets, refer to this [repository](https://github.com/ArrowLuo/CLIP4Clip).

For LSMDC, you must obtain permission from MPII to download and use the data, so we do not provide the split and caption files in the `data/` directory.

<a name="eval"/>

## Evaluation
The following commands can be used to reproduce the main results of our paper using the supplied checkpoint files for each dataset. The commands will by default generate results for text-to-video retrieval (t2v). For video-to-text retrieval (v2t) results, add the argument `--metric=v2t` to the command.

If the `outputs/` folder does not exist, first run `mkdir outputs` to create the directory. For each dataset, create a directory in `outputs/` and store the corresponding checkpoint file. For each command below, replace `{exp_name}` with the name of that directory.

Also, replace `{videos_dir}` with the path to the dataset's videos.

For evaluation, you can change the `batch_size` without affecting results.
  

<a name="eval-commands"/>

| Dataset | Command | Checkpoint File | t2v R@1 Result |
|:-----------:|:-----------:| :-----------: | :-----------: |
|MSR-VTT-9k|`python test.py --exp_name={exp_name} --videos_dir={videos_dir} --batch_size=32 --huggingface --load_epoch=-1 --dataset_name=MSRVTT --msrvtt_train_file=9k`| [Link](https://drive.google.com/file/d/1M2Y41B3a3AxzSJn-n-Xh0Edds97NU1ND/view?usp=sharing)       | 46.9| |
|MSR-VTT-7k|`python test.py --exp_name={exp_name} --videos_dir={videos_dir} --batch_size=32 --huggingface --load_epoch=-1 --dataset_name=MSRVTT --msrvtt_train_file=7k`| [Link](https://drive.google.com/file/d/1KW6VQiiTHpfMcK8GIgRq-5aWAgf7rGPj/view?usp=sharing)       | 43.9| |
|MSVD|`python test.py --exp_name={exp_name} --videos_dir={videos_dir} --batch_size=32 --huggingface --load_epoch=-1 --dataset_name=MSVD`| [Link](https://drive.google.com/file/d/1c1iV6V00hnvZPTfLdWSFV2adUNWC2-zk/view?usp=sharing)       | 47.2| |
|LSMDC|`python test.py --exp_name={exp_name} --videos_dir={videos_dir} --batch_size=32 --huggingface --load_epoch=-1 --dataset_name=LSMDC`| [Link](https://drive.google.com/file/d/1vQiQjVg6kX1u4T2HmalrydSZYQ0fAbX_/view?usp=sharing)       |25.2| |

<a name="train"/>

## Training
The following commands can be used to train our X-Pool model for each dataset. Again, the evaluation is by default set to generate results for text-to-video retrieval (t2v). For video-to-text retrieval (v2t) results, add the argument `--metric=v2t` to the command.

For each command below, replace `{exp_name}` with your choice name of experiment. Also, replace `{videos_dir}` with the path to the dataset's videos.
  

<a name="train-commands"/>

| Dataset | Command |
|:-----------:|:-----------:|
|MSR-VTT-9k|`python train.py --exp_name={exp_name} --videos_dir={videos_dir} --batch_size=32 --noclip_lr=3e-5 --transformer_dropout=0.3 --huggingface --dataset_name=MSRVTT --msrvtt_train_file=9k`|
|MSR-VTT-7k|`python train.py --exp_name={exp_name} --videos_dir={videos_dir} --batch_size=32 --noclip_lr=1e-5 --transformer_dropout=0.4 --huggingface --dataset_name=MSRVTT --msrvtt_train_file=7k`|
|MSVD|`python train.py --exp_name={exp_name} --videos_dir={videos_dir} --batch_size=32 --noclip_lr=1e-5 --transformer_dropout=0.4 --huggingface --dataset_name=MSVD`|
|LSMDC|`python train.py --exp_name={exp_name} --videos_dir={videos_dir} --batch_size=32 --noclip_lr=1e-5 --transformer_dropout=0.3 --huggingface --dataset_name=LSMDC`|

<a name="train-commands"/>

## Citation

If you find this work useful in your research, please cite the following paper:

```
@inproceedings{gorti2022xpool,
  title={X-Pool: Cross-Modal Language-Video Attention for Text-Video Retrieval},
  author={Gorti, Satya Krishna and Vouitsis, No{\"e}l and Ma, Junwei and Golestan, Keyvan and Volkovs, Maksims and Garg, Animesh and Yu, Guangwei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```
