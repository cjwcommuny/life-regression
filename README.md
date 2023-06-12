# Life Regression

## Dependencies

```
Python==3.8.6
torch==1.10.0
torchprofile==0.0.4
torchvision==0.11.1
transformers==4.15.0
fvcore==0.1.5
matplotlib==3.4.3
numpy==1.20.3
pytorch-lightning==1.4.4
seaborn==0.11.0
timm==0.5.4
```

## Data preparation

You have to download **ImageNet** from [image-net.org](http://image-net.org/). The directory structure is the same as `torchvision.datasets.ImageNet` in `torchvision`:

```
/path/to/ILSVRC2012/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

## Training

Run the following command to train the proposed model with `DeiT-S` backbone:

```
python train_deit.py --keep_rate=KEEP_RATE --temperature=TEMPERATURE --data_root=/PATH/TO/ILSVRC2012
```

where `KEEP_RATE`, `TEMPERATURE` are hyper-parameters. `/PATH/TO/ILSVRC2012` is the path of the dataset.

## Evaluation

Run the following command to evaluate the proposed model:

```
python inference.py --checkpoint_dir=CHECKPOINT_DIR
```

where `CHECKPOINT_DIR` is the directory where the checkpoint resides.

## Benchmark

Run the following command to compute the throughput of models:

```
python benchmark_sota.py
```

Run the following command to compute GFLOPS of models:

```
python compute_efficiency.py
```

## Visualization

Fill the relevant path (workspace, checkpoint etc.) in `./notebooks/visualize.ipynb` and run. You can obtain the visualization results.

