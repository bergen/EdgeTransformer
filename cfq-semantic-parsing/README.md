This folder contains the code to train Edge on the [CFQ Semantic Parsing task](https://arxiv.org/abs/1912.09713).

The code uses [Weights & Biases](https://wandb.ai). Please set yourb `WANDB_API_KEY` as follows:

```
export WANDB_API_KEY=XXX
```

You can the run the training on MCD1:

```
python -m torch.distributed.launch --nproc_per_node 4 train.py
```

Training on 4 V100 GPUs takes around 25 hours. On average you should be getting the test accuracy of around 44-45%, but individual runs can be getting +- 2% around that.