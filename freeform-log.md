### 6/15 2 PM
On Windows + CUDA, using the CNN+LSTM architecture
Training Sagittal T2/STIR:  67%|██████▋   | 8/12 [3:45:55<1:53:31, 1702.84s/it]

Will try running on WSL with additional dataloader workers
That works! GPU utilization showing up at 100% even. Hoping to see train time improvement when I am back.

Assuming it went smooth, I will next implement some config file system to make hyperparam sweeps easier, add observability etc.

Fuck yeah...
Training Sagittal T2/STIR:   8%|████▌                                                 | 1/12 [13:21<2:26:51, 801.02s/it]

Lesson here: data loading is a suspect bottleneck for when GPU utilization fails to reach 100% during training

### 6/15 5 PM
Apparently I had forgotten the scheduler step not to mention flushing the pyplot figure. Not seeing the model training, so going to try a higher starting lr, and actually add the scheduler stepping this time.
This time will also use 30 epochs instead of 11. If it still does not start converging I must be missing something fundamental.
Oh and also going to try resnet18 instead of resnet50. Features are not really that complex, right?

Let's also try the head architecture from the 2022 paper as well, sans the batch normalization when it is per series

### 6/15 8 PM
Well it stil ldoes not do great after epoch 1. t2/stir Val acc stuck around .84 and ce loss around .72. Let's try with image augmentation
Before:
Training Sagittal T2/STIR:  9/30 [1:05:09<2:32:01, 434.38s/it]
After: 1/30 [07:10<3:27:56, 430.24s/it]

Seems the transforms have no impact on train time at least
But the val loss and acc still not changing per epoch...
Let's try a smaller LSTM to see if it is even training