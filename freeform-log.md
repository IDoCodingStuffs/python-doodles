### 6/15 2 PM
On Windows + CUDA, using the CNN+LSTM architecture
Training Sagittal T2/STIR:  67%|██████▋   | 8/12 [3:45:55<1:53:31, 1702.84s/it]

Will try running on WSL with additional dataloader workers
That works! GPU utilization showing up at 100% even. Hoping to see train time improvement when I am back.

Assuming it went smooth, I will next implement some config file system to make hyperparam sweeps easier, add observability etc.

Fuck yeah...
Training Sagittal T2/STIR:   8%|████▌                                                 | 1/12 [13:21<2:26:51, 801.02s/it]

Lesson here: data loading is a suspect bottleneck for when GPU utilization fails to reach 100% during training

### 6/15 3 PM
Apparently I had forgotten the scheduler step not to mention flushing the pyplot figure. Not seeing the model training, so going to try a higher starting lr, and actually add the scheduler stepping this time.