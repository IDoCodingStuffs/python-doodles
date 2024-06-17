## 6/15 
### 2 PM
On Windows + CUDA, using the CNN+LSTM architecture
Training Sagittal T2/STIR:  67%|██████▋   | 8/12 [3:45:55<1:53:31, 1702.84s/it]

Will try running on WSL with additional dataloader workers
That works! GPU utilization showing up at 100% even. Hoping to see train time improvement when I am back.

Assuming it went smooth, I will next implement some config file system to make hyperparam sweeps easier, add observability etc.

Fuck yeah...
Training Sagittal T2/STIR:   8%|████▌                                                 | 1/12 [13:21<2:26:51, 801.02s/it]

Lesson here: data loading is a suspect bottleneck for when GPU utilization fails to reach 100% during training

### 5 PM
Apparently I had forgotten the scheduler step not to mention flushing the pyplot figure. Not seeing the model training, so going to try a higher starting lr, and actually add the scheduler stepping this time.
This time will also use 30 epochs instead of 11. If it still does not start converging I must be missing something fundamental.
Oh and also going to try resnet18 instead of resnet50. Features are not really that complex, right?

Let's also try the head architecture from the 2022 paper as well, sans the batch normalization when it is per series

### 8 PM
Well it stil ldoes not do great after epoch 1. t2/stir Val acc stuck around .84 and ce loss around .72. Let's try with image augmentation
Before:
Training Sagittal T2/STIR:  9/30 [1:05:09<2:32:01, 434.38s/it]
After: 1/30 [07:10<3:27:56, 430.24s/it]

Seems the transforms have no impact on train time at least
But the val loss and acc still not changing per epoch...
Let's try a smaller LSTM to see if it is even training

### 10 PM
Model still refuses to train... Gonna try BCEWithLogits loss before tearing everything down to start from a minimal model.
Just realized... the severities are different per disc. No wonder it fails to train, duh.
I need to change the output shape to account for all discs of interest.

So I am outputting a concatenated one-hot encoding per each of the 5 discs of interests, scaled to sum up to 1. Let's see if this trains...

It's seeming to work. For some reason training faster even [05:45<2:47:11, 345.92s/it]. I think it is because I moved the `tensor(np_array_input).to(device)` call out of the training loop, which makes obvious sense. Big lesson there.

## 6/16

### 10 AM
Well I forgot nan handling so the other two models did not get to train. But the T2/STIR model seems to have trained somewhat even if very little. That or it is a false impression from changing the output feature vector size.

Let's try training the other two and submitting to see. If it gets a bad score still, the next step will be setting up a proper local benchmark.

### 12 PM
I just realized... There is also a component for different conditions like stenosis, foraminal narrowing etc.
Which means I need to increase the output feature vector size further. Subarticular stenosis and foraminal narrowing times per side, and also spinal canal narrowing. So 5 x 5 x 3 makes 75 output features.
I also need to upgrade the model to patient level vs just series level to accommodate this.

### 3 PM
Let's start with disabling flip, rotation etc. augmentations and then enabling patient-level data loading. Then, I can figure how to set up the model for that.
Or before all that, let's try expanding the output features first.
While doing that, I discovered a bug with the label generation. No wonder the model was basically not training at all.
That said, still not training despite fixing that bug. So I should have different features for different modalities then?

So not all studies have all conditions. Which means I might need to use some ensembling on top of series level inference vs patient level inference.