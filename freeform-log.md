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

### 7 PM
T2STIR model still not training. 
I will go back to the resnet only model. If it overfits, then the problem is with the model. If not, then the problem is upstream. But before that, let me try removing the no_grad call wrapping the whole ResNet layer. Maybe the final FC is not getting trained that's what's up.

Nope. Still not training.

Even the starter notebook said this. So the lesson is, to pay better attention to the starting data.
> Saggital T1 images map to Neural Foraminal Narrowing <br>
Axial T2 images map to Subarticular Stenosis <br>
Saggital T2/STIR map to Canal Stenosis <br>

Let me try the approach here:
https://visualstudiomagazine.com/articles/2021/10/04/ordinal-classification-pytorch.aspx

Which says to map categories to target floats rather than one-hot encoded categories. And still not training.
Next, let me try L1Loss and after, this other approach here: https://towardsdatascience.com/how-to-perform-ordinal-regression-classification-in-pytorch-361a2a095a99

L1Loss is not making much difference either. I should dump some sort of confusion matrix before continuing on, to understand what's going on better

### 10 PM
Found a bug where I was dividing by the number of levels. Let's see if it trains with that fix.

Well. Seems it is training now, finally. Might get a useful T2STIR model for sure, although I need to see what to do for the other two. Also need to figure how to make other loss functions play nice.

## 6/16

### 10 AM
So today, I will look into adding an inference notebook with a confusion matrix to figure how well the model is actually training.

And the answer is, not well at all. It seems the "mild" cases are overrepresented in data at least for t2/stir, so the model predicts everything as such. To mitigate, I will try oversampling the cases with moderate or severe conditions. If that fails, I will try some penalized loss.

So the boost will need to be 10x for moderate and 20x for severe given this.
> moderate,732
normal_mild,8552
severe,469

And it will not be a trivial thing to do since there are multiple classes.
Let's try something silly like this first
```python
self.sampling_weights[key] = 1 + (np.sum(self.labels[key]) - len(self.levels) * 0.25) * 8
```

Well it is not training much better that way either. So I should try switching the feature encoding and loss functions.

### 1 PM
Fixed a bug with device output being float32 and label being float64 (double). Let's see if fixing that and using MSE loss helps. Probably not though. Then I can also try KL Divergence loss

MSELoss still not great. Everything gets predicted one class or another, horizontal line on confusion matrix. Let's see how KLDiv will do.

### 3 PM
Well it was a bad idea. It is not a probability distribution after all. Back to feature engineering. 2 labels for 3 classes per level should do it? I also disabled oversampling, will focus on making the mode overfit first.

Per https://discuss.pytorch.org/t/multi-label-classification-in-pytorch/905/45 one can apparently do multilabel classification with BCELoss too. So multi-hot encoding plus that and let's see if the model starts training.
If not, I can try oversampling again to see if it helps.

Getting weird large negative numbers for loss. Wonder what's up with that. Ah. Just a typo.

### 4 PM
I should try multi-label soft margin loss actually. BCE is not appropriate at all, this is a multilabel classification problem after all.

### 5 PM 
No help from that. I noticed my outputs are all almost exactly the same. So found this SO link that might help: https://stackoverflow.com/questions/74200607/all-predicted-values-of-lstm-model-is-almost-same

### 6 PM
Let me try initializing the model with Kaiming weights. The SO post did not help much, so I will shrink the model back.

Or what if, the feature is just hard to learn (which it is) and that is why the output keeps being the same? Maybe my assumption that it should immediately start learning is wrong.
Might also involve the encoder CNN layer not being sufficiently fine-tuned or even appropriate.

### 7 PM
Here is another idea: one fully connected head per level. Each has one label, and then I can call CrossEntropyLoss on each separately.
I will also look into the ideas from this post such as FocalLoss: https://www.kaggle.com/code/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb

## 6/18

### 9 AM
It turned out the detail I was missing was a very obvious one -- the CNN backbone needs training too! So I will try the multihot approach from last night, but will start training the backbone at epoch 10. Let's see what it looks like when I return from work.

### 12 PM
Actually... I should use a smaller resnet and also use a single head basic model first. Overfit first, expand and optimize later. I want to see if the 2.5D model backbone can even learn features without preprocessing first.

### 4 PM
Not training as of epoch 22. Need to fix a mistake I had made with the continuous labels, then try multi-hot single tensor 

### 6 PM
Just realized I should use BCELoss instead of BCEWithLogits loss for multi-hot. The loss gets wacky with small numbers.

### 7 PM
Still no signs of loss going down. Let me try training the backbone immediately, with higher lr.

### 11 PM
After hours of trying to figure the memory issue with UNet, I gave up and found a pretrained model for brain MRIs. I will use that. And the thing took 200 epochs apparently ?! Guess I should be more patient.

## 6/19
### 12 AM
I also just realized... ResNet input size is 224x224, but the features are likely to get lost that way. 
So what I need to do instead is having a landmark detector first. Then crop around the landmarks per spinal joint.
And *then* train and/or infer on those cropped images.

### 12 PM
Okay, so ResNet with an output of 2x5 coordinates. MSELoss, should be straightforward. Hell if it does not overfit I know nothing.

### 1 PM
Well I realized I needed a temporal element still. Training will still be time-consuming and I really want a CNN only approach, but let's see if it at least manages to train

### 2 PM
I have a feeling there is a bug with that LSTM setup and it does not learn that great. I should just try a regular CNN first.

### 4 PM
Finally! The CNN overfits. I finally have a fucking baseline to go off of.
Let's try adding some image augmentations, dropping the nulls from input, and see what happens...
Oh shit, I need to scale the labels by image resize. Need to do it in the dataset impl.

### 5 PM
Well it stops fitting after a certain point. I guess I should double check the coordinates are all there for all levels, not too many are dropped, and if all else fails, figure how to prevent UNet from blowing up so that I can train it as the backbone instead.

### 6 PM
It looks like simply dropping null is not sufficient to clean the data from the faulty labels. They also seem to be getting misplaced.
I need to figure that one out.

Yeah same image gets multiple labels or something. I am not passing those right.

### 7 PM
After fixing that, finally getting the slope going downwards. Sweet. I will even have my backbone trained this way, so 2 birds. 
Next thing will be figuring how to clean and then impute the rows with missing joint coords.

Ooh just realized, I should do the train-test split by study ids. So that I am not ending up with blank levels within same series.