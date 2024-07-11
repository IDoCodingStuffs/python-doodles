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

### 9 PM
Well something is still off, the labels are still offset. But I will figure it out later, at least it is way better than how it was yesterday.

### 10 PM
Just realized it is because I was flipping the x and y coordinates while resizing. So ridiculous lol.
After fixing that, the model is converging real fast and well.

## 6/20
### 7 AM
That said the model performance is not that amazing. I should figure something else, but I will take a break for the rest of the week.
One last thing -- let me disable the gaussian blur and try Huber loss. If it still does not get the ordering of the vertebrae, I will take that break.

### 9 AM
Doing somewhat better. Val loss still does not go down but at least the labels are not all over the place. Let me try ResNet34 and if that does not work, add an auxiliary loss function to keep things neat.
I can also try getting more aggressive with the Gaussian blur. And try to add more transforms, assuming I can propagate it to the labels.

### 12 PM
The trick here can be handy to use the same transform on label: https://discuss.pytorch.org/t/use-a-subset-of-composed-transforms-with-same-random-seed/47550
I can kinda hack it initially with a single-pixel image, and later use something more proper to transform the label vector

### 1 PM
This too can be interesting: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
Might have to get more elaborate with the coordinate regression
This as well https://python.plainenglish.io/single-object-detection-with-pytorch-step-by-step-96430358ae9d
Which describes IOU based loss, which might be better than just Huber loss across the full set of points

### 2 PM
So the M1 Pro laptop takes 20 minutes per epoch, but my 4090 setup takes just 2. Good investment.

### 3 PM
Gonna try with mean IOU loss instead of Huber. Let's see if it helps.
Ooh better yet, let me try combined loss. And TIL, you can put loss in CUDA.

### 5 PM
Well that was a fiasco, the model convergence is awful. I wonder what will standalone distance IOU will look like.

### 7 PM
Answer is, somewhat better. But the validation loss refuses to go down, and inference snaps to wrong things. I figured how to add horizontal and vertical flips, so maybe that plus Gaussian blur will yield better results?

### 8 PM
I will let it train all the way but starting to feel like the model itself is just insufficient. 
But I definitely met the initial part "overfit first, expand later". So time to try a UNet backbone instead.
Afterwards, I can tack on the temporal components. Hoping I get to define the bar these upcoming 3 months... Man I really want to win this.

### 10 PM
Well forget the UNet backbone. I will try to use that new public solution with 0.84 score. No need to reinvent the wheel, let me learn the timm model and make architecture changes later.

### 11 PM
Manage to have it at least start running. Trains pretty fast with batch size 16. I'll call it a day after it finishes training, takes just 1 hr.

## 6/21
### 1 PM
Okay time to try labeling all levels. If it trains with just 3 labels, it should train with more ideally.

### 2 PM
Finished adding multihot labels but now it is training suspiciously fast. Like 15 mins for the whole net. And loss goes down only 10% or even less.

### 3 PM
Trying multi-head model again, one per level. Maybe it will overfit successfully?

### 4 PM 
Well it is definitely picking up something. Trying 100 epochs, significant gains by 25. I have high hopes from this one.

Turns out it does not like flips. Let's omit those, the orientation does not change in GT anyway. I might reintroduce hflips only though.

### 5 PM
So the loss that was going down before I accidentally stopped that run? It is not dropping after starting again. I might have to random seed fish I guess. But at least now I know it is possible.

### 6 PM
What helped was chilling down the learning rates of the mid layer and backbone. I might also need to swap the scheduler for something else than cosine annealing though.

### 11 PM
Nah it just converged to local minima because predicting everything as 0 is overrepresented. I need to add FocalLoss and oversample.

## 6/22
### 8 AM
I need to prepare for the AI midterm today. Did not implement FocalLoss but did implement oversampling. Let's see if it helps first.

### 10 AM
It is somewhat helping. For one of the levels it started getting pink in the middle of the diagonal. Let me try changing up the sample weights (2^class vs class * 4) a little and try again.

### 11 AM
Initial epochs showing mispredictions now tend to skew towards the other classes. Man this is hard. 
Guess I should track loss separately per level too? So 5 focal loss functions, with the added benefit of plotting them separately.

### 2 PM
Worked on the confusion matrix viz a bit, now it is a lot more informative. From the earlier looks, it seems the model can finally try to pick up some features now that I am using focal loss.
Some more tuning might be useful but next step will likely be figuring how to have it do the test-train split better so that half the images are not ending up in the valset, and then expand the model.
But it remains to be seen if it will keep training or flatline after 100 or so epochs (1 hour per 100 with the current efficientnet + single lstm + 5-head setup)

### 4 PM
Looking at the confusion matrix, it does not seem to be able to extract features well. Also starts flatlining around 160 epochs.
Let's see if it fares better with a bigger efficientnet. Say b4.

## 6/23

### 1 AM
You know what, I will try a ViT model. It has the temporality and also serves as a vision model after all.
And also very easy to implement with Timm, don't even need the heads per class.
And from the way it looks so far, it is converging very fast within the first epochs. Hoping it won't turn out to be just overfitting to the skewed data.

Well the confusion matrix looks kinda meh, but it does seem to be picking up stuff and I will have to see how it looks at 500 epochs.

### 2 AM
I restarted it, set to go up to 200 epochs and using a 512 res instead of 384 res variant. I have high hopes.
If it fails, I can start looking into training some larger ViT and doing transfer learning. But it might get finicky since those things are HUGE.

### 7 AM
It might be having issues due to the weird very high gamma I have been using. I wonder what it will look like with the default gamma.

### 10 AM
Looks better halfway through. I think part of the issue was ^6 is a bit too much, propagating basically no loss.
That said I might need to figure a better loss function or fix some other issue with FocalLoss. It decreases alright, 
but that decrease should show up as a perfect diagonal on the confusion matrix. Which it is not.

### 1 PM
Let me try with no Gaussian blur now. Maybe it gets stuck in minima without clear distinctions, identifies label 0s as 2s etc because of it.
I can re-add augmentation later if it ends up overtraining.

### 2 PM
Actually, let me try the 2.5D model approach again. I have definitely confirmed the backbone can learn, so now I can refine it by adding stuff.
Chances are the per image model was having those weird precision issues due to the images on the distal ends.

### 3 PM
Now I can verify ViT absolutely can and does learn features per-image. Next will be confirming something that can learn per series.
So back to efficientnet so that it can fit in memory, and let's see if a transformer mid layer will work.

### 4 PM
No, mem usage still blows up. It exceeds 24G. Same thing plagued me trying UNet too, just fitting the series into memory is a whole ordeal.
I have to figure just wtf is going on. It has to be the way I pass in a series since I don't encounter this in batching per image, and even setting the transformer layer to identity reproes this.

It's not the number of images passed either -- just `model(images[0:2].to(device))` still reproes also. Break time, I have a midterm to submit tonight after all.

## 6/24
### 9 AM
Managed to submit midterm 1 hour before it was due, at 3 AM. Meanwhile found some 7 year old discussion suggesting to call
`del loss` and `del output` to avoid some Python scoping shenanigans. Seems to be working.

Well it still goes over 24Gs so something is still fucky. But it stays stable at least.

### 10 AM
Swapping the spatial component with an identity did not work. Disabling layer norm keeps it stable for longer.
Although it still increases, at least is below 24G for now. Let's see if it will stay that way, and what epoch times will be like.

No, it still keeps increasing. Back to the drawing board. Not to mention, the model is taking forever in an epoch.

### 11 AM
Let me track the iteration times to see how terrible it really is.
Ok it starts with like 2-3 iterations per sec. So 10-20x longer than per image, which is actually reasonable.
Next thing to see is if the loss actually goes down.

Oh and the memory leak kicks in later with overflow into swap memory, which drops it to 1-2 seconds per iteration instead. So like a 4-6x perf loss.
I definitely need to fix the root cause, learn how to profile.

Towards the end some iterations take even more, up to like 30. But then following iterations run better.
Also, the validation set runs similarly slow at 1 seconds per iter. Effect of which is, 25 minutes for training and 10 for validation per epoch.
And also also, why the fuck is trainset 1898 samples and val is 782 despite a 0.1 split? Makes no sense.

Time to add profiling.

### 2 PM
Profiling did not do much, but after adding a cache emptying call per training iteration, it is no longer exploding.
But the downside is, I am getting bottlenecked by IO instead now. A lot of copy to GPU.

And when I finally make the purge happen only every 20 iterations, things start looking much better.
Sure, `empty_cache()` is just a band-aid but it still works... Until it stops working, around 220th iteration.
Also the same effect from every 10...

Guess I will just put up with it for now. One day I will figure it out.

1.5it/s consistent up to 900 with the empty call
2-5s/it right after the 30th without

### 3 PM
Added global average pooling to the head instead of naive mean. And re enabled layer normalization, if that will mean anything for just 38 features.
Now it trains at 1 it/s and infers at 8 it/s, so I will leave it alone for a few hours to see what the loss will end up looking like. It is starting off good as of epoch 1.
Probably not realistic to train it for 100 epochs on local though. I might need to rent an H100 eventually.

### 4 PM
I wonder if ViT can actually classify these only with some average pooling. I will probably need to get more elaborate with what I do with the image embeddings and use something more sophisticated.
But as a start, I wonder if it can learn to ignore the front 0-padding for example.

### 6 PM
Nah the loss decrease is way too slow. I need to add some spatial learning layer in addition, and do some sort of pooling at the head of it.

### 7 PM
It's even worse that way. Let me try adaptive pooling and remove the pooling from the head. So the temporal layer only gets some 512-dim vector, 
and then I can run the transformer layer on it plus the head.

To speed up experiment speed, I will also use a tiny portion of the data to see if it can overfit.

The answer is yes, it does. Which means the real training data will actually lead to food results, even if it takes forever to train.
Oh and I should probably make adaptive pooling use max. With the padding and all.
Speaking of, let me try using equal padding on each side vs padding only on front. Probably going to work out better with max pooling too.

### 8 PM
It's definitely looking promising. Next, let me add some layer norm right after the max pool into the attention block and see how that affects things.

### 9 PM
At some point, I can look into accumulated gradients. Loss fluctuates the first few epochs I bother checking, but then my batch size is just 1.

### 10 PM
So it just fluctuates between 0.18 and 0.23 training loss per iter so the tiny data overfit won't tell me much more. 
Next, what if I remove the transformer blocks?
Not much difference. I'll have it run overnight with the transformer block, then remove them and run without next.
After that, I can also try stuff like LSTM or RNNs too.

By the way, purging cache once every 10 iters vs 1 gets me from 1.05 s/it to 1.3it/s. Pretty neat.

## 6/25
### 1 AM
Let me add dropout too, why not. I want to see how well this thing trains by the morning. Should be like 12 epochs by then.

### 6 AM
Not training, how about without the transformers? Should start fitting in theory.

### 9 AM
Moved the avg pooling to after the transformer encoder call. Let's see if it trains now.
Actually nevermind that, I will try the BERT trick and also realized I was doing the padding wrong, so fixed that.

### 10 AM
I have been missing the final softmax all along lmfao. No wonder the loss was fluctuating so much.

### 11 AM
Next iteration, I will try removing the padding since these are transformers after all. And maybe remove the intermediate encoder layer while at it, just pass directly to head.

### 1 PM
So the softmax + bert trick + padding leads to model just not learning at all. To rectify, I will try removing padding first.
I think the problem was softmax actually so gonna remove that too.
Oh and it's sorta faster without the padding as a bonus. About 1.8 it/s

### 5 PM
Well as of epoch 11, loss is flatlining at around 0.177, down from 0.208 at epoch 0. so similar performance to pooling.
Maybe my layers are too few? Let's try 6 instead of 2.
But yeah it is trash so far. I will crank up the backbone lr too, otherwise it is just getting stuck at the local minima predicting everything as moderate.

### 11 PM
Yeah still stuck in the same local minima. I guess it sorta is going down albeit very extremely slowly. 

## 6/26
### 12 AM
So some things to try next:
- Higher alpha on FocalLoss
- Figure how to batch the input, or alternatively use gradient accumulation (second option is lazier)
- Add some image augments. Start with reintroducing Gaussian blur, then flips and then rotations etc.

Starting with grad acc first. Let's do Gaussian blur too, why not. Also cranked up alpha and gamma.

Also, note to future self for the next thing to do once the model starts training: start looking into classical CV techniques to get a real boost. Even just for preprocessing.

### 1 AM
I noticed it has hiccups every few hundred iters or so. Might be worth fixing since it takes epoch time from 19 mins to 27.

### 2 AM
I just realized I misread the previous iteration's results and saw ` 0.040` as `0.4` lol. I cranked down loss alpha as a consequence.
But if it does not perform better this time, I will crank it back up.

Or wait, nvm that nvm. it's due to the normalizing division, so equivalent to `0.32`, and I will likely not know if it is behaving better until this morning.
I should get some sleep.

At least reducing the no of iterations per gradient step helped. `23` mins per epoch with `6` vs `30` per with `8`.
Might end up slower to converge but hopefully not going to get stuck in that minima.

I guess it is even better, more like just `18` when I am not using the machine to multitask. Maybe I should get a dual GPU setup...

I will probably need to re-introduce oversampling also. To ensure there are some moderate and severe cases every gradient update batch.
That way I can avoid it learning to predict everything as mild.

### 8 AM
No, still gets stuck. Time to retry the more aggressive focal loss trick.

### 9 AM
Started off with higher loss but it is going down as of first 2 epochs. I will need to see how it looks this evening though.

### 10 AM
Here are some links of interest to get more elaborate with data preprocess:
https://medium.com/@abdallahashraf90x/oversampling-for-better-machine-learning-with-imbalanced-data-68f9b5ac2696
https://www.tensorflow.org/tutorials/structured_data/imbalanced_data

If naive oversampling seems to help the model start learning, I can get more elaborate with these approaches next.

### 11 AM
Keras-style metrics can be another thing to add to get a far better picture of training performance
Another thing can be undersampling mild/moderates rather than oversampling the other way around, at first.
Which can help extract features more quickly.
Also, I should start reserving a test set in case I start overfitting val further down the road.

Oh, training pause-resume by file. I put in some `.pause `file and it pauses if that file is seen. That way I can run metrics on the side.

### 12 PM
Another agenda item is adding competition scoring metrics. It will help provide a more solid "final score" after the standard metrics start looking promising"


### 2 PM
Seems the loss just started stagnating after the first couple of epochs. I will let it run all the way through to see if it is one of the cases where it takes some time to discover new minima.
But meanwhile, I should probably train the backbone per image first and then load.
The other thing is, I should shift focus to preprocessing.

### 6 PM
Still just predicting the same values regardless of input. I definitely need to shift my approach.
I think the Gaussians are way too aggressive for one. Since the model is failing to overfit, I should disable them first.
Priority is to make the model overfit first. That's how I moved to this stage from the per-image stage.

### 7 PM
Noticed one issue -- conversion from float to uint 8 introduces a bunch of artifacts. And is unnecessary, since the images are loaded fine anyway. 
So the lesson is, check the dumb code you are copypasting lol. I guess it counts as a preprocess improvement at least.

Yeah now that I am comparing before vs after, it was fucking up the input data big time. Could have saved so much electricity :(

### 9 PM
Figured the bug with the train-test split. So many silly bugs, so much carelessness...

## 6/27
### 8 AM
Still not training after fixing that bug with input data. Back to the drawing board -- try disabling the middle layer and make it overfit
I should go back a few steps actually.

Ok how about just 1 transformer encoder layer? Not too hopeful but worth trying I guess.

### 9 AM
After that, I will try just per-image training again, and then an average pooling layer to pass things to the head.
Train the backbone separately, after ensuring it can even train after the fix.

Also need to figure passing per-image labels since forcing 5 levels per image loses most of the training set lol.
Really hoping it won't also end up with the same "stuck predicting the exact same output" issue

### 11 AM
Found another copypasta "did not verify" mishap. The training data from the starter notebook was for coordinate viz. So I was discarding whole bunch of training data.
So embarrassing.

### 12 PM
Yep, yet again paying the tech debt of laziness lol. I think I finished fixing the per-image set though

### 6 PM
Well it is still not training. So back to the CNN-LSTM approach, which is much better proven.

## 6/28
### 6 AM
Good news is it can overfit, bad news is it overfits right off the bat. One thing I can do is adding transforms now.

### 11 AM
It looks far more promising after I add the transforms. Next, to use the alternate oversampling logic.

### 9 PM
Things are looking good so far, although there seems to be a thing with transforms being applied separately per channel.

So let's start over with a 1-channel model this time.

## 6/29
### 12 PM
Seems it is overfitting after the 20th epoch or so, which is when val stops going down meaningfully.
Next, let me see if it does better without the LSTM on per image.

### 2 PM
No, it is not learning without the LSTM per image. So next thing is, loading that model from the 20th epoch as the backbone to a series level model.

Also found a bug where I was softmaxing before the loss function... It was still learning though somehow?

Let's see if LSTM will work with the CLS trick

### 6 PM
It's seeming to overfit a lot by epoch 10, although the recall for severe at L4/L5 is sorta promising. I will let it keep running to see if val goes down later on.
After all the distributions for train and val are different because of oversampling.
Then I will check the performance without the per-image LSTM layer but with the backbone trained alongside it.
Afterwards, I will try training a ViT backbone again since its performance was similar.

Another thing to experiment with is LSTM vs transformer for the middle layer.

### 8 PM
Still no improvement, so I will try without the per-image LSTM and see if that helps things.

### 10 PM
I don't know why, but val loss is absurdly unstable with this.

And as of epoch 20, not great performance. Time to retry ViT

## 6/30
### 10 AM
ViT per image seems to be overfitting after epoch 10. Interestingly, that translates to losing recall in L1/L2 and L5/S1.

Turns out the oversampling is not good enough for those two. Even at a rate of 20:1.

### 11 AM
Let's see how it behaves with the revamped oversampling. It is still hardly equal across all labels but at least somewhat better.

### 3 PM
So train loss goes down but not val. Let me guess, I should be oversampling val too?
Also, the oversampling is still not perfect even if better than before. Mild labels still dwarf the other two even if not as badly as before.

### 8 PM
It's looking more promising now. Only L5/S1 is failing to pop up for some reason by epoch 20, but promising results for the rest.
I will give it a few more hours, and then if L5/S1 features also start getting acquired, I will move on to setting up the 2.5D model next with this backbone.

### 10 PM
One idea to explore is, readjusting the final probabilities per the actual data distribution. Let's see how it turns out after the backbone keeps training overnight.
In the morning, I will take the backbone with the best recalls on all 5 levels and train the series level model with it.

## 7/1
### 11 AM
Still training, at epoch 55 now after 23:54 hours. I just want to see some F1 on all categories for all classes, so that I can move on to the next step lol.
L5S1 still not popping up.

### 3 PM
I might need a preprocess layer. This is still kinda performant though. 

### 5 PM
As of epoch 70, it is looking promising except for L5/S1. I wonder if it will eventually start picking up? Let's have it run till 100 why not.
1500 seconds per epoch...

### 9 PM
Damn thing crashed on its own. I should probably address that... I was curious what it would look like by epoch 200.
But I don't think it was going anywhere anyway. Next thing will be trying the series level training on a promising backbone.

So 4 layer transformer encoder, take the 576 features per image and run those through, see if it can learn anything.
Also freezing the backbone for the first 10 epochs so that the attention layer and head gets a good start instead of passing gradients breaking everything.
Might increase that 10 to 20 or 30 depending on if loss seems to keep going down for that first 10.

### 10 PM
Realized I am probably ending up rotating the image with the swapaxes call lol. Let's see if my fix translates to better loss on the first 10 epochs.

## 11 PM
I think this will end up being way too slow. Let's see what it looks like in the morning, but I probably will revert to an efficientnet backbone.
ViT will probably require a machine off RunPod.

## 7/2

### 8 AM
Efficientnet backbone seems to have comparable performance to the ViT backbone.
I wonder if it will capture the L5/S1 features with further training.
And more importantly, if it will fit in memory with series level

### 10 AM
I wonder if I should modify the loss function so that moderate and severe categories are weighted -- trade precision for recall
Or better yet, just adjust the prediction thresholds

Another thing - instead of 21M ViT, I can try one of the T2T-ViT-7 10 or 12 models which are significantly smaller.
Or any of the extra-small vits on Timm

### 2 PM
There is also an `efficientvit`... Param counts are comfortably below 10M for the m models, or even b0/b1, so should be good enough for training on the 4090.
Wow there is also a timm_3d lib. I can use the 3D variants like that??? Of course the efficientvits are not available, as the downside.

And the models are large at the tiniest around 10M params, but the flipside is, I don't have to duplicate gradients per image as in the 2.5D approach.

I also need to figure the approach for inhomogenous series sizes and whether if it is a component of the data, or some bug

Downside of 3D -- might have to sacrifice the flexibility with the series length, assuming it is not some bug.
Series size might be too small for 3D models as well -- getting "height must be divisible by 8 errors"

### 3 PM
timm_3d lib refusing to work out so far. Btw, worth looking at `volumentations` for training augs.
Might have to pad or 3D resize the loaded data to stretch to 256 depth for the 3D models

### 5 PM
More architectures to consider: `deit`

### 6 PM
Starting to train series-level for `efficientnetv2b3` -- backbone from epoch 85 plus 4-layer transformer encoder plus head.
It takes up like 12G of memory during training with this approach, so this is a good gauge for the upper level of model size.

### 9 PM
I just realized -- why am I adding a whole transformer encoder block where I can just add a single self-attention layer?
The model can finally start learning in series. And I can finally start adding the other series data types, then submit and finally start iterating through model architectures and extra data and shit.

### 10 PM
Turns out it was too early to celebrate -- model just predicts the same value for all input, so no learning done.

### 11 PM
I guess missing normalization can lead to that? Let's see how it behaves with layernorm before the self attention layer.

Similar, at least for the initial epochs. Let's see if it is one of those "hard to learn" cases where it will take some hours before it can start continuing to learn.

## 7/3
### 12 AM
So one problem I can now check for -- head weights are all 0 in that faulty case, so whatever comes before is failing. 
Maybe due to the attention head approach being incorrect? In any case, it's just learning the bias at least for now.

Yeah no, the normalization is not helping at least initially. Time to leave it overnight.

### 1 AM
One other discovery -- the sampling weights are not working as well as expected. Somewhat better, but the first, second and last joints still get overrepresented mild.
Oh well, at least the model is not immediately overfitting like it did without normalization. Really hoping it does manage to pick something up eventually.

### 8 AM
No such luck. Time to explore other options. Let's see if architecture change helps -- 3D model vs 2.5

### 9 AM
I should go through this https://www.kaggle.com/code/hugowjd/rsna2024-lsdc-training-densenet
Also, need to learn about K-Fold Cross Validation.

### 11 AM
Another approach to explore: pass in each image in the series in a different channel. Might work especially well with ViTs.
For the attention layer, this can come in handy: https://stackoverflow.com/questions/61619007/pytorch-how-to-add-a-self-attention-to-another-architecture

3D Model is not doing great. Time to retry the transformer with some positional embeddings this time.

### 3 PM
No luck so far as of epoch 25. At least it's not overfitting though, so let's see if might eventually overcome the barrier.

### 5 PM
Still no luck. Let me see if there is some loss function bug on some off chance, however unlikely.
If not, I will focus on passing the sequence across channels. And setting up the eval pipeline for 3D.

### 8 PM
Passing sequence across channels shows promise for efficientnet. Not so much for tinyvit.
So let's train this baby with b4 efficientnet.

For some reason training tends to have a lot of hiccups -- keeps pausing and resuming a lot. Wonder why.
I should also look into cuda amp stuff. Might enable me to train beefier models.
Ditto for quantization aware training. Later though, let's get a submission out of this all this week.

### 11 PM
It's working! It's finally fucking working!
Next thing is training the other two series and submitting this shit!

## 7/4
### 12 AM
For the handed dataset -- vertical mirror is an awesome way to augment the data. It will be the exact next thing I will try after the initial submission in fact!

### 1 AM
Another TODO before I call it a night: dump the confusion and prec/recall/f1 graphs automatically
Also to consider, using the same model for both sagittal series?

Shit, also, turns out the per-channel approach won't support the axial series. So that one needs a 3D model. How exciting!

### 10 AM
So, turns out that is not feasible for memory reasons. And the median is like 50 slices, 25th percentile is 20 something.
Meaning I will either need to downsample or get way elaborate with it. Well, at least the multichannel model can accommodate.

### 3 PM
Trying resizing to 100 channels for T2 this time. Hoping it will work out better.
Training is still bottlenecked and GPU utilization sucks, but at least training time does not completely suck.

### 5 PM
Switched back to padding. The memory bottleneck sucks... Wondering if I can move the transforms to GPU or something.

## 7/5
### 11 AM
The performance is kinda meh if I have to be honest. I should probably introduce 3D transforms and also figure 3D model use instead of multichannel.
Also, I should check if there is some leakage between train and val just in case.

### 2 PM
Finally made a submission with the 3 multichannel models. Fingers crossed on no bugs, hoping to get something better than random.

### 5 PM
Working on volumetric transforms now.

Newsflash, I did not get something better than random. Fuck.

Next thing is revamping my whole approach I guess. I should also set up the competition metric on local so that I am not flying blind and spending 2 weeks to end up not even beating random.

### 8 PM
Starting training from pretrained and using avg pool, and suddenly validation loss is going wild.
That makes more sense, I think default pooling was making it fail to generalize or something.

Let's try random sampling 10 images per series.

## 7/6
### 1 AM
Shuffling the order is yielding better results. My initial assumption of passing it the images in order was incorrect, and it was overfitting like crazy.

Now I am seeing the val loss going down properly for the initial 10 or so epochs at least.

### 2 AM
I wonder how it will behave with volumentations too. I can make another submission and see how the score changes either way, at least for the sagittal series.

### 1 PM
It improved by a .1 so not terrible, probably mostly improvements from the T2/STIR. 
I think I need a 3D model for sure. Which requires some preprocessing step so that I am not processing massive volumes.
Maybe I can start with 64^3 voxels then pass them to a channel per condition. Just rough clipping through the middle for a start.

### 2 PM
I finally started looking into PyDicom docs... Boy have I been doing this all wrong all along.
There is a bunch of MRI metadata 3D reconstruction and also downsampling vs resizing...
So if I downsample width and height by 4, then it will be in the same order of magnitude as a 384^3
Time to test it out

### 5 PM
Actually, let's try the sequential approach again first now that the sequences are actually right.

### 6 PM
Not working with transformers + CLS. Avg pool maybe?
Oh before that, I realized I forgot to add the positional embeddings. Let's see if that changes things.

### 9 PM
Seems like a no, let's try avg pooling

## 7/7
### 1 AM
Still no. Wonder if I can use multichannel alongside the transformer. 
It won't see the features from each image independently, but might end up learning some sequentiality?
Fuck, there is gotta be some trick to this I am missing. Get an embedding per image and run through a sequential layer?
Too expensive and fails to train.
Run each image into its own channel? Now that misses sequentiality.
Data loading was one little part I was missing. Now there has to be just another tiny part like that.

Timm3D models? Too expensive to even fit without preprocessing to get the image sizes down.
Although I guess I can downsample a bit... Huh, it can actually kinda run 128^3 as opposed to 384^3. Let's try that then.

Also, finally had the sense to check the impact of augments on the 3d data. Did not even look like spines.
Now trying to see if it can fit without absurdly aggressive 2d transforms that break everything about the 3d relationships.
If it manages to fit, I can introduce volumetric transforms on top.
Also also, I wonder if this is why the sequential models would just not fit. Not that it matters with how slow they are.

### 3 AM
Calling it a night now, gonna see how the 128x128 resize turns out.
Next thing will be setting up some model to chop up the images and feed it in 64^3 sized channels, without downscaling.
Could be as simple as chopping evenly, but I do not want to chop in the middle of vertebral discs.
So, some sort of segmentation model to get vertebral contours, then finding axial and sagittal midlines, and then chopping up along those.

### 7 PM
So to start on that, first I will try to pass in patient level data as a whole. Each series in its own channel first.
Also, discussion on Kaggle showed he file indices were better for sorting, somehow. But I will still need to figure the orientations to match.

### 8 PM
Oh of course there are nans in severity. Beautiful.
Well I think I have a patient level dataset with 3 channels put together now. So let's try and see if it performs with a 3D model.

### 11 PM
Val loss still not going down. Gonna iterate through the loss function again. 
BCE with logits instead of FocalLoss, which is somehow insanely faster. Like, it's bottlenecked by CPU but flies through once on GPU.
But class imbalance will be something to tweak around.

## 7/8
### 12 AM
Val loss goes down with BCE, but the problem is if it will just learn to predict mild for everything.
Guess I will see in the morning. Leaving it to train with efficientnetb0 and efficientnetb3 overnight. 
When I am up, I can check the test set.

### 8 AM
Val loss went down for the first few before coming back up, so that's promising.
Sadly inference is very slow without workers, so I will need to fix that data loading bottleneck.

Profiler says -- it is the scipy zoom that is so time consuming. So I could try 0 padding + downsampling.

### 11 AM
Padding is significantly faster. Still bottlenecked by IO though so I will need to figure it out. 
Also val loss decreases more, so that's a bonus.

### 1 PM
Well the downside is, it is worse at picking up the minority classes. So I should probably reintroduce oversampling.
Also, the mirror trick.
And also, Focal Loss but implemented correctly so that alpha is a vector.

### 2 PM
Fuck my val loss has been all borked up since I switched to BCE.

### 3 PM
Gave up trying to figure the implementation for FocalLoss for now. Gonna use weighted BCE loss, see if that works.

Also started getting some `IOError` which went away upon reboot... for now.

## 7/9
### 12 PM
Val loss was failing to converge while train loss was overfitting after the first handful. 
Gonna see if not applying 2D augmentations helps.

Realized I was deep frying the input again with a `.to(uint8)` call again. No wonder it is immediately overfitting.
Let's see if it does better without it now.

Although, I will need to do that conversion at some point so that I can use augmentations beyond dropout and rotate.
Added back with `cv2.convertScaleAbs` since val loss was getting absurdly high

### 1 PM
Adding randombrightnesscontrast or gaussian noise per slice was leading to slicewise artifacts which I did not like.
It might be beneficial anyway, but I will try and find or implement 3D equivalents.

Ditto for stuff like stretch, random crop, dropout etc.

### 2 PM
!TODO: Check this out https://colab.research.google.com/drive/1CT9nIGME_M4kIDc3BfEF4pCb_8JdFLpH#scrollTo=DLqhO16yXQq1

Or better yet, this: https://torchio.readthedocs.io

### 9 PM
After some work, I figured how to use the torchio transforms, and boy do they work much better
Val loss goes down significantly for some 20 epochs, so I will run to 100 overnight with more aggressive augmentations.

I think this will be good enough for a submission and wrap up before I move onto some segmentation approach.
The idea is -- one model to grab each disc + foramina + radial nerve bits and another to just diagnose the condition.
That will help me get around the issue with the absurdly underrepresented condition classes for the L1/L2 and L5/S1 levels especially.
Instead:
1. Some 3D segmentation model to determine the vertebrae
2. Some sort of model to get the centers and pose
3. Slice along the center of each vertebra axially
4. Grab a cube just large enough to contain the relevant features for each of the 5 levels
5. Some diagnostic model to finally get the condition level (1-3) for each of the 5 conditions

## 7/10
### 12 AM
One last thing -- some images have patients moving and such, so simple stacking is not sufficient. I will need to add some affine mapping per slice
based on the imageposition and imageorientation metadata.

### 1 AM
Realized I was using the wrong dimension for channel unsqueeze. Fixed that, which should enable me to use resampling instead of resizing tomorrow.

### 10 AM
Got a cool finding: `RSNACervicalSpineFracture` from 2022 has segmentation data. Can be repurposed for lumbar probably.

### 11 AM
Another cool/worrying finding -- those images are not natively uint16 for nothing. Converting to uint8 loses gray levels.
So I might want to fix that eventually. By moving the data loading completely into torchio.

This could also help with the resizing narrow slices https://github.com/fepegar/torchio/discussions/828

### 7 PM
Turns out volume stiching is actually tricky as hell.
Oh well, hopefully it will be fine.