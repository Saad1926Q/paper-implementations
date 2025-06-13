# My YOLO Implementation Experience

So umm I was just having a bit of imposter syndrome and I was like let's try to implement YOLO.

So I skimmed through the paper and I was like *oh it's just a CNN, looks simple enough* (note: it was not).

Implementing the architecture was actually pretty straightforward 'coz it's just a CNN.

So first we have 20 convolutional layers followed by adaptive avg pooling and then a linear layer, and this is supposed to be pretrained on the ImageNet dataset (which is like 190 GB in size so yeah I obviously am not going to be training this thing but yeah).

So after that we use the first 20 layers and extend the network by adding some more convolutional layers and 2 linear layers.

Then this is trained on the PASCAL VOC dataset which has 20 labelled classes.

Seems easy enough, right?

---

## The Real Challenge

This is where the real challenge was.

First of all, just comprehending the output of this thing took me quite some time (like quite some time).  
Then I had to sit down and try to understand how the loss function (which can definitely benefit from some vectorization 'coz right now I have written a version which I find kinda inefficient) will be implemented — which again took quite some time.  
And yeah, during the implementation of the loss fn I also had to implement IoU and format the bbox coordinates.

Then yeah, the training loop was pretty straightforward to implement.

---

## Inference

Then it was time to implement inference (which was honestly quite vaguely written in the paper IMO but yeah I tried to implement whatever I could comprehend).

So in the implementation of inference, first we check that the confidence score of the box is greater than the threshold which we have set — only then it is considered for the final predictions.

Then we apply Non-Max Suppression which basically keeps only the best box. So what we do is: if there are 2 boxes which basically represent the same box, only then we remove the one with the lower score. This is like a very high-level understanding of NMS without going into the details.

Then after this we get our final output...

---

## Final Thoughts

Also, one thing is that I know there is a pretty good chance that I might have messed up here and there, but I think overall I am pretty happy 'coz at one point I was pretty close to giving up.

During implementing this thing I was definitely able to get a much better understanding of YOLO than before and also a sort of sense of appreciation for how everything goes together.
