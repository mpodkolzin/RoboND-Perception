[//]: # (Image References)

[world1]: ./images/world_1.jpg
[world2]: ./images/world_2.jpg
[world3]: ./images/world_3.jpg

[confusion_1]: ./images/confusion_1.jpg
[confusion_2]: ./images/confusion_2.jpg

![demo-1](https://user-images.githubusercontent.com/20687560/28748231-46b5b912-7467-11e7-8778-3095172b7b19.png)

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

#### Exercise 1,2 pipeline for filtering, segmentation and clustering implemented.  

* step 1: Voxel Downsampling -> voxel_downsample() function
* step 3: Passthrough Filter -> passthrough_filter() function
* step 2: Statistical Outlier Removal -> filter_outliers() function
* step 4: RANSAC plane fitting -> ransac_filter() function
* step 5: Cluster Segmentation -> find_cluster_indices() functions

#### Exercise 3.  Features extracted and SVM trained.  Object recognition implemented.
The feature extraction is done in sensor_stick/src/sensor_stick/features.py . The feature vectors consist of:
* HSV components, each has 32 entries (from the 32 bins of the histogram) , describing the distribution of the component in the trained object
* x,y components of the normal vectors, with 10 entries each. describing the distribution of normal vector component on the trained object.

Trained model with ~96-99% accuracy. Below are the confusion matrix from the training session

![Confusion matrix][confusion_1]

Normalized Confusion matrix

![Confusion matrix normalized][confusion_2]



### Pick and Place Setup

The 3 test worlds are used to test out the object recognition model. The outputs are saved in output_1.yaml, output_2.yaml, output_3.yaml, inside /perception/pr2_robot/scripts folder. At the place coordinates, I added random noise in the x and y coordinates, to avoid the stacking pieces problem as reported in the lessons

The pipeline achieved the following results:

* world 1: 3/3 (100% - required 100%)
* world 2: 4/5 (80% - required 80%) (book misclassified as biscuits)
* world 3: 7/8 (87.5% - required 75%) (snacks misclassified as biscuits)

##### World 1
![World 1][world1]

##### World 2
![World 2][world2]

##### World 3 
![World 3][world3]

Summary and Closing remarks

1. I actually implemented Pick & Place routine, but it worked well only for world_1, because
it only has 3 objects, with more 'packed' worlds2&3 robot arm was constantly kick objects off 
the table. Most likely collision cloud was not computed properly.
2. There is some room for improvement with object recognition. I would probably experiment further with number of histogram bins, SVC kernel types, adding more features etc.
3. Interesting observation is that 'linear' classifier yielded better results than 'rbf'.






