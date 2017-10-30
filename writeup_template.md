## Project: Perception Pick & Place
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


# Required Steps for a Passing Submission:
1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify). 
2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  [See the example `output.yaml` for details on what the output should look like.](https://github.com/udacity/RoboND-Perception-Project/blob/master/pr2_robot/config/output.yaml)  
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.
9. Congratulations!  Your Done!

# Extra Challenges: Complete the Pick & Place
7. To create a collision map, publish a point cloud to the `/pr2/3d_map/points` topic and make sure you change the `point_cloud_topic` to `/pr2/3d_map/points` in `sensors.yaml` in the `/pr2_robot/config/` directory. This topic is read by Moveit!, which uses this point cloud input to generate a collision map, allowing the robot to plan its trajectory.  Keep in mind that later when you go to pick up an object, you must first remove it from this point cloud so it is removed from the collision map!
8. Rotate the robot to generate collision map of table sides. This can be accomplished by publishing joint angle value(in radians) to `/pr2/world_joint_controller/command`
9. Rotate the robot back to its original state.
10. Create a ROS Client for the “pick_place_routine” rosservice.  In the required steps above, you already created the messages you need to use this service. Checkout the [PickPlace.srv](https://github.com/udacity/RoboND-Perception-Project/tree/master/pr2_robot/srv) file to find out what arguments you must pass to this service.
11. If everything was done correctly, when you pass the appropriate messages to the `pick_place_routine` service, the selected arm will perform pick and place operation and display trajectory in the RViz window
12. Place all the objects from your pick list in their respective dropoff box and you have completed the challenge!
13. Looking for a bigger challenge?  Load up the `challenge.world` scenario and see if you can get your perception pipeline working there!

[//]: # (Image References)

[world1]: ./images/world_1.jpg
[world2]: ./images/world_2.jpg
[world3]: ./images/world_3.jpg

[confusion_1]: ./images/confusion_1.jpg
[confusion_2]: ./images/confusion_2.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

#### Exercise 1,2 pipeline for filtering, segmentation and clustering implemented.  

* step 1: Voxel Downsampling -> voxel_downsample() function
* step 3: Passthrough Filter -> passthrough_filter() function
* step 2: Statistical Outlier Removal -> detection_workflow.py -> filter_outliers() function
* step 4: RANSAC plane fitting -> ransac_filter() function
* step 5: Cluster Segmentation -> find_cluster_indices() functions

#### Exercise 3.  Features extracted and SVM trained.  Object recognition implemented.
The feature extraction is done in sensor_stick/src/sensor_stick/features.py . The feature vectors consist of:
* HSV components, each has 32 entries (from the 32 bins of the histogram) , describing the distribution of the component in the trained object
* x,y components of the normal vectors, with 10 entries each. describing the distribution of normal vector component on the trained object.

Trained model with ~96% accuracy. Below are the confusion matrix from the training session

Here is an example of how to include an image in your writeup.

![demo-1](https://user-images.githubusercontent.com/20687560/28748231-46b5b912-7467-11e7-8778-3095172b7b19.png)

### Pick and Place Setup

The 3 test worlds are used to test out the object recognition model. The outputs are saved in output_1.yaml, output_2.yaml, output_3.yaml, inside /perception/pr2_robot/scripts folder. At the place coordinates, I added random noise in the x and y coordinates, to avoid the stacking pieces problem as reported in the lessons

The pipeline achieved the following results:

* world 1: 3/3 (100% - required 100%)
* world 2: 4/5 (80% - required 80%)
* world 3: 7/8 (87.5% - required 75%)

##### World 1
![World 1][world1]

##### World 2
![World 2][world2]

##### World 3
![World 3][world3]





