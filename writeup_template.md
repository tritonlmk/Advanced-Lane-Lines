## Advanced Lane Lines Finding


**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration4.jpg "calibration4.jpg"
[image2]: ./output_images/calibration5.jpg "calibration5.jpg"
[image3]: ./output_images/original_cali.jpg "Original Image"
[image4]: ./output_images/undistortedimagetest.jpg "Undistorted"
[image5]: ./output_images/warped_cali.jpg "Warped&Undistorted"
[image6]: ./output_images/test_image_undistort.jpg "Single Picture Undistorted"
[image7]: ./output_images/color_comb.JPG "Color Threshold Combined"
[image8]: ./output_images/grad_comb.JPG "Gradient Threshold Combined"
[image9]: ./output_images/stacked.JPG "Color and Gradient Threshold Combined"
[image10]: ./output_images/total_comb.JPG "Final Result of Threshold Combination"
[image11]: ./output_images/birdeye.JPG "Birdeye Prespective Transform"
[video1]: ./project_video.mp4 "Video" 

---

### Camera Calibration

#### 1. This process will calibrate the camera in orderto make the picture reflects the real words better.

The code for this step is contained in the first three code cell of the IPython notebook located in "./LaneLines_finding.ipynb" 

It start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here assumes the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time when all chessboard corners in a test image are successfully detected.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. And `imagepoints` could be found using `cv2.findChessboardCorners` function.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained the undistorted image.

I used images in the file `./camera_cal` to calibrate the camera. However, there are two images that the `cv2.findChessboardCorners`, which are shown below:

#### Image that cv2.findChessboardCorners failed to find corners (calibration4.jpg & calibration5.jpg)
![alt text][image1]
![alt text][image2]

And I used the `cv2.getPerspectiveTransform` to get the prespective transform matrix and use `cv2.warpPerspective` to change the image prespective(birdview for lane lines).
Below are the results:

#### Original Image vs Undistorted Image vs Warped&Undistorted Image

![alt text][image3]
![alt text][image4]
![alt text][image5]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image6]

#### 2. Using different thresholds and find a good combination to find lane lines. Then change the picture to a birdeye presprctive.

Firstly I tried all the following thresholds to find the optimal combination for lane lines detection result:
* grayscale
* R threshold in RGB
* S threshold in HLS
* Sobel Gradient: magnitude
* Sobel Gradient: dirtction
* Sobel Gradient: absolute value

This part is included in the 5th code cell in the .ipyb file.

I used a combination of color(R threshold & S threshold) and all three gradient thresholds to generate a binary image .  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

#### Binary Result of Color Threshold combination(R & S)
![alt text][image7]
#### Binary Result of Gradient Threshold combination(absolute value & direction & magnitude)
![alt text][image8]
#### Binary Result of the Color Threshold and Gradient Threshold Stacked
![alt text][image9]
#### Final Binary Result of Thresholds Combination
![alt text][image10]

Then the opencv function `cv2.warpPrespective` and the transform matrix is used to transform the result to a birdeye prespective.


#### 3. Prespective Transformation.

In order to make the lane lines finding have better results and calculate the curvature of the lane lines, the binary result need to be changed to a birdeye perspective. And the change lane lines found back to the original prespection for visulation.
So I calculated the transform matrix and the inverse transform matrix in the third code cell and print out the transform matrix in the 4th cell of the .ipyb file.

The cell takes a source (`src`) and destination (`dst`) points to calculate the matrixs needed.  I chose the hardcode the source and destination points in the following manner:

```python
    src = np.float32 ([
        [220, 651],
        [350, 577],
        [828, 577],
        [921, 651]
    ])

    dst = np.float32 ([
            [220, 651],
            [220, 577],
            [921, 577],
            [921, 651]
        ])
    M_bird = cv2.getPerspectiveTransform(src, dst)
    M_bird_inv = cv2.getPerspectiveTransform(dst, src)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 320, 651      | 220, 651      | 
| 350, 557      | 220, 577      |
| 828, 557      | 960, 577      |
| 921, 651      | 921, 651      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

#### Birdeye View Binary Result
![alt text][image11]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
