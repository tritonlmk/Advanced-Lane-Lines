## Advanced Lane Lines Finding

* The Project used traditional method(sobel gradient and sliding windows) to detect lane lines in a video and output it as project_video_output.mp4

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
[image12]: ./output_images/slide_win.JPG "Result of the Sliding Window"
[image13]: ./output_images/histogram.JPG "Histogram used to find starting points"
[image14]: ./output_images/test_final.JPG "Visualization Result"
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

One important thing about the Thresholding is that: do not use the gray scale threshold, for it will give too much noise when the color of the road base changes. Also, the gradient direction threshold also have too much noise, so the `|` or logic is used(while other thresholds use `&` and logic).

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

This part is done in the 7th code cell of the .ipyb file.

Then I used the sliding window method introduced in the udacity courses to detect the lane lines. To find the starting point I used the histogram of the lower half of the picture along x direction to find the peak and use its y location as the starting point.
The histogram is shown below:

#### Histogram used to find starting points to slid windows
![alt text][image13]

This part is also done in the 7th code cell of the .ipyb file.
I also tried the convolution window method, but it performs really bad and I found it hard to define a appropriate kernel to do the convolution. So I just give up this method.

I later write a fuction to find the best order(form 1 to 3) to fit the lanelines, and than using the function to calculate the curvature and the displacement between the car and the middle of the lane lines. Then I choose a polynomial to fit the valid lane lines points. I make the program to automatically find the best order o polynomial(from 1 to 3) to fit.
You can see the code below:

```python
def best_poly_order(y, x):
    # 1st order
    poly1 = np.polyfit(y,x,1)
    p1 = np.poly1d(poly1)
    pred1 = p1(y)
    mse1 = mean_squared_error(x, pred1)
    # 2nd order
    poly2 = np.polyfit(y,x,2)
    p2 = np.poly1d(poly2)
    pred2 = p2(y)
    mse2 = mean_squared_error(x, pred2)
    # 3rd order
    poly3 = np.polyfit(y,x,2)
    p3 = np.poly1d(poly3)
    pred3 = p3(y)
    mse3 = mean_squared_error(x, pred3)
    minimum = mse1
    order = 1
    if minimum > mse2:
        minimum = mse2
        order = 2
    elif minimum > mse3:
        minimum = mse3
        order = 3
    
    if order == 1:
        return order, poly1
    elif order == 2:
        return order, poly2
    else:
        return order, poly3
```

Later, I could the previous result(lane lines fit) to find the starting point of the sliding window in a frame.
You can see the code:

```python
   else:
        left_lane_poly = np.poly1d(left_fit)
        left_lane_inds = ((nonzerox > (left_lane_poly(nonzeroy) - margin)) & \
                          (nonzerox < (left_lane_poly(nonzeroy) + margin)))
        right_lane_poly = np.poly1d(right_fit)
        right_lane_inds = ((nonzerox > (right_lane_poly(nonzeroy) - margin)) & \
                           (nonzerox < (right_lane_poly(nonzeroy) + margin)))
```

#### result of the lane lines( valid points)
![alt text][image12]

#### 5. Curvature of lane lines and Positon of vehicle

I did this in 6th code cell of the .ipyb file. The function of calculatin the curvature from a polynomial already exists, so I just need to implement it into the code.(Do not forget the order of the polynomial

#### 6. Visualization of the lane lines detected

I implemented also in code cell 7 in the .ipyb file. Using the inverse prespective transform matrix to transform the result back to the original perspective. Then use `cv2.addWeighted` function to combine it with the origianl one

# Lane Lines Visualization of the test image
![alt text][image14]

---

### Pipeline (video)

#### 1. Below is a link to the result video

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion/Improvement

#### 1. Improvement that may be applied further

I just used the result got from the sliding windows as a result and then output it. However, a filter should be used to judge whether the result is good or not. If it is bad or no valid lane line points is detected in a window, we should use the existing points to deduce and complete the missing or bad curves. It may also happen that some frams are too bad that either no lane lines could be found or the result are too bad. How to fix it from the previous frames remains a problem.

We also need to somehow synchornize the left and right lane lines fit from the valid points detected by sliding windows. Thus the curvature could be more accurate and stable.

Maybe those could explain why this program did not perform well on the challenge_video.mp4 and harder_challenge_video.mp4
