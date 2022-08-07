
%% Object Recognition and Tracking for Augmented Reality in Matlab
% A webcam is needed.
% Extracted features are SURF.
% Tracking algorithm is KLT.
% You can use your own reference image and overlayed video.
% “NR” is number of repeats. By increasing it, stream gets longer. 
% Enjoy 😊


%% Load reference image, detect SURF points, and extract descriptors
clear;
warning('off');
referenceimage = imread('ts.jpg'); 

%% Detect and extract SURF features 
referenceimageGray = rgb2gray(referenceimage); 
referencePts = detectSURFFeatures(referenceimageGray); 
referenceFeatures = extractFeatures(referenceimageGray, referencePts); 

%% Display SURF features for a reference image 
% figure; 
% imshow(referenceimage); hold on; 
% plot(referencePts.selectStrongest(50)); 

%% Initialize replacement video 
video = vision.VideoFileReader('v3.mp4','VideoOutputDataType', 'uint8'); 

% Skip past the first few black frames
for k = 1:30 
    step(video);
end 

%% Prepare video input from webcam 
camera = webcam('USB2.0 PC CAMERA'); 
set(camera, 'Resolution', '640x480'); 

%% Detect SURF features in webcam frame

NR=300;% Number of repeats 
for i=1:NR

cameraFrame = snapshot(camera); 
cameraFrameGray = rgb2gray(cameraFrame);
cameraPts = detectSURFFeatures(cameraFrameGray); 
% figure(1)
% imshow(cameraFrame), hold on;
% plot(cameraPts.selectStrongest(500)); 


%% Try to match the reference image and camera frame features 
cameraFeatures = extractFeatures(cameraFrameGray, cameraPts); 
idxPairs = matchFeatures(cameraFeatures, referenceFeatures); 
% Store the SURF points that were matched 
matchedCameraPts = cameraPts(idxPairs(:,1)); 
matchedReferencePts = referencePts(idxPairs(:,2)); 
% figure(1)
% showMatchedFeatures(cameraFrame, referenceimage, matchedCameraPts, matchedReferencePts, 'Montage'); 

%% Get geometric transformation between reference image and webcam frame
[referenceTransform, inlierReferencePts, inlierCameraPts] ...
    = estimateGeometricTransform( matchedReferencePts, matchedCameraPts, 'affine'); 
% Show the inliers of the estimated geometric transformation
% figure(1) 
% showMatchedFeatures(cameraFrame, referenceimage, inlierCameraPts, inlierReferencePts, 'Montage');

%% Rescale replacement video frame 
% Load replacement video frame 
videoFrame = step(video); 
% Get replacement and reference dimensions 
repDims = size(videoFrame(:,:,1)); 
refDims = size(referenceimage); 
% Find transformation that scales video frame to replacement image size
% preserving aspect ratio 
scaleTransform = findScaleTransform(refDims,repDims); 
%
outputView = imref2d(size(referenceimage));
videoFrameScaled = imwarp(videoFrame, scaleTransform, 'outputView', outputView); 

% figure(1)
% imshowpair(referenceimage,videoFrameScaled, 'montage');

%% Apply estimated geometric transform to scaled replacement video frame 
outputView = imref2d(size(cameraFrame)); 
videoFrameTransformed = imwarp(videoFrameScaled, referenceTransform,'OutputView', outputView);

% figure(1)
% imshowpair(cameraFrame, videoFrameTransformed, 'Montage'); 

%% Insert transformed replacement video frame into webcam frame 
alphaBlender = vision.AlphaBlender( 'Operation', 'Binary mask', 'MaskSource', 'Input port'); 
mask = videoFrameTransformed(:,:,1) | ... 
       videoFrameTransformed(:,:,2) | ... 
       videoFrameTransformed(:,:,3) > 0 ; 
outputFrame = step(alphaBlender, cameraFrame, videoFrameTransformed, mask); 

% figure(1) 
% imshow(outputFrame); 

%% Initialize Point Tracker  
pointTracker=vision.PointTracker('MaxBidirectionalError', 2); 
initialize(pointTracker, inlierCameraPts.Location, cameraFrame); 
% Display the points being used for tracking
trackingMarkers = insertMarker(cameraFrame, inlierCameraPts.Location, 'Size', 7, 'Color', 'yellow'); 

% figure(1)
% imshow(trackingMarkers); 

%% Track points to next frame 
% Store previous frame just for visual comparison 
prevCameraFrame = cameraFrame; 
% Get next webcam frame 
cameraFrame = snapshot(camera); 
% Find newly tracked points
[trackedPoints, isValid] = step(pointTracker, cameraFrame); 
% Use only the locations that have been reliably tracked 
newValidLocations = trackedPoints(isValid,:); 
oldValidLocations = inlierCameraPts.Location(isValid,:); 

%% Estimate geometric transformation between two frames 
% if (nnz(isValid) >= 2) % Must have at least 2 tracked points between frames
   [trackingTransform, oldlnlierLocations, newInlierLocations] = ...
       estimateGeometricTransform( oldValidLocations, newValidLocations, 'affine');
% end 
% Show the valid of the geometric transformation 
% figure(1) 
% showMatchedFeatures(prevCameraFrame, cameraFrame, oldlnlierLocations, newInlierLocations, 'Montage'); 

% Reset Point Tracker for tracking in next frame 
setPoints(pointTracker, newValidLocations); 

%% Accumulate geometric transformations from reference to current frame 
trackingTransform.T = referenceTransform.T * trackingTransform.T; 

%% Rescale new replacement video frame 
repFrame = step(video); 
outputView = imref2d(size(referenceimage));
videoFrameScaled = imwarp(videoFrame, scaleTransform, 'OutputView', outputView);

% figure(1)
% imshowpair(referenceimage, videoFrameScaled, 'Montage'); 

%% Apply total geometric transformation to new replacement video frame 
outputView = imref2d(size(cameraFrame));
videoFrameTransformed = imwarp(videoFrameScaled, trackingTransform, 'OutputView', outputView); 
% figure(1)
% imshowpair(cameraFrame, videoFrameTransformed, 'Montage'); 

%% Insert transformed replacement frame into webcam input 
mask = videoFrameTransformed(:,:,1) | ... 
       videoFrameTransformed(:,:,2) | ... 
       videoFrameTransformed(:,:,3) > 0 ; 

outputFrame = step(alphaBlender, cameraFrame, videoFrameTransformed, mask); 

% figure(1)
% imshowpair(cameraFrame, outputFrame, 'Montage'); 

imshow(outputFrame);

end

%% Final cleanup
release(video);
delete(camera);
