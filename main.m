clc;

% Detect Moving Cars In Video
% Create system objects to read file.
videoSource = vision.VideoFileReader('salitona.mp4',...
    'VideoOutputDataType','uint8');

% Setting frames to 5 because it is a short video. Set initial standard deviation.
g = 10;
detector = vision.ForegroundDetector(...
       'NumTrainingFrames', g, ... 
       'InitialVariance', 30*30);

% Perform blob analysis.
blob = vision.BlobAnalysis(...
       'CentroidOutputPort', false, 'AreaOutputPort', false, ...
       'BoundingBoxOutputPort', true, ...
       'MinimumBlobAreaSource', 'Property', 'MinimumBlobArea', 250);

% Insert a border.
shapeInserter = vision.ShapeInserter('BorderColor','White');

%T = 10;
finalMask = zeros(288,352);
finalFrame = uint8(zeros(288,352,3));
thr = 0.6;

f1 = figure('Position',[100, 200, 600, 400]); imshow(finalMask);

% Play results. Draw bounding boxes around cars.
videoPlayer = vision.VideoPlayer();
%videoPlayerSubtractor = vision.VideoPlayer();
t = 1;
while ~isDone(videoSource)
%     if isDone(videoSource)
%         videoPlayer.reset();
%     end
    frame  = step(videoSource);
    fgMask = step(detector, frame);
    bgMask = ~fgMask;
    
    %finalMask = finalMask + (bgMask - finalMask)/t;
    if t > g
        tempMask = finalMask == 0 & bgMask == 1;
        %finalMask = (1-a)*finalMask + a*tempMask;
        finalMask = finalMask + tempMask;
        finalFrame = finalFrame + bsxfun(@times, frame, cast(tempMask, 'like', frame));

        figure(f1);
        imshow(finalFrame);
    end
    
    bbox   = step(blob, fgMask);
    out    = step(shapeInserter, frame, bbox);
    %outBG = finalFrame;
    step(videoPlayer, out);
    %step(videoPlayerSubtractor, finalFrame);

    t = t + 1;
    pause(5);
end

% Release objects.
release(videoPlayer);
%release(videoPlayerSubtractor);
release(videoSource);
