clc; close all; clear V;

% Create system objects to read file.
videoReader = vision.VideoFileReader('salitona.mp4', 'VideoOutputDataType','uint8');
% w = 640;
% h = 360;
h = 288;
w = 352;

videoPlayer = vision.VideoPlayer();
t = 1;
while ~isDone(videoReader)
    frame  = step(videoReader);
    V(:,:,:,t) = frame;
    t = t + 1;
end

% compute the background estimation
bg = uint8(zeros(h,w,3));
bg(:,:,1) = median(V(:,:,1,:), 4);
bg(:,:,2) = median(V(:,:,2,:), 4);
bg(:,:,3) = median(V(:,:,3,:), 4);
figure('Position',[100, 200, 600, 400]); imshow(bg);


videoPlayer.reset();
reset(videoReader);
cont = ~isDone(videoReader);
%f2 = figure('Position',[450, 200, 600, 400]);
while cont
    frame  = step(videoReader);
    frame2 = frame;
    
    mask = abs(frame - bg);
    %mask = sum(mask,3);
    frame2(mask > 5) = 0;
    
    step(videoPlayer, frame);
    %figure(f2); imshow(frame2);
    
    pause(0.02);
    cont = ~isDone(videoReader) && isOpen(videoPlayer);
end
release(videoPlayer);
release(videoReader);
