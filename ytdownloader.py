import pafy
import cv2
import os
import math
# import time


#url = "https://www.youtube.com/watch?v=Zwt3ZOej3_U"
DOWNLOAD_FOLDER = 'dataset'         # download folder
SAVE_VIDEO = True                   # set to false to save single frames
FPS = 30
VIDEO_CODEC = cv2.VideoWriter_fourcc(*'mp4v')
VIDEO_CHUNKS_FRAMES = 30*(FPS*60)    # save video at chunks of frames (30 min @ 25 fps)

# url = "https://www.youtube.com/watch?v=1EiC9bvVGnk"
# url = 'https://www.youtube.com/watch?v=ZTa4ap3i278'
print('\nInsert youtube video url: ')
url = input()
video = pafy.new(url)
print('> Info:')
print(video)

# best = video.getbest(preftype="mp4")
print("> Streams:")
streams = [s for s in video.streams if s.extension == 'mp4']
index = 0
for s in streams:
    print('[{}]'.format(index), s.resolution, s.extension, s.get_filesize())
    index += 1

print('Choose video resolution: ', end='')
try:
    choice = int(input()[0])
except Exception as exc:
    print('Wrong input, exiting...')
    exit(1)

stream = streams[choice]
videourl = stream.url

filename_max_length = min(20, len(video.title))
video_folder = "{}/{}".format(DOWNLOAD_FOLDER, video.title[0:filename_max_length])
if not os.path.isdir(video_folder):
    os.makedirs(video_folder)

def get_frame(capture):
    ret, frame = capture.read()
    cv2.imshow('yt', frame)
    return ret, frame

def print_progress(current, tot, prefix=''):
    scale = 60 / tot
    left = math.floor((tot-current) * scale)
    curr = math.ceil(current * scale)
    print('{}: |={}{}|'.format(prefix, '='*curr, ' '*left), end='\r')

def print_minutage(frames_count):
    tot_min = frames_count / (FPS * 60)
    print('Frame {} - Tot duration: {}min'.format(frames_count, tot_min))

shape = stream.dimensions[::-1]
if SAVE_VIDEO:
    cap = None
else:
    cap = cv2.VideoCapture()
    cap.open(videourl)

ret = True
frames_count = 0
video_chunk = 0
while True:

    if SAVE_VIDEO:
        current_frame = frames_count % VIDEO_CHUNKS_FRAMES
        # check if a new chunk must be created
        if current_frame == 0:
            if cap is None:
                cap = cv2.VideoCapture()
                cap.open(videourl)
            else:    # release the previous video session
                out.release()
            out = cv2.VideoWriter('{}/out{}.mov'.format(video_folder, video_chunk), VIDEO_CODEC, FPS, stream.dimensions)
            video_chunk += 1
            print()
        
        ret, frame = get_frame(cap)
        if frame.shape[:2] == shape:
            out.write(frame)

        print_progress(current_frame, VIDEO_CHUNKS_FRAMES, video_chunk)
    else:   # save single frames
        ret, frame = get_frame(cap)
        framename = '{}/frame{:016d}.jpg'.format(video_folder, frames_count)
        cv2.imwrite(framename, frame)
        
        print_minutage(frames_count)

    frames_count += 1
    if cv2.waitKey(10) & 0xFF == ord('q'): break
    if not cap.isOpened() or ret == False: break

if SAVE_VIDEO: out.release()
cap.release()
cv2.destroyAllWindows()