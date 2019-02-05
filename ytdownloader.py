import pafy
import cv2
import os
# import time

#url = "https://www.youtube.com/watch?v=Zwt3ZOej3_U"
DOWNLOAD_FOLDER = 'dataset'
SAVE_VIDEO = True   # set to false to save single frames
FPS = 20

url = "https://www.youtube.com/watch?v=1EiC9bvVGnk"
video = pafy.new(url)
print("- Info:")
print(video)

# best = video.getbest(preftype="mp4")
print("- Streams:")
streams = [s for s in video.streams if s.extension == 'mp4']
index = 0
for s in streams:
    print('[{}]'.format(index), s.resolution, s.extension, s.get_filesize())
    index += 1

print('Choose video resolution: ', end='')
choice = int(input()[0])

stream = streams[choice]
videourl = stream.url

filename_max_length = min(20, len(video.title))
video_folder = "{}/{}".format(DOWNLOAD_FOLDER, video.title[0:filename_max_length])
if not os.path.isdir(video_folder):
    os.makedirs(video_folder)

if SAVE_VIDEO:
    output_codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('{}/out.mov'.format(video_folder), output_codec, FPS, stream.dimensions)

cap = cv2.VideoCapture()
cap.open(videourl)

ret = True
i = 0
while cap.isOpened() and ret:
    ret, frame = cap.read()

    cv2.imshow('yt', frame)

    if SAVE_VIDEO:
        #cv2.flip(frame,0)
        out.write(frame)
    else:
        framename = '{}/frame{:016d}.jpg'.format(video_folder, i)
        cv2.imwrite(framename, frame)
        i += 1

    if cv2.waitKey(10) & 0xFF == ord('q'): break

if SAVE_VIDEO: out.release()
cap.release()
cv2.destroyAllWindows()