import pafy
import cv2
# import time

url = "https://www.youtube.com/watch?v=Zwt3ZOej3_U"
video = pafy.new(url)
print("- Info:")
print(video)

# print("- Best:")
# best = video.getbest(preftype="mp4")
# print(best)

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
#exit()

filename_max_length = min(20, len(video.title))
name = video.title[0:filename_max_length]
#stream.download(filepath=video.title[0:filename_max_length], quiet=False)
#stream.download(filepath='dataset/' + name, quiet=False)

# =======
cap = cv2.VideoCapture()
cap.open(videourl)

while True:
    ret, frame = cap.read()

    cv2.imshow('yt', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()