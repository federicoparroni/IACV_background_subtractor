import os
from datetime import datetime
import time
import urllib.request
import re

def timetostring():
    now = datetime.now()
    return '{}_{}_{}__{}_{}'.format(now.year, now.month, now.day, now.hour, now.minute)

downloads_folder = 'dataset'
check_every = 10 # minutes
cam_ids = [ ('salitona', 554),
            ('foreste', 1258),
            ('milanoest', 174),
            ('tunnelligure', 1296) ]
prev_hour = datetime.now().hour

request_prefix_url = 'https://www.autostrade.it/autostrade-gis/popupVideocam.do?tlc='
request_suffix_url = '&cq=LQM4&tipo=V'
# base_url = 'http://video.autostrade.it/video-mp4_hq/rav/f48ce3b8-8c06-47c6-b17f-ac1612cf4150-3.mp4'

if not os.path.isdir(downloads_folder):
    os.mkdir(downloads_folder)
    #os.makedirs(downloads_folder)

check_every *= 60
while True:
    #h = datetime.now().hour
    #if prev_hour != h:
    #    prev_hour = h
    for cam_id in cam_ids:
        resource_url = '{}{}{}'.format(request_prefix_url, cam_id[1], request_suffix_url)
        response = urllib.request.urlopen(resource_url)
        html = str(response.read())
        #print(html)

        results = re.findall('(?<=<source src=")http://video\.autostrade\.it.*?(?=")',str(html), re.IGNORECASE)
        if len(results) > 0:
            video_url = results[0]
            folder_to_save = '{}/{}'.format(downloads_folder, cam_id[0])
            video_name = '{}.mp4'.format(timetostring())
            if not os.path.isdir(folder_to_save):
                os.mkdir(folder_to_save)
            path_to_save = '{}/{}'.format(folder_to_save, video_name)
            urllib.request.urlretrieve(video_url, filename=path_to_save)
            
    #else:
    #    time.sleep(check_every)

