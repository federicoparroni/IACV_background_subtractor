import os
from datetime import datetime
import time
import urllib.request
import re

def timetostring():
    now = datetime.now()
    return '{}_{}_{}__{}_{}'.format(now.year, now.month, now.day, now.hour, now.minute)

downloads_folder = 'dataset'
download_every = 60*10 # seconds
check_every = 60 # seconds
cam_ids = [ ('salitona', 554),
            ('foreste', 1258),
            ('milanoest', 174),
            #('tunnelligure', 1296),
            ('traffico', 1039) ]
prev_time = 0

request_prefix_url = 'https://www.autostrade.it/autostrade-gis/popupVideocam.do?tlc='
request_suffix_url = '&cq=LQM4&tipo=V'

if not os.path.isdir(downloads_folder):
    os.mkdir(downloads_folder)
    #os.makedirs(downloads_folder)

while True:
    t = time.time()
    if t >= prev_time + download_every:
        prev_time = t
        print('\n{}'.format(datetime.now()))

        n = 0
        for cam_id in cam_ids:
            resource_url = '{}{}{}'.format(request_prefix_url, cam_id[1], request_suffix_url)
            response = urllib.request.urlopen(resource_url)
            html = str(response.read())

            results = re.findall('(?<=<source src=")http://video\.autostrade\.it.*?(?=")',str(html), re.IGNORECASE)
            if len(results) > 0:
                video_url = results[0]
                folder_to_save = '{}/{}'.format(downloads_folder, cam_id[0])
                video_name = '{}.mp4'.format(timetostring())
                if not os.path.isdir(folder_to_save):
                    os.mkdir(folder_to_save)
                path_to_save = '{}/{}'.format(folder_to_save, video_name)

                print('({}/{}) Downloading {} (cam_id: {})...'.format(n,len(cam_ids), cam_id[1], cam_id[0]), end='\t')
                urllib.request.urlretrieve(video_url, filename=path_to_save)
                print('ok')
            else:
                print('Cannot get video url from html:\n{}'.format(html))

            n+= 1
        print('Next download at {}'.format(datetime.fromtimestamp(t+download_every)))
    else:
        time.sleep(check_every)

