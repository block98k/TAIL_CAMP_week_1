import os

video_path='/home/kk/TAIL_week_1/datasets/imgs/'
videos = os.listdir(video_path)
f = open('TrainTestFileList/classes.txt','w')
for i,video in enumerate(videos):
    f.write(str(i)+' '+str(video)+'\n')
f.close


