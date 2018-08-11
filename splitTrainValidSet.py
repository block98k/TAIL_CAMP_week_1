import os
from tqdm import tqdm

img_path = '/home/kk/TAIL_week_1/datasets/imgs/'
f1 = open('TrainTestFileList/trainfilelist.txt','w')
f2 = open('TrainTestFileList/validfilelist.txt','w')

action_list = os.listdir(img_path)

ratio = 0.8

label = 0

for action in tqdm(action_list):
    actionimg_path = img_path+action
    videos = os.listdir(actionimg_path)
    videosnum = len(videos)
    for i,video in enumerate(videos):
        if i<videosnum*ratio:
            f1.write(action+'/'+video+' '+str(label)+'\n')
        else:
            f2.write(action+'/'+video+' '+str(label)+'\n')
    label+=1
    


f1.close()
f2.close()

