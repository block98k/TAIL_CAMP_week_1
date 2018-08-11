# coding=utf8
from models import c3d_model
from keras.optimizers import SGD
import numpy as np
import cv2


def main():
    with open('./TrainTestFileList/classes.txt', 'r') as f:
        class_names = f.readlines()
        f.close()

    # init model
    model = c3d_model()
    lr = 0.005
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    model.load_weights('C3D01--3.766.hdf5', by_name=True)

    # read video
    video = './videos/Best_Of_Skype_Laughter_Chain_laugh_h_nm_np1_fr_goo_13.avi'
    cap = cv2.VideoCapture(video)

    clip = []
    while True:
        ret, frame = cap.read()
        if ret:
            tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            clip.append(cv2.resize(tmp, (120, 90)))
            if len(clip) == 16:
                inputs = np.array(clip).astype(np.float32)
                inputs = np.expand_dims(inputs, axis=0)
                inputs /=255.
                inputs -= 0.5
                inputs *=2.
                inputs = np.transpose(inputs, (0, 2, 3, 1, 4))
                pred = model.predict(inputs)
                label = np.argmax(pred[0])
                cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 1)
                cv2.putText(frame, "prob: %.4f" % pred[0][label], (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 1)
                clip.pop(0)
            cv2.imshow('result', frame)
            cv2.waitKey(10)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
