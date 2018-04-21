import cv2
import numpy as np

writer = cv2.VideoWriter('test1.avi', cv2.VideoWriter_fourcc(*'PIM1'), 25, (640, 480))
for i in range(100):
    x = np.random.randint(255, size=(480, 640, 3)).astype('uint8')
    writer.write(x)
    break;
