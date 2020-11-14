import numpy as np
import copy
import matplotlib.pyplot as plt
from openAEDAT import aedatUtils
import cv2


############  FOR 1 ##################
def main1():
    path = 'DVS_1.aedat'
    video = cv2.VideoCapture('video_1.mp4')
    write_obj = cv2.VideoWriter('vid//test_video_1.mkv', cv2.VideoWriter_fourcc(*'XVID'), 20, (128, 128))

    t, x, y, p = aedatUtils.loadaerdat(path)

    tI = 38300
    # tI = 38649.583550737 #50 ms for video 1

    totalImages = []
    totalImages = aedatUtils.getFramesTimeBased(t, p, x, y, tI)
    handle = None
    imageVector = []

    print('-----------------------------------------')
    print(len(totalImages))
    print(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # j = np.zeros((128, 128, 3), dtype=np.uint8)
    # l = video.read()[1]
    # k = cv2.resize(cv2.cvtColor(l, cv2.COLOR_BGR2GRAY), (128, 128))
    # for temp1 in range(len(totalImages[0])):
    #     for temp2 in range(len(totalImages[0][0])):
    #         if totalImages[0][temp1][temp2] == 0:
    #             j[temp1][temp2] = [k[temp1][temp2], 255, 0]
    #         elif totalImages[0][temp1][temp2] == 255:
    #             j[temp1][temp2] = [k[temp1][temp2], 0, 255]
    #         else:
    #             j[temp1][temp2] = cv2.resize(l, (128, 128))[temp1][temp2]
    # print(j.shape, j.dtype, j.max(), j.min())
    # print(j)
    # print(k.shape, k.dtype, k.max(), k.min())
    # print(k)
    print('-----------------------------------------')

    for f in totalImages:
        ret, frame = video.read()
        if ret:
            mat_1 = np.array(([1, 0, 20], [0, 1, 25]), dtype=np.float32)
            l = cv2.warpAffine((cv2.resize(cv2.rotate(cv2.flip(frame, 1), cv2.ROTATE_90_COUNTERCLOCKWISE), (128, 128))), mat_1, (128, 128))
            j = np.zeros((128, 128, 3), dtype=np.uint8)
            k = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
            for temp1 in range(len(f)):
                for temp2 in range(len(f[0])):
                    if f[temp1][temp2] == 0:
                        j[temp1][temp2] = [k[temp1][temp2], 255, 0]
                    elif f[temp1][temp2] == 255:
                        j[temp1][temp2] = [k[temp1][temp2], 0, 255]
                    else:
                        j[temp1][temp2] = l[temp1][temp2]
            j = cv2.fastNlMeansDenoisingColored(j, None, 10, 10, 7, 15)
            cv2.imshow('Output', j)
            write_obj.write(j)
            if cv2.waitKey(1) & 0xff == 27:
                break
    #     f = f.astype(np.uint8)
    #     imagem = copy.deepcopy(f)
    #     if handle is None:
    #         plt.subplot(121)
    #         plt.imshow(np.dstack([f, f, f]))
    #         plt.subplot(122)
    #         plt.imshow(cv2.flip(cv2.rotate(video.read()[1], cv2.ROTATE_90_CLOCKWISE), 1))  # for 1
    #     else:
    #         handle.set_data(np.dstack([f, f, f]))
    #
    #     plt.pause(tI / 1000000)
    #     plt.draw()
    write_obj.release()
    cv2.destroyAllWindows()


############  FOR 2 ##################
def main2():
    path = 'DVS_2.aedat'
    video = cv2.VideoCapture('video_2.mp4')
    write_obj = cv2.VideoWriter('vid//test_video_2.mkv', cv2.VideoWriter_fourcc(*'XVID'), 20, (128, 128))

    t, x, y, p = aedatUtils.loadaerdat(path)

    tI = 35000
    # tI = 28507.165275992 #for video 2

    totalImages = []
    totalImages = aedatUtils.getFramesTimeBased(t, p, x, y, tI)
    handle = None
    imageVector = []

    print('-----------------------------------------')
    print(len(totalImages))
    print(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # j = np.zeros((128, 128, 3), dtype=np.uint8)
    # l = video.read()[1]
    # k = cv2.resize(cv2.cvtColor(l, cv2.COLOR_BGR2GRAY), (128, 128))
    # for temp1 in range(len(totalImages[0])):
    #     for temp2 in range(len(totalImages[0][0])):
    #         if totalImages[0][temp1][temp2] == 0:
    #             j[temp1][temp2] = [k[temp1][temp2], 255, 0]
    #         elif totalImages[0][temp1][temp2] == 255:
    #             j[temp1][temp2] = [k[temp1][temp2], 0, 255]
    #         else:
    #             j[temp1][temp2] = cv2.resize(l, (128, 128))[temp1][temp2]
    # print(j.shape, j.dtype, j.max(), j.min())
    # print(j)
    # print(k.shape, k.dtype, k.max(), k.min())
    # print(k)
    print('-----------------------------------------')

    for f in totalImages:
        ret, frame = video.read()
        if ret:
            # l = cv2.flip(cv2.rotate(cv2.resize(frame, (128, 128)), cv2.ROTATE_90_CLOCKWISE), 1)
            mat_1 = np.array(([1, 0, 20], [0, 1, 28]), dtype=np.float32)
            l = cv2.warpAffine((cv2.flip(cv2.rotate(cv2.resize(frame, (128, 128)), cv2.ROTATE_90_COUNTERCLOCKWISE), 0)), mat_1, (128, 128))
            j = np.zeros((128, 128, 3), dtype=np.uint8)
            k = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
            for temp1 in range(len(f)):
                for temp2 in range(len(f[0])):
                    if f[temp1][temp2] == 0:
                        j[temp1][temp2] = [k[temp1][temp2], 255, 0]
                    elif f[temp1][temp2] == 255:
                        j[temp1][temp2] = [k[temp1][temp2], 0, 255]
                    else:
                        j[temp1][temp2] = l[temp1][temp2]
            j = cv2.fastNlMeansDenoisingColored(j, None, 10, 10, 7, 15)
            cv2.imshow('Output', j)
            write_obj.write(j)
            if cv2.waitKey(1) & 0xff == 27:
                break
    #     f = f.astype(np.uint8)
    #     imagem = copy.deepcopy(f)
    #     if handle is None:
    #         plt.subplot(121)
    #         plt.imshow(np.dstack([f, f, f]))
    #         plt.subplot(122)
    #         plt.imshow(cv2.flip(cv2.rotate(video.read()[1], cv2.ROTATE_90_CLOCKWISE), 1))  # for 1
    #     else:
    #         handle.set_data(np.dstack([f, f, f]))
    #
    #     plt.pause(tI / 1000000)
    #     plt.draw()
    write_obj.release()
    cv2.destroyAllWindows()


############  FOR 3 ##################
def main3():
    path = 'DVS_3.aedat'
    video = cv2.VideoCapture('video_3.mp4')
    write_obj = cv2.VideoWriter('vid//test_video_3.mkv', cv2.VideoWriter_fourcc(*'XVID'), 20, (128, 128))

    t, x, y, p = aedatUtils.loadaerdat(path)

    tI = 46800
    # tI = 47291.638093820 #for video 3

    totalImages = []
    totalImages = aedatUtils.getFramesTimeBased(t, p, x, y, tI)
    handle = None
    imageVector = []

    print('-----------------------------------------')
    print(len(totalImages))
    print(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # j = np.zeros((128, 128, 3), dtype=np.uint8)
    # l = video.read()[1]
    # k = cv2.resize(cv2.cvtColor(l, cv2.COLOR_BGR2GRAY), (128, 128))
    # for temp1 in range(len(totalImages[0])):
    #     for temp2 in range(len(totalImages[0][0])):
    #         if totalImages[0][temp1][temp2] == 0:
    #             j[temp1][temp2] = [k[temp1][temp2], 255, 0]
    #         elif totalImages[0][temp1][temp2] == 255:
    #             j[temp1][temp2] = [k[temp1][temp2], 0, 255]
    #         else:
    #             j[temp1][temp2] = cv2.resize(l, (128, 128))[temp1][temp2]
    # print(j.shape, j.dtype, j.max(), j.min())
    # print(j)
    # print(k.shape, k.dtype, k.max(), k.min())
    # print(k)
    print('-----------------------------------------')

    for f in totalImages:
        ret, frame = video.read()
        if ret:
            # mat_1 = np.array(([1, 0, 20], [0, 1, 28]), dtype=np.float32)
            # l = cv2.warpAffine((cv2.flip(cv2.resize(frame, (128, 128)), 1)), mat_1, (128, 128))
            l = cv2.flip(cv2.resize(frame, (128, 128)), 1)
            j = np.zeros((128, 128, 3), dtype=np.uint8)
            k = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
            for temp1 in range(len(f)):
                for temp2 in range(len(f[0])):
                    if f[temp1][temp2] == 0:
                        j[temp1][temp2] = [k[temp1][temp2], 255, 0]
                    elif f[temp1][temp2] == 255:
                        j[temp1][temp2] = [k[temp1][temp2], 0, 255]
                    else:
                        j[temp1][temp2] = l[temp1][temp2]
            j = cv2.fastNlMeansDenoisingColored(j, None, 10, 10, 7, 15)
            cv2.imshow('Output', j)
            write_obj.write(j)
            if cv2.waitKey(1) & 0xff == 27:
                break
    #     f = f.astype(np.uint8)
    #     imagem = copy.deepcopy(f)
    #     if handle is None:
    #         plt.subplot(121)
    #         plt.imshow(np.dstack([f, f, f]))
    #         plt.subplot(122)
    #         plt.imshow(cv2.flip(cv2.rotate(video.read()[1], cv2.ROTATE_90_CLOCKWISE), 1))  # for 1
    #     else:
    #         handle.set_data(np.dstack([f, f, f]))
    #
    #     plt.pause(tI / 1000000)
    #     plt.draw()
    write_obj.release()
    cv2.destroyAllWindows()


############  FOR 4 ##################
def main4():
    path = 'DVS_4.aedat'
    video = cv2.VideoCapture('video_4.mp4')
    write_obj = cv2.VideoWriter('vid//test_video_4.mkv', cv2.VideoWriter_fourcc(*'XVID'), 20, (128, 128))

    t, x, y, p = aedatUtils.loadaerdat(path)

    # tI = 46800
    tI = 37053.505261598 #for video 4

    totalImages = []
    totalImages = aedatUtils.getFramesTimeBased(t, p, x, y, tI)
    handle = None
    imageVector = []

    print('-----------------------------------------')
    print(len(totalImages))
    print(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # j = np.zeros((128, 128, 3), dtype=np.uint8)
    # l = video.read()[1]
    # k = cv2.resize(cv2.cvtColor(l, cv2.COLOR_BGR2GRAY), (128, 128))
    # for temp1 in range(len(totalImages[0])):
    #     for temp2 in range(len(totalImages[0][0])):
    #         if totalImages[0][temp1][temp2] == 0:
    #             j[temp1][temp2] = [k[temp1][temp2], 255, 0]
    #         elif totalImages[0][temp1][temp2] == 255:
    #             j[temp1][temp2] = [k[temp1][temp2], 0, 255]
    #         else:
    #             j[temp1][temp2] = cv2.resize(l, (128, 128))[temp1][temp2]
    # print(j.shape, j.dtype, j.max(), j.min())
    # print(j)
    # print(k.shape, k.dtype, k.max(), k.min())
    # print(k)
    print('-----------------------------------------')

    for f in totalImages:
        ret, frame = video.read()
        if ret:
            mat_1 = np.array(([1, 0, 0], [0, 1, 0]), dtype=np.float32)
            l = cv2.warpAffine((cv2.flip(cv2.resize(frame, (128, 128)), 1)), mat_1, (128, 128))
            j = np.zeros((128, 128, 3), dtype=np.uint8)
            k = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
            for temp1 in range(len(f)):
                for temp2 in range(len(f[0])):
                    if f[temp1][temp2] == 0:
                        j[temp1][temp2] = [k[temp1][temp2], 255, 0]
                    elif f[temp1][temp2] == 255:
                        j[temp1][temp2] = [k[temp1][temp2], 0, 255]
                    else:
                        j[temp1][temp2] = l[temp1][temp2]
            j = cv2.fastNlMeansDenoisingColored(j, None, 10, 10, 7, 15)
            cv2.imshow('Output', j)
            write_obj.write(j)
            if cv2.waitKey(1) & 0xff == 27:
                break
    #     f = f.astype(np.uint8)
    #     imagem = copy.deepcopy(f)
    #     if handle is None:
    #         plt.subplot(121)
    #         plt.imshow(np.dstack([f, f, f]))
    #         plt.subplot(122)
    #         plt.imshow(cv2.flip(cv2.rotate(video.read()[1], cv2.ROTATE_90_CLOCKWISE), 1))  # for 1
    #     else:
    #         handle.set_data(np.dstack([f, f, f]))
    #
    #     plt.pause(tI / 1000000)
    #     plt.draw()
    write_obj.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main1()
    main2()
    main3()
    main4()

