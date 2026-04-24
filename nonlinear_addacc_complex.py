import numpy as np
import argparse
import os
import copy
import sys
from datetime import datetime

path = os.path.dirname(os.path.dirname(__file__))+r"\simulate"
sys.path.append(path)

# scenario 2
def scenario_2(physical_width, physical_height, frame, trackID, seed):
    
    np.random.seed(seed)
    # Frame to which the current track belongs
    frame = frame
    # Trajectory to which the point belongs
    trackID = trackID

    # make sure those point not be on the image egde =>  physical_width/8, physical_width - (physical_width/8)
    x = np.random.uniform(physical_width/8, physical_width - (physical_width/8), 1)
    z = np.random.uniform(physical_height/8, physical_height-(physical_height/8), 1)

    # velocity
    v = np.random.uniform(10.0,15.0)
    # acceleration
    a = np.random.uniform(0.5,1.5)
    # inital phase 0~2Π
    pha = np.random.uniform(0,2) * np.pi
    # angular velocity -10Π/s~10Π/s.     w*delta_t  = w*0.02 
    w = np.random.uniform(-10, 10) * np.pi   
    # 1 = true(this point is belong to the track)   0 = false
    label = 1

    state_space = np.array([frame, trackID, x, z, v, pha, w, a, label], dtype=object).reshape((1, 9))
    return state_space



# non linear motion
def non_linear_iterate(state_space, delta_t, frame):

    x = state_space[:, 2]
    z = state_space[:, 3]
    v = state_space[:, 4]
    pha = state_space[:, 5]
    w = state_space[:, 6]
    a = state_space[:, 7]

    pha_tmp = (pha + w * delta_t) 
    r_tmp = v / w

    
    x = x + r_tmp * (np.sin(pha.astype('float')) - np.sin(pha_tmp.astype('float'))) + np.random.normal(0,0.01,1)
    z = z + r_tmp * (np.cos(pha_tmp.astype('float')) - np.cos(pha.astype('float'))) + np.random.normal(0,0.01,1)
    
    pha = pha_tmp
    v = v + a * delta_t
    w = v / r_tmp
    label = 1

    state_space[:, 0] = frame
    state_space[:, 2] = x
    state_space[:, 3] = z
    state_space[:, 4] = v
    state_space[:, 5] = pha
    state_space[:, 6] = w
    state_space[:, 7] = a
    state_space[:, 8] = label
    
    return state_space



# linear motion
def linear_iterate(state_space, delta_t, frame):

    x = state_space[:, 2]
    z = state_space[:, 3]
    v = state_space[:, 4]
    pha = state_space[:, 5]
    w = state_space[:, 6]
    a = state_space[:, 7]

    x = x + v*delta_t
    z = z + v*delta_t
    label = 1

    state_space[:, 0] = frame
    state_space[:, 2] = x
    state_space[:, 3] = z
    state_space[:, 8] = label

    return state_space



# To create false point  
def false_candidate(state_space, frame):
    delta_t = 0.2
    
    x = state_space[:, 2] 
    z = state_space[:, 3]
    v = state_space[:, 4]
    label = state_space[:,-1]

    # sudden change of velocity to simulate false point  
    x = x + v*delta_t
    z = z + v*delta_t
    label = 0

    state_space[:, 0] = frame
    state_space[:, 2] = x
    state_space[:, 3] = z
    state_space[:, -1] = label
    
    return state_space


# generate a track with length
def generate_tracks(physical_width, physical_height, delta_t, nums, track_min_length, track_max_length, pixel, scenario, imgs, total_frames, max_disp_frame, noise_ratio=0.1):

    # img_widths = int(physical_width/pixel)
    # img_height = int(physical_height/pixel)

    # for i in range(total_frames):
    #     img = np.zeros((img_widths, img_height, 3), np.uint8)
    #     imgs.append(img)

    tracks = np.zeros((0,9))  # 保存完整的轨迹
    tracks_temp = np.zeros((0,9))  # 保存带有消失点的轨迹
    frame_points_count = np.zeros(total_frames+1, dtype=int) 
    for num in range(nums):
        trackID = num + 1
        # seed = datetime.now().microsecond   
        seed = trackID + datetime.now().microsecond + datetime.now().second
        # Randomly set the initial length of the trajectory : [10, 30)
        length = np.random.randint(track_min_length, track_max_length)
        if length >= total_frames:
            length = np.random.randint(track_min_length, total_frames+1)
            start_frame = 1
        else:
            # Randomly set the initial frame
            start_frame = np.random.randint(1, total_frames-length+1)  

        flag = False   # 标记该轨迹是否有消失点
        disp = np.random.randint(1, max_disp_frame)  # 随机生成消失帧数
        if disp/length*1.0 > 0.5:  # 确保实际消失帧数<<轨迹长度
            disp = length - disp 
        start_disp_frame = np.random.randint(start_frame+3, start_frame+length-disp+1)  # 随机生成起始消失帧数并确保起始消失帧数不超过轨迹的结束帧数        

        for k in range(start_frame, start_frame+length):
            frame = k
            if frame == start_frame:
                ss = scenario_2(physical_width, physical_height, frame, trackID, seed)
                track_true  = copy.copy(ss)
                track_true_temp  = copy.copy(ss)
                # visual track motion
                # true_width = int(ss[0, 2]/pixel)
                # true_height = int(ss[0, 3]/pixel)
                # imgs[k] = visual(imgs[k], img_widths, img_height,true_width, true_height, "white")
                # cv.imshow("Frame:"+str(k+1), imgs[k])
                # cv.waitKey(-1)
                frame_points_count[frame] += 1
                continue

            # ID为3的倍数的轨迹从任意帧开始消失
            if trackID % 3 == 0 and start_disp_frame <= k < start_disp_frame + disp:
                flag = True  # 标志轨迹有消失点
                ss_temp = np.hstack((np.reshape(np.array([frame, trackID]),(1,2)), np.full((1, 7), -1)))
                track_true_temp = np.concatenate((track_true_temp, ss_temp), axis=0)

                # 假设轨迹在继续前进         
                if trackID % 2 == 0:
                    ss = non_linear_iterate(ss, delta_t, frame)
                else:
                    ss = linear_iterate(ss, delta_t, frame)
                track_true = np.concatenate((track_true, ss), axis=0)
                frame_points_count[frame] += 1
                continue

            else:
                # 若轨迹ID是偶数，则为非线性轨迹
                if trackID % 2 == 0:
                    ss = non_linear_iterate(ss, delta_t, frame)
                else:
                # 若轨迹ID是奇数，则为线性轨迹
                    ss = linear_iterate(ss, delta_t, frame)
                track_true = np.concatenate((track_true, ss), axis=0)
                track_true_temp = np.concatenate((track_true_temp, ss), axis=0)
                frame_points_count[frame] += 1
            
    
            # visual track motion
            # true_width = int(ss[0, 2]/pixel)
            # true_height = int(ss[0, 3]/pixel)
            # imgs[k] = visual(imgs[k], img_widths, img_height, true_width, true_height, "white")



        tracks = np.concatenate((tracks, track_true))
        tracks_temp = np.concatenate((tracks_temp, track_true_temp))

    # 生成噪声点
    print(f"Generating noise points with ratio {noise_ratio}...")
    noise_tracks = np.zeros((0,9))
    for frame in range(1, total_frames + 1):
        # 计算该帧应该生成的噪声点数量
        true_points_in_frame = frame_points_count[frame]
        noise_count = int(true_points_in_frame * noise_ratio)

        if noise_count > 0:
            # 从该帧的所有真实轨迹点中随机选择作为噪声点的基础
            frame_mask = tracks[:, 0] == frame
            frame_true_points = tracks[frame_mask]

            if len(frame_true_points) > 0:
                # 随机选择噪声点的基础
                selected_indices = np.random.choice(len(frame_true_points),
                                                  min(noise_count, len(frame_true_points)),
                                                  replace=False)

                for idx in selected_indices:
                    base_point = frame_true_points[idx:idx+1].copy()
                    noise_point = false_candidate(base_point, frame)
                    noise_tracks = np.concatenate((noise_tracks, noise_point))

                frame_points_count[frame] += len(selected_indices)

    # 将噪声点加入到轨迹数据中
    tracks = np.concatenate((tracks, noise_tracks))
    tracks_temp = np.concatenate((tracks_temp, noise_tracks))  # 噪声点也加入complex版本

    print(f"Generated {len(noise_tracks)} noise points across {total_frames} frames")

    # Save imgs
    # for i in range(length):
    #     # cv.imshow("Frame: "+str(i+1), imgs[i])
    #     # cv.waitKey(-1)
    #     cv.imwrite("./dataset/Imgs/{}.png".format(i+1), imgs[i])
    os.makedirs(os.path.join(os.getcwd(), 'dataset'), exist_ok=True)
    np.savetxt(os.path.join(os.getcwd(), 'dataset', 'data_density_32000.txt'), frame_points_count)

    return tracks, tracks_temp


def main(scenario):

    if scenario != '2':
        print("Only scenario 2 is supported")
        exit(0)

    # 设置基于当前目录的路径
    base_dir = os.path.join(os.getcwd(), 'simulate', 'data', 'nonlinear', 'scenario_2')
    os.makedirs(base_dir, exist_ok=True)

    dataset_dir = os.path.join(os.getcwd(), 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'test'), exist_ok=True)

    train_path = os.path.join(base_dir, 'train')
    train_temp_path = os.path.join(base_dir, 'train_complex')
    test_path = os.path.join(base_dir, 'test')
    test_temp_path = os.path.join(base_dir, 'test_complex')
    
    physical_width = 15  # 15mm
    physical_height = 15   
    delta_t = 0.02  # 20ms
    pixel = 0.0192

    # tracks for train + validate
    train_num = 40000        # train: validate : test = 4:2:1
    # tracks for test
    test_num = 40000

    train_tracks = {}
    train_imgs = []
    test_tracks = {}
    test_imgs = []

    train_total_frames = 2000  
    test_total_frames = 2000
     
    track_min_length = 10     # 10-30
    track_max_length = 30

    max_disp_frame = 3

    # 噪声点生成参数
    noise_ratio = 0.1  # 每帧噪声点占真实轨迹点的比例

    # 1. 生成轨迹
    train_tracks,train_tracks_temp  = generate_tracks(physical_width, physical_height, delta_t, train_num, track_min_length, track_max_length, pixel, scenario, train_imgs, train_total_frames, max_disp_frame, noise_ratio)
    np.save(train_path, train_tracks)
    np.save(train_temp_path, train_tracks_temp)

    # 2. 生成图片
    # for i in range(train_track_length):
    #     cv.imshow("Frame: "+str(i+1), train_imgs[i])
    #     cv.waitKey(-1)
    #     cv.imwrite("./dataset/train/imgs/{}.png".format(i+1), train_imgs[i])

    # 1. 生成轨迹
    test_tracks, test_tracks_temp = generate_tracks(physical_width, physical_height, delta_t, test_num, track_min_length, track_max_length, pixel, scenario, test_imgs, test_total_frames, max_disp_frame, noise_ratio)
    np.save(test_path, test_tracks)
    np.save(test_temp_path, test_tracks_temp)
 
    # 2. 生成图片
    # for i in range(test_track_length):
    #     cv.imshow("Frame: "+str(i+1), test_imgs[i])
    #     cv.waitKey(-1)
    #     cv.imwrite("./dataset/test/imgs/{}.png".format(i+1), test_imgs[i])


    # 加载数据并保存为文本和CSV格式
    train_data1 = np.load(train_path + '.npy', allow_pickle=True)
    np.savetxt(os.path.join(dataset_dir, 'train', 'train_tracks.txt'), train_data1, delimiter=',')
    np.savetxt(os.path.join(dataset_dir, 'train', 'train_tracks.csv'), train_data1, delimiter=',')

    train_data2 = np.load(train_temp_path + '.npy', allow_pickle=True)
    np.savetxt(os.path.join(dataset_dir, 'train', 'train_tracks_complex.txt'), train_data2, delimiter=',')
    np.savetxt(os.path.join(dataset_dir, 'train', 'train_tracks_complex.csv'), train_data2, delimiter=',')

    test_data1 = np.load(test_path + '.npy', allow_pickle=True)
    np.savetxt(os.path.join(dataset_dir, 'test', 'test_tracks.txt'), test_data1, delimiter=',')
    np.savetxt(os.path.join(dataset_dir, 'test', 'test_tracks.csv'), test_data1, delimiter=',')

    test_data2 = np.load(test_temp_path + '.npy', allow_pickle=True)
    np.savetxt(os.path.join(dataset_dir, 'test', 'test_tracks_complex.txt'), test_data2, delimiter=',')
    np.savetxt(os.path.join(dataset_dir, 'test', 'test_tracks_complex.csv'), test_data2, delimiter=',')



if __name__ == '__main__':
    print("*"*60)
    print('Working path is '+os.getcwd())
    print("*"*60)
    parser = argparse.ArgumentParser(prog='input',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="ZhangYt")
                                        
    # Scenario 2 is used by default
    parser.add_argument('-s', type=str, help='Choose scenario.', default="2")
    args = parser.parse_args()
    main(args.s)
