import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
from PIL import Image
import argparse
import kitti_util
from models import *

def png_to_npy(args, idx, model):

    def test(imgL,imgR):
        model.eval()

        if args.cuda:
            imgL = imgL.cuda()
            imgR = imgR.cuda()     

        with torch.no_grad():
            start_time = time.time()
            output = model(imgL,imgR)
            print('여기서 시간 다 쓰는데 어떡하냐? = %.3f' %(time.time() - start_time))
        output = torch.squeeze(output).data.cpu().numpy()
        return output

    #main###################################################
    normal_mean_var = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(**normal_mean_var)])    

    imgL_o = Image.open(args.datapath + 'image_2/' + idx + '.png').convert('RGB')
    imgR_o = Image.open(args.datapath + 'image_3/' + idx + '.png').convert('RGB')
    # 여기서 계산량을 줄여야함

    imgL = infer_transform(imgL_o)
    imgR = infer_transform(imgR_o)         

    # pad to width and hight to 16 times
    if imgL.shape[1] % 16 != 0:
        times = imgL.shape[1]//16       
        top_pad = (times+1)*16 -imgL.shape[1]
    else:
        top_pad = 0

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2]//16                       
        right_pad = (times+1)*16-imgL.shape[2]
    else:
        right_pad = 0    

    imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
    imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)
    
    start_time = time.time()
    pred_disp = test(imgL,imgR)
    print('disparity process = %.3f' %(time.time() - start_time))

    if top_pad !=0 or right_pad != 0:
        img = pred_disp[top_pad:,:-right_pad]
    else:
        img = pred_disp

    return img


def npy_to_bin(args, disp_map, predix):
    def project_disp_to_points(calib, disp, max_high):
        disp[disp < 0] = 0
        baseline = 0.54
        mask = disp > 0
        depth = calib.f_u * baseline / (disp + 1. - mask)
        print(len(depth))
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        points = np.stack([c, r, depth])
        points = points.reshape((3, -1))
        points = points.T
        points = points[mask.reshape(-1)]
        cloud = calib.project_image_to_velo(points)
        valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
        return cloud[valid]
    #main############################################
    calib = kitti_util.Calibration('{}/{}.txt'.format(args.calib_dir, predix))

    disp_map = (disp_map*256).astype(np.uint16)/256.
    lidar = project_disp_to_points(calib, disp_map, args.max_high)
    
    lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
    lidar = lidar.astype(np.float32)

    lidar.tofile('{}/{}.bin'.format(args.save_dir, predix))
    #return lidar


def main():
    # png to npy
    parser = argparse.ArgumentParser(description='marge')
    parser.add_argument('--masterpath', default='KITTI', help='KITTI PATH')
    parser.add_argument('--KITTI', default='2015', help='KITTI version')
    parser.add_argument('--loadmodel', default='finetune_300.tar', help='loading model')
    parser.add_argument('--maxdisp', default=192, help='maxium disparity')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    # npy to bin
    parser.add_argument('--max_high', type=int, default=1)

    args = parser.parse_args()

    args.datapath = args.masterpath + '/object/training/'
    args.save_dir = './'
    args.calib_dir = args.masterpath + '/object/training/calib/'

    assert os.path.isdir(args.datapath)
    assert os.path.isdir(args.save_dir)
    assert os.path.isdir(args.calib_dir)
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    model = stackhourglass(args.maxdisp)
    start_time = time.time()
    model = nn.DataParallel(model, device_ids=[0])
    print('Parallel set time = %.3f' %(time.time() - start_time))
    model.cuda()

    if args.loadmodel is not None:
        state_dict = torch.load(args.loadmodel)
        model.load_state_dict(state_dict['state_dict'])
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    for i in ['000000','000001','000002','000003','000004']:
        idx = i
        img = png_to_npy(args, idx, model)

        start_time = time.time()
        npy_to_bin(args, img, idx)
        print('pointcloud process = %.3f' %(time.time() - start_time))
        print('Finish Depth ' + idx)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print('ALL = %.3f' %(time.time() - start_time))