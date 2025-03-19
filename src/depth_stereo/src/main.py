import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

import logging
import torch
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
import numpy as np
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge
import cv2
import copy

import sys
from os.path import join as opj
import os
sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), "../../../devel/lib/python3/dist-packages"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from tc_stereo import TCStereo
# from core.utils.utils import InputPadder
from utils import *
import torchvision.transforms as T
from network import GcNet
from airsim_ros.msg import VelCmd
from depth2command.model import LSTMNetVIT_Modified

class DepthStereoNode:
    def __init__(self, args):
        rospy.init_node("depth_stereo_node", anonymous=False,)
        print("init DepthStereoNode")
        
        # init model
        self.model = GcNet(256, 512, 96).float()
        if args.restore_ckpt is not None:
            logging.info("Loading checkpoint...")
            checkpoint = torch.load(args.restore_ckpt)
            new_state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                namekey = k[7:]
                new_state_dict[namekey] = v
            checkpoint['state_dict'] = new_state_dict
            self.model.load_state_dict(checkpoint['state_dict'])
            logging.info(f"Done loading checkpoint")
        
        self.model = torch.nn.DataParallel(self.model, device_ids=[args.device])
        self.model = torch.quantization.quantize_dynamic(self.model, 
                                        {torch.nn.Linear, torch.nn.Conv2d},
                                        dtype=torch.qint8)
        self.model.cuda(args.device)
        self.model.eval()
        
        
        self.command_model = LSTMNetVIT_Modified()
        self.command_model.load_state_dict(torch.load(args.command_model_path))
        
        self.command_model = torch.nn.DataParallel(self.command_model, device_ids=[args.device])
        # self.command_model = torch.quantization.quantize_dynamic(self.command_model,
        #                                 {torch.nn.Linear, torch.nn.Conv2d},
        #                                 dtype=torch.qint8)
        self.command_model = self.command_model.cuda()
        self.command_model.eval()
        
        
        # init params
        self.idx = 0
        self.args = args
        self.last_orientation = None
        self.last_pos = None
        self.cv_bridge = CvBridge()
        
        # model params
        self.K = getKMatrix()
        self.K = torch.tensor(self.K)
        self.baseline = getBaseline()
        
        self.desired_vel = 5.0
        self.hidden_state = None
        # self.params = dict()    # model'params
        # self.previous_T = None
        # self.flow_q = None
        # self.net_list = None
        # self.fmap1 = None
        
        
        # self.rgb_cam_suber = rospy.Subscriber(
        #     "/airsim_node/drone_1/front_left/Scene",
        #     Image,
        #     self.rgb_callback,
        #     tcp_nodelay=True
        # )
        
        # 通过当前的旋转矩阵减去上一刻的旋转矩阵可以得到 这一瞬间变化的旋转矩阵，
        # 通过当前的位置减去上一时刻的位置 可以得到 平移向量
        # self.pos_suber 是 可以获取得到每个时刻的无人机的 位置和位姿 数据
        self.pos_suber = Subscriber(
            "/eskf_odom",
            Odometry,
            tcp_nodelay = True
        )
        
        # 前置摄像头的左视图
        self.rgb_front_left_suber = Subscriber(
            "/airsim_node/drone_1/front_left/Scene",
            Image,
            tcp_nodelay=True
        )
        
        self.rgb_front_right_suber = Subscriber(
            "/airsim_node/drone_1/front_right/Scene",
            Image,
            tcp_nodelay=True
        )
        self.ts = ApproximateTimeSynchronizer([self.rgb_front_left_suber, self.rgb_front_right_suber,
                                                self.pos_suber], queue_size=100000, slop=0.1)
        if self.args.use_model:
            self.ts.registerCallback(self.image_callback_model)
        else:
            self.ts.registerCallback(self.image_callback)
        
        # 模型深度图计算发布
        self.depth_puber = rospy.Publisher(
            "/depth_stereo/Depth",
            Image,
            queue_size=1,
            tcp_nodelay=True        
        )
        
        #模型控制信息发布
        self.control_puber = rospy.Publisher(
            "/airsim_node/drone_1/vel_cmd_body_frame",
            VelCmd,
            queue_size=1,
            tcp_nodelay=True
        )
    
    def rgb_callback(self, image):
        left_image = self.cv_bridge.imgmsg_to_cv2(image, "passthrough")
        cv2.imshow("image", left_image)
        cv2.waitKey(1)

    def image_callback(self, left_image: Image, right_image: Image, msg: Odometry):
        raise NotImplementedError("Not implemented")
        ori = msg.pose.pose.orientation
        if abs(ori.x) < 1e-5 and abs(ori.y) <1e-5 and abs(ori.z) <1e-5 and abs(ori.w) < 1e-5:
            rospy.loginfo("Rechieved all zero orientation data, return!")
            return
        # 相机内参
        K_left = np.array([[831.3843994140625, 0.0, 480.0],
                        [0.0, 831.3843994140625, 360.0],
                        [0.0, 0.0, 1.0]])

        K_right = np.array([[831.3843994140625, 0.0, 480.0],
                            [0.0, 831.3843994140625, 360.0],
                            [0.0, 0.0, 1.0]])

        # 畸变系数（假设没有畸变）
        D_left = np.zeros(5)
        D_right = np.zeros(5)

        # 旋转矩阵和投影矩阵（假设没有旋转和平移）
        curr_orientation_list = [ori.x, ori.y, ori.z, ori.w]
        rotation_matrix = R.from_quat(curr_orientation_list).as_matrix()
        
        # 投影矩阵
        P_left = np.array([[831.3843994140625, 0.0, 480.0, 0.0],
                        [0.0, 831.3843994140625, 360.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0]])

        P_right = np.array([[831.3843994140625, 0.0, 480.0, -831.3843994140625 * 0.1],
                            [0.0, 831.3843994140625, 360.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0]])
        
        img_left = self.cv_bridge.imgmsg_to_cv2(left_image, "passthrough")
        img_right = self.cv_bridge.imgmsg_to_cv2(right_image, "passthrough")
        h, w = img_left.shape[:2]
        map1_left, map2_left = cv2.initUndistortRectifyMap(K_left, D_left, rotation_matrix, P_left, (w, h), cv2.CV_32FC1)
        map1_right, map2_right = cv2.initUndistortRectifyMap(K_right, D_right, rotation_matrix, P_right, (w, h), cv2.CV_32FC1)

        rectified_left = cv2.remap(img_left, map1_left, map2_left, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(img_right, map1_right, map2_right, cv2.INTER_LINEAR)

        # 将图像转换为灰度图像
        rectified_left_gray = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
        rectified_right_gray = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

        # 确保图像格式为 CV_8UC1
        rectified_left_gray = rectified_left_gray.astype(np.uint8)
        rectified_right_gray = rectified_right_gray.astype(np.uint8)

        # 计算视差图
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(rectified_left_gray, rectified_right_gray)

        # 计算深度图
        focal_length = K_left[0, 0]
        baseline = 0.3  # 基线距离
        depth = (focal_length * baseline) / (disparity + 1e-6)  # 避免除以零
        # 归一化深度图
        depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow('Disparity', disparity)
        cv2.imshow('Depth', depth)  # 归一化显示
        cv2.waitKey(1)
    
    @torch.no_grad()
    def image_callback_model(self, left_image: Image, right_image: Image, msg: Odometry):
        """根据输入的左右图像 计算深度图

        Args:
            left_image (Image): _description_
            right_image (Image): _description_
            msg (Odometry): _description_
        """
        torch.cuda.empty_cache()
        ori = msg.pose.pose.orientation
        if abs(ori.x) < 1e-5 and abs(ori.y) <1e-5 and abs(ori.z) <1e-5 and abs(ori.w) < 1e-5:
            rospy.loginfo("Rechieved all zero orientation data, return!")
            return
        
        # 随机采样 10% 的数据 降低计算量
        if self.idx % 10 != 0:
            self.idx += 1
            return
        
        
        left_image = self.cv_bridge.imgmsg_to_cv2(left_image, "passthrough")
        right_image = self.cv_bridge.imgmsg_to_cv2(right_image, "passthrough")
        # 输入图像预处理
        left, right = load(left_image, right_image)
        disp = self.model(left, right)
        disp = disp.squeeze().detach().cpu().numpy()
        depth = disp2depth(disp, self.baseline, self.K[0, 0])
        depth = np.array(depth)
        
        depth = cv2.resize(depth, (960, 720)).astype(np.float32)
        depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # cv2.imshow("depth", depth.astype(np.uint8))
        # cv2.waitKey(1)
        depth = torch.from_numpy(depth).cuda().reshape(1, 1, 720, 960).float()

        
        # 计算深度图的平均值
        # average_depth = np.mean(depth)ze(depth, depth, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 使用深度图计算控制信息 模型输入分别是深度图， 位姿四元组，期望速度， hidden_state
        quaternion = [ori.w, ori.x, ori.y, ori.z]
        quaternion = torch.tensor(quaternion).reshape(1, 4).cuda()
        # 输入的期望速度
        expected_speed = torch.tensor([[self.desired_vel]]).cuda()
        
        # 输出三个方向的线速度和角速度
        command, self.hidden_state = self.command_model([depth, quaternion, expected_speed, self.hidden_state])
        command = command.squeeze().detach().cpu().numpy()
        # 数据类型  airsim_ros/VelCmd
        # 数据转换
        rospy.loginfo("command: %s", command)
        
        twist_msg = VelCmd()
        twist_msg.twist.angular.x = command[0] * self.desired_vel
        twist_msg.twist.angular.y = command[1] * self.desired_vel
        twist_msg.twist.angular.z = command[2] * self.desired_vel
        twist_msg.twist.linear.x = command[3] * self.desired_vel
        twist_msg.twist.linear.y = command[4] * self.desired_vel
        twist_msg.twist.linear.z = command[5] * self.desired_vel
        self.control_puber.publish(twist_msg)


def load(limg, rimg):
        mean = [0.406, 0.456, 0.485]
        std = [0.225, 0.224, 0.229]
        transform = T.Compose([Normalize(mean, std), ToTensor(), Pad(256, 512)])
        pairs = {'left': limg, 'right': rimg}
        pairs = transform(pairs)
        left = pairs['left'].to('cuda').unsqueeze(0)
        right = pairs['right'].to('cuda').unsqueeze(0)
        return left, right

def disp2depth(disp, baseline, focal_length):
    '''
    disp: disparity map
    baseline: baseline distance
    focal_length: focal length
    '''
    depth = focal_length * baseline / disp
    return depth

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", type=str,default=None)
    parser.add_argument('--device', default=0, type=int, help='the device id')
    parser.add_argument('--use_model', action='store_true', help='use model to inference the depth')
    parser.add_argument('--command_model_path', type=str, default='command_model_epoch116.pth')
    # Architecure choices
    
    args, unknown = parser.parse_known_args()
    
    
    depth_stereo_node = DepthStereoNode(args)
    
    rospy.spin()



