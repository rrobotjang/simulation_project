#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
# ... 기타 필요한 임포트 ...

import torch
import numpy as np

# 정책 신경망 모델 구조 정의 (Isaac Lab에서 사용한 것과 동일해야 함)
class ActorCritic(torch.nn.Module):
    # Isaac Lab 모델 구조에 맞춰 Actor (정책) 네트워크를 정의합니다.
    # 예시: 256x256 은닉층을 가진 MLP
    def __init__(self, num_observations, num_actions):
        super().__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(num_observations, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_actions),
            # 출력 레이어는 Isaac Lab 설정에 따라 tanh 등을 포함할 수 있음
        )
    
    def forward(self, x):
        return self.actor(x)


class IntegratedRobotSystem(Node):
    def __init__(self):
        super().__init__('integrated_robot_system')
        # ... (기존 비전, HRI 설정 유지) ...
        
        self.task_state = "IDLE"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- 강화 학습 모델 로드 ---
        self.num_observations = 10 # 관측값 차원 (관절각, 물체위치 등)
        self.num_actions = 6       # 액션 차원 (6DOF 로봇팔)
        self.policy_model = ActorCritic(self.num_observations, self.num_actions).to(self.device)
        
        # Isaac Lab에서 학습된 .pth 파일 경로
        model_path = "/ros_ws/install/share/manipulation_planning/models/doosan_policy.pth"
        try:
            # 체크포인트 파일에서 Actor (정책) 상태만 로드
            checkpoint = torch.load(model_path, map_location=self.device)
            # RSL-RL은 'actor'라는 키로 모델 상태를 저장합니다.
            self.policy_model.load_state_dict(checkpoint['actor_state_dict'])
            self.policy_model.eval() # 추론 모드 설정
            self.get_logger().info(f"Successfully loaded policy model from {model_path}")
        except FileNotFoundError:
            self.get_logger().error(f"Model file not found at {model_path}! Cannot use RL control.")
            # 이 경우 MoveIt! 폴백 로직을 사용하거나 종료

        # MoveIt! 관련 코드는 제거되거나 비활성화됩니다.
        # self.move_group = MoveGroupPythonInterface(self) # 이 줄은 사용하지 않음

    # ... (기존 콜백 함수들 유지) ...

    def process_vision_data(self):
        # ... (비전 처리 후 target_object_pose 업데이트) ...
        with self.lock:
            self.target_object_pose = object_pose_msg.pose
            self.task_state = "RL_ACTUATE" # 상태 변경: RL 추론 및 동작 실행

    def background_worker_loop(self):
        while rclpy.ok():
            if self.task_state == "RL_ACTUATE":
                self.run_rl_policy_inference()
                with self.lock:
                    self.task_state = "IDLE" # 동작 완료 후 대기
            # ... (time.sleep(0.1) 등) ...

    def run_rl_policy_inference(self):
        """
        MoveIt! 플래너 대신 강화 학습 모델을 사용하여 행동 결정
        """
        with torch.no_grad():
            # 1. 현재 관측값 수집 (ROS 2에서 받은 데이터로 구성)
            # 이 부분은 실제 로봇의 상태(encapsulate_observations 함수 필요)를 numpy 배열로 만듭니다.
            # 예시: [joint_pos_1, ..., joint_pos_6, gripper_pos, obj_x, obj_y, obj_z]
            observations = self.get_current_observations() 
            obs_tensor = torch.as_tensor(observations, dtype=torch.float32).to(self.device)
            
            # 2. 정책 신경망 추론 (행동 예측)
            # 모델은 정규화된 행동(-1 ~ 1)을 출력합니다.
            actions_tensor = self.policy_model(obs_tensor)
            actions_np = actions_tensor.cpu().numpy()

            # 3. 행동 스케일링 및 로봇 드라이버에게 명령 전송
            # 행동을 실제 관절 위치/토크 범위로 변환합니다.
            desired_joint_commands = self.scale_actions_to_joints(actions_np)
            
            # 로봇 드라이버 토픽 발행 (예: joint_trajectory_controller로 바로 발행)
            self.publish_joint_commands(desired_joint_commands)
            self.get_logger().info(f"RL Policy executed action: {desired_joint_commands}")

    # --- 헬퍼 함수 구현 필요 ---
    def get_current_observations(self):
        # 실제 관측값을 Numpy 배열로 반환하는 로직 구현 필요
        # (카메라 Pose, 조인트 상태 구독 필요)
        pass

    def scale_actions_to_joints(self, actions):
        # -1~1 범위의 액션을 실제 로봇의 제어 범위로 스케일링하는 로직 구현 필요
        pass
    
    def publish_joint_commands(self, commands):
        # control_msgs/action/FollowJointTrajectory 메시지 생성 및 발행/액션 호출
        pass

# main 함수는 동일하게 MultiThreadedExecutor 사용
