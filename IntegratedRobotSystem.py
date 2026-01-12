#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
# ... (기타 필요한 임포트 및 RL 모델 정의) ...

class IntegratedRobotSystem(Node):
    def __init__(self):
        # ... (기존 초기화 코드 유지) ...
        self.task_state = "IDLE" 
        self.retry_count = 0
        self.MAX_RETRIES = 2
        
        # MoveIt! MoveGroup 인터페이스 (fallback 또는 초기 홈 포지션 이동용)
        # 실제 로봇 드라이버의 홈 포지션 이동 방식에 맞춰 수정 필요
        # self.move_group_interface = MoveGroupInterface(self, "manipulator") 

    # ... (기존 콜백 및 헬퍼 함수 유지) ...

    def background_worker_loop(self):
        """
        백그라운드 워커 루프: 태스크 상태에 따라 동작 제어
        """
        while rclpy.ok():
            if self.task_state == "RL_ACTUATE":
                success = self.run_rl_policy_inference()
                if success:
                    self.get_logger().info("Task successful!")
                    with self.lock:
                        self.task_state = "IDLE"
                        self.retry_count = 0
                else:
                    self.handle_task_failure()
            
            elif self.task_state == "GO_HOME":
                self.go_to_home_position()
                with self.lock:
                    self.task_state = "IDLE" # 홈 이동 후 대기 상태로 전환
            
            # ... (time.sleep(0.1) 등) ...

    def run_rl_policy_inference(self):
        """
        RL 추론 실행 및 성공/실패 반환
        """
        try:
            # ... (이전 코드의 RL 추론 및 명령 전송 로직) ...
            
            # 실제 동작 완료까지 대기 (예: 특정 시간 또는 센서 피드백)
            # is_task_completed() 함수를 사용하여 성공 여부 판단 필요
            if self.is_task_completed():
                return True
            else:
                return False

        except Exception as e:
            self.get_logger().error(f"Error during RL inference/execution: {e}")
            return False

    def handle_task_failure(self):
        """
        태스크 실패 시 재시도 횟수를 확인하고 처리
        """
        with self.lock:
            self.retry_count += 1
            if self.retry_count <= self.MAX_RETRIES:
                self.get_logger().warn(f"Task failed. Retrying... (Attempt {self.retry_count}/{self.MAX_RETRIES})")
                self.task_state = "DETECTING" # 비전 파이프라인부터 재시작
            else:
                self.get_logger().error("Max retries reached. Moving to home position and stopping.")
                self.task_state = "GO_HOME" # 최대 재시도 실패 시 홈 이동
                self.retry_count = 0 # 카운트 초기화

    def go_to_home_position(self):
        """
        로봇을 미리 정의된 안전한 홈 포지션으로 이동
        """
        self.get_logger().info("Moving to home position...")
        # MoveIt! 인터페이스를 사용하거나, 특정 조인트 목표값을 발행
        # self.move_group_interface.set_named_target("home")
        # self.move_group_interface.go(wait=True)
        self.get_logger().info("Arrived at home position. IDLE.")
        
    # --- 헬퍼 함수 ---
    def is_task_completed(self):
        # 물체가 성공적으로 파지되어 목표 위치에 있는지 확인하는 로직 (센서 피드백)
        # 하루 만에 구현이 어려울 경우, 간단한 타이머나 목업 센서 사용
        return True # 임시로 항상 성공 반환

# main 함수는 동일하게 MultiThreadedExecutor 사용
def main(args=None):
    rclpy.init(args=args)
    integrated_system = IntegratedRobotSystem()
    
    worker_thread = threading.Thread(target=integrated_system.background_worker_loop)
    worker_thread.daemon = True
    worker_thread.start()

    executor = MultiThreadedExecutor()
    executor.add_node(integrated_system)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    
    integrated_system.destroy_node()
    rclpy.shutdown()
    worker_thread.join()
