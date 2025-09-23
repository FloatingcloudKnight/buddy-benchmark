import subprocess
import os
import signal
import sys
import time
from typing import List, Tuple

process_list = [
    ['/home/chenweiwei/rvv/buddy-mlir/build/bin/buddy-llama-input-run'],  # 第一个启动的输入进程
    ['/home/chenweiwei/rvv/buddy-mlir/build/bin/buddy-llama-rms-run'],
    ['/home/chenweiwei/rvv/buddy-mlir/build/bin/buddy-llama-rms-0-run'],
    ['/home/chenweiwei/rvv/buddy-mlir/build/bin/buddy-llama-mha-run'],
    ['/home/chenweiwei/rvv/buddy-mlir/build/bin/buddy-llama-mha-0-run'],
    ['/home/chenweiwei/rvv/buddy-mlir/build/bin/buddy-llama-add-run'],
    ['/home/chenweiwei/rvv/buddy-mlir/build/bin/buddy-llama-add-0-run'],
    ['/home/chenweiwei/rvv/buddy-mlir/build/bin/buddy-llama-rms-mlp-run'],
    ['/home/chenweiwei/rvv/buddy-mlir/build/bin/buddy-llama-rms-mlp-0-run'],
    ['/home/chenweiwei/rvv/buddy-mlir/build/bin/buddy-llama-mlp-run'],
    ['/home/chenweiwei/rvv/buddy-mlir/build/bin/buddy-llama-mlp-0-run'],
    ['/home/chenweiwei/rvv/buddy-mlir/build/bin/buddy-llama-add-mlp-run'],
    ['/home/chenweiwei/rvv/buddy-mlir/build/bin/buddy-llama-add-mlp-0-run'],
    ['/home/chenweiwei/rvv/buddy-mlir/build/bin/buddy-llama-output-run']
]

class ProcessController:
    def __init__(self):
        self.running_procs: List[Tuple[subprocess.Popen, str]] = []
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def generate_log_path(self, cmd: List[str]) -> str:
        """生成带时间戳的日志路径[6](@ref)"""
        base_name = os.path.basename(cmd[0])
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        return f"logs/{base_name}-{timestamp}.log"

    def signal_handler(self, signum: int, frame: any) -> None:
        """终止所有进程[7](@ref)"""
        print("\n接收到终止信号，正在清理进程...")
        self.terminate_all()
        sys.exit(1)

    def start_process(self, cmd: List[str]) -> subprocess.Popen:
        """启动单个进程并重定向输出[4](@ref)"""
        log_path = self.generate_log_path(cmd)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        with open(log_path, 'w') as log_file:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE if cmd == process_list[0] else None,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )
            self.running_procs.append((proc, log_path))
            print(f"已启动进程: {' '.join(cmd)} [PID:{proc.pid}] 日志: {log_path}")
            return proc

    def terminate_all(self) -> None:
        """终止所有运行中的进程[8](@ref)"""
        for proc, log_path in self.running_procs:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=3)
                    print(f"已终止进程 [PID:{proc.pid}]")
            except Exception as e:
                print(f"终止进程错误: {str(e)}")

    def execute_sequence(self) -> None:
        """执行主流程控制[2](@ref)"""
        try:
            # 第一阶段：启动输入进程（保持stdin管道）
            input_proc = self.start_process(process_list[0])
            
            # 第二阶段：顺序执行其他进程
            for cmd in process_list[1:]:
                proc = self.start_process(cmd)
                
                # 阻塞等待完成[1](@ref)
                while proc.poll() is None:
                    try:
                        proc.wait(timeout=0.5)
                        if proc.returncode != 0:
                            raise RuntimeError(f"进程异常退出: {cmd[0]} (代码:{proc.returncode})")
                    except subprocess.TimeoutExpired:
                        continue

                print(f"进程完成: {cmd[0]}")

            # 第三阶段：所有进程完成后处理输入
            if input_proc.poll() is None:
                user_input = input("\n所有进程执行完毕，请输入数据：")
                input_proc.communicate(input=user_input + "\n")
                print("输入处理完成")

        except Exception as e:
            print(f"执行错误: {str(e)}")
            self.terminate_all()
            sys.exit(1)

if __name__ == "__main__":
    controller = ProcessController()
    controller.execute_sequence()
