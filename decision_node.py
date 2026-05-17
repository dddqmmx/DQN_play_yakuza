import socket
import threading
import time

from config import DEVICE, FILE_PATHS, GAME_CONFIG, MODEL_PROFILES
from distributed_protocol import configure_socket, recv_message, send_message
from dqn_agent import DQNAgent


class DecisionNode:
    """AI 节点：同一进程内完成决策和训练。"""

    def __init__(self, host="0.0.0.0", port=15001, model_profile="large", checkpoint=None):
        self.host = host
        self.port = port
        self.model_profile = model_profile if model_profile in MODEL_PROFILES else "large"
        self.checkpoint_file = checkpoint or self._default_checkpoint(self.model_profile)
        self.agent = self._create_agent(self.model_profile, self.checkpoint_file)
        self.hidden_by_client = {}
        self.is_running = True
        self.train_thread = threading.Thread(target=self._train_worker, daemon=True)
        self.console_thread = threading.Thread(target=self._console_worker, daemon=True)
        self.agent_lock = threading.RLock()

    def _default_checkpoint(self, profile):
        base = FILE_PATHS["checkpoint"]
        if profile == "large":
            return base
        stem, dot, suffix = base.rpartition(".")
        if not dot:
            return f"{base}_{profile}"
        return f"{stem}_{profile}.{suffix}"

    def _create_agent(self, profile, checkpoint_file):
        log_dir = f"{FILE_PATHS['tensorboard_logs']}_{profile}"
        return DQNAgent(
            GAME_CONFIG["num_actions"],
            network_config=MODEL_PROFILES[profile],
            checkpoint_file=checkpoint_file,
            log_dir=log_dir,
        )

    def _train_worker(self):
        while self.is_running:
            try:
                with self.agent_lock:
                    agent = self.agent
                    ready = agent.memory.is_ready(agent.batch_size)
                    if ready:
                        agent.update_model()
                time.sleep(0.001 if ready else 0.02)
            except Exception as exc:
                print(f">> AI 节点训练线程错误: {exc}")
                time.sleep(1.0)

    def _select_action(self, client_id, msg):
        hidden = self.hidden_by_client.get(client_id)
        with self.agent_lock:
            agent = self.agent
            if agent.use_recurrent:
                action, next_hidden = agent.select_action(
                    msg["state"], msg["boss_health"], msg["self_health"], hidden
                )
                self.hidden_by_client[client_id] = next_hidden
            else:
                action = agent.select_action(msg["state"], msg["boss_health"], msg["self_health"])
        return action

    def _store_transition(self, msg):
        with self.agent_lock:
            self.agent.store_transition(
                msg["state"], msg["boss_health"], msg["self_health"], msg["action"],
                msg["reward"], msg["next_state"], msg["next_boss_health"],
                msg["next_self_health"], msg["done"]
            )

    def _save(self, filename=None):
        with self.agent_lock:
            self.agent.save_checkpoint(filename or self.checkpoint_file)

    def _switch_checkpoint(self, filename):
        with self.agent_lock:
            self.checkpoint_file = filename
            loaded = self.agent.load_checkpoint(filename)
        print(f">> 保存点已切换到: {filename} | load={'ok' if loaded else 'new'}")

    def _switch_model(self, profile, checkpoint=None):
        if profile not in MODEL_PROFILES:
            print(f">> 未知模型档位: {profile}，可用: {', '.join(MODEL_PROFILES)}")
            return
        checkpoint = checkpoint or self._default_checkpoint(profile)
        with self.agent_lock:
            old_agent = self.agent
            old_agent.save_checkpoint(self.checkpoint_file)
            self.model_profile = profile
            self.checkpoint_file = checkpoint
            self.hidden_by_client.clear()
            self.agent = self._create_agent(profile, checkpoint)
            loaded = self.agent.load_checkpoint(checkpoint)
            old_agent.close()
        print(f">> 已切换模型: {profile} | checkpoint={checkpoint} | load={'ok' if loaded else 'new'}")

    def _print_status(self):
        with self.agent_lock:
            agent = self.agent
            memory_len = len(agent.memory)
            print(
                f">> status | model={self.model_profile} | checkpoint={self.checkpoint_file} | "
                f"steps={agent.steps} | memory={memory_len} | device={DEVICE}"
            )

    def _console_worker(self):
        print(">> AI 控制台命令: help, status, save [path], checkpoint <path>, model <small|medium|large> [path], exit")
        while self.is_running:
            try:
                raw = input("ai> ").strip()
            except EOFError:
                break
            if not raw:
                continue
            parts = raw.split()
            cmd = parts[0].lower()
            try:
                if cmd in ("help", "?"):
                    print("命令: status | save [path] | checkpoint <path> | model <small|medium|large> [path] | exit")
                elif cmd == "status":
                    self._print_status()
                elif cmd == "save":
                    self._save(parts[1] if len(parts) > 1 else None)
                    print(">> 保存任务已提交")
                elif cmd in ("checkpoint", "ckpt"):
                    if len(parts) < 2:
                        print("用法: checkpoint <path>")
                    else:
                        self._switch_checkpoint(parts[1])
                elif cmd == "model":
                    if len(parts) < 2:
                        print(f"当前模型: {self.model_profile}，可用: {', '.join(MODEL_PROFILES)}")
                    else:
                        self._switch_model(parts[1].lower(), parts[2] if len(parts) > 2 else None)
                elif cmd in ("exit", "quit", "q"):
                    print(">> 正在退出 AI 节点...")
                    self.is_running = False
                    self._save()
                    break
                else:
                    print(f">> 未知命令: {cmd}")
            except Exception as exc:
                print(f">> 控制台命令失败: {exc}")

    def _handle_client(self, conn, addr):
        client_id = f"{addr[0]}:{addr[1]}"
        print(f">> Game 节点已连接: {client_id}")
        try:
            while self.is_running:
                msg = recv_message(conn)
                msg_type = msg.get("type")
                if msg_type == "action_request":
                    if msg.get("episode_done"):
                        self.hidden_by_client.pop(client_id, None)
                    started = time.perf_counter()
                    action = self._select_action(client_id, msg)
                    with self.agent_lock:
                        train_steps = self.agent.steps
                    send_message(conn, {
                        "type": "action",
                        "action": action,
                        "decision_ms": (time.perf_counter() - started) * 1000.0,
                        "train_steps": train_steps,
                    })
                elif msg_type == "transition":
                    self._store_transition(msg)
                    send_message(conn, {"type": "ack"})
                elif msg_type == "reset_hidden":
                    self.hidden_by_client.pop(client_id, None)
                    send_message(conn, {"type": "ack"})
                elif msg_type == "save":
                    self._save()
                    send_message(conn, {"type": "ack"})
                else:
                    send_message(conn, {"type": "error", "message": f"unknown type: {msg_type}"})
        except ConnectionError:
            print(f">> Game 节点断开: {client_id}")
        except Exception as exc:
            print(f">> AI 节点连接错误 {client_id}: {exc}")
        finally:
            self.hidden_by_client.pop(client_id, None)
            conn.close()

    def serve_forever(self):
        if self.agent.load_checkpoint(self.checkpoint_file):
            print(">> AI 节点已加载本地模型")
        self.train_thread.start()
        self.console_thread.start()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            configure_socket(server)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.settimeout(1.0)
            try:
                server.bind((self.host, self.port))
            except PermissionError as exc:
                raise PermissionError(
                    f"无法监听 {self.host}:{self.port}。请换一个端口，比如 15001/16001，"
                    "或检查 Windows 保留端口/防火墙/管理员权限。"
                ) from exc
            server.listen()
            print(f">> AI 节点监听 {self.host}:{self.port}，device={DEVICE}")
            try:
                while self.is_running:
                    try:
                        conn, addr = server.accept()
                    except socket.timeout:
                        continue
                    configure_socket(conn)
                    threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()
            finally:
                self.shutdown()

    def shutdown(self):
        self.is_running = False
        with self.agent_lock:
            self.agent.save_checkpoint(self.checkpoint_file)
            self.agent.close()
