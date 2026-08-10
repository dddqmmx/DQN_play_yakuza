# -*- coding: utf-8 -*-
"""兼容层：GameNode = core.GameClient + 当前平台后端。"""
from core.game_loop import GameClient, GameNode  # noqa: F401
from backends import load_platform


def create_game_node(
    decision_host,
    decision_port=15001,
    target_fps=20.0,
    freeze_process=False,
    input_settle_ms=5.0,
    capture_method=None,
):
    bundle = load_platform(capture_method=capture_method)
    return GameClient(
        platform=bundle,
        decision_host=decision_host,
        decision_port=decision_port,
        target_fps=target_fps,
        freeze_process=freeze_process,
        input_settle_ms=input_settle_ms,
    )


# 保持 `from game_node import GameNode` 可用：自动注入平台
class GameNode(GameClient):
    def __init__(
        self,
        decision_host,
        decision_port=15001,
        target_fps=20.0,
        freeze_process=False,
        input_settle_ms=5.0,
        capture_method=None,
        platform=None,
    ):
        if platform is None:
            platform = load_platform(capture_method=capture_method)
        super().__init__(
            platform=platform,
            decision_host=decision_host,
            decision_port=decision_port,
            target_fps=target_fps,
            freeze_process=freeze_process,
            input_settle_ms=input_settle_ms,
        )
