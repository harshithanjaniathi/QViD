from qvid.scoring import qvid_scores, select_topk

__all__ = ["qvid_scores", "select_topk", "apply_qvid"]


def __getattr__(name):
    # Imported lazily so `qvid.scoring` works without LLaVA installed.
    if name == "apply_qvid":
        from qvid.llava import apply_qvid
        return apply_qvid
    raise AttributeError(name)
