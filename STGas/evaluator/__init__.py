import copy

from .detection import DetectionEvaluator


def build_evaluator(cfg, dataset):
    evaluator_cfg = copy.deepcopy(cfg)
    name = evaluator_cfg.pop("name")
    if name == "DetectionEvaluator":
        return DetectionEvaluator(dataset)
    else:
        raise NotImplementedError
