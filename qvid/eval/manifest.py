from lmms_eval.models.registry_v2 import ModelManifest

# Picked up by lmms-eval through the `lmms_eval.models` entry point in pyproject.toml.
MANIFEST = ModelManifest(
    model_id="qvid_llava",
    simple_class_path="qvid.eval.lmms_llava.QViDLlava",
)
