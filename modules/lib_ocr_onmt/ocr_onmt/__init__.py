from .main_legacy import ONMTModelLegacy
from .main_triton import ONMTModelTriton

def load_onmt_model(flow_mode, ocr_vocab_path, ocr_weight_path, **kwargs):
    if flow_mode == "legacy":
        model = ONMTModelLegacy(ocr_vocab_path=ocr_vocab_path, ocr_weight_path=ocr_weight_path, **kwargs)
    else:
        model = ONMTModelTriton(ocr_vocab_path=ocr_vocab_path, ocr_weight_path=ocr_weight_path, **kwargs)
    return model