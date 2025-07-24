from .main_legacy import VietOCRLegacy
from .main_triton import VietOCRTriton

def load_ocr_model(flow_mode, ocr_vocab_path, ocr_weight_path, **kwargs):
    if flow_mode == "legacy":
        model = VietOCRLegacy(ocr_vocab_path=ocr_vocab_path, ocr_weight_path=ocr_weight_path, **kwargs)
    else:
        model = VietOCRTriton(ocr_vocab_path=ocr_vocab_path, ocr_weight_path=ocr_weight_path, **kwargs)
    return model