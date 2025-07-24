def load_layout_db_model(flow_mode:str, layout_model_path:str, **kwargs):
    if flow_mode == "legacy":
        from .main_legacy import DBTextDetectorLegacy

        model = DBTextDetectorLegacy(layout_model_path=layout_model_path, **kwargs)
    else:
        from .main_triton import DBTextDetectorTriton

        model = DBTextDetectorTriton(layout_model_path=layout_model_path, **kwargs)
    return model