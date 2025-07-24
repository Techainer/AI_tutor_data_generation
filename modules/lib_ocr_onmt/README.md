# lib_ocr_onmt  
[![CI](https://github.com/Techainer/lib_ocr_onmt/actions/workflows/ci.yml/badge.svg?branch=idcard)](https://github.com/Techainer/lib_ocr_onmt/actions/workflows/ci.yml)
 [![Code Smells](https://sonarqube.techainer.com/api/project_badges/measure?project=lib_ocr_onmt&metric=code_smells)](https://sonarqube.techainer.com/dashboard?id=lib_ocr_onmt) [![Vulnerabilities](https://sonarqube.techainer.com/api/project_badges/measure?project=lib_ocr_onmt&metric=vulnerabilities)](https://sonarqube.techainer.com/dashboard?id=lib_ocr_onmt) [![Coverage](https://sonarqube.techainer.com/api/project_badges/measure?project=lib_ocr_onmt&metric=coverage)](https://sonarqube.techainer.com/dashboard?id=lib_ocr_onmt)

IDCard OCR with Open Source Neural Machine Translation (ONMT).

If you encounter any problem while using this module, please open an issue and cc [po](mailto:po@techainer.com)!
## Current checkpoint
- Accuracy by Char: 0.92 / Accuracy by Line: 0.65
- Evaluation report: [link]()
- Training Commit ID: [1574f3d]()
- Checkpoint link: [Download]()
- [Training report]() / [ClearML log]()

## Setup Environment
This module required `python>=3.8` with `torch==1.8.1` and `torchvision==0.9.1` to run. Other python dependencies can be install with:
```
pip3 install -r requirements.txt
```
To get model weight run: `dvc pull -r techainer`
## Service Instruction
**Run the service:**
1. Check config in `mlconfig.yaml`.
2. Run the service with specific mode: `mlchain run -m <mode_name>`. `mode_name` can be set as `triton` or `prod`.

**Test the service:** After run the service successfully
- Check out Swagger if you want to test this service after running
- Additionally, this repo also provide [test_prediction.py](tests/test_prediction.py) and [test_batch_prediction.py](tests/test_batch_prediction.py) to measure model performance.
