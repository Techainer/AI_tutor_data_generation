# lib_layout_db 
[![CI](https://github.com/Techainer/lib_layout_db/actions/workflows/ci.yml/badge.svg)](https://github.com/Techainer/lib_layout_db/actions/workflows/ci.yml) [![Quality Gate Status](https://sonarqube.techainer.com/api/project_badges/measure?project=lib_layout_db&metric=alert_status)](https://sonarqube.techainer.com/dashboard?id=lib_layout_db)
 [![Code Smells](https://sonarqube.techainer.com/api/project_badges/measure?project=lib_layout_db&metric=code_smells)](https://sonarqube.techainer.com/dashboard?id=lib_layout_db) [![Vulnerabilities](https://sonarqube.techainer.com/api/project_badges/measure?project=lib_layout_db&metric=vulnerabilities)](https://sonarqube.techainer.com/dashboard?id=lib_layout_db) [![Coverage](https://sonarqube.techainer.com/api/project_badges/measure?project=lib_layout_db&metric=coverage)](https://sonarqube.techainer.com/dashboard?id=lib_layout_db)

Text detection model using Differentiable Binarization (DB)

## How to setup
- This module required `paddlepaddle==2.0.2` to run.
- Other dependencies can be install with: `pip3 install -r requirements.txt`
- Model weights can be downloaded with: `dvc pull -r techainer`. Ask [linus](mailto:linus@techainer.com) if you don't have your dvc credentials
## How to run
- To test the model: `python3 test.py`
- To run the server: `mlchain run`
- To integrate this with your application, check out [swagger](0.0.0.0:8095/swagger) after you run the server.

##
If you encouter any problem using this module. Please open an issue and cc [linus](mailto:linus@techainer.com). Thanks for checking by ;)