

```bash
git clone --recursive https://github.com/Raghav-B/accelerated_features

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install kornia fork with onnx exportability

cd third_party/kornia
pip install -e .


```