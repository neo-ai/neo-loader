## Neo Framework -> Relay IR converters
This is a set of utility classes for converting ML models to [TVM](https://github.com/neo-ai/tvm) Relay IR. 

## Installation
This package does not assume any framework dependencies.

## Usage
```python
import tarfile

from neo_loader import load_model

model_artifacts = []
with tarfile.open('/path/to/model.tar.gz', 'r:gz') as tf:
    tf.extractall()
    model_artifacts = tf.getnames()

relay_artifacts = load_model(
    model_artifacts=model_artifacts,
    input_shape={'data': [1, 3, 224, 224]},
    framework='tensorflow'
)

def relay_func(relay_artifacts: Dict) -> object:
    return relay_artifacts['model_objects'][0]

def relay_params(relay_artifacts: Dict) -> object:
    return relay_artifacts['model_objects'][1]

def relay_dtype(relay_artifacts: Dict) -> object:
    if len(relay_artifacts['model_objects']) == 2:
        return {}
    else:
        return relay_artifacts['model_objects'][2]
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

