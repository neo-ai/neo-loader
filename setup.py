from setuptools import find_packages, setup
import io
import os

CURRENT_DIR = os.path.dirname(__file__)

test_deps=[
    'pytest',
    'numpy',
]

extras = {
    'test': test_deps,
}

setup(
    name="NeoCompilerModelLoaders",
    version="1.0.1",
    # declare your packages
    packages=find_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},
    # include data files
    root_script_source_version=True,
    tests_require=test_deps,
    extras_require=extras,

    author = 'AWS Neo',
    author_email = 'aws-neo-ai@amazon.com',
    url='https://github.com/neo-ai/neo-ai-dlr',
    license = "Apache-2.0",
    description = 'Common libraries for converting machine learning models to TVM Relay IR',
    long_description=io.open(os.path.join(CURRENT_DIR, 'README.md'), encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Utilities",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires = '>=3.5',

)
