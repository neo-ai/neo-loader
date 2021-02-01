from setuptools import find_packages, setup

test_deps=[
    'pytest',
    'numpy',
]

extras = {
    'test': test_deps,
}

setup(
    name="NeoCompilerModelLoaders",
    version="1.0",
    # declare your packages
    packages=find_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},
    # include data files
    root_script_source_version=True,
    tests_require=test_deps,
    extras_require=extras,
)
