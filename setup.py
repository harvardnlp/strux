from setuptools import setup

setup(
    name="strux",
    version="0.1",
    author="Sasha Rush",
    author_email="arush@cornell.edu",
    packages=[
        "strux",
    ],
    package_data={"strux": []},
    url="https://github.com/harvardnlp/strucx",
    install_requires=["jax"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)
