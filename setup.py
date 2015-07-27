from setuptools import setup

setup(
    name = "trees",
    version = "0.0.1",
    author = "Sharad Vikram",
    author_email = "sharad.vikram@gmail.com",
    entry_points = {
        'console_scripts' : [
            'interactive_server=trees.server.server:main'
        ],
    },
    packages=[
        'trees'
    ],
    classifiers=[
    ],
)
