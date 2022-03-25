from setuptools import setup

setup(
    name='catenae',
    description='Extract catenae from treebank',
    author=['Ludovica Pannitto'],
    author_email=['ellepannitto@gmail.com'],
    url='https://github.com/ellepannitto/Catenae',
    version='0.1.0',
    license='MIT',
    packages=['catenae', 'catenae.logging_utils', 'catenae.utils', 'catenae.core'],
    package_data={'catenae': ['logging_utils/*.yml']},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'catenae = catenae.main:main'
        ],
    },
    install_requires=[],
)