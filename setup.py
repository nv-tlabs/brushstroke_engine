from setuptools import setup, find_packages

# TODO: periodically update these
reqs = ['numpy>=1.19', 'scikit-image>=0.17']

setup(
    name='forger',
    version='0.0.1',
    author='Maria Shugrina, Chin-Ying Li',
    author_email='mshugrina@nvidia.com',
    description='Neural Brushstroke Engine project.',
    long_description_content_type='text/markdown',
    py_modules=['forger', 'thirdparty'],
    python_requires='>=3.6',
    install_requires=reqs,
    classifiers=[
        "Operating System :: OS Independent"
    ],
)