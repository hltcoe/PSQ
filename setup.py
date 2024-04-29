import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='fast_psq',
    version='0.0.1',
    author='Eugene Yang',
    author_email='eugene.yang@jhu.edu',
    description="Efficient Implementation of Probabilistic Structured Queries",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hltcoe/PSQ',
    packages=setuptools.find_packages(),
    install_requires=["numpy", "scipy", "tqdm", "nltk", "mosestokenizer", "huggingface_hub"],
    include_package_data=True,
    python_requires='>=3.8',
)