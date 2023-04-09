import setuptools

setuptools.setup(
    name='daamt2i',
    version=eval(open('daamt2i/_version.py').read().strip().split('=')[1]),
    author='Rishi Dey Chowdhury',
    license='MIT',
    url='https://github.com/RishiDarkDevil/daam-t2i',
    author_email='rishi8001100192@gmail.com',
    description='DAAM-Text2Image: Extension of DAAM for Text-Image Cross-Attention in Diffusion Models',
    install_requires=open('requirements.txt').read().strip().splitlines(),
    packages=setuptools.find_packages(),
    python_requires='>=3.8'
)
