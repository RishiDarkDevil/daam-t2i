import setuptools

setuptools.setup(
    name='daamt2i',
    version=eval(open('daam/_version.py').read().strip().split('=')[1]),
    author='RishiDarkDevil',
    license='MIT',
    url='https://github.com/RishiDarkDevil/daam-t2i',
    author_email='r33tang@uwaterloo.ca',
    description='What the DAAM: Interpreting Stable Diffusion Using Cross Attention.',
    install_requires=open('requirements.txt').read().strip().splitlines(),
    packages=setuptools.find_packages(),
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'daam = daam.run.generate:main',
            'daam-demo = daam.run.demo:main',
        ]
    }
)
