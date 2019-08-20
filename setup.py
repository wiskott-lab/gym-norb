import setuptools

setuptools.setup(
        name='gym-norb',
        version='0.1',
        author='Robin Schiewer',
        author_email='robin.schiewer@ini.rub.de',
        description='Norb object dataset presented as an openai gym environment',
        packages=['norb'],
        install_requires=['gym', 'numpy'],
        classifiers=['Programming Language :: Python :: 3'],
        include_package_data=True
)
