from setuptools import setup

setup(name='Information Retrival System',
      packages=['irs'],
      install_requires=[
          'google-api-python-client',
      ],
      entry_points={
          'console_scripts': [
              'run = irs.__main__:main'
          ]
      },
)