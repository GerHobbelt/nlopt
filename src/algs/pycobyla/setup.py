from pathlib import Path
from setuptools import setup

package_namespace = 'colorsensing'
package_name = 'pycobyla' 
setup_url = 'https://github.com/josepsanz/nlopt'
setup_name = 'pycobyla'
setup_author = 'Josep Maria Sanz'
setup_author_email = 'jsanz83@gmail.com'
version = '0.1.0'
packages = ('pycobyla',)

with open('requirements.txt', 'r') as f:
    requirements = tuple(f.readlines())

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name=setup_name,
    version=version,
    packages=packages,
    install_requires=requirements,
    url=setup_url,
    author=setup_author,
    author_email=setup_author_email,
    description='',
    long_description=long_description,
)
