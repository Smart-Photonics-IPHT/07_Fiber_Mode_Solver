#!/usr/bin/env python3

# This file is part of FiberModes.
#
# FiberModes is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FiberModes is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with FiberModes.  If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup, find_packages

setup(
    name='fibermodes',
    version='0.3.0',
    description='Multilayers optical fiber mode solver',
    author='Behnam Pishnamazi, Mario Chemnitz (based on v.0.2.0 by Charles Brunet)',
    author_email='mario.chemnitz@leibniz-ipht.de',
    url='https://github.com/Smart-Photonics-IPHT/07_Fiber_Mode_Solver_2.0',
    packages=find_packages(exclude=['plots', 'scripts', 'tests']),
    include_package_data=True,
    python_requires='>=3.4',
    entry_points={
        'gui_scripts': [
            'fibereditor=fibermodesgui.fibereditorapp:main',
            'materialcalculator=fibermodesgui.materialcalculator:main',
            'modesolver=fibermodesgui.modesolverapp:main',
            'wavelengthcalculator=fibermodesgui.wavelengthcalculator:main'
        ]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Environment :: X11 Applications :: Qt',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Telecommunications Industry',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    install_requires=[
        'numpy>=1.9.0',
        'scipy>=0.15.0',
        'pyqtgraph>=0.9.10',
    ],
    extras_require={
        'test': [
            'nose>=1.3.2',
            'coverage>=3.7'
        ]
    }
)