========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/pybayts/badge/?style=flat
    :target: https://readthedocs.org/projects/pybayts
    :alt: Documentation Status

.. |travis| image:: https://api.travis-ci.com/developmentseed/pybayts.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.com/github/developmentseed/pybayts

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/developmentseed/pybayts?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/developmentseed/pybayts

.. |requires| image:: https://requires.io/github/developmentseed/pybayts/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/developmentseed/pybayts/requirements/?branch=master

.. |codecov| image:: https://codecov.io/gh/developmentseed/pybayts/branch/master/graphs/badge.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/developmentseed/pybayts

.. |version| image:: https://img.shields.io/pypi/v/pybayts.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/pybayts

.. |wheel| image:: https://img.shields.io/pypi/wheel/pybayts.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/pybayts

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/pybayts.svg
    :alt: Supported versions
    :target: https://pypi.org/project/pybayts

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/pybayts.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/pybayts

.. |commits-since| image:: https://img.shields.io/github/commits-since/developmentseed/pybayts/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/developmentseed/pybayts/compare/v0.0.0...master



.. end-badges

A python port of R's bayts package based on Reiche 2015 and Reiche 2018.

* Free software: GNU Lesser General Public License v3 (LGPLv3)

Installation
============

::

    pip install pybayts

You can also install the in-development version with::

    pip install https://github.com/developmentseed/pybayts/archive/master.zip


Documentation
=============


https://pybayts.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
