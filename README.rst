IC: Invisible Cities
==============================================

IC stands for Invisible Cities and also for Italo Calvino, the author of the master piece.

Quickstart guide
----------------

+ Make sure you have `git lfs` installed (https://git-lfs.github.com/)

+ Clone the repository

+ :code:`cd` into the `IC` directory

+ :code:`source manage.sh work_in_python_version 3.7`

The last step will, if necessary, install conda and create an
appropriate conda environment, as well as setting environment
variables which are needed for the correct functioning of IC.

The installation steps will be skipped if conda and the required
environment are already available, so subsequent invocations of the
command should be much quicker than the first.

You will need to perform the last two steps in every shell in which
you want to run IC.

To check your progress when you are developing, you will want to
compile any Cython components that have changed and run the test
suite. This can be done with

.. code-block::

   source manage.sh compile_and_test_par

If you check out a commit which requires an older set of dependencies,
the :code:`compile_and_test` commands will automatically switch to an
appropriate environment, creating it on the fly if necessary.

[![Test suite](https://github.com/gonzaponte/IC/actions/workflows/test_suite.yml/badge.svg)](https://github.com/gonzaponte/IC/actions/workflows/test_suite.yml)

[![Calibration check](https://github.com/gonzaponte/IC/actions/workflows/check_db_calibration.yml/badge.svg)](https://github.com/gonzaponte/IC/actions/workflows/check_db_calibration.yml)
