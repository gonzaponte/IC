=======================
Python Code Style Guide
=======================

Overview
========

This document defines the coding style guidelines for this repository.

Our primary guiding principle is **readability**. While we are broadly aligned
with the philosophy of PEP8, we do not follow it strictly. Instead, we favor
clarity, visual structure, and symmetry when writing code.

Whenever a rule conflicts with readability, **readability takes precedence**.


General Principles
==================

- Code is written for humans first, machines second.
- Prefer clarity over cleverness.
- Use visual alignment and symmetry to make structure explicit.
- Maintain consistency within a file or module.
- When in doubt, fall back to PEP8 unless it harms readability.


Imports
=======

- Use **one import per line**, even when importing from the same module.

  .. code-block:: python

      from math import log
      from math import sqrt

- Separate import blocks with **one blank line**. Each block should refer to one
  of these categories
  - Standard library (except typing)
  - Scientific libraries
  - This repository
    - May be splitted into smaller blocks when many imports are necessary
  - Typing

  .. code-block:: python

      import os
      import sys

      import tables as tb
      import pandas as pd

      from .. core.core_functions import in_range

      from typing import Tuple

- Don't you dare to use a **star import** :)


Whitespace and Vertical Structure
=================================

Blank lines are used to structure code visually:

- **Two blank lines** between:
  - Imports and code
  - Top-level function definitions
  - Top-level class definitions

- **One blank line**:
  - To separate logical blocks within a function or method

Example:

.. code-block:: python

    def process_data(data):
        cleaned = clean_data(data)

        transformed0 = transform0(cleaned)
        transformed1 = transform1(transformed0)
        transformed2 = transform2(transformed1)

        return transformed2

**Avoid spurious blank lines. For instance, at the beginning of a function definition:

.. code-block:: python

    def process_data(data):

        cleaned = clean_data(data)
        # ...


Alignment and Symmetry
======================

We intentionally use alignment to highlight structure and relationships
between lines.

- Extra spaces around operators (e.g., ``=``) are allowed when they improve
  readability. Use the minimum number of spaces that provide symmetry.
- Align related assignments to emphasize symmetry.
- Align repeated words across lines
- With repeated function calls, it's usually better to align arguments across lines.
- Whenever reasonable, align numbers to the right

Example:

.. code-block:: python

    x_low  =    0
    x_high =   10
    y_low  = -  5
    y_high =  105

This is preferred over:

.. code-block:: python

    x_low = 0
    x_high = 10
    y_low = -5
    y_high = 105


Naming Conventions
==================

- Variables and functions: ***snake_case**

  .. code-block:: python

      total_energy = compute_energy(event)

- Classes: **CamelCase**

  .. code-block:: python

      class EventReconstructor:
          pass

- Constants (if used): **UPPER_CASE**

  .. code-block:: python

      MAX_ITERATIONS = 100


Function Definitions
====================

- Use **two blank lines** between top-level function definitions.

- For functions with many arguments:
  - Split arguments across multiple lines
  - Align them for readability

Examples (without annotations):

.. code-block:: python

    def simulate_event(position,
                       energy,
                       time,
                       medium,
                       detector,
                      ):
        pass

.. code-block:: python

    def simulate_event( position
                      , energy
                      , time
                      , mediume
                      , detectore
                      ):
        pass

- Closing parenthesis should align with the start of the statement.



Line Length
===========

- Follow PEP8's general guidance (~79–100 characters), but:
  - Prefer breaking lines to preserve clarity
  - Do not sacrifice readability to meet strict limits


Comments
========

- Use comments to explain **why**, not **what**.
  - Anyone that understands python, should be able to understand **what** the
    code is doing. They might not understand **why**, if that's the case, leave
    a comment.
- Avoid redundant comments.

Bad:

.. code-block:: python

    x = x - 1  # decrement x

Good:

.. code-block:: python

    # np.digitize uses 0 and N+1 to denote values out of the given range. We
    # ensured that those cases are not present beforehand and it's more
    # convenient to have numbers between 0 and N for indexing.
    x = x - 1


Type Hints
==========

- Document your functions with type hints. It helps the reader understand what
  the function expects without reading the documentation or the code.
- Use the most general type hint that applies to your function. For instance, if
  you only need to apply `len`, do not type hint it with `List` or `np.ndarray`,
  but with `Sequence`.


Variable Naming
===============

- Use meaningful variable names. Prefer `distance` over `d`.
- Find a compromise. Prefer `distance_ab` over `distance_voxel_a_to_voxel_b`.


Avoid unnecessary nesting
=========================

When possible, handle exceptional or trivial cases early to avoid unnecessary
nesting.

Prefer:

.. code-block:: python

    def normalize(values):
        if len(values) == 0:
            return values

        scale = max(values)
        return values / scale

over:

.. code-block:: python

    def normalize(values):
        if len(values) == 0:
            return values
        else:
            scale = max(values)
            return values / scale

Do one thing, but do it well
============================

A function should implement a coherent task. If you find yourself writing a long
function, identify logical blocks and extract them into helper functions. This
is better both for readability and testing. This is not a strict rule about
length, but about clarity of purpose.

For instance, consider:

Example
=======

Avoid functions that perform multiple unrelated tasks. Instead, split the logic
into smaller, focused functions.

Bad example:

.. code-block:: python

    def process_event(file_path, threshold, calibration, lifetime):
        # read data
        data = load_file(file_path)
        hits = decode_hits(data)

        # calibrate
        hits = apply_calibration(hits, calibration)
        hits = apply_lifetime_correction(hits, lifetime)

        # filter
        selected = [h for h in hits if h.energy > threshold]

        # compute observables
        total_energy = sum(h.energy for h in selected)
        mean_time    = sum(h.time for h in selected) / len(selected) if selected else 0

        # format output
        result = {
            "n_hits"      : len(selected),
            "energy"      : total_energy,
            "mean_time"   : mean_time,
        }

        return result


Clean version:

.. code-block:: python

    def process_event(file_path, threshold, calibration, lifetime):
        hits     = load_hits(file_path)
        hits     = calibrate_hits(hits, calibration, lifetime)
        selected = select_hits(hits, threshold)

        return summarize_hits(selected)


    def load_hits(file_path):
        data = load_file(file_path)
        return decode_hits(data)


    def calibrate_hits(hits, calibration, lifetime):
        hits = apply_calibration(hits, calibration)
        hits = apply_lifetime_correction(hits, lifetime)

        return hits


    def select_hits(hits, threshold):
        return [h for h in hits if h.energy > threshold]


    def summarize_hits(hits):
        total_energy = sum(h.energy for h in hits)
        mean_time    = sum(h.time for h in hits) / len(hits)

        return {
            "n_hits"    : len(hits),
            "energy"    : total_energy,
            "mean_time" : mean_time,
        }
