Code style guidelines and conventions
=====================================

The following principle guides our approach to code style:

    *"Code is read more often than it is written -
    write for the reader, not for the machine."*



These guidelines aim to make new IC code easier to read, understand, and
maintain. They are not strict rules but principles to help developers write
clear, self-explanatory Python code. We prioritize readability above conventions
like PEP8. Here we list some conventions that are a bit more generic and widely
adopted and other ones that are more specific to this repository.

.. note::

   Bear in mind that many of the following guidelines are subjective and do not
   have well-defined boundaries. Try to be reasonable and argue your points, but
   in most cases, there is room for interpretation.


General style guidelines
------------------------

1. Keep lines a reasonable length, typically below 80 characters

2. Docstrings are helpful, but don't rely on them for someone to understand your
   code. The code should be as self-explanatory as possible

3. Don't show off. Prefer readability over cleverness. A simple, clear solution
   is almost always better than an elegant but obscure one.

4. Use meaningful names for your variables and functions, they help keep track
   of the story.
   - Avoid cryptic or very abbreviated names, unless widely understood in the
   context.
   - Be sensible about the length; nobody likes
   ``a_very_very_very_long_variable_name``. Find a compromise.
   - A variable name should reflect its purpose, not the implementation details.

5. Keep functions small and focused on one thing. This helps both readability
   and testing. Use `functional decomposition
   <en.wikipedia.org/wiki/Functional_decomposition>`_ to achieve more complex
   behavior instead of writing monolithic functions. A function exceeding 15-20
   lines probably does more than one thing; consider splitting it.

6. Avoid repetition. `DRY principle: don't repeat yourself
   <en.wikipedia.org/wiki/Don't_repeat_yourself>`_. If you end up repeating a
   piece of code multiple times, factor it out. But bear in mind that
   duplication of small snippets can sometimes be preferable to premature
   abstraction. There isn't alwaya a clear cut, but large chunks of repeated
   code or chunks of any size that end up repeating multiple times tend to
   benefit from the DRY principle.

7. Comment your code as much as needed, but as little as necessary. That is, add
   comments wherever they add value; avoid superfluous comments. If you need to
   comment your code, focus on the "why", the "what" should be apparent.
   Needless to say, update the comments at the same time as the code. Outdated
   comments are worse than no comments.

8. Use clear and meaningful error messages where needed. Example:

   .. code-block:: python

      raise SomeException("this function failed")

   would be more useful like

   .. code-block:: python

      raise SomeException("this function failed because of this and here "
                          "are some hints on what you should look for")


Notes
~~~~~
1. When performance is critically impacted and a piece of code must be written
   in a more obscure fashion, it is recommended to have both the fast and slow
   versions of the code available and verified using the same tests.
   Documentation of the obscure code is also helpful.

2. We assume that the reader is familiar and comfortable with the main
   scientific packages used in the repository:

   - numpy

   - pandas

   - scipy

   This means that expressions using these libraries are preferred over longer
   equivalents using builtin functions.


Specific style guidelines
-------------------------

1. We use ``snake_case`` for local variables, ``UPPERCASE`` for global variables
   and ``CamelCase`` for classes.

2. Do not use tabs, but spaces. We use 4 spaces per indentation level.

3. Idiomatic python expressions are favored. For instance
   ``if len(some_list) > 0: ...``
   is written more idiomatically as
   ``if some_list: ...``

   - Avoid superfluous parentheses. For instance:
     ``if (a > b): ...`` reads better as ``if a > b: ...``

4. Add spaces around operators such as ``+ - * / = > < ^``. Somehow ``**`` is
   different and feels better without spaces around it.

5. Avoid extraneous whitespaces, unless they favour another guideline (e.g.
   they improve the symmetry). Examples of scenarios to avoid:

   - before a comma or colon ( ``if condition :`` or ``(1 , 2 , 3)``)

   - immediately before or after opening/closing parentheses or brackets
     (``a_function ( x, y )``)

6. Symmetry helps readability. Try to align repeated characters or tokens across
   connected lines. For instance

   .. code-block:: python

      result = fun(a = "1", b = 2)
      result_2 = fun(a = "3.14", b = 4)

   reads better like

   .. code-block:: python

      result   = fun(a = "1"   , b = 2)
      result_2 = fun(a = "3.14", b = 4)

   Other cases where it is convenient to exploit the alignment:

   - if/else clauses:
   .. code-block:: python

      if   a<b: c= 1
      elif a>b: c=-1
      else    : c= 0

   - multiline variable definition:
   .. code-block:: python

      matrix = [
          [  1  ,   2  ,   3  , ],
          [  4.5,   6.7,   8.9, ],
          [ 10  ,  11  ,  12  , ],
      ]

   Note how in these cases we bent some of the other rules to improve the
   readability:

   - Removing spaces around `< > =`

   - Used a double whitespace in between matrix columns


7. Similarly, when writing a long expression, it is helpful to split it in
   multiple lines with alignment. For instance:

   .. code-block:: python

      total = (  first_term
              + second_term
              -  third_term * factor)

   Note how the factor is written together with the third term to highlight its
   precedence in the operation.

8. Use two blank lines to separate chunks in the global scope. Use small gaps to
   separate chunks within any local scope. Examples:

   - Use two blank lines between imports and global variables (if present)

   - Use two blank lines between global variables and function definitions in
     the main scope

   - Use a single blank line to differentiate steps within a function

   Example:

   .. code-block:: python

      import os
      import sys

      import numpy as np

      from invisible_cities.database.load_db import DataPMT

      from typing import Optional


      EPSILON = np.finfo(np.float64).eps


      def a_function(...):
          step_11 = ...
          step_12 = ...

          step_2 = ...

          step_31 = ...
          step_32 = ...


      def another_function(....):
          [...]

9. Type-hint your functions' arguments and output. Declare custom local types if
   necessary. For instance

   .. code-block:: python

      def f(a, b, c):
          [...]

   would become

   .. code-block:: python

      def f(a: int, b:float, c:np.ndarray) -> pd.DataFrame:
          [...]

10. If a function has many arguments, it's better to split them across multiple
    lines, specially when the line becomes long. For instance:

    .. code-block:: python

       def f(var_a: float, var_b: int, var_c: Union[str, NoneType]) -> (str, bool):
           [...]

    reads better like

    .. code-block:: python

       def f( var_a: float
            , var_b: int
            , var_c: Union[str, NoneType]
            ) -> (str, bool):
            [...]

    Note: using the commas at the beginning of the line instead of at the end is a
    personal preference of the editor, but it has two benefits:

    - provides a reference for the eye, making it easier to follow

    - helps to avoid noise in the diff algorithm used by `git` when adding or
      removing lines

11. Document your functions using docstrings with the numpy style:

    .. code-block:: python

       """
       Description

       Parameters
       ----------
       arg1: np.ndarray, shape (n,m)
           description

       arg2: float, optional
           description... (default = 1.0)

       Returns
       -------
       whatever: pd.DataFrame
           description

       Raises
       ------
       exception1:
           description

       exception2:
           description

       Notes
       -----
       notes

       References
       ----------
       references

       Examples
       --------
       examples
       """

    Not all sections are necessary. Most of the time only the first two are
    needed.

12. A procedural approach is preferred over an object-oriented one. In general,
    we tend to keep the data containers and algorithmic parts separate.
    Specifically, we avoid classes as understood in OOP. Dataclasses (classes
    containing only data), namedtuples (tuples with named positional elements),
    and similar structures are welcome. Class methods are implemented as
    independent functions.

13. We use a one-import per line scheme. Multiple imports from the same module
    are split across multiple lines. If a module contains many useful functions,
    it is usually preferred to import the whole module (usually abbreviated) and
    qualify the function usage with `module.function`. Star imports are not
    accepted.

    - Order of imports (blank line between groups):

    .. code-block:: python

        standard lib

        third-party libraries

        IC modules

        typing

14. If your function has a behaviour that depends on a parameter with a reduced
    amount of possible values, consider using a `symbol
    <en.wikipedia.org/wiki/Symbol_(programming)>`_ (a.k.a. enumerated
    constants or atoms). Most programmers use an argument in the form of a string and
    modify their behaviour if this string matches a few specific values. Symbols
    are both safer and clearer. The mere existence of a symbol indicates the
    limited range of options available. And their definition indicates which
    ones are valid. Besides, if you use an IDE, you can figure out the possible
    values and have it type-checked on-the-fly. Errors due to typos are much
    simpler to understand with symbols. Example:

    .. code-block:: python

       def subtract_baseline(wf: np.ndarray, method: str) -> np.ndarray:
           if method == "mode"  : pass #subtract mode
           if method == "mean"  : pass #subtract mean
           if method == "median": pass #subtract median

    would be better as

    .. code-block:: python

       def subtract_baseline(wf: np.ndarray, method: BaselineMethod) -> np.ndarray:
           if method is BaselineMethod.mode  : pass #subtract mode
           if method is BaselineMethod.mean  : pass #subtract mean
           if method is BaselineMethod.median: pass #subtract median

    Note how just from the signature of the function we can infer some useful
    information. In the first function we accept a string. Which values are
    valid? The second function already tells us that it must be one of the
    fields of ``BaselineMethod``. The second advantage is in the comparison
    step. Symbols allow us to compare *identity* instead of *equality*. In the
    first case there is some room for trickery, as we could implement ``__eq__``
    to satisfy some of thoe equalities. We cannot do the same with ``is``.

15. City "components" are functions that usually just delegate on another
    function but fixes some configuration and/or precalculates some data for
    them. These components are usually named as nouns while their inner
    functions are usually named as verbs. Example: the component
    ``hits_thresholder`` calls the inner function ``threshold_hits``.
