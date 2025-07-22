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
   and testing. Use function composition to achieve more complex behavior. A
   function exceeding 15-20 lines probably does more than one thing, consider
   splitting it.

6. Avoid repetition. DRY principle: don't repeat yourself. If you end up
   repeating a piece of code multiple times, factor it out. But bear in mind
   that duplication of small snippets can sometimes be preferable to premature
   abstraction.

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
   Documentation of the obscure code is obviously also helpful.

2. We assume that the reader is familiar and comfortable with the main
   scientific packages used in the repository:
   - numpy
   - pandas
   - scipy

   This means that expressions using these libraries are preferred over longer
   equivalents using builtin functions.


Specific style guidelines
-------------------------

1. We use `snake_case` for local variables, `UPPERCASE` for global variables and
   `CamelCase` for classes.

2. Do not use tabs, but spaces. We use 4 spaces per indentation level.

3. Symmetry helps readability. Try to align repeated characters or tokens across
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


4. Similarly, when writing a long expression, it is helpful to split it in
   multiple lines with alignment. For instance:

   .. code-block:: python

      total = (  first_term
              + second_term
              -  third_term * factor)

   Note how the factor is written together with the third term to highlight its
   precedence in the operation.

5. Use two blank lines to separate chunks in the global scope. Use small gaps to
   separate chunks within any local scope. Examples:

   - Use two blank lines between imports and global variables (if present)

   - Use two blank lines between global variables and function definitions in
     the main scope

   - Use a single blank line to differentiate steps within a function

6. Type-hint your functions' arguments and output. Declare custom local types if
   necessary.

7. If a function has many arguments, it's better to split them across multiple
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

   Note: using the commas at the beginning of the line instead of at the end has
   two benefits:

   - provides a reference for the eye, making it easier to follow

   - helps to avoid noise in the diff algorithm used by `git` when adding or
     removing lines

8. Document your functions using docstrings with the numpy style:

   .. code-block:: python

       """
       Description

       Parameters
       ----------
       arg1: np.ndarray, shape (n,m)
           description

       arg2: float, optional
           description... (default = 1.0)

       ...

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

9. Function composition is preferred over inheritance. In general, we avoid
   classes as understood in OOP. Dataclasses (classes containing only data),
   namedtuples (tuples with named positional elements), and similar structures
   are accepted.

10. We use a one-import per line scheme. Multiple imports from the same module
    are split across multiple lines. If a module contains many useful functions,
    it is usually preferred to import the whole module (usually abbreviated) and
    qualify the function usage with `module.function`. Star imports are not
    accepted.

    - Order of imports (blank line between groups):

      .. code-block::python

          standard lib

          third-party libraries

          IC modules

          typing

11. Idiomatic python expressions are favoured. For instance
    ``if len(some_list) > 0: ...``
    is written more idiomatically as
    ``if some_list: ...``

    - Avoid superfluous parentheses. For instance:
      ``if (a > b): ...`` reads better as ``if a > b: ...``

12. Add spaces around operators such as ``+ - * / = > < ^``. Somehow ``**`` is
    different and feels better without spaces around it.

13. Avoid extraneous whitespaces, unless they favour another guideline (e.g.
    they improve the symmetry). Examples:

    - before a comma or colon

    - immediately before or after opening/closing parentheses or brackets

14. If your function has a behaviour that depends on a parameter with a reduced
    amount of possible values, consider using a symbol. Most programmers use an
    argument in the form of a string and modify their behaviour if this string
    matches a few specific values. Symbols are both safer and clearer. The mere
    existence of a symbol indicates the limited range of options available. And
    their definition indicates which ones are valid. Besides, if you use an IDE,
    you can figure out the possible values and have it type-checked on-the-fly.
    Errors due to typos are much simpler to understand with symbols.

15. City "components" are usually named as nouns while their inner functions are
    usually named as verbs. Example: ``hits_thresholder`` (component) vs.
    ``threshold_hits`` (inner function)
