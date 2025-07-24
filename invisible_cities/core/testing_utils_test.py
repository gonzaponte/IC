import numpy as np

from pytest                       import mark
from pytest                       import raises
from hypothesis                   import given
from hypothesis.strategies        import floats
from hypothesis.     extra.pandas import data_frames
from hypothesis.     extra.pandas import column
from hypothesis.     extra.pandas import range_indexes
from . testing_utils              import all_elements_close
from . testing_utils              import ignore_warning
from . testing_utils              import assert_tables_equality


@mark.parametrize("  mu  t_rel".split(),
                  ((   123,  1e-1),
                   ( 45678,  1e-3)))
def test_all_elements_close_rel(mu, t_rel):
    eps = np.finfo(np.float32).eps
    x   = np.linspace( mu - t_rel*mu + eps
                     , mu + t_rel*mu - eps
                       , 101)
    x = np.roll(x, -50) # put mu at index 0
    assert all_elements_close(x, t_rel=t_rel, t_abs=0)


@ignore_warning.divide_by_zero
@mark.parametrize("   mu  t_abs".split(),
                  ((   0,  1e-6),
                   ( 123,  1e-1)))
def test_all_elements_close_abs(mu, t_abs):
    eps = np.finfo(np.float32).eps
    x   = np.linspace( mu - t_abs + eps
                     , mu + t_abs - eps
                     , 101)
    x = np.roll(x, -50) # put mu at index 0

    assert all_elements_close(x, t_rel=0, t_abs=t_abs)


@mark.parametrize("opts", (dict(t_rel=0.1), dict(t_abs=1.234)))
def test_all_elements_close_par(opts):
    n  = 100
    t  = np.linspace(0, 2*np.pi, n) # parametric variable
    x  = np.ones(n) * 12.34
    x += np.sin (t) * 1.234 # variations are up to 10% of baseline value
    assert all_elements_close(x, **opts)


@given(data_frames([column('A', dtype=int  ),
                    column('B', dtype=float),
                    column('C', dtype=str  )],
                   index = range_indexes(max_size=5)))
def test_assert_tables_equality(df):
    table = df.to_records(index=False)
    assert_tables_equality(table, table)


def test_assert_tables_equality_withNaN():
    table = np.array([('Rex', 9, 81.0), ('Fido', 3, np.nan)],
                     dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
    assert_tables_equality(table, table)


@mark.parametrize("index value".split(), ((0, "three"), (1, 3), (2, 3.0)))
def test_assert_tables_equality_fails_different_values(index, value):
    # modify a value in the second row and check that the function
    # picks up the difference
    table1 = np.array([ ('one', 1, 1.0)
                      , ('two', 2, 2.0)],
                      dtype=[('text', 'U10'), ('integer', 'i4'), ('float', 'f4')])

    table2 = table1.copy()
    table2[1][index] = value
    with raises(AssertionError):
        assert_tables_equality(table1, table2)


def test_assert_tables_equality_fails_different_names():
    # modify the type of a column and check that the function picks up
    # the difference
    dtypes1 = [('text', 'U10'), ('integer', 'i4'), ('float', 'f4')]
    table1  = np.array([ ('one', 1, 1.0)
                       , ('two', 2, 2.0)],
                       dtype=dtypes1)

    dtypes2 = [('different_name', 'U10'), ('integer', 'i4'), ('float', 'f4')]
    table2  = np.array([ ('one', 1, 1.0)
                       , ('two', 2, 2.0)],
                       dtype=dtypes2)

    with raises(AssertionError):
        assert_tables_equality(table1, table2)


def test_assert_tables_equality_fails_different_types():
    # modify the type of a column and check that the function picks up
    # the difference
    dtypes1 = [('text', 'U10'), ('integer', 'i4'), ('float', 'f4')]
    table1  = np.array([ ('one', 1, 1.0)
                       , ('two', 2, 2.0)],
                       dtype=dtypes1)

    dtypes2    = list(dtypes1)
    dtypes2[1] = ("integer", "f4")
    table2     = table1.copy().astype(dtypes2)

    with raises(AssertionError):
        assert_tables_equality(table1, table2)


@mark.parametrize("dtype", (int, float))
def test_assert_tables_equality_equal_arrays(dtype):
    array1 = np.arange(20, dtype=dtype)
    array2 = array1.copy()
    assert_tables_equality(array1, array2)


@mark.parametrize("dtype", (int, float))
def test_assert_tables_equality_different_arrays(dtype):
    array1 = np.arange(20, dtype=dtype)
    array2 = array1 + 1
    with raises(AssertionError):
        assert_tables_equality(array1, array2)


@mark.parametrize(    "shape1  shape2".split()
                 , ( ( (3, 4), (4, 4) )    # different lengths
                   , ( (3, 3), (3, 4) )))  # different widths
def test_assert_tables_equality_different_array_shapes(shape1, shape2):
    array1 = np.ones(shape1)
    array2 = np.ones(shape2)
    with raises(AssertionError):
        assert_tables_equality(array1, array2)


def test_assert_tables_equality_different_table_shapes():
    table1 = np.array([ (1, 1.0)
                      , (2, 2.0)],
                      dtype=[('integer', 'i4'), ('float', 'f4')])

    table2 = np.array([ (1, 1.0)
                      , (2, 2.0)
                      , (3, 3.0)],
                      dtype=[('integer', 'i4'), ('float', 'f4')])

    table3 = np.array([ ('one', 1, 1.0)
                      , ('two', 2, 2.0)],
                      dtype=[('text', 'U10'), ('integer', 'i4'), ('float', 'f4')])

    # different lengths
    with raises(AssertionError):
        assert_tables_equality(table1, table2)

    # different widths
    with raises(AssertionError):
        assert_tables_equality(table1, table3)
