def sum_list(numbers):
    """Given a list of integers 'numbers'
    return the sum of this list."""
    return sum(numbers)


def max_value(numbers):
    """Given a list of numbers 'numbers'
    return the maximum value of this list."""
    return max(numbers)


def reverse_string(s):
    """Given a string 'string'
    return the reversed version of the input string."""
    return s[::-1]


def filter_even(numbers):
    """Given a list of numbers 'numbers'
    return a list containing only the even numbers from the input list."""
    even = []
    for n in numbers:
      if n % 2 == 0:
        even.append(n)
    return even


def get_fifth_row(df):
    """Given a dataframe 'df'
    return the fifth row of this as a pandas DataFrame."""
    return df.iloc[4]


def column_mean(df, column):
    """Given a dataframe 'df' and the name of a column 'column'
    return the mean of the specified column in a pandas DataFrame."""
    return df[column].mean()


def lookup_key(d, key):
    """Given a dictionary 'd' and a key 'key'
    return the value associated with the key in the dictionary."""
    return d.get(key)


def count_occurrences(lst):
    """Given a list 'lst'
    return a dictionary with counts of each unique element in the list."""
    counts = {}
    for item in lst:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1
    return counts


def drop_missing(df):
    """Given a dataframe 'df' with some rows containing missing values,
    return a DataFrame with rows containing missing values removed."""
    return df.dropna()


def value_counts_df(df, column):
    """Given a dataframe 'df' with various columns and the name of one of those columns 'column',
    return a DataFrame with value counts of the specified column."""
    vc = df[column].value_counts()
    out = vc.reset_index()
    out.columns = [column, "count"]
    return out
