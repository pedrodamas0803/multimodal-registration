import os


def get_extension(path: str) -> str:
    """Return the file extension without the leading dot, lower-cased.

    Examples
    --------
    >>> get_extension("scan.TIFF")
    'tiff'
    >>> get_extension("data.h5")
    'h5'
    """
    _, ext = os.path.splitext(path)
    return ext.lstrip(".").lower()
