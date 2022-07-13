import json
import os
import pathlib
import random
import sys
import uuid
import warnings
from os.path import expanduser
from typing import Any, Dict, Optional, Sequence, Tuple, Union

__resources_path__ = os.path.join(
    os.path.dirname(
        sys.modules.get('docarray').__file__ if 'docarray' in sys.modules else __file__
    ),
    'resources',
)

__cache_path__ = f'{expanduser("~")}/.cache/{__package__}'


def typename(obj):
    """
    Get the typename of object.

    :param obj: Target object.
    :return: Typename of the obj.
    """
    if not isinstance(obj, type):
        obj = obj.__class__
    try:
        return f'{obj.__module__}.{obj.__name__}'
    except AttributeError:
        return str(obj)


def deprecate_by(new_fn, removed_at: str):
    """A helper function to label deprecated function

    Usage: old_fn_name = deprecate_by(new_fn)

    :param new_fn: the new function
    :param removed_at: removed at which version
    :return: a wrapped function with old function name
    """

    def _f(*args, **kwargs):
        import inspect

        old_fn_name = inspect.stack()[1][4][0].strip().split("=")[0].strip()
        warnings.warn(
            f'`{old_fn_name}` is renamed to `.{new_fn.__name__}()` with the same usage, please use the latter instead. The old function will be removed in {removed_at}.',
            FutureWarning,
        )
        return new_fn(*args, **kwargs)

    return _f


def dunder_get(_dict: Any, key: str) -> Any:
    """Returns value for a specified dunderkey

    A "dunderkey" is just a fieldname that may or may not contain
    double underscores (dunderscores!) for referencing nested keys in
    a dict. eg::
     >>> data = {'a': {'b': 1}}
     >>> dunder_get(data, 'a__b')

    key 'b' can be referrenced as 'a__b'

    :param _dict: (dict, list, struct or object) which we want to index into
    :param key: (str) that represents a first level or nested key in the dict
    :return: (mixed) value corresponding to the key

    """

    if not _dict:
        return None

    try:
        part1, part2 = key.split('__', 1)
    except ValueError:
        part1, part2 = key, ''

    try:
        part1 = int(part1)  # parse int parameter
    except ValueError:
        pass

    if isinstance(part1, int):
        result = _dict[part1]
    elif isinstance(_dict, dict):
        if part1 in _dict:
            result = _dict[part1]
        else:
            result = None
    elif isinstance(_dict, Sequence):
        result = _dict[part1]
    else:
        result = getattr(_dict, part1)

    return dunder_get(result, part2) if part2 else result


def random_identity(use_uuid1: bool = False) -> str:
    """
    Generate random UUID.

    ..note::
        A MAC address or time-based ordering (UUID1) can afford increased database performance, since it's less work
        to sort numbers closer-together than those distributed randomly (UUID4) (see here).

        A second related issue, is that using UUID1 can be useful in debugging, even if origin data is lost or not
        explicitly stored.

    :param use_uuid1: use UUID1 instead of UUID4. This is the default Document ID generator.
    :return: A random UUID.

    """
    return random_uuid(use_uuid1).hex


def random_uuid(use_uuid1: bool = False) -> uuid.UUID:
    """
    Get a random UUID.

    :param use_uuid1: Use UUID1 if True, else use UUID4.
    :return: A random UUID.
    """
    return uuid.uuid1() if use_uuid1 else uuid.uuid4()


def download_mermaid_url(mermaid_url, output) -> None:
    """
    Download the jpg image from mermaid_url.

    :param mermaid_url: The URL of the image.
    :param output: A filename specifying the name of the image to be created, the suffix svg/jpg determines the file type of the output image.
    """
    from urllib.request import Request, urlopen

    try:
        req = Request(mermaid_url, headers={'User-Agent': 'Mozilla/5.0'})
        with open(output, 'wb') as fp:
            fp.write(urlopen(req).read())
    except:
        raise RuntimeError('Invalid or too-complicated graph')


def get_request_header() -> Dict:
    """Return the header of request.

    :return: request header
    """
    return {k: str(v) for k, v in get_full_version().items()}


def get_full_version() -> Dict:
    """
    Get the version of libraries used in Jina and environment variables.

    :return: Version information and environment variables
    """
    import google.protobuf, platform
    from . import __version__
    from google.protobuf.internal import api_implementation
    from uuid import getnode

    return {
        'docarray': __version__,
        'protobuf': google.protobuf.__version__,
        'proto-backend': api_implementation.Type(),
        'python': platform.python_version(),
        'platform': platform.system(),
        'platform-release': platform.release(),
        'platform-version': platform.version(),
        'architecture': platform.machine(),
        'processor': platform.processor(),
        'uid': getnode(),
        'session-id': str(random_uuid(use_uuid1=True)),
        'ci-vendor': get_ci_vendor(),
    }


def get_ci_vendor() -> str:
    with open(os.path.join(__resources_path__, 'ci-vendors.json')) as fp:
        all_cis = json.load(fp)
        for c in all_cis:
            if isinstance(c['env'], str) and c['env'] in os.environ:
                return c['constant']
            elif isinstance(c['env'], dict):
                for k, v in c['env'].items():
                    if os.environ.get(k, None) == v:
                        return c['constant']
            elif isinstance(c['env'], list):
                for k in c['env']:
                    if k in os.environ:
                        return c['constant']
        return 'unset'


assigned_ports = set()
unassigned_ports = []
DEFAULT_MIN_PORT = 49153
MAX_PORT = 65535


def reset_ports():
    def _get_unassigned_ports():
        # if we are running out of ports, lower default minimum port
        if MAX_PORT - DEFAULT_MIN_PORT - len(assigned_ports) < 100:
            min_port = int(os.environ.get('JINA_RANDOM_PORT_MIN', '16384'))
        else:
            min_port = int(
                os.environ.get('JINA_RANDOM_PORT_MIN', str(DEFAULT_MIN_PORT))
            )
        max_port = int(os.environ.get('JINA_RANDOM_PORT_MAX', str(MAX_PORT)))
        return set(range(min_port, max_port + 1)) - set(assigned_ports)

    unassigned_ports.clear()
    assigned_ports.clear()
    unassigned_ports.extend(_get_unassigned_ports())
    random.shuffle(unassigned_ports)


def random_port() -> Optional[int]:
    """
    Get a random available port number.

    :return: A random port.
    """

    def _random_port():
        import socket

        def _check_bind(port):
            with socket.socket() as s:
                try:
                    s.bind(('', port))
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    return port
                except OSError:
                    return None

        _port = None
        if len(unassigned_ports) == 0:
            reset_ports()
        for idx, _port in enumerate(unassigned_ports):
            if _check_bind(_port) is not None:
                break
        else:
            raise OSError(
                f'can not find an available port in {len(unassigned_ports)} unassigned ports, assigned already {len(assigned_ports)} ports'
            )
        int_port = int(_port)
        unassigned_ports.pop(idx)
        assigned_ports.add(int_port)
        return int_port

    try:
        return _random_port()
    except OSError:
        assigned_ports.clear()
        unassigned_ports.clear()
        return _random_port()


class cached_property:
    """The decorator to cache property of a class."""

    def __init__(self, func):
        """
        Create the :class:`cached_property`.

        :param func: Cached function.
        """
        self.func = func

    def __get__(self, obj, cls):
        cached_value = obj.__dict__.get(f'CACHED_{self.func.__name__}', None)
        if cached_value is not None:
            return cached_value

        value = obj.__dict__[f'CACHED_{self.func.__name__}'] = self.func(obj)
        return value

    def __delete__(self, obj):
        cached_value = obj.__dict__.get(f'CACHED_{self.func.__name__}', None)
        if cached_value is not None:
            if hasattr(cached_value, 'close'):
                cached_value.close()
            del obj.__dict__[f'CACHED_{self.func.__name__}']


def compress_bytes(data: bytes, algorithm: Optional[str] = None) -> bytes:
    if algorithm == 'lz4':
        import lz4.frame

        data = lz4.frame.compress(data)
    elif algorithm == 'bz2':
        import bz2

        data = bz2.compress(data)
    elif algorithm == 'lzma':
        import lzma

        data = lzma.compress(data)
    elif algorithm == 'zlib':
        import zlib

        data = zlib.compress(data)
    elif algorithm == 'gzip':
        import gzip

        data = gzip.compress(data)
    return data


def decompress_bytes(data: bytes, algorithm: Optional[str] = None) -> bytes:
    if algorithm == 'lz4':
        import lz4.frame

        data = lz4.frame.decompress(data)
    elif algorithm == 'bz2':
        import bz2

        data = bz2.decompress(data)
    elif algorithm == 'lzma':
        import lzma

        data = lzma.decompress(data)
    elif algorithm == 'zlib':
        import zlib

        data = zlib.decompress(data)
    elif algorithm == 'gzip':
        import gzip

        data = gzip.decompress(data)
    return data


def get_compress_ctx(algorithm: Optional[str] = None, mode: str = 'wb'):
    if algorithm == 'lz4':
        import lz4.frame

        compress_ctx = lambda x: lz4.frame.LZ4FrameFile(x, mode)
    elif algorithm == 'gzip':
        import gzip

        compress_ctx = lambda x: gzip.GzipFile(fileobj=x, mode=mode)
    elif algorithm == 'bz2':
        import bz2

        compress_ctx = lambda x: bz2.BZ2File(x, mode)
    elif algorithm == 'lzma':
        import lzma

        compress_ctx = lambda x: lzma.LZMAFile(x, mode)
    else:
        compress_ctx = None
    return compress_ctx


def dataclass_from_dict(klass, dikt):
    try:
        fieldtypes = klass.__annotations__
        return klass(**{f: dataclass_from_dict(fieldtypes[f], dikt[f]) for f in dikt})
    except AttributeError:
        if isinstance(dikt, (tuple, list)):
            return [dataclass_from_dict(klass.__args__[0], f) for f in dikt]
        return dikt


def protocol_and_compress_from_file_path(
    file_path: str,
    default_protocol: Optional[str] = None,
    default_compress: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Extract protocol and compression algorithm from a string, use defaults if not found.

    :param file_path: path of a file.
    :param default_protocol: default serialization protocol used in case not found.
    :param default_compress: default compression method used in case not found.

    Examples:

    >>> protocol_and_compress_from_file_path('./docarray_fashion_mnist.protobuf.gzip')
    ('protobuf', 'gzip')

    >>> protocol_and_compress_from_file_path('/Documents/docarray_fashion_mnist.protobuf')
    ('protobuf', None)

    >>> protocol_and_compress_from_file_path('/Documents/docarray_fashion_mnist.gzip')
    (None, gzip)
    """

    ALLOWED_PROTOCOLS = {'pickle', 'protobuf', 'protobuf-array', 'pickle-array'}
    ALLOWED_COMPRESSIONS = {'lz4', 'bz2', 'lzma', 'zlib', 'gzip'}

    protocol = default_protocol
    compress = default_compress

    file_extensions = [e.replace('.', '') for e in pathlib.Path(file_path).suffixes]
    for extension in file_extensions:
        if extension in ALLOWED_PROTOCOLS:
            protocol = extension
        elif extension in ALLOWED_COMPRESSIONS:
            compress = extension

    return protocol, compress


def add_protocol_and_compress_to_file_path(
    file_path: str, protocol: Optional[str] = None, compress: Optional[str] = None
) -> str:
    """Creates a new file path with the protocol and compression methods as extensions.

    :param file_path: path of a file.
    :param protocol: chosen protocol.
    :param compress: compression algorithm.

    Examples:

    >>> add_protocol_and_compress_to_file_path('docarray_fashion_mnist.bin')
    'docarray_fashion_mnist.bin'

    >>> add_protocol_and_compress_to_file_path('docarray_fashion_mnist', 'protobuf', 'gzip')
    'docarray_fashion_mnist.protobuf.gzip'
    """

    file_path_extended = file_path
    if protocol:
        file_path_extended += '.' + protocol
    if compress:
        file_path_extended += '.' + compress

    return file_path_extended


def filter_dict(d: Dict) -> Dict:
    """Removes `None` values from dict `d`.

    :param d: input dict
    :return: filtered dict
    """
    return dict(filter(lambda item: item[1] is not None, d.items()))


def _safe_cast_int(value: Union[str, int, float]) -> int:
    """Safely cast string and float to an integer
    It mainly avoids silently rounding down the float value
    :param value: value to be cast
    :return: cast integer
    """
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"Can't safely cast {value} to an int")
    return int(value)
