import warnings
from dataclasses import dataclass, field
from typing import (
    Dict,
    Optional,
    TYPE_CHECKING,
    Union,
    List,
    Iterable,
    Mapping,
)

if TYPE_CHECKING:
    from ....typing import DocumentArraySourceType, ArrayType
import numpy as np

import clickhouse_driver
from clickhouse_driver import connect

from .helper import initialize_table
from ..base.backend import BaseBackendMixin
from ....helper import random_identity, dataclass_from_dict


def _sanitize_table_name(table_name: str) -> str:
    ret = ''.join(c for c in table_name if c.isalnum() or c == '_')
    if ret != table_name:
        warnings.warn(f'The table name is changed to {ret} due to illegal characters')
    return ret


@dataclass
class ClickHouseConfig:
    n_dim: int = 768  # dims
    host: Union[
        str, List[Union[str, Mapping[str, Union[str, int]]]], None
    ] = 'localhost'
    user: str = 'default'
    password: str = ''
    port: int = 9000
    database: str = ''
    table_name: Optional[str] = None
    serialize_config: Dict = field(default_factory=dict)
    conn_config: Dict = field(default_factory=dict)


class BackendMixin(BaseBackendMixin):
    """Provide necessary functions to enable this storage backend."""

    schema_version = '0'

    def _init_storage(
        self,
        _docs: Optional['DocumentArraySourceType'] = None,
        config: Optional[Union[ClickHouseConfig, Dict]] = None,
        **kwargs,
    ):
        if not config:
            config = ClickHouseConfig()

        if isinstance(config, dict):
            config = dataclass_from_dict(ClickHouseConfig, config)

        from docarray import Document

        _conn_kwargs = dict()
        _conn_kwargs.update(config.conn_config)
        self._client = self.build_cur(config)

        self._table_name = (
            _sanitize_table_name(self.__class__.__name__ + random_identity())
            if config.table_name is None
            else _sanitize_table_name(config.table_name)
        )
        self._persist = bool(config.table_name)
        config.table_name = self._table_name
        self.n_dim = config.n_dim
        initialize_table(
            self._table_name, self.__class__.__name__, self.schema_version, self._client
        )
        self._config = config

        super()._init_storage()

        if _docs is None:
            return

        self.clear()

        if isinstance(_docs, Iterable):
            self.extend(_docs)
        elif isinstance(_docs, Document):
            self.append(_docs)

    def build_cur(self, config):
        conn = connect(
            host=config.host,
            user=config.user,
            password=config.password,
            port=config.port,
            database=config.database,
        )
        cursor = conn.cursor()
        return cursor

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['_client']
        return d

    def __setstate__(self, state):
        self.__dict__ = state
        _conn_kwargs = dict()
        _conn_kwargs.update(state['_config'].conn_config)
        self._client = self.build_cur(state['_config'])

    def _map_embedding(self, embedding: 'ArrayType') -> List[float]:
        from ....math.helper import EPSILON

        if embedding is None:
            embedding = np.zeros(self.n_dim) + EPSILON
        else:
            from ....math.ndarray import to_numpy_array

            embedding = to_numpy_array(embedding)

            if embedding.ndim > 1:
                embedding = np.asarray(embedding).squeeze()

        if np.all(embedding == 0):
            embedding = embedding + EPSILON

        return embedding.tolist()
