import clickhouse_driver


def initialize_table(
    table_name: str,
    container_type_name: str,
    schema_version: str,
    cur: clickhouse_driver.dbapi.cursor.Cursor,
) -> None:
    if not _is_metadata_table_initialized(cur):
        _do_initialize_metadata_table(cur)

    if not _is_table_initialized(table_name, container_type_name, schema_version, cur):
        _do_create_table(table_name, cur)
        _do_tidy_table_metadata(table_name, container_type_name, schema_version, cur)


def _is_metadata_table_initialized(cur: clickhouse_driver.dbapi.cursor.Cursor) -> bool:
    try:
        _ = cur.execute('SELECT 1 FROM metadata LIMIT 1')
        return True
    except Exception as _:
        pass
    return False


def _do_initialize_metadata_table(cur: clickhouse_driver.dbapi.cursor.Cursor) -> None:
    cur.execute(
        '''
        CREATE TABLE metadata
        (
            `table_name` TEXT NOT NULL,
            `schema_version` TEXT NOT NULL,
            `container_type` TEXT NOT NULL
        )
        ENGINE = MergeTree
        PRIMARY KEY table_name
        ORDER BY (table_name, container_type)
        '''
    )


def _do_create_table(
    table_name: str,
    cur: clickhouse_driver.dbapi.cursor.Cursor,
) -> None:
    cur.execute(
        f'''
        CREATE TABLE IF NOT EXISTS {table_name}
        (
            `doc_id` TEXT NOT NULL,
            `serialized_value` BLOB NOT NULL,
            `embedding` Array(Float32) NOT NULL,
            `text` String,
            `item_order` INTEGER,
            `indx` INTEGER
        )
        ENGINE = MergeTree
        ORDER BY indx
        '''
    )


def _is_table_initialized(
    table_name: str,
    container_type_name: str,
    schema_version: str,
    cur: clickhouse_driver.dbapi.cursor.Cursor,
) -> bool:
    try:
        cur.execute(
            'SELECT schema_version FROM metadata WHERE table_name=? AND container_type=?',
            (table_name, container_type_name),
        )
        buf = cur.fetchone()
        if buf is None:
            return False
        version = buf[0]
        if version != schema_version:
            return False
        cur.execute(f'SELECT 1 FROM {table_name} LIMIT 1')
        _ = list(cur)
        return True
    except Exception as _:
        pass
    return False


def _do_tidy_table_metadata(
    table_name: str,
    container_type_name: str,
    schema_version: str,
    cur: clickhouse_driver.dbapi.cursor.Cursor,
) -> None:
    cur.execute(
        f"INSERT INTO metadata VALUES ('{table_name}', '{schema_version}', '{container_type_name}')"
    )
