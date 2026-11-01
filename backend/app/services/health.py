from sqlalchemy import text
from sqlalchemy.engine import Engine


def verify_database_connection(engine: Engine) -> None:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
