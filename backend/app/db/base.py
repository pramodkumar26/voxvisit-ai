import uuid
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, ClassVar

from sqlalchemy import BigInteger, CheckConstraint, DateTime, MetaData, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map: ClassVar[dict[Any, Any]] = {
        datetime: DateTime(timezone=True),
        dict[str, Any]: JSONB,
    }


uuid_pk = Annotated[
    uuid.UUID,
    mapped_column(primary_key=True, server_default=text("gen_random_uuid()")),
]
created_at = Annotated[datetime, mapped_column(server_default=func.now())]
updated_at = Annotated[
    datetime,
    mapped_column(server_default=func.now(), onupdate=func.now()),
]
row_version = Annotated[int, mapped_column(BigInteger, server_default=text("1"))]


def one_of(column: str, values: type[StrEnum], name: str | None = None) -> CheckConstraint:
    allowed = ", ".join(f"'{value}'" for value in values)
    return CheckConstraint(f"{column} IN ({allowed})", name=name or f"{column}_valid")


def not_blank(column: str) -> CheckConstraint:
    return CheckConstraint(f"length(btrim({column})) > 0", name=f"{column}_not_blank")
