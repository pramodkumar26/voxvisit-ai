from datetime import datetime
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy import Engine

from app.auth.service import Clock


def get_engine(request: Request) -> Engine:
    return request.app.state.db_engine


def get_clock(request: Request) -> Clock:
    return request.app.state.clock


def get_now(request: Request) -> datetime:
    return request.app.state.clock()


EngineDep = Annotated[Engine, Depends(get_engine)]
NowDep = Annotated[datetime, Depends(get_now)]
