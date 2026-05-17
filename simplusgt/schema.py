"""Typed input schema for SimplusGT cases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _num(value: Any) -> Any:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "nan":
            return float("nan")
        if lowered in {"inf", "+inf", "infinity", "+infinity"}:
            return float("inf")
        if lowered in {"-inf", "-infinity"}:
            return float("-inf")
    return value


def _as_records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    raise TypeError(f"Expected a record or list of records, got {type(value).__name__}")


@dataclass(frozen=True)
class Basic:
    Fs: float
    Fbase: float
    Sbase: float
    Vbase: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Basic":
        return cls(**{key: float(_num(data[key])) for key in ("Fs", "Fbase", "Sbase", "Vbase")})


@dataclass(frozen=True)
class Advance:
    DiscretizationMethod: int = 2
    LinearizationTimes: int = 1
    DiscretizationDampingFlag: int = 1
    DirectFeedthrough: int = 0
    PowerFlowAlgorithm: int = 1
    EnableCreateSimulinkModel: int = 0
    EnablePlotPole: int = 0
    EnablePlotAdmittance: int = 0
    EnablePrintOutput: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Advance":
        values = {field_name: int(_num(data.get(field_name, getattr(cls, field_name))))
                  for field_name in cls.__dataclass_fields__
                  if field_name != "raw"}
        values["raw"] = dict(data)
        return cls(**values)


@dataclass(frozen=True)
class Bus:
    BusNo: int
    BusType: int
    Voltage: float
    Theta: float
    PGi: float
    QGi: float
    PLi: float
    QLi: float
    Qmin: float
    Qmax: float
    AreaNo: int
    AcDc: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Bus":
        return cls(
            BusNo=int(_num(data["BusNo"])),
            BusType=int(_num(data["BusType"])),
            Voltage=float(_num(data["Voltage"])),
            Theta=float(_num(data["Theta"])),
            PGi=float(_num(data["PGi"])),
            QGi=float(_num(data["QGi"])),
            PLi=float(_num(data["PLi"])),
            QLi=float(_num(data["QLi"])),
            Qmin=float(_num(data["Qmin"])),
            Qmax=float(_num(data["Qmax"])),
            AreaNo=int(_num(data["AreaNo"])),
            AcDc=int(_num(data["AcDc"])),
        )


@dataclass(frozen=True)
class NetworkLine:
    FromBus: int
    ToBus: int
    R: float
    wL: float
    wC: float
    G: float
    TurnsRatio: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NetworkLine":
        return cls(
            FromBus=int(_num(data["FromBus"])),
            ToBus=int(_num(data["ToBus"])),
            R=float(_num(data["R"])),
            wL=float(_num(data["wL"])),
            wC=float(_num(data["wC"])),
            G=float(_num(data["G"])),
            TurnsRatio=float(_num(data["TurnsRatio"])),
        )


@dataclass(frozen=True)
class NetworkLineIEEE:
    FromBus: int
    ToBus: int
    R: float
    X: float
    B: float
    G: float
    TurnsRatio: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NetworkLineIEEE":
        return cls(
            FromBus=int(_num(data["FromBus"])),
            ToBus=int(_num(data["ToBus"])),
            R=float(_num(data["R"])),
            X=float(_num(data["X"])),
            B=float(_num(data["B"])),
            G=float(_num(data["G"])),
            TurnsRatio=float(_num(data["TurnsRatio"])),
        )


@dataclass(frozen=True)
class Apparatus:
    BusNo: tuple[int, ...]
    Type: int
    Para: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Apparatus":
        bus = data["BusNo"]
        if isinstance(bus, list):
            buses = tuple(int(_num(item)) for item in bus)
        else:
            buses = (int(_num(bus)),)
        para = data.get("Para", {})
        return cls(BusNo=buses, Type=int(_num(data["Type"])), Para=dict(para) if isinstance(para, dict) else {})


@dataclass(frozen=True)
class CaseData:
    Basic: Basic
    Advance: Advance
    Bus: list[Bus]
    NetworkLine: list[NetworkLine]
    NetworkLineIEEE: list[NetworkLineIEEE]
    Apparatus: list[Apparatus]
    raw: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaseData":
        return cls(
            Basic=Basic.from_dict(data["Basic"]),
            Advance=Advance.from_dict(data.get("Advance", {})),
            Bus=[Bus.from_dict(item) for item in _as_records(data.get("Bus", []))],
            NetworkLine=[NetworkLine.from_dict(item) for item in _as_records(data.get("NetworkLine", []))],
            NetworkLineIEEE=[NetworkLineIEEE.from_dict(item) for item in _as_records(data.get("NetworkLineIEEE", []))],
            Apparatus=[Apparatus.from_dict(item) for item in _as_records(data.get("Apparatus", []))],
            raw=data,
        )
