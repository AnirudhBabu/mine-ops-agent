"""Mock Oracle Fusion-style REST API.

Loosely modelled on Fusion Cloud REST conventions so the agent integration
code looks like the real thing:

* resources under /fscmRestApi/resources/latest/<resource>
* collection responses shaped as {"items": [...], "count": n, "hasMore": false}
* simple ``q=Field=value;Field2=value`` row filters
* POST creates a row and returns it with a generated identifier

This is NOT the Oracle API. Field names and resource paths are simplified.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

SEED_PATH = Path(__file__).resolve().parent.parent / "data" / "seed.json"
BASE = "/fscmRestApi/resources/latest"


class Store:
    """In-memory data store, reset from the seed file for every app instance."""

    def __init__(self, seed_path: Path = SEED_PATH) -> None:
        self._seed = json.loads(seed_path.read_text())
        self.reset()

    def reset(self) -> None:
        self.data: dict[str, Any] = copy.deepcopy(self._seed)
        self._next_wo = 5100
        self._next_req = 9000

    def next_work_order_id(self) -> int:
        self._next_wo += 1
        return self._next_wo

    def next_requisition_id(self) -> int:
        self._next_req += 1
        return self._next_req


def _filter(rows: list[dict], q: str | None) -> list[dict]:
    if not q:
        return rows
    clauses = []
    for part in q.split(";"):
        if "=" not in part:
            raise HTTPException(400, f"Invalid q clause: {part!r}")
        field, value = part.split("=", 1)
        clauses.append((field.strip(), value.strip()))
    return [r for r in rows if all(str(r.get(f)) == v for f, v in clauses)]


def _collection(rows: list[dict]) -> dict:
    return {"items": rows, "count": len(rows), "hasMore": False}


class WorkOrderIn(BaseModel):
    AssetId: str
    Description: str
    WorkOrderType: str = "PREVENTIVE"
    Priority: str = Field("MEDIUM", pattern="^(LOW|MEDIUM|HIGH)$")


class RequisitionLineIn(BaseModel):
    ItemNumber: str
    Quantity: int = Field(gt=0)


class RequisitionIn(BaseModel):
    Description: str
    Priority: str = Field("MEDIUM", pattern="^(LOW|MEDIUM|HIGH)$")
    RequestedBy: str
    ApprovedBy: str | None = None
    lines: list[RequisitionLineIn] = Field(min_length=1)


def create_app(store: Store | None = None) -> FastAPI:
    store = store or Store()
    app = FastAPI(title="Mock Fusion-style Maintenance and Procurement API")
    app.state.store = store

    @app.get(f"{BASE}/assets")
    def list_assets(q: str | None = Query(None)):
        return _collection(_filter(store.data["assets"], q))

    @app.get(f"{BASE}/assets/{{asset_id}}")
    def get_asset(asset_id: str):
        for a in store.data["assets"]:
            if a["AssetId"] == asset_id:
                return a
        raise HTTPException(404, f"Asset {asset_id} not found")

    @app.get(f"{BASE}/maintenanceWorkOrders")
    def list_work_orders(q: str | None = Query(None)):
        return _collection(_filter(store.data["workOrders"], q))

    @app.post(f"{BASE}/maintenanceWorkOrders", status_code=201)
    def create_work_order(body: WorkOrderIn):
        get_asset(body.AssetId)
        wo_id = store.next_work_order_id()
        row = {"WorkOrderId": wo_id, "WorkOrderNumber": f"WO-{wo_id}", "Status": "OPEN", **body.model_dump()}
        store.data["workOrders"].append(row)
        return row

    @app.get(f"{BASE}/inventoryItems")
    def list_items(q: str | None = Query(None)):
        return _collection(_filter(store.data["inventoryItems"], q))

    @app.get(f"{BASE}/serviceKits/{{asset_type}}")
    def get_service_kit(asset_type: str):
        kit = store.data["serviceKits"].get(asset_type)
        if kit is None:
            raise HTTPException(404, f"No service kit for {asset_type}")
        return _collection(kit)

    @app.get(f"{BASE}/purchaseRequisitions")
    def list_requisitions(q: str | None = Query(None)):
        return _collection(_filter(store.data["purchaseRequisitions"], q))

    @app.post(f"{BASE}/purchaseRequisitions", status_code=201)
    def create_requisition(body: RequisitionIn):
        items = {i["ItemNumber"]: i for i in store.data["inventoryItems"]}
        lines, total = [], 0.0
        for n, line in enumerate(body.lines, start=1):
            item = items.get(line.ItemNumber)
            if item is None:
                raise HTTPException(422, f"Unknown item {line.ItemNumber}")
            amount = round(item["UnitPrice"] * line.Quantity, 2)
            total += amount
            lines.append({"LineNumber": n, "ItemNumber": line.ItemNumber, "Quantity": line.Quantity,
                          "UnitPrice": item["UnitPrice"], "Amount": amount})
        req_id = store.next_requisition_id()
        row = {
            "RequisitionHeaderId": req_id,
            "Requisition": f"PR-{req_id}",
            "Description": body.Description,
            "Priority": body.Priority,
            "RequestedBy": body.RequestedBy,
            "ApprovedBy": body.ApprovedBy,
            "DocumentStatus": "PENDING_APPROVAL" if body.ApprovedBy is None else "APPROVED",
            "Currency": "SAR",
            "TotalAmount": round(total, 2),
            "lines": lines,
        }
        store.data["purchaseRequisitions"].append(row)
        return row

    return app


app = create_app()
