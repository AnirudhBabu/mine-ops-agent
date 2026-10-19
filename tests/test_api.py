from fastapi.testclient import TestClient

from mine_ops_agent.fusion_api import BASE, create_app


def client():
    return TestClient(create_app())


def test_collection_shape_and_filter():
    body = client().get(f"{BASE}/assets", params={"q": "AssetType=HAUL_TRUCK;Site=North Pit"}).json()
    assert set(body) == {"items", "count", "hasMore"}
    assert [a["AssetId"] for a in body["items"]] == ["HT-101", "HT-102"]


def test_bad_filter_is_400():
    assert client().get(f"{BASE}/assets", params={"q": "nonsense"}).status_code == 400


def test_requisition_prices_lines_from_catalogue():
    r = client().post(f"{BASE}/purchaseRequisitions", json={
        "Description": "test", "RequestedBy": "t", "lines": [{"ItemNumber": "FLT-OIL-240", "Quantity": 3}]})
    assert r.status_code == 201
    assert r.json()["TotalAmount"] == 255.0
    assert r.json()["DocumentStatus"] == "PENDING_APPROVAL"


def test_unknown_item_and_asset_rejected():
    c = client()
    assert c.post(f"{BASE}/purchaseRequisitions", json={
        "Description": "x", "RequestedBy": "t", "lines": [{"ItemNumber": "NOPE", "Quantity": 1}]}).status_code == 422
    assert c.post(f"{BASE}/maintenanceWorkOrders", json={"AssetId": "XX-999", "Description": "x"}).status_code == 404
