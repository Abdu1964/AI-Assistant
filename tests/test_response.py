
# test_api.py
import pytest
import os
import jwt
import io
import json
from dotenv import load_dotenv
from app import create_app

# Load test environment variables (only if .env exists)
if os.path.exists('.env'):
    load_dotenv('.env')

# Generate token once at module level
JWT_SECRET = os.getenv("JWT_SECRET")
if JWT_SECRET is None:
    raise ValueError("JWT_SECRET environment variable is not set. Check your GitHub secrets or .env file.")

TEST_USER_ID = "some_id"
TEST_TOKEN = jwt.encode({"user_id": TEST_USER_ID}, JWT_SECRET, algorithm="HS256")

@pytest.fixture
def client():
    """Create test client with test config"""
    app,socket = create_app()
    with app.test_client() as client:
        yield client

@pytest.fixture
def auth_headers():
    """Return headers with valid auth token"""
    return {
        'Authorization': f'Bearer {TEST_TOKEN}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }

def test_health_check(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.json == "This is health check"

def test_query_endpoint(client, auth_headers):
    """Test endpoint with valid token"""
    response = client.post(
        '/query',
        headers=auth_headers,
        data={
            "query": "hello"
            }
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response Data: {response.get_data(as_text=True)}")
    print(f"Response Headers: {dict(response.headers)}")
    assert response.status_code == 200

def test_query_responses(client, auth_headers):
    """Test various query types"""
    test_cases = [
        {"input": {"query": "explain gene FTO"}, "expected_key": "json_format"},
        {"input": {"query": "what is Rejuve Bio"}, "expected_key": "text"}
    ]

    for case in test_cases:
        response = client.post(
            '/query',
            headers=auth_headers,
            data={
                **case["input"],
                "context": json.dumps({"id": None, "resource": "annotation"})
            }
        )
        assert response.status_code == 200
        assert case["expected_key"] in response.json
