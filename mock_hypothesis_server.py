from flask import Flask, request, jsonify
import uuid
import time
import datetime

app = Flask(__name__)

# In-memory storage for mock data
hypothesis_requests = {}

@app.route('/api/mock/hypothesis/enrich', methods=['POST'])
def start_enrich():
    """Step 1: Start enrichment"""
    req_id = f"hyp_{uuid.uuid4().hex[:8]}"
    hypothesis_requests[req_id] = {
        "status": "pending",
        "created_at": datetime.datetime.utcnow(),
        "enrich_id": f"enrich_{uuid.uuid4().hex[:8]}"
    }
    return jsonify({
        "hypothesis_id": req_id,
        "project_id": request.json.get("project_id")
    }), 202

@app.route('/api/mock/hypothesis/hypothesis', methods=['GET'])
def check_status():
    """Step 2: Check status (Polling)"""
    req_id = request.args.get("id")
    if not req_id or req_id not in hypothesis_requests:
        return jsonify({"error": "Hypothesis ID not found"}), 404
    
    data = hypothesis_requests[req_id]
    
    # Simulate processing time (e.g., 5 seconds)
    time_diff = (datetime.datetime.utcnow() - data["created_at"]).total_seconds()
    if time_diff > 5:
        data["status"] = "completed"
    
    return jsonify({
        "id": req_id,
        "status": data["status"],
        "phenotype": "Obesity",
        "enrich_id": data["enrich_id"]
    })

@app.route('/api/mock/hypothesis/enrich', methods=['GET'])
def get_enrich():
    """Step 3: Get enrichment results"""
    return jsonify({
        "id": request.args.get("id"),
        "causal_gene": "FTO",
        "GO_terms": [
            {"id": "GO:1904177", "name": "Regulation of Adipose Tissue Development"}
        ]
    })

@app.route('/api/mock/hypothesis/hypothesis', methods=['POST'])
def final_hypothesis():
    """Step 4: Generate final hypothesis"""
    return jsonify({
        "summary": "Mocked hypothesis summary",
        "graph": {
            "nodes": [
                {"id": "rs1421085", "type": "snp", "name": "rs1421085"},
                {"id": "FTO", "type": "gene", "name": "FTO"},
                {"id": "GO:1904177", "type": "go", "name": "Regulation of Adipose Tissue Development"}
            ],
            "edges": [
                {"source": "rs1421085", "target": "FTO", "label": "affects"},
                {"source": "FTO", "target": "GO:1904177", "label": "involved_in"}
            ],
            "probability": 0.95
        }
    }), 201

if __name__ == "__main__":
    print("Starting Mock Hypothesis Server on port 9001...")
    app.run(host='0.0.0.0', port=9001, debug=True)
